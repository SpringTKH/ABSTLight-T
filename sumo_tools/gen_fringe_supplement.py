#!/usr/bin/env python3
"""
gen_fringe_supplement.py

Generates per-scenario fringe vehicles originating from fringe edge
135488883#0 (Jalan Tun Perak dead-end entry), which randomTrips.py
normally skips due to the --min-distance constraint.

Per-file targets – bell-curve profile over 3000 steps (ratio 1:4:1):
    osm.peak.rou.xml     : 750  (warmup 125 / surge 500 / cooldown 125)
    osm.moderate.rou.xml : 600  (warmup 100 / surge 400 / cooldown 100)
    osm.light.rou.xml    : 450  (warmup  75 / surge 300 / cooldown  75)

rain_h / rain_m / acc1-4 are NOT processed here — they are re-derived
from peak/moderate by generate_rainy_xml.py and gen_accidents_routes.py.

Vehicle IDs use prefixes f1_/f2_/f3_ to avoid collisions with i1_/i2_/i3_.
Re-running the script strips any existing fringe vehicles and replaces them,
so updated counts are always applied correctly.
"""

from __future__ import annotations

import gzip
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

# ── Config ───────────────────────────────────────────────────────────────────

SOURCE_EDGE     = "135488883#0"
RNG_SEED        = 99          # separate from gen_routes.py (seed 42)
FRINGE_PREFIXES = ("f1_", "f2_", "f3_")

# Bell-curve interval windows (begin_s, end_s)  — ratio 1 : 4 : 1
_WINDOWS = [(0, 600), (600, 2400), (2400, 3000)]
_LABELS  = ["warmup", "surge", "cooldown"]
_PFXS    = ["f1_", "f2_", "f3_"]

# Per-file vehicle targets.  rain_h/rain_m/acc1-4 inherit from peak/moderate.
TARGET_ROUTE_FILES: Dict[str, int] = {
    "osm.peak.rou.xml":     600,
    "osm.moderate.rou.xml": 480,
    "osm.light.rou.xml":    360,
}

DEFAULT_NET_FILE = "osm.net.xml.gz"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_valid_sink_edges(net_file: Path) -> List[str]:
    """Non-internal edges that are incoming to at least one junction."""
    if net_file.suffix == ".gz":
        with gzip.open(net_file, "rb") as f:
            tree = ET.parse(f)
    else:
        tree = ET.parse(net_file)
    root = tree.getroot()
    incoming: set = set()
    for junc in root.findall("junction"):
        for lane_id in (junc.get("incLanes") or "").split():
            incoming.add(lane_id.rsplit("_", 1)[0])
    return [
        e.get("id", "") for e in root.findall("edge")
        if not e.get("id", "").startswith(":")
        and e.get("id") != SOURCE_EDGE
        and e.get("id") in incoming
    ]


def compute_intervals(total: int) -> List[Tuple[int, int, str, int, str]]:
    """
    Return (begin, end, label, count, prefix) for each interval.
    Ratio 1:4:1 — warmup and cooldown are each 1/6 of total, surge is 4/6.
    """
    unit     = total / 6
    warmup   = round(unit)
    cooldown = round(unit)
    surge    = total - warmup - cooldown
    return [
        (*_WINDOWS[i], _LABELS[i], cnt, _PFXS[i])
        for i, cnt in enumerate([warmup, surge, cooldown])
    ]


# ── Trip generation + routing ─────────────────────────────────────────────────

def build_trip_file(
    sink_edges: List[str],
    out_path: Path,
    begin: int, end: int,
    count: int, prefix: str,
    rng: random.Random,
) -> None:
    root = ET.Element("routes")
    step = (end - begin) / count
    for i in range(count):
        trip = ET.SubElement(root, "trip")
        trip.set("id",     f"{prefix}{i}")
        trip.set("depart", f"{begin + i * step:.2f}")
        trip.set("from",   SOURCE_EDGE)
        trip.set("to",     rng.choice(sink_edges))
    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)


def run_duarouter(net_file: Path, trips: Path, out: Path, cwd: Path) -> None:
    cmd = ["duarouter", "-n", str(net_file), "-r", str(trips),
           "-o", str(out), "--ignore-errors"]
    print("      >", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def filter_short_routes(route_file: Path) -> int:
    """Remove vehicles whose route has ≤ 2 edges. Returns number removed."""
    tree  = ET.parse(route_file)
    root  = tree.getroot()
    before = len(root.findall("vehicle"))
    for v in list(root.findall("vehicle")):
        r = v.find("route")
        if r is None or len((r.get("edges") or "").split()) <= 2:
            root.remove(v)
    ET.ElementTree(root).write(route_file, encoding="utf-8", xml_declaration=True)
    return before - len(root.findall("vehicle"))


def generate_fringe_pool(
    intervals: List[Tuple[int, int, str, int, str]],
    sinks: List[str],
    work_dir: Path,
    net_file: Path,
    rng: random.Random,
    tag: str,
) -> Path:
    """Route all intervals for one file tag; return merged fringe route path."""
    route_files: List[Path] = []
    for begin, end, label, count, prefix in intervals:
        trip_path  = work_dir / f"tmp.fringe.{tag}.{label}.trips.xml"
        route_path = work_dir / f"tmp.fringe.{tag}.{label}.rou.xml"
        build_trip_file(sinks, trip_path, begin, end, count, prefix, rng)
        run_duarouter(net_file, trip_path, route_path, work_dir)
        trip_path.unlink(missing_ok=True)
        alt = work_dir / f"tmp.fringe.{tag}.{label}.rou.alt.xml"
        if alt.exists():
            alt.unlink()
        route_files.append(route_path)

    merged_root = ET.Element("routes")
    for rf in route_files:
        for v in ET.parse(rf).getroot().findall("vehicle"):
            merged_root.append(v)
        rf.unlink(missing_ok=True)

    merged_path = work_dir / f"tmp.fringe.{tag}.rou.xml"
    ET.ElementTree(merged_root).write(merged_path, encoding="utf-8", xml_declaration=True)
    return merged_path


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_into_target(fringe_path: Path, target_file: Path) -> Tuple[int, int]:
    """
    Strip any existing fringe vehicles (f1_/f2_/f3_) from target, append
    new ones, re-sort ALL vehicles by depart time.
    Returns (stripped_count, added_count).
    """
    fringe_vehicles = ET.parse(fringe_path).getroot().findall("vehicle")
    target_tree     = ET.parse(target_file)
    target_root     = target_tree.getroot()

    # Strip old fringe vehicles so re-runs replace rather than double-add.
    old_fringe = [v for v in target_root.findall("vehicle")
                  if v.get("id", "").startswith(FRINGE_PREFIXES)]
    for v in old_fringe:
        target_root.remove(v)

    for v in fringe_vehicles:
        target_root.append(v)

    non_vehicles = [el for el in target_root if el.tag != "vehicle"]
    vehicles     = sorted(
        [el for el in target_root if el.tag == "vehicle"],
        key=lambda v: float(v.get("depart", 0)),
    )
    for el in list(target_root):
        target_root.remove(el)
    for el in non_vehicles:
        target_root.append(el)
    for v in vehicles:
        target_root.append(v)

    target_tree.write(target_file, encoding="utf-8", xml_declaration=True)
    return len(old_fringe), len(fringe_vehicles)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    work_dir       = Path(__file__).resolve().parent.parent / "sumo_files"
    net_file_plain = work_dir / "osm.net.xml"
    net_file       = net_file_plain if net_file_plain.exists() else work_dir / DEFAULT_NET_FILE
    if not net_file.exists():
        print(f"ERROR: network file not found in {work_dir}", file=sys.stderr)
        return 1

    print(f"=== Fringe Supplement: {SOURCE_EDGE} ===")
    print(f"Network : {net_file.name}")
    print("Targets :")
    for fname, total in TARGET_ROUTE_FILES.items():
        ivs = compute_intervals(total)
        warmup, surge, cooldown = ivs[0][3], ivs[1][3], ivs[2][3]
        print(f"  {fname:<30}  target={total}  "
              f"(warmup={warmup} / surge={surge} / cooldown={cooldown})")

    # ── 1. Scan network ──────────────────────────────────────────────────
    print("\n[1/3] Scanning network for valid sink edges...")
    sinks = load_valid_sink_edges(net_file)
    if not sinks:
        print("ERROR: no valid sink edges found.", file=sys.stderr)
        return 1
    print(f"  {len(sinks)} eligible sink edges")

    # ── 2. Per-file: generate pool → filter → merge ───────────────────────
    print("\n[2/3] Generating and merging fringe vehicles per file...")
    n = len(TARGET_ROUTE_FILES)
    for idx, (fname, total) in enumerate(TARGET_ROUTE_FILES.items(), 1):
        target = work_dir / fname
        if not target.exists():
            print(f"  [{idx}/{n}] SKIP : {fname} not found")
            continue

        tag       = fname.replace("osm.", "").replace(".rou.xml", "")
        intervals = compute_intervals(total)
        # Use a per-file RNG so each file's vehicles are reproducible.
        file_rng  = random.Random(RNG_SEED + idx * 1000)

        print(f"\n  [{idx}/{n}] {fname}  (target={total})")
        fringe_path = generate_fringe_pool(intervals, sinks, work_dir, net_file, file_rng, tag)

        removed   = filter_short_routes(fringe_path)
        surviving = len(ET.parse(fringe_path).getroot().findall("vehicle"))
        print(f"    Routing: {removed} removed after filtering, {surviving} survive")

        if surviving == 0:
            print(f"    WARNING: no vehicles survived for {fname}")
            fringe_path.unlink(missing_ok=True)
            continue

        stripped, added = merge_into_target(fringe_path, target)
        fringe_path.unlink(missing_ok=True)
        total_now   = len(ET.parse(target).getroot().findall("vehicle"))
        strip_note  = f"  (replaced {stripped} previous)" if stripped else ""
        print(f"    +{added} fringe vehicles → {total_now} total{strip_note}")

    # ── 3. Summary ────────────────────────────────────────────────────────
    print("\n[3/3] Done.  Files updated:")
    for fname in TARGET_ROUTE_FILES:
        print(f"  sumo_files/{fname}")
    print("\nNote: re-run gen_accidents_routes.py and generate_rainy_xml.py")
    print("      to propagate fringe vehicles into acc/rain route files.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
