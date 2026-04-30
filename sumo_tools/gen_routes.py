#!/usr/bin/env python3
"""
generate_clean_routes.py

Generate SUMO route files for three traffic scenarios using a bell-curve profile:
  warm-up (0-1000s) -> surge (1000-2000s) -> cooldown (2000-3000s)
with a strict flush window from 3000s to 3600s (no new vehicle departures).

For each scenario:
1) Call randomTrips.py three times (one per interval) to create temp trips
2) Merge temp trips into one trip file
3) Run duarouter to create final .rou.xml
4) Post-filter vehicles whose route has <= 2 edges
"""

from __future__ import annotations

import os
import sys
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple


# ------------------------- Global Config -------------------------

MIN_DISTANCE = 500
RNG_SEED = 42

# Bell-curve generation window and flush window
GEN_START = 0
GEN_END = 3000
FLUSH_END = 3600  # no generation in [3000, 3600]

INTERVALS: List[Tuple[int, int, str]] = [
    (0, 600, "warmup"),
    (600, 2400, "surge"),
    (2400, 3000, "cooldown"),
]

# Scenario profile: interval counts + output names
SCENARIOS = {
    "light": {
        "counts": [240, 720, 240],  # total ~1200
        "trips_file": "osm.light.trips.xml",
        "route_file": "osm.light.rou.xml",
    },
    "moderate": {
        "counts": [270, 1080, 270],  # total ~1600
        "trips_file": "osm.moderate.trips.xml",
        "route_file": "osm.moderate.rou.xml",
    },
    "peak": {
        "counts": [300, 1440, 300],  # total ~2040
        "trips_file": "osm.peak.trips.xml",
        "route_file": "osm.peak.rou.xml",
    },
}

DEFAULT_NET_FILE = "osm.net.xml.gz"


# ------------------------- Utility -------------------------

def period_from_target(target_count: int, begin: int, end: int) -> float:
    """
    Compute period p such that expected vehicles ~= (end - begin) / p.
    """
    if target_count <= 0:
        raise ValueError("target_count must be > 0")
    duration = end - begin
    if duration <= 0:
        raise ValueError("Invalid interval duration")
    return duration / float(target_count)


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def get_sumo_tools_paths() -> Tuple[Path, str]:
    """
    Returns:
      randomTrips.py path, duarouter executable path
    """
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise EnvironmentError("SUMO_HOME is not set.")

    random_trips = Path(sumo_home) / "tools" / "randomTrips.py"
    if not random_trips.exists():
        raise FileNotFoundError(f"Cannot find randomTrips.py at: {random_trips}")

    duarouter = shutil.which("duarouter")
    if duarouter is None:
        raise FileNotFoundError("duarouter not found in PATH.")

    return random_trips, duarouter


def count_vehicles_in_xml(xml_file: Path) -> int:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return len(root.findall("trip")) + len(root.findall("vehicle"))


# ------------------------- Trip Generation -------------------------

def generate_interval_trips(
    *,
    python_exec: str,
    random_trips_py: Path,
    net_file: Path,
    begin: int,
    end: int,
    target_count: int,
    output_trips: Path,
    seed: int,
    prefix: str,
    cwd: Path,
) -> None:
    """
    Generate one interval trip file using randomTrips.py.
    """
    p = period_from_target(target_count, begin, end)

    cmd = [
        python_exec,
        str(random_trips_py),
        "-n", str(net_file),
        "-o", str(output_trips),         # trips output
        "-b", str(begin),
        "-e", str(end),
        "-p", f"{p:.10f}",               # exact period from target
        "--min-distance", str(MIN_DISTANCE),
        "--seed", str(seed),
        "--prefix", prefix,
        "--validate",
    ]
    run_cmd(cmd, cwd=cwd)


def merge_trip_files(temp_trip_files: List[Path], merged_trip_file: Path) -> int:
    """
    Merge multiple trips files into one <routes> XML.
    Keeps all <trip> elements.
    """
    routes_root = ET.Element("routes")
    total_trips = 0

    for tf in temp_trip_files:
        tree = ET.parse(tf)
        root = tree.getroot()
        for trip in root.findall("trip"):
            routes_root.append(trip)
            total_trips += 1

    merged_tree = ET.ElementTree(routes_root)
    merged_tree.write(merged_trip_file, encoding="utf-8", xml_declaration=True)
    return total_trips


def run_duarouter(net_file: Path, merged_trip_file: Path, route_file: Path, cwd: Path) -> None:
    """
    Build final routes from merged trips via duarouter.
    """
    cmd = [
        "duarouter",
        "-n", str(net_file),
        "-r", str(merged_trip_file),
        "-o", str(route_file),
        "--ignore-errors",
    ]
    run_cmd(cmd, cwd=cwd)


# ------------------------- Route Post-Filter -------------------------

def filter_short_routes(route_file: Path) -> int:
    """
    Remove any <vehicle> with route edges length <= 2.
    Condition:
      len(route_edges.split()) <= 2  => remove vehicle
    """
    tree = ET.parse(route_file)
    root = tree.getroot()

    removed = 0
    for vehicle in list(root.findall("vehicle")):
        route = vehicle.find("route")
        if route is None:
            # Strict mode: malformed vehicle removed.
            root.remove(vehicle)
            removed += 1
            continue

        edges = (route.get("edges") or "").strip().split()
        if len(edges) <= 2:
            root.remove(vehicle)
            removed += 1

    tree.write(route_file, encoding="utf-8", xml_declaration=True)
    return removed


def count_final_vehicles(route_file: Path) -> int:
    tree = ET.parse(route_file)
    root = tree.getroot()
    return len(root.findall("vehicle"))


def cleanup_intermediate_files(work_dir: Path) -> None:
    """
    Delete intermediate route-generation artifacts after final .rou.xml files exist.

    Removes:
      - *.trips.xml
      - *.rou.alt.xml
    """
    patterns = ("*.trips.xml", "*.rou.alt.xml")
    removed = 0

    for pattern in patterns:
        for path in work_dir.glob(pattern):
            if path.is_file():
                try:
                    path.unlink()
                    removed += 1
                    print(f"  Cleanup: removed {path.name}")
                except Exception as exc:
                    print(f"  Cleanup warning: could not remove {path.name}: {exc}")

    if removed == 0:
        print("  Cleanup: no *.trips.xml or *.rou.alt.xml files found.")
    else:
        print(f"  Cleanup done: removed {removed} intermediate files.")


# ------------------------- Main -------------------------

def build_scenario(
    scenario_name: str,
    scenario_cfg: Dict[str, object],
    random_trips_py: Path,
    net_file: Path,
    work_dir: Path,
) -> None:
    counts: List[int] = scenario_cfg["counts"]  # type: ignore[assignment]
    merged_trip_name: str = scenario_cfg["trips_file"]  # type: ignore[assignment]
    final_route_name: str = scenario_cfg["route_file"]  # type: ignore[assignment]

    if len(counts) != len(INTERVALS):
        raise ValueError(f"{scenario_name}: counts length must be {len(INTERVALS)}")

    print(f"\n=== Scenario: {scenario_name.upper()} ===")
    print(f"Target interval counts: {counts} (total ~{sum(counts)})")
    print(f"Generation window: {GEN_START}-{GEN_END}s; flush window: {GEN_END}-{FLUSH_END}s")

    temp_files: List[Path] = []

    # 1) Generate 3 temporary interval trip files
    for idx, ((begin, end, label), target_count) in enumerate(zip(INTERVALS, counts), start=1):
        temp_trip = work_dir / f"tmp.{scenario_name}.{idx}.{label}.trips.xml"
        temp_files.append(temp_trip)

        period = period_from_target(target_count, begin, end)
        print(
            f"  Interval {idx} [{begin}-{end}] {label}: "
            f"target~{target_count}, period={period:.6f}"
        )

        generate_interval_trips(
            python_exec=sys.executable,
            random_trips_py=random_trips_py,
            net_file=net_file,
            begin=begin,
            end=end,
            target_count=target_count,
            output_trips=temp_trip,
            seed=RNG_SEED + idx,          # deterministic but distinct per interval
            prefix=f"i{idx}_",            # unique ID namespace per interval
            cwd=work_dir,
        )

    # 2) Merge temp trips
    merged_trip_file = work_dir / merged_trip_name
    merged_count = merge_trip_files(temp_files, merged_trip_file)
    print(f"  Merged trips: {merged_count} -> {merged_trip_file.name}")

    # 3) Run duarouter
    final_route_file = work_dir / final_route_name
    run_duarouter(net_file, merged_trip_file, final_route_file, work_dir)

    # 4) Strict post-filter for short/dead-end-like routes
    removed = filter_short_routes(final_route_file)
    final_count = count_final_vehicles(final_route_file)
    print(
        f"  Post-filter: removed={removed} vehicles with <=2 edges; "
        f"final_vehicles={final_count} -> {final_route_file.name}"
    )

    # 5) Cleanup temporary interval files
    for tf in temp_files:
        if tf.exists():
            tf.unlink()


def main() -> int:
    work_dir = Path(__file__).resolve().parent.parent / "sumo_files"
    net_file = work_dir / DEFAULT_NET_FILE

    if not net_file.exists():
        print(f"ERROR: network file not found: {net_file}", file=sys.stderr)
        return 1

    try:
        random_trips_py, _ = get_sumo_tools_paths()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("=== Bell-Curve SUMO Route Generation ===")
    print(f"Network: {net_file.name}")
    print(f"Windows: generate [{GEN_START}, {GEN_END}], flush ({GEN_END}, {FLUSH_END})")
    print(f"Min distance: {MIN_DISTANCE}")
    print("Scenarios: light(~700), moderate(~900), peak(~1100)")

    try:
        for name, cfg in SCENARIOS.items():
            build_scenario(name, cfg, random_trips_py, net_file, work_dir)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: subprocess failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nDone. Generated files:")
    for _, cfg in SCENARIOS.items():
        print(f" - {cfg['trips_file']}")
        print(f" - {cfg['route_file']}")

    # Final cleanup of intermediate artifacts not needed for runtime simulation.
    cleanup_intermediate_files(work_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())