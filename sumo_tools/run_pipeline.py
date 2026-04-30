#!/usr/bin/env python3
"""
run_pipeline.py

Full route-file generation pipeline.  Run from the project root or from
sumo_tools/ — the script resolves the correct working directory automatically.

Step order
----------
1. gen_routes.py            — generate light / moderate / peak route files
2. gen_fringe_supplement.py — add fringe cars from 135488883#0
3. classify_traffic.py      — assign realistic vehicle-type mix (x3 files)
4. generate_rainy_xml.py    — build rain_m and rain_h from peak base (x2)
5. gen_accidents_routes.py  — build acc1-4 from moderate base
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# ── Resolve project root (parent of this script's directory) ────────────────
_HERE        = Path(__file__).resolve().parent   # .../sumo_tools
PROJECT_ROOT = _HERE.parent                      # .../ABSTLight_for_GCP
SUMO_FILES   = PROJECT_ROOT / "sumo_files"


def run(description: str, args: list[str]) -> None:
    """Run a subprocess from PROJECT_ROOT; abort the pipeline on failure."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  cmd: {' '.join(args)}")
    print(f"{'='*60}")
    result = subprocess.run(args, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"\nERROR: step failed with exit code {result.returncode}. "
              "Pipeline aborted.", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    python = sys.executable   # same interpreter that launched this script

    # ── Step 1: base route files ────────────────────────────────────────────
    run(
        "Step 1/5 — gen_routes.py: generate light / moderate / peak",
        [python, "sumo_tools/gen_routes.py"],
    )

    # ── Step 2: fringe supplement ───────────────────────────────────────────
    run(
        "Step 2/5 — gen_fringe_supplement.py: add fringe cars from 135488883#0",
        [python, "sumo_tools/gen_fringe_supplement.py"],
    )

    # ── Step 3: vehicle-type classification (in-place, seed=42) ────────────
    for scenario in ("light", "moderate", "peak"):
        route_file = f"sumo_files/osm.{scenario}.rou.xml"
        run(
            f"Step 3/5 — classify_traffic.py: {scenario}",
            [
                python, "sumo_tools/classify_traffic.py",
                "--input",  route_file,
                "--output", route_file,
                "--seed", "42",
            ],
        )

    # ── Step 4: rainy scenarios (derived from peak) ─────────────────────────
    for condition, out_name in [
        ("Moderate Rain", "osm.rain_m.rou.xml"),
        ("Heavy Rain",    "osm.rain_h.rou.xml"),
    ]:
        run(
            f"Step 4/5 — generate_rainy_xml.py: {condition}",
            [
                python, "sumo_tools/generate_rainy_xml.py",
                "--num-vehicles",  "2000",
                "--output-file",   str(SUMO_FILES / out_name),
                "--rain-condition", condition,
            ],
        )

    # ── Step 5: accident scenarios (derived from moderate) ──────────────────
    run(
        "Step 5/5 — gen_accidents_routes.py: acc1 / acc2 / acc3 / acc4",
        [python, "sumo_tools/gen_accidents_routes.py"],
    )

    print("\n" + "="*60)
    print("  Pipeline complete.  All route files are up to date.")
    print("="*60)


if __name__ == "__main__":
    main()
