"""Run all test.py models over a fixed seed set for one route file."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

SEEDS = [42, 100, 1234, 2026, 8888]
MODELS = [1, 2, 3, 4]

MODEL_LABEL_MAP = {
    "SUMO_DEFAULT": "Actuated",
    "SUMO_FIXED_TIME": "Fixed-Time",
    "ABSTLIGHT": "ABSTLight_Base",
    "ABSTLIGHT_FINETUNED": "ABSTLight_FT",
}

TARGET_METRICS = [
    "arrived_vehicles",
    "avg_delay_per_vehicle",
    "max_queue_length",
    "queue_length_variance",
    "time_loss",
    "phase_switch_count",
    "teleportation_by_collisions",
]

_ROUTE_RESULTS_SUBDIR_MAP = {
    "osm.moderate.rou.xml": Path("normal"),
    "osm.peak.rou.xml": Path("peak-hours"),
    "osm.rain_m.rou.xml": Path("rainy") / "moderate_rain",
    "osm.rain_h.rou.xml": Path("rainy") / "heavy_rain",
    "osm.acc1.rou.xml": Path("accident") / "acc1",
    "osm.acc2.rou.xml": Path("accident") / "acc2",
    "osm.acc3.rou.xml": Path("accident") / "acc3",
    "osm.acc4.rou.xml": Path("accident") / "acc4",
}


def _route_results_dir(route_file: str) -> Path:
    route_name = Path(route_file).name.lower()
    subdir = _ROUTE_RESULTS_SUBDIR_MAP.get(route_name, Path("misc"))
    return Path("results") / subdir


def _route_summary_name(route_file: str) -> str:
    route_name = Path(route_file).name.lower()
    if route_name.endswith(".rou.xml"):
        return route_name[: -len(".rou.xml")].replace(".", "_")
    return Path(route_name).stem.replace(".", "_")


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def aggregate_seed_results(target_dir: Path, route_file: str) -> Path:
    """Aggregate per-seed test results into one summary JSON per route."""
    target_dir.mkdir(parents=True, exist_ok=True)

    # model_name -> seed -> (mtime, run_entry)
    grouped: dict[str, dict[int, tuple[float, dict]]] = {}

    for file_path in sorted(target_dir.glob("test_results_*.json")):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        model_type = str(payload.get("model_type", "")).upper()
        if model_type not in MODEL_LABEL_MAP:
            continue

        payload_route = str(payload.get("route_file", ""))
        if payload_route != route_file:
            continue

        seed_value = payload.get("seed", None)
        if seed_value is None:
            continue

        try:
            seed_int = int(seed_value)
        except (TypeError, ValueError):
            continue

        if seed_int not in SEEDS:
            continue

        aggregate = payload.get("aggregate", {})
        if not isinstance(aggregate, dict):
            continue

        metrics = {key: _safe_float(aggregate.get(key, 0.0)) for key in TARGET_METRICS}
        run_entry = {
            "seed": seed_int,
            "file": str(file_path).replace("\\", "/"),
            "metrics": metrics,
        }

        model_name = MODEL_LABEL_MAP[model_type]
        grouped.setdefault(model_name, {})
        current = grouped[model_name].get(seed_int)
        mtime = file_path.stat().st_mtime
        if current is None or mtime >= current[0]:
            grouped[model_name][seed_int] = (mtime, run_entry)

    final_summary: dict[str, object] = {
        "route_file": route_file,
        "target_dir": str(target_dir).replace("\\", "/"),
        "seeds_expected": SEEDS,
        "models": {},
    }

    models_out: dict[str, object] = {}
    for model_name in sorted(grouped.keys()):
        runs_by_seed = [grouped[model_name][seed][1] for seed in sorted(grouped[model_name].keys())]
        statistical_summary: dict[str, dict[str, float]] = {}

        for metric in TARGET_METRICS:
            values = [float(run["metrics"].get(metric, 0.0)) for run in runs_by_seed]
            if values:
                mean_value = float(statistics.fmean(values))
                std_value = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
            else:
                mean_value = 0.0
                std_value = 0.0
            statistical_summary[metric] = {
                "mean": mean_value,
                "std_dev": std_value,
            }

        models_out[model_name] = {
            "runs_by_seed": runs_by_seed,
            "statistical_summary": statistical_summary,
        }

    final_summary["models"] = models_out

    summary_name = f"final_summary_{_route_summary_name(route_file)}.json"
    summary_path = target_dir / summary_name
    summary_path.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    return summary_path


def _scenario_flags(route_file: str, model: int) -> list[str]:
    """Infer scenario flags from route file name for model-specific behavior."""
    route_lower = route_file.lower()
    flags: list[str] = []

    # Only model 4 (ABSTLight_FT) receives explicit scenario flags.
    if model != 4:
        return flags

    if "acc" in route_lower:
        flags.append("--accident")
    elif "rain" in route_lower:
        flags.append("--rainy")

    return flags


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run test.py for all models (1/2/3/4) using fixed seeds "
            "[42, 100, 1234, 2026, 8888]."
        )
    )
    parser.add_argument("--route-file", required=True, help="Route file passed to test.py")
    args = parser.parse_args()

    route_file = args.route_file
    test_script = Path(__file__).resolve().parent / "test.py"
    target_dir = _route_results_dir(route_file)

    if not test_script.exists():
        raise FileNotFoundError(f"Cannot find test.py at: {test_script}")

    total_runs = len(SEEDS) * len(MODELS)
    run_idx = 0

    print("=" * 72)
    print("Seed Suite Runner")
    print("=" * 72)
    print(f"Route file   : {route_file}")
    print(f"Results dir  : {target_dir}")
    print(f"Seeds        : {SEEDS}")
    print(f"Models       : {MODELS}")
    print(f"Total runs   : {total_runs}")
    print("=" * 72)

    for seed in SEEDS:
        for model in MODELS:
            run_idx += 1
            cmd = [
                sys.executable,
                str(test_script),
                "--model",
                str(model),
                "--route-file",
                route_file,
                "--seed",
                str(seed),
                "--test-episodes",
                "1",
            ]
            cmd.extend(_scenario_flags(route_file, model))

            print(f"\n[{run_idx}/{total_runs}] model={model} seed={seed}")
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)

            summary_path = aggregate_seed_results(target_dir=target_dir, route_file=route_file)
            print("\nAll seed-suite runs completed.")
            print(f"Aggregated summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
