"""Run the same TrustTrace experiment protocol under A0, A1, A2, A3 and A4.

Quick mode is the default and is diagnostic only. Use --full for the configured
full experiment. Results are saved separately per ablation and in a matrix index.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ABLATIONS = ("A0", "A1", "A2", "A3", "A4")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Run full configured scenario counts and tuning.")
    parser.add_argument("--trials", type=int, default=100, help="Optuna trials in full mode.")
    parser.add_argument("--output-dir", default="reports/ablation_matrix", help="Directory for JSON outputs.")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1")

    os.environ.setdefault("TRUSTTRACE_STRICT_EVAL", "1")
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from experiment.full_pipeline import run_full_experiment

    config_path = ROOT / "config.yaml"
    original_config = config_path.read_bytes() if config_path.exists() else None
    config_hash = hashlib.sha256(original_config).hexdigest() if original_config is not None else None

    started = dt.datetime.now(dt.timezone.utc).isoformat()
    matrix = {
        "project": "TrustTrace",
        "started_utc": started,
        "mode": "full" if args.full else "quick_diagnostic",
        "ablation_order": list(ABLATIONS),
        "fixed_config_sha256": config_hash,
        "results": {},
    }

    previous_ablation = os.environ.get("TRUSTTRACE_ABLATION")
    failures = []
    try:
        for ablation_id in ABLATIONS:
            print(f"\n========== RUNNING {ablation_id} ==========")
            os.environ["TRUSTTRACE_ABLATION"] = ablation_id
            # Each ablation starts from the same original configuration. This
            # prevents tuning or side effects from A0 changing A1–A4 settings.
            if original_config is not None:
                config_path.write_bytes(original_config)
            try:
                result = run_full_experiment(
                    quick=not args.full,
                    optuna_trials=args.trials if args.full else 3,
                )
                effective_config = config_path.read_bytes() if config_path.exists() else b""
                effective_config_hash = hashlib.sha256(effective_config).hexdigest() if effective_config else None
                scenarios = result.get("scenarios", [])
                scenario_manifest = json.dumps(scenarios, sort_keys=True, separators=(",", ":"), default=str)
                scenario_manifest_hash = hashlib.sha256(scenario_manifest.encode("utf-8")).hexdigest()
                previous_scenario_hash = matrix.get("scenario_manifest_sha256")
                if previous_scenario_hash and previous_scenario_hash != scenario_manifest_hash:
                    raise RuntimeError(
                        f"Scenario manifest changed between ablations: {previous_scenario_hash} != {scenario_manifest_hash}"
                    )
                matrix["scenario_manifest_sha256"] = scenario_manifest_hash
                record = {
                    "status": "completed",
                    "ablation_id": ablation_id,
                    "effective_config_sha256": effective_config_hash,
                    "scenario_manifest_sha256": scenario_manifest_hash,
                    "scenarios": scenarios,
                    "n_scenarios": result.get("n_scenarios"),
                    "combined_metrics": result.get("combined_metrics", {}),
                    "baseline": result.get("baseline", {}),
                    "attack_results": result.get("attack_results", []),
                    "best_params": result.get("best_params", {}),
                }
            except Exception as exc:
                record = {
                    "status": "failed",
                    "ablation_id": ablation_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(ablation_id)
            matrix["results"][ablation_id] = record
            _write_json(output_dir / f"{ablation_id}_results.json", record)
            _write_json(output_dir / "ablation_matrix.json", matrix)
    finally:
        if original_config is not None:
            config_path.write_bytes(original_config)
        if previous_ablation is None:
            os.environ.pop("TRUSTTRACE_ABLATION", None)
        else:
            os.environ["TRUSTTRACE_ABLATION"] = previous_ablation

    matrix["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    matrix["failed_ablations"] = failures
    _write_json(output_dir / "ablation_matrix.json", matrix)
    print(f"\nAblation matrix saved to: {output_dir / 'ablation_matrix.json'}")
    if failures:
        print("Failed ablations:", ", ".join(failures))
        return 1
    print("All five ablation runs completed. Review the recorded outputs before interpreting metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
