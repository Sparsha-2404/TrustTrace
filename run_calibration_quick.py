#!/usr/bin/env python3
"""
Quick calibration runner - Tests baseline collection without full attack scenarios.

Usage:
    python run_calibration_quick.py [n_runs]

Example:
    python run_calibration_quick.py 10  # Quick 10-run test
    python run_calibration_quick.py 80  # Full calibration
"""

import sys
import time
import os
import json
from logger.interaction_logger import InteractionLogger
from drift.behavioral_drift import BehavioralDriftModule
from calibration.calibrate import run_calibration

def main():
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    if n_runs < 1:
        print("Calibration run count must be at least 1.")
        return 2
    
    print("=" * 70)
    print("🧪 TRUSTTRACE CALIBRATION RUNNER")
    print("=" * 70)
    print(f"Starting calibration with {n_runs} benign runs...")
    print()
    
    start = time.time()
    os.environ.setdefault("TRUSTTRACE_STRICT_EVAL", "1")
    
    try:
        logger = InteractionLogger()
        drift = BehavioralDriftModule()
        outputs = run_calibration(logger, drift, n_runs=n_runs)

        # Validation: ensure baseline files are written
        baseline_dir = os.path.join(os.path.dirname(__file__), "calibration", "baselines")
        baseline_files = [
            f for f in os.listdir(baseline_dir)
            if f.endswith(".json") and f != "manifest.json"
        ]
        if baseline_files:
            print(f"[Calibration Validation] Baseline files written: {len(baseline_files)} files found.")
        else:
            print("[Calibration Validation] No baseline files found!")

        # Validation: load baselines to confirm loading works.
        loaded = 0
        failed_files = []
        for filename in baseline_files:
            try:
                with open(os.path.join(baseline_dir, filename), encoding="utf-8") as handle:
                    vector = json.load(handle)
                if not isinstance(vector, list) or not vector:
                    raise ValueError("baseline vector is empty or malformed")
                loaded += 1
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                failed_files.append(f"{filename}: {exc}")
        required = ("Retriever", "Planner", "Executor", "Generator", "UserPrompt")
        manifest_valid = drift.baselines_valid_for(required)
        print(f"[Calibration Validation] Baselines loaded: {loaded}/{len(baseline_files)}")
        print(f"[Calibration Validation] Embedding manifest valid: {manifest_valid}")
        if failed_files or not manifest_valid:
            raise RuntimeError(
                "Calibration artifacts failed integrity validation: "
                + "; ".join(failed_files or ["baseline manifest missing or mismatched"])
            )
        
        elapsed = time.time() - start
        
        print()
        print("=" * 70)
        print(f"✓ CALIBRATION COMPLETE ({elapsed:.1f}s)")
        print("=" * 70)
        print(f"Baselines collected for: {list(outputs.keys())}")
        print(f"Average outputs per agent: {sum(len(v) for v in outputs.values()) // len(outputs)}")
        print()
        print("Next: Run 'python main.py' to execute full attack scenarios")
        print("=" * 70)
        
        return 0
    
    except KeyboardInterrupt:
        print("\n✗ Calibration cancelled by user")
        return 1
    except Exception as e:
        print(f"\n✗ Error during calibration: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
