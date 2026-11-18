"""Run full validation and print final report."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from experiment.full_pipeline import run_full_experiment


def main():
    config.EXPERIMENT_MODE = True
    print("Running TrustTrace final validation (quick mode)...")
    report = run_full_experiment(quick=True, force_calibrate=False)
    metrics = report["combined_metrics"]
    print("\nDiagnostic experiment execution completed.")
    print("SECURITY/RESEARCH ACCEPTANCE: NOT CERTIFIED BY THIS QUICK RUN")
    print(f"  FPR: {metrics['FPR']:.2%}")
    print(f"  PZ Accuracy: {metrics['Patient_Zero_Accuracy']:.2%}")
    print(f"  Chroma size: {metrics['Chroma_Collection_Size_Mean']:.0f}")
    print(f"  ASR measurement: {report.get('baseline', {}).get('success_measurement', 'unknown')}")
    print("  Review the generated report and full A0-A4/held-out protocol before making research claims.")
    # Exit 0 means only that the diagnostic pipeline completed without raising;
    # it is deliberately not a security acceptance or publication-readiness gate.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
