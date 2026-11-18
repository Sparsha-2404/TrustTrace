"""Quick smoke test: generate an experiment HTML report with synthetic data."""

from __future__ import annotations

import os
import sys
import json
from time import time

# Ensure repo root is on sys.path so imports work when running the script directly
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from reports.report_generator import generate_experiment_report


def make_synthetic(n=10):
    scenarios = []
    attack_results = []
    for i in range(1, n + 1):
        scenarios.append({
            "query": f"Prompt number {i}",
            "type": "direct" if i % 3 == 0 else ("indirect" if i % 3 == 1 else "memory"),
            "source": "handcrafted",
        })
        attack_results.append({
            "succeeded": i % 7 == 0,
            "detected": i % 5 == 0,
            "recovered": i % 11 == 0,
            "outputs": {"Generator": "Synthetic output"},
            "pz_confidence": 0.05 * i,
            "avg_trust": 0.9 - 0.01 * i,
            "execution_time": 0.1 * i,
        })
    return scenarios, attack_results


def main():
    scenarios, attack_results = make_synthetic(n=12)
    data = {
        "project_name": "TrustTrace",
        "experiment_date": "2026-07-14T00:00:00Z",
        "duration_s": 12.34,
        "llm_provider": "mock",
        "dataset_used": "data/deepset_injections",
        "total_prompts": len(scenarios),
        "author": "Smoke Test",
        "version": "test-1",
        "scenarios": scenarios,
        "attack_results": attack_results,
        "metrics": {"ASR": 0.1, "Detection_Rate": 0.8, "FPR": 0.02},
        "combined_metrics": {"ASR": 0.1},
        "baseline": {},
        "avg_irs": 0.12,
        "avg_trust": 0.85,
        "avg_latency": 0.5,
    }

    out_dir = os.path.join(ROOT, "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "experiment_report_test.html")
    generate_experiment_report(out_path, data)
    print("Wrote test report:", out_path)


if __name__ == "__main__":
    main()
