> **SUPERSEDED HISTORICAL SNAPSHOT.** This report documents an earlier audit and is not a current certification. Its "no code changes required" conclusion is outdated after the 18 November follow-up fixes. See `../ARCHITECTURE_AUDIT.md` and `../RESEARCH_READINESS.md`.

# TrustTrace Audit Report

## Summary
- Audit completed for the deepset prompt-injections benchmark and TrustTrace inference pipeline.
- No ground-truth label leakage was found in the runtime inference path.
- Benchmark labels are only used for post-hoc evaluation.

## Checklist
- [x] Verified deepset dataset label distribution
- [x] Confirmed no exact duplicate prompts in train/test
- [x] Confirmed `scripts/run_deepset_full_benchmark.py` does not pass labels into runtime
- [x] Confirmed `TrustTraceRuntime.execute()` accepts only prompt input and runtime controls
- [x] Confirmed no label leakage in runtime or experiment entry points
- [x] No code changes required for the audit findings

## Dataset audit
- `data/deepset_injections/train.json`
  - 546 samples
  - 343 benign, 203 malicious
- `data/deepset_injections/test.json`
  - 116 samples
  - 56 benign, 60 malicious
- Exact duplicate text count across each file: 0 (train/test individually contain no duplicate exact prompt texts).

## Benchmark runner audit
- `scripts/run_deepset_full_benchmark.py` loads prompt text and ground-truth label pairs from the dataset.
- It calls `runtime.execute(text, auto_detect=True)` with only `text` and no ground-truth label.
- The field `ground_truth_attack` is written to the results file only after execution, for evaluation metrics.
- Evaluation metrics are computed from `ground_truth_attack` vs. reported `detected` only.

## Runtime inference audit
- `runtime/trusttrace_runtime.py::TrustTraceRuntime.execute` accepts:
  - `user_query`, `attack_type`, `payload_index`, `auto_detect`, `simulate_attack`
- The runtime decision and detection logic operate on `user_query` and runtime detectors only.
- No dataset labels are consumed by `execute()` during normal benchmark inference.

## Scanner training audit
- `scanner/injection_scanner.py::InjectionScanner.train` trains the scanner on `data/deepset_injections/train.json`.
- This is a standard supervised training step and does not imply label leakage at inference time.
- `InjectionScanner.score()` is later used at runtime to compute suspicion scores from prompt text.

## Experiment entry points
- `main.py` bootstraps the TrustTrace runtime without any ground-truth label input.
- `experiment/full_pipeline.py` and `experiment/batch_processor.py` manage attack scenarios and recovery evaluation but do not inject deepset benchmark labels into runtime inference.

## Findings
- The evaluated benchmark implementation is correctly separated: inference receives prompt text only, labels remain external to the security decision flow.
- The only usage of ground-truth labels is for metrics such as TP/FN/FPR/ASR after the runtime result is generated.

## Notes
- The existing `reports/deepset_full_benchmark_results.jsonl` may contain mixed output from multiple runs, including error rows and success rows. For a clean run, delete or archive that file before rerunning `scripts/run_deepset_full_benchmark.py`.

## Conclusion
The audit is complete for the current repository state. No evidence of ground-truth label leakage into `runtime.execute()` or the deepset benchmark inference path was found.
