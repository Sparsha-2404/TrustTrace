> **HISTORICAL SNAPSHOT: This progress note records an earlier implementation stage. Current implementation and research limitations are described in ARCHITECTURE_AUDIT.md and RESEARCH_READINESS.md.**

# TrustTrace Version 3 — Implementation Progress

This is a working copy derived from the uploaded `TrustTrace-integrated-main(2).zip`. The original upload was left unchanged. No full LLM experiment has been run and no manuscript result has been generated from this working copy.

## Changes made in this first pass

1. **Trust update:** replaced the additive `rho*T - w_m*MS - w_b*BD + eta*H` update with the Version 3 EMA base update `T = rho*T + (1-rho)*H`, where `H = 1-(MS+BD)/2`. Graph propagation remains a separate subsequent operation. Removed the undocumented extra 20% sender-compromised penalty so the update path no longer adds an unlisted penalty on top of the specified propagation operation.
2. **Trust configuration:** removed `w_m`, `w_b`, and `eta` from `config.yaml` and `config.defaults.yaml`. `rho` is validated to be in `[0,1]`; MS and BD are validated to be in `[0,1]`.
3. **Trust tests:** added focused tests for benign fixed-point behaviour, exact EMA-plus-propagation arithmetic, score validation, and scenario origin labels.
4. **Scenario provenance and Patient Zero denominator:** added explicit `origin` metadata to handcrafted scenarios: `UserInput` for direct/jailbreak, `Document` for indirect, and `MemoryEntry` for memory poisoning. Corrected memory-poisoning component-level ground truth from `Retriever` to `MemoryStore`, since Retriever reads the poisoned entry but is not the poisoned store. Results now include `origin` and `pz_evaluable`; an entry-gated prompt is marked not evaluable for component attribution because it never reached a component. The attribution algorithm itself still needs the separate three-method comparison and further validation.
5. **Benchmark split handling:** rewrote `scripts/run_deepset_full_benchmark.py` to evaluate one split at a time, defaulting to the 116-row test split. It no longer concatenates train and test rows. Fresh runs overwrite the split-specific result file rather than append to stale results; resume is explicit; duplicate indices and errored rows cause summary generation to fail. Goal-based ASR is not misreported from `1-recall`; the script explicitly marks it as not applicable to the prompt-detection benchmark.
6. **Strict mock protection:** added `TRUSTTRACE_STRICT_EVAL` checks that reject MockProvider and deterministic `[mock:...]` fallback output in evidence-generating runs.
7. **Timing:** changed measured elapsed time in the full-pipeline scenario loop and recovery manager to `time.perf_counter()`.
8. **Split regression test:** added a test asserting the official local split sizes (546 train, 116 test) and independent index spaces.

## Validation performed

- `python -m compileall -q .` — passed.
- Focused tests from the first pass: `tests/test_trust_ema.py`, `tests/test_deepset_split.py`, `tests/test_regressions.py`, and `tests/test_behavioral_drift_fallback.py` previously reported **9 passed** in the initial run.
- Current focused regression check: `python -m pytest -q tests/test_patient_zero_ground_truth.py tests/test_trust_ema.py tests/test_deepset_split.py` — **8 passed**.
- `python -m compileall -q experiment trust graph tests` — passed.
- A broader combined test command exceeded the execution time limit after printing progress dots; it is not counted as a pass.
- Full `pytest -q` remains blocked during collection because the current Python environment does not have `chromadb` installed (`ModuleNotFoundError`). This is an environment dependency issue; a complete suite result is not yet available.
- Local dataset files contain 546 train rows and 116 test rows. The test split has 60 positive labels in the local copy; this is a data inventory, not a model-performance result.

## Remaining blockers before Version 3 experiments

- Implement and test A0–A4 ablation modes from Table 7, including A0's observing-but-never-blocking behaviour and `mu=0` in A3.
- Implement the three Patient Zero comparison methods and define origin attribution so content source (`UserInput`, `Document`, `MemoryEntry`) is not confused with first receiving agent.
- Review all attack scenario success predicates and ensure every malicious episode has a machine-checkable goal; separate this from detection metrics.
- Add complete run provenance (seed, model/provider identifier, configuration hash, scenario ID) to each episode.
- Ensure block events retain ground-truth labels and that counts reconcile with TP/FP definitions.
- Install the project's declared dependencies in a supported environment and run the full suite.
- Calibrate `delta` only on a disjoint benign validation set; freeze it before evaluation.
- Conduct a small real-provider smoke run with strict mode, inspect all logs, then run the preregistered full experiments with fixed seeds/repeats.
- Update the Version 3 manuscript only after validated results exist.

## Research caution

The code changes are implementation work, not evidence of improved performance. The revised trust dynamics, detection, attribution, recovery, and ablation claims must be validated empirically. No final performance number should be copied from stale output files.

## Second pass: ablation harness and ASR accounting

9. **A0–A4 configuration layer:** added `experiment/ablations.py` with the five configurations matching Version 3 manuscript Table 9. `BatchTrustTrace` accepts `ablation_id`, and the full experiment pipeline reads `TRUSTTRACE_ABLATION` (defaults to `A4`). A0 disables the entry gate and detection, A1 uses the entry gate only, A2 adds event-level content IRS checks without trust-state updates, A3 enables trust dynamics with `mu=0`, and A4 enables the propagation coefficient. Recovery is enabled only in A3/A4 so the simpler configurations do not call the trust-dependent recovery manager.
10. **Important timing limitation:** the current pipeline still executes an attack before `process_run_events` evaluates the logged events. Therefore A2's event-level IRS flags are *post-hoc detection*, not online L1–L6 blocking. They must not be described as interception or as preventing an attack in the results. The A0–A4 switch makes the current differences reproducible, but true per-hop enforcement still requires moving checks into the pipeline before each agent consumes the content.
11. **ASR metric correction:** moved metric helpers to dependency-light `experiment/metrics.py` and changed the ASR success function to follow the manuscript's Table 8 definition: goal-based ASR is based on whether the scenario success predicate holds, independently of whether a detector flagged the episode. This prevents post-hoc detection from being misreported as mitigation. The marker-only goal predicate is still a placeholder and needs scenario-specific predicates before the reported ASR is manuscript-ready.
12. **Ablation tests:** added tests that verify A0–A4 configuration differences, case-insensitive configuration selection, and recovery flags.

### Second-pass validation

- `python -m compileall -q experiment trust graph tests` — passed.
- `python -m pytest -q tests/test_ablations.py tests/test_asr.py tests/test_patient_zero_ground_truth.py tests/test_trust_ema.py tests/test_deepset_split.py` — **15 passed**.
- No LLM-backed benchmark has been run. The ablation harness has not yet been empirically validated end-to-end. In particular, A2 is currently a post-hoc event-level check, not the online per-hop enforcement implied by the manuscript's L1–L6 wording.
