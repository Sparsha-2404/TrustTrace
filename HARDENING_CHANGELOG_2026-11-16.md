# Extended hardening change log — 16 November 2026

Source archive: `TrustTrace_V3_Extended_Hardening_2026-11-16.zip`
SHA-256 verified before modification:
`b9540a266be9979bd9c1e00191be4c31566e105f8446138c0bd3deb1c60f6b7d`

## Changes in this pass

### Attack outcome measurement
- Restricted the legacy marker proxy to the final `Generator` response.
- Explicitly documented that lexical markers are not semantic goal-completion.
- Added regression coverage proving that markers in retrieved context or
  intermediate agent outputs do not count as user-visible attack success.

### Benchmark inference integrity
- Batch threshold selection now uses the same global `lambda` regardless of
  the ground-truth attack family. Legacy attack-specific threshold keys no
  longer influence the batch inference path.
- Batch A2+ checks retrieved documents with `RetrieverValidator`.
- Batch A2+ validates Planner goals before the Planner LLM call.
- Rejected Planner tasks stop downstream execution and produce blocked outputs.
- Retrieval-hit counts reset per episode and count actual retrieved documents.

### Baseline delivery
- Indirect-injection and memory-poisoning baseline episodes now verify that
  the injected payload was actually present in Retriever output.
- Undelivered attacks raise an error and are not silently counted as valid
  attack trials.

### Patient Zero
- Missing compromise timestamps no longer create `inf - inf` / NaN PCS values.
- Invalid or absent backward-traversal results fall back deterministically.
- Missing Patient Zero predictions count as incorrect when ground truth exists.
- Method-wise Patient Zero accuracy is recorded separately.

### Metrics and recovery reporting
- Attack detection confusion counts are independent of goal-achievement outcome.
- System availability is `null` if no task observations exist.
- Removed synthetic `record_recovery(0, duration, 1, 1)` availability evidence.
- Recovery is marked complete only when the recovery routine actually returns.
- The final HTML report is generated after benign episodes, so FPR is included.

### Trust and tuning
- Trust score inputs reject booleans, non-numeric values and non-finite values.
- Tuning no longer searches obsolete `w_m`, `w_b` and `eta` parameters.
- Failed tuning episodes now fail the trial instead of being imputed as missed
  detections.
- Tuning split is stratified by attack type/source and holdout indices are
  returned; holdout is not falsely claimed as evaluated by the current
  full-pipeline report.

## Verification performed in the build environment

- Python syntax compilation: PASS.
- Automated suite: 90 passed, 0 failed.
- Compliance checker: PASS.
- Ablation matrix CLI help: PASS.
- ZIP integrity check: PASS.
- Source environment: Python 3.13.5.

The test run emitted 11 expected warnings because the build environment does
not have the semantic Sentence Transformers model available and the explicit
non-semantic hash fallback was exercised by tests. These fallback baselines
are not valid for research claims.

## Not verified here

- LM Studio live inference: no server was running at `127.0.0.1:1234`.
- Windows/Python 3.11 execution on the user's laptop.
- Persistent ChromaDB restart, corruption and rollback on the target machine.
- Full A0–A4 benchmark execution.
- Held-out benchmark results or publication metrics.

This is a hardened implementation candidate, not a guarantee of zero
vulnerabilities and not a certificate of publication readiness.


## Follow-up hardening — 18 November 2026

- Isolated unit tests from machine `.env` strict-mode flags and external embedding-model downloads using a test-only deterministic embedder; production strict mode remains intact.
- Corrected calibration diagnostics to use component-appropriate baselines and removed unsupported pass/fail claims from those descriptive checks.
- Hardened deepset benchmark summaries against out-of-range/missing indices, altered labels, split mixing, dataset changes, model/provider changes and config changes during resume.
- Added atomic summary writes and support for custom output directories.
- Preserved stable scenario IDs and a scenario-manifest hash in A0–A4 outputs.
- Pinned PyArrow 17.0.0 for the NumPy <2 / Python 3.11 dependency set.
- Removed local secrets/config, generated vector databases, bytecode caches and legacy run outputs from the clean source deliverable.

- Repaired the standalone setup entry point: no longer requires an untracked `logs/` directory, validates both data splits, and reuses the canonical downloader.
- Added regression coverage for setup structure and dataset validation.
