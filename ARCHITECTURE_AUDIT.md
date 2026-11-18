# TrustTrace V3 — follow-up source audit (18 November 2026)

This follow-up supersedes the implementation-status snapshot below where noted.
It was performed against the uploaded `TrustTrace_Source_Only.zip` source tree.
It is a code and unit-test audit, not a live integration or security certification.

## Follow-up findings and changes

1. **Unit-test environment isolation:** the project `.env` enables strict research mode. Earlier tests accidentally inherited that flag while trying to exercise the non-strict embedding fallback, and several tests depended on a downloaded model. Added a test-only deterministic embedding fixture; the fallback-specific tests explicitly clear strict mode. The production strict-mode guard remains unchanged.
2. **Calibration diagnostic mismatch:** three benign samples and two attack prompts were all compared against the Retriever baseline. This compared raw user prompts with retrieved-document embeddings and generated answer text with the wrong component. Diagnostics now use `UserPrompt`, `Retriever`, and `Generator` baselines according to the sample source. These diagnostics are not treated as acceptance thresholds.
3. **Held-out benchmark integrity:** summary generation now checks official split indices, label agreement with the source dataset, per-row split identity, dataset SHA-256, provider/model identity and configuration SHA-256. Full runs must cover every official index. Summary writes are atomic and custom output directories are created. Resume refuses to append rows from a different provider, model, config or dataset.
4. **A0–A4 paired episode traceability:** each attack result now retains its scenario ID/index/source, and the matrix records the full scenario list plus a canonical scenario-manifest SHA-256. A changed scenario manifest between ablations fails that matrix run.
5. **Dependency compatibility:** pinned `pyarrow==17.0.0` alongside the existing NumPy `<2` constraint for the documented Python 3.11 setup.
6. **Setup entry point:** `setup.py` no longer requires a pre-created `logs/` folder, verifies both bundled dataset splits, and delegates downloads to the canonical atomic downloader. Its environment check explicitly targets Python 3.11.
7. **Packaging hygiene:** the final source package excludes the local `.env`, virtual environments, Chroma databases/backups, Python bytecode caches, and archived legacy run outputs. Use `.env.example` to create a local environment file.

## Validation performed in this audit environment

- Python: 3.13.5
- `python -m compileall -q .`: PASS
- `python -m pytest -q`: **97 passed, 0 failed** (two expected warnings from tests that deliberately exercise the non-semantic fallback)
- `python scripts/compliance_check.py`: PASS
- Synthetic benchmark-summary integrity check: PASS; this validated summary/index/hash handling only and did not run inference.
- Real Sentence Transformers model loading: NOT performed in this environment.
- Live LM Studio inference: NOT performed.
- Persistent ChromaDB restart, quarantine and rollback: NOT performed.
- Full A0–A4 experiment and held-out model evaluation: NOT performed.
- User's Windows/Python 3.11 environment: NOT verified.

## Remaining research blockers

- Run and preserve real target-machine integration tests for LM Studio and persistent ChromaDB.
- Demonstrate attack delivery/retrieval and downstream blocking with the configured real model.
- Repeat A0–A4 with frozen model/config/data, record model stochasticity and compare matching scenario IDs. Scenario IDs do not make stochastic LLM responses deterministic.
- Use an independently validated goal-completion oracle; the current final-output lexical marker remains only a proxy.
- Run a leakage-controlled held-out evaluation after freezing thresholds and hyperparameters.
- Reconcile all manuscript equations, thresholds, dataset splits, ablation semantics and metric denominators against the exact final source.

---

# TrustTrace V3 — implementation status (16 November 2026)

This file replaces earlier checklist-style claims. A passing compliance script is
not proof that every research requirement or security control has been validated.

## Implemented controls present in this source tree

- Real-provider configuration with LM Studio/OpenAI-compatible endpoint support.
- Strict-evaluation mode rejects mock provider output.
- Persistent ChromaDB is required by production runtime/benchmark constructors;
  the in-memory collection is for isolated tests only.
- Planner and Retriever validators are wired into the interactive runtime.
- Batch A2–A4 path now performs retrieved-document and Planner pre-execution
  checks, while A0/A1 retain their configured ablation boundaries.
- Version 3 EMA trust update is implemented, with graph propagation as a
  separately applied operation.
- Multi-layer compromise confirmation uses a consecutive-observation window.
- Patient Zero scoring handles missing timestamps deterministically.
- Experiment thresholds no longer vary with the ground-truth attack family.
- Marker-based attack outcome proxy inspects only the final Generator response.
- Missing Patient Zero predictions count as incorrect for method comparisons.
- Availability is reported as unknown when no recovery-window workload was
  observed; no synthetic 1/1 task record is inserted.
- Indirect and memory-poisoning baseline episodes fail closed if the injected
  payload was not actually retrieved.
- Hyperparameter search no longer silently converts failed episodes into
  missed detections and no longer tunes obsolete trust-equation parameters.
- Final HTML report is generated after benign runs so FPR includes those runs.

## Known research limitations — do not omit from a manuscript

1. The final-response marker predicate is a lexical proxy, not a semantic
   goal-completion judge. A paper must describe it as a proxy or replace it
   with a scenario-specific, independently validated oracle before reporting
   goal-based ASR as semantic compliance.
2. The multi-agent handcrafted experiment is a development benchmark. Its
   fixed scenarios and expanded payload repetitions are not a substitute for
   an independently held-out, leakage-controlled test set.
3. Hyperparameter search exposes a stratified tuning/holdout index split, but
   the full pipeline's headline multi-agent metrics are not a separate frozen
   holdout evaluation. The run is therefore marked diagnostic.
4. System availability is not measured unless actual tasks are instrumented
   during recovery. It is intentionally `null` when there are no observations.
5. LM Studio, Windows-specific execution, and persistent ChromaDB restart/
   rollback behaviour require validation on the target laptop.
6. Security controls are defence-in-depth heuristics and validators, not a
   formal guarantee against prompt injection or arbitrary tool misuse.
7. The test environment used for this build is Python 3.13.5. The user's
   intended Python 3.11 + Windows environment must still be checked separately.
8. Research metrics must be generated from fresh run IDs, frozen configuration,
   recorded model/version, dataset hashes, and raw per-episode outputs.
9. No experiment results in the archived legacy folder are certified as
   publication-ready.

## Recommended paper protocol

- Freeze the exact code commit/ZIP and configuration before collecting results.
- Train scanner and fit all calibration/threshold artifacts on training data only.
- Tune hyperparameters on a development split only.
- Evaluate once on an untouched held-out split, without tuning after viewing it.
- Run A0–A4 on identical scenario IDs, seeds, model version, and hardware.
- Record per-episode attack delivery, retrieval confirmation, detector output,
  goal-oracle outcome, Patient Zero predictions for every method, recovery
  actions, and real task-availability observations.
- Report denominators, missing predictions, confidence intervals, and all
  failed/aborted episodes. Do not impute missing observations as successes.
