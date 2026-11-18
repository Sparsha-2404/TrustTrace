> **HISTORICAL SNAPSHOT: This earlier 16 November hardening pass is superseded by HARDENING_CHANGELOG_2026-11-16.md. Use the latter for the consolidated changes and current validation limits.**

# TrustTrace V3 — extended hardening pass (16 November 2026)

## Baseline and scope

This build was based on `TrustTrace_V3_Hardened_2026-11-16.zip`. The work focused on additional implementation defects and failure modes discovered during a second code review. It does not claim that all possible vulnerabilities have been eliminated.

## Additional fixes in this pass

1. **Persistent vector store:** added `victim_pipeline/chroma_store.py`; pipeline and default `MemoryManager` now share a `PersistentClient` at `data/chroma_db`, with `TRUSTTRACE_CHROMA_PATH` override.
2. **No fake production store:** main interactive startup, interactive demo, calibration and full experiments reject the in-memory test fallback.
3. **Knowledge-base reset:** batch episode resets now reseed the exact trusted baseline corpus. Interactive benign turns remove only attacker-marked documents and preserve trusted/user memory.
4. **Persistent memory quarantine:** quarantine metadata is written to Chroma; reads filter persisted quarantine flags after process restart; release can operate on persisted quarantine after restart.
5. **Persistent rollback:** rollback enumerates stored Chroma records by timestamp and agent, not only the current process's write log.
6. **Memory input validation:** content, agent, trust/IRS/memory-trust scores and Chroma metadata types are validated before writes.
7. **Reference exemption tightening:** only exact immutable baseline reference documents receive the benign-seed exemption; arbitrary `Reference:`-prefixed content remains untrusted.
8. **Scanner dataset integrity:** removed the synthetic fallback dataset. Training now validates row shape, text, binary labels and presence of both classes; model/scaler saves are atomic and scanner scoring validates class index and finite probabilities.
9. **Scanner startup:** both model and scaler are required before treating the scanner as trained; missing either triggers retraining.
10. **Calibration integrity:** baselines are written atomically and bound to the embedding backend and file hashes in `calibration/baselines/manifest.json`. Stale, missing or modified baseline manifests trigger recalibration.
11. **Strict research mode:** strict evaluation rejects the character-hash embedding fallback and mock LLM; calibration uses strict mode and requires persistent Chroma.
12. **Interactive execution:** the interactive demo now uses `TrustTraceRuntime.execute()` rather than running the unmonitored pipeline and analysing outputs afterwards.
13. **No synthetic injection during classification:** auto-detection of real user content no longer plants synthetic indirect/memory payloads in persistent memory; only explicit `simulate_attack=True` does that.
14. **Containment-safe reset:** runtime reset is refused while agents remain quarantined or held tasks await human review.
15. **Validation hardening:** risk scores, graph suspicion weights, trust values, decision evidence, planner/retriever/tool inputs and metadata are range/type checked.
16. **Tool permissions:** an explicitly empty permission set is now treated as no permissions, not as permission checking being disabled.
17. **Patient-zero timestamps:** timestamp `0.0` is handled as a valid earliest timestamp; generic path exceptions no longer hide unexpected programming errors.
18. **Chroma integrity:** generic initialization/version errors no longer trigger automatic database moves or destructive rebuilds. Existing persistent data is left untouched and the error is surfaced.
19. **Recovery reports:** incident filenames are validated and report writes are atomic.
20. **Cleanup safety:** baseline indirect/memory payloads are removed in `finally` blocks even when model execution fails.
21. **Calibration helper:** fixed the baseline directory path, excludes the manifest from vector counts, and validates the manifest rather than swallowing malformed-file errors.
22. **Documentation and regression coverage:** updated setup instructions and added tests for the above behaviours.
23. **Calibration retrieval screening:** calibration runs now validate retrieved documents and exact Planner tasks before they are sent to the LLM, and reuse the same validated retrieval result rather than issuing a second unvalidated query.
24. **Calibration reproducibility:** startup checks the baseline manifest before constructing the runtime; full experiments refresh the IRS benign centroid after recalibration.
25. **Persistent memory lifecycle:** interactive benign requests preserve trusted/user knowledge-base content; only isolated batch scenarios perform a full baseline reset.
26. **Attack cleanup:** runtime-simulated indirect/memory payloads are cleaned in a `finally` block even if the LLM, retriever or validation callback raises.
27. **Configuration validation:** per-agent compromise persistence windows are configurable, and IRS/MS weights, trust recovery increments and graph crossings are range-checked.
28. **Tool/retriever validation:** mismatched document-type lists are rejected; an explicitly empty tool-permission set denies tools that require permissions.
29. **Chroma migration safety:** removed automatic database movement/rebuild on generic errors and made version mismatch non-destructive.
30. **Interactive lifecycle:** moved runtime construction after scanner training and calibration, and ensured strict mode is active before embedding/provider initialization.

## Validation performed in the build environment

- Python syntax compilation (`compileall`): passed.
- Pytest: **80 passed, 0 failed** (11 expected warnings from tests explicitly exercising the non-semantic fallback).
- `scripts/compliance_check.py`: **PASS**.
- `scripts/run_ablation_matrix.py --help`: passed.
- Persistent Chroma and LM Studio end-to-end checks: **not executed** in this build environment. No LM Studio server was listening at `127.0.0.1:1234`; actual Windows Python 3.11 and Chroma installation also remain unverified here.
- Full A0–A4 benchmark: not run; no performance metrics are claimed.

## Remaining limitations

- Rule-based prompt-injection and response validators are not semantic proofs and can miss novel, multilingual, obfuscated or context-dependent attacks.
- The handcrafted attack-success predicate remains marker-based and is not a comprehensive semantic task-success judge.
- Chroma persistent-store behaviour is unit-covered with a Chroma-like fake but still requires a real local Chroma smoke test on the target laptop.
- LM Studio generation and the full runtime have not been end-to-end exercised in this build environment.
- The project is a research prototype, not a production security boundary for real secrets, unrestricted tools or high-impact actions.
