# TrustTrace V3 — research readiness gate

**Status: NOT READY for final conference results or submission.**

This source package is a hardened engineering candidate. Passing unit tests or
completing the quick diagnostic does not certify the security design, semantic
attack-success measurement, or paper claims.

## Code-level corrections in the 18 Nov 2026 follow-up

- Retrieval security callbacks can now replace the retrieved context with a
  sanitized list; the original rejected context is not forwarded to Planner.
- A2-A4 inspect direct user input before retrieval and check retrieved
  documents before Planner consumption.
- Confirmed live blocks are recorded in compromise state so Patient Zero and
  recovery do not depend solely on trust-threshold crossings.
- Known poisoned fixture IDs remain present until recovery and are explicitly
  rolled back; recovery checks its postconditions.
- Baseline and defended episodes use the same scenario manifest and the
  pipeline refuses a comparative report if their scenario IDs differ.
- Lexical output-marker measurements are labelled as proxies. Missing recovery
  observations are reported as N/A rather than 0 seconds.
- AgentDojo tasks are excluded until a real task/document/tool/oracle adapter
  exists.
- Quick validation reports diagnostic execution only, not acceptance.
- The ChromaDB 0.5.0/PostHog 8 API mismatch observed in the uploaded virtual
  environment is constrained to the compatible legacy PostHog API range.

## Remaining blockers before research claims

- [ ] Run the patched full test suite on the user's Windows machine with Python
      3.11.9 and confirm the exact result.
- [ ] Re-run real LM Studio completion and semantic SentenceTransformer checks
      on the patched package.
- [ ] Re-run ChromaDB persistence and rollback tests on the patched package,
      and verify that the PostHog telemetry signature warnings have disappeared.
- [ ] Run the A0-A4 matrix and verify identical scenario-manifest hashes,
      provider/model, data snapshot, configuration, and random seeds.
- [ ] Inspect per-hop traces to prove rejected direct and retrieved payloads
      never appear in downstream Planner/Executor/Generator prompts.
- [ ] Verify actual memory rollback and trust restoration across a process
      restart, not only in-process fixture cleanup.
- [ ] Implement scenario-specific, machine-checkable goal oracles. Current ASR
      remains a final-Generator lexical marker proxy and is not semantic ASR.
- [ ] Evaluate Patient Zero methods on an adequately sized, independently
      labelled dataset; include no-prediction cases.
- [ ] Instrument actual task arrivals/completions during recovery before
      reporting availability.
- [ ] Create and preserve an untouched held-out test split; freeze thresholds
      and parameters before evaluating it.
- [ ] Reconcile manuscript equations, thresholds, ablation definitions,
      denominators, and claims with the exact frozen source version.
- [ ] Add a real AgentDojo adapter before making AgentDojo benchmark claims.

## Do not claim

- Zero vulnerabilities or complete prompt-injection immunity.
- Semantic ASR based only on lexical response markers.
- Recovery success from detection alone or from a zero-duration placeholder.
- 100% availability when no tasks were observed during recovery.
- Patient Zero accuracy calculated only on successful predictions.
- Held-out performance from quick diagnostic runs.
- AgentDojo performance when only handcrafted payloads were exercised.
