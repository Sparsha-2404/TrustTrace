# TrustTrace follow-up hardening patch — 18 November 2026

## Source changes in this package

1. **Attack fixture cleanup** — added `MemoryManager.delete_entry()` and connected indirect-injection/memory-poisoning cleanup to it. Cleanup now removes persistent Chroma records, temporary-memory entries, write-log entries and local quarantine state, then verifies that the identifier is absent from persistent and temporary storage.
2. **Default-deny defended injection** — attack injection helpers now require both `trust_engine` and `irs_assessor` unless the caller explicitly selects `control_mode=True`. Only the deliberately undefended A0 baseline and the explicit classifier diagnostic opt into control mode.
3. **Memory trust integrity** — `MemoryManager.write()` recomputes memory trust from IRS and trust-at-write and rejects inconsistent caller-supplied values.
4. **Reviewed promotion/release** — temporary-memory approval and quarantine release require a reviewer identifier, reason, and a revalidator callback. IRS, trust and memory-trust are re-derived; below-threshold entries remain blocked/quarantined. The calling application must authenticate reviewer identity; the fields persisted here are audit metadata, not authentication.
5. **Quarantine-aware defended retrieval** — A2-A4 retrieve with persisted quarantine filtering. Filtering occurs before top-k truncation so quarantined top-ranked records do not starve eligible results. A0/A1 retrieval remains unchanged to preserve the ablation definitions.
6. **More realistic indirect fixtures** — indirect and memory-poisoning payloads are embedded in topic-relevant documents so delivery depends on ordinary semantic retrieval, rather than artificially injecting the payload into the downstream prompt or increasing top-k.
7. **Metric honesty** — code comments and printed summaries identify current ASR as a final-Generator lexical marker proxy, not a scenario-specific goal oracle. The report metadata already records this limitation.

## Validation performed on this package

- `python -m py_compile` on modified runtime, memory, attack and experiment modules: passed.
- `pytest -q`: **104 passed, 2 warnings**. The warnings are expected tests of the non-semantic hash-embedding fallback.
- Added regression coverage for temporary and persistent cleanup, forged memory-trust rejection, reviewed promotion/release, quarantine-aware retrieval, defended-injection fail-closed behaviour, and topic-framed indirect payload storage.

## Not validated by this build environment

- The build environment used Python 3.13.5, not the target laptop's Python 3.11.9.
- ChromaDB was not installed in this build environment; therefore this patch's retrieval and cleanup tests use a Chroma-compatible test collection, not a live persistent ChromaDB instance.
- No live LM Studio inference was run from this build environment. The user's earlier local connection checks remain separate evidence.
- The full final-validation run and A0-A4 matrix have not been rerun here. In particular, the baseline delivery failure must be rechecked against the actual Chroma embedding/retrieval backend after these changes.
- **ASR remains a lexical marker proxy.** Scenario-specific, independently verifiable goal oracles are still required before reporting headline ASR as malicious-goal completion.
- This is a research-prototype hardening update, not a guarantee of zero vulnerabilities or production security.

## Follow-up source review and containment corrections (18 Nov 2026)

The user's Windows quick-validation output exposed four additional defects. The
following source changes and regression tests were added in this follow-up:

1. **Retrieval filtering is now effective, not observational.** The
   `KnowledgeBase` callback may return a sanitized list, and
   `run_pipeline_stepwise` now uses that returned list as the Planner context.
   Previously, a callback could reject a retrieved item but the original
   unfiltered `context` was still passed to the Planner.
2. **Direct input is inspected before retrieval.** The A2-A4 path now applies
   the Retriever validation and per-hop risk check to the incoming user
   content before ChromaDB is queried. Known benchmark payload patterns were
   added to the retriever validator. These rules are not a proof of complete
   prompt-injection coverage.
3. **Live containment is visible to Patient Zero and recovery.** A confirmed
   live block records the blocked component as compromised. Memory-store
   sourced poison records also mark `MemoryStore`, allowing the attribution
   methods to evaluate the declared source component.
4. **Recovery sees poisoned fixtures before cleanup.** The attack fixture is
   retained until recovery runs. Recovery explicitly removes known fixture IDs,
   checks that deletion succeeded, restores trust, and fails closed if the
   recovery postconditions are not met. The caller still performs unconditional
   idempotent cleanup.
5. **Baseline and defended experiments now share one scenario manifest.** The
   baseline no longer runs a hard-coded 12 episodes while the quick defended
   run evaluates 8. It receives the exact same scenario IDs, queries, attack
   types, and payload indices, and the full pipeline refuses to report a
   comparison if the IDs differ.
6. **Metric labels are corrected.** Final-generator marker hits are explicitly
   labelled as a lexical proxy, not as malicious-goal achievement. When no
   recovery observations exist, recovery-time metrics are `N/A`, not zero.
7. **AgentDojo is not falsely counted.** The package being installed is not a
   benchmark adapter. AgentDojo scenarios are excluded until task documents,
   tool execution, and task-specific goal oracles are implemented.
8. **Quick validation is not an acceptance gate.** Its successful process exit
   means the diagnostic run executed without an exception; it no longer
   implies security acceptance or publication readiness.

### Regression evidence for this source revision

- `pytest -q`: **119 passed, 2 warnings** in the isolated review environment.
- The warnings deliberately exercise the non-semantic embedding fallback in
  unit tests. They are not evidence that the user's real SentenceTransformer
  installation is unavailable.
- The review environment does not have ChromaDB, the real SentenceTransformer
  model, or the user's LM Studio endpoint. No live end-to-end experiment was
  run here. Windows/LM Studio/ChromaDB integration must be rerun by the user on
  the patched package.
9. **ChromaDB telemetry dependency pinned.** The uploaded Windows environment
   contained `chromadb==0.5.0` with `posthog==8.0.1`. ChromaDB 0.5.0 calls the
   legacy three-argument `posthog.capture(user_id, event, properties)` API;
   PostHog 8 removed it, producing the observed `capture() takes 1 positional
   argument but 3 were given` warnings. `constraints.txt` now pins
   `posthog>=2.4.0,<4.0.0` to retain the API expected by this ChromaDB release.
   The actual fix must be confirmed by reinstalling under the user's Windows
   Python 3.11 environment.

10. **System refusal text is no longer counted as attacker success.** The prior
    final-output marker proxy searched for the substring `compromised`, while
    TrustTrace's own `[BLOCKED]` refusal said "blocked by the compromised
    pipeline". That made a successful block look like a successful attack. The
    proxy now excludes `[BLOCKED]` and `[QUARANTINED]` outputs, with a regression
    test. The remaining proxy is still not a scenario-specific goal oracle.
