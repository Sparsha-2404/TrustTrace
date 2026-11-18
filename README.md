# TrustTrace V3

**Real-time trust monitoring, prompt-injection screening, propagation analysis and recovery for a multi-agent LLM pipeline.**

TrustTrace is a research prototype. It instruments a four-stage pipeline (Retriever → Planner → Executor → Generator) with input and retrieval screening, per-hop checks, trust updates, graph-based incident analysis, memory quarantine/rollback and experiment ablations A0–A4.

> **Security statement:** no software can honestly guarantee “zero loopholes.” TrustTrace combines deterministic rules, a trained injection classifier, behavioural signals and trust controls. These mechanisms can still miss novel attacks or flag benign content. Treat it as a defence-in-depth research prototype, not a production security boundary for secrets, financial actions, shell execution or other high-impact operations.

## Current implementation

- Configured for LM Studio's OpenAI-compatible API at `http://127.0.0.1:1234/v1` and model `qwen/qwen3-4b-2507`.
- Real provider initialisation failures now stop execution; the runtime does not silently substitute mock output.
- Planner goals are validated before the Planner LLM call.
- Retrieved documents are validated before their content is forwarded to the Planner.
- Final-response validation checks unsafe disclosure patterns before accepting a refusal phrase.
- Trust compromise requires the configured consecutive confirmation window; confirmed compromise remains latched until explicit recovery.
- Recovery does not automatically replay held tasks through an unprotected pipeline. Held tasks are retained for human review.
- A2 per-hop checks and A3/A4 trust updates are enforced online between agent steps in the batch experiment runner.
- Direct/jailbreak entry screening inspects the actual crafted prompt, not only its benign seed query.
- Quick experiment config restoration preserves the selected LLM provider and endpoint.
- Interactive prompts run through the monitored TrustTrace runtime; automatic classification does not insert synthetic attack payloads into user memory.
- The application and memory manager share one persistent ChromaDB collection at `data/chroma_db` (override with `TRUSTTRACE_CHROMA_PATH`). The in-memory collection is test-only; interactive sessions, calibration and benchmarks reject it.
- Memory quarantine is stored in Chroma metadata, survives process restarts, is respected by MemoryManager reads, and is explicitly filtered during A2-A4 retrieval. A0/A1 retrieval semantics remain unchanged.
- Trusted seed-document exemptions use exact corpus matching; arbitrary documents beginning with `Reference:` are not exempt.
- Scanner training rejects empty, malformed or single-class datasets and never fabricates a synthetic replacement dataset.
- Calibration baselines are tied to the embedding backend and file hashes through `calibration/baselines/manifest.json`; missing or stale manifests trigger recalibration.
- Calibration drift diagnostics now compare user prompts, retrieved references and generated text with their corresponding component baselines. Their values are explicitly diagnostic, not unvalidated pass/fail thresholds.
- Strict evaluation rejects the mock LLM and the character-hash embedding fallback.
- The held-out benchmark validates official split indices, ground-truth labels, dataset hashes, model/provider identity and configuration hashes; resume cannot silently mix different datasets or configurations.
- A0–A4 episode outputs retain stable scenario IDs, and the matrix records a scenario-manifest hash so paired scenario coverage can be checked.

## Recommended environment

- Windows 10/11 or Linux/macOS
- Python **3.11** (recommended; other versions are not the validated target)
- LM Studio with a loaded chat model and its local server enabled
- At least 8 GB RAM; 16 GB recommended for local embedding and LLM use

## Install (Windows PowerShell)

From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt -c constraints.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, use the venv interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -c constraints.txt
```

Optional integrations (AgentDojo, JailbreakBench, Mistral API, Plotly image export):

```powershell
python -m pip install -r requirements-optional.txt
```

## Configure LM Studio

1. Open LM Studio and load `qwen/qwen3-4b-2507` (or update `config.yaml` to the exact model identifier shown by your LM Studio server).
2. Open **Developer → Local Server** and start the OpenAI-compatible server on port `1234`.
3. Keep the server running while TrustTrace is using the model.
4. Confirm `config.yaml` contains:

```yaml
llm:
  provider: lm_studio
  base_url: http://127.0.0.1:1234/v1
  model: qwen/qwen3-4b-2507
  api_key: null
```

5. Check the actual endpoint with a real inference request:

```powershell
python scripts/check_llm_connection.py
```

An HTTP server being reachable is not sufficient: the check must return a non-empty model response. If it fails, do not run the benchmark.

You can override the endpoint without editing YAML:

```powershell
$env:TRUSTTRACE_LLM_BASE_URL = "http://127.0.0.1:1234/v1"
```

## First-time data and model setup

The persistent vector store is created automatically at `data/chroma_db` on first use. To use another location, set:

```powershell
$env:TRUSTTRACE_CHROMA_PATH = "D:\TrustTraceData\chroma_db"
```

The deepset train/test JSON files are included in this project. If they are missing or you deliberately want a fresh copy, run `python data/download_deepset.py`. Then prepare the scanner and calibration baselines:

```powershell
python train_scanner.py
python run_calibration_quick.py 80
```

The downloader does not create a synthetic substitute if Hugging Face is unavailable. Scanner training requires both benign and injection classes and uses only the training split. First-time setup may download the Sentence Transformers model files. The dependency set pins PyArrow 17.0.0 to avoid the known NumPy 1.x/PyArrow compatibility failure on the Python 3.11 target. Calibration uses the configured real LLM and writes an embedding-backend/hash manifest; do not edit baseline JSON files manually after calibration.

## Run the interactive system

```powershell
python main.py
```

The interactive entry point bootstraps the scanner, calibration baselines, memory and runtime. Use the menu to submit prompts and inspect reports.

## Validate and run experiments

For a quick diagnostic run of the configured A4 pipeline:

```powershell
$env:TRUSTTRACE_STRICT_EVAL = "1"
python scripts/run_final_validation.py
```

Run all five ablations with the same scenario-generation protocol:

```powershell
$env:TRUSTTRACE_STRICT_EVAL = "1"
python scripts/run_ablation_matrix.py
```

The matrix script defaults to **quick diagnostic mode**. Its results are not publication-grade. For the configured full run:

```powershell
python scripts/run_ablation_matrix.py --full --trials 100
```

Outputs are written to `reports/ablation_matrix/`. Inspect each per-ablation JSON and verify that all runs completed without errors before comparing metrics. Record the LM Studio model ID, dataset version, configuration, code version and run date in any paper or presentation.

## Tests

Run the local test suite from the project root:

```powershell
python -m pytest -q
```

The 18 Nov 2026 follow-up hardening source was syntax-compiled and **104 unit tests passed** in a Python 3.13.5 build environment. Two expected warnings exercise the deliberately non-semantic hash-embedding fallback; strict research runs must use the real Sentence Transformers model. This test run did not exercise the user's Windows Python 3.11 environment, live LM Studio endpoint, persistent ChromaDB restart/rollback, or the complete A0-A4 benchmark. Those still require validation on the target machine.

## Ablations

| Mode | Mechanisms |
|---|---|
| A0 | No TrustTrace entry/per-hop/trust/recovery controls |
| A1 | Entry gate only |
| A2 | Entry gate + online per-hop content checks |
| A3 | A2 + trust dynamics with propagation coefficient `mu = 0` |
| A4 | A3 + graph trust propagation |

The batch experiment runner applies A2 checks and A3/A4 trust decisions before the next downstream agent step. A0/A1 intentionally do not receive those mechanisms, to preserve the ablation definitions.

## Important limitations and interpretation rules

- Pattern validators are not semantic proofs. Obfuscation, multilingual attacks, indirect prompt injection and novel payloads can evade them.
- The final response validator is heuristic; it is not a reliable secret detector or a substitute for output policy enforcement.
- The local LLM is not a trusted security oracle. Never expose real secrets or give the prototype unrestricted tools.
- Patient-zero accuracy is meaningful only for scenarios with defensible ground-truth origins and a detected component. Entry-gated prompts do not have a component-level patient zero.
- The ASR success predicate in the current handcrafted episode harness is a final-Generator lexical marker proxy. Reports now label this limitation explicitly. It is not a scenario-specific goal oracle and must not be presented as comprehensive semantic goal verification. Use independently validated task-specific success predicates before reporting headline ASR.
- Quick runs are smoke/diagnostic runs. Do not use them as final research evidence.
- Held tasks are not automatically replayed after recovery; review and revalidate them before any release workflow.
- Attack-fixture cleanup removes both persistent Chroma entries and process-local temporary-memory entries.
- Defended injection helpers fail closed unless both the trust engine and IRS assessor are supplied. The intentionally undefended A0 control must opt in explicitly with `control_mode=True`.
- Memory writes recompute memory trust from IRS and trust-at-write; a caller cannot override the policy by supplying a forged `memory_trust` value.
- Temporary-memory promotion and quarantine release are default-deny and require an independent revalidator, reviewer identifier and reason. The caller must authenticate the reviewer; these audit fields are not an authentication system.

## Repository map

- `runtime/` — interactive TrustTrace orchestrator
- `victim_pipeline/` — four-stage target pipeline and stepwise execution callbacks
- `irs/`, `scanner/`, `validation/`, `detection/` — input, retrieval, agent and output checks
- `trust/`, `graph/`, `detector/` — trust state, propagation graph and patient-zero methods
- `memory/`, `recovery/` — memory trust, rollback, quarantine and human-reviewed held tasks
- `experiment/` — ablations, scenario generation, online batch processing and metrics
- `scripts/` — connection check, final validation and A0–A4 matrix runner
- `tests/` — regression and security-hardening tests
