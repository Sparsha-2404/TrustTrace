# TrustTrace V3 — Quick Start

> **Research caution:** This is a hardened implementation candidate, not a
> guarantee of zero vulnerabilities or a publication-ready benchmark. Read
> `RESEARCH_READINESS.md` and `ARCHITECTURE_AUDIT.md` before collecting results.
> The build-time tests were run under Python 3.13.5; verify under Python 3.11
> and against a live LM Studio + persistent ChromaDB setup on the target PC.
> The 18 Nov follow-up review suite reports 119 passed / 2 expected fallback
> warnings in an isolated test environment; it did not run live ChromaDB or
> LM Studio integration. A quick-validation exit code means execution completed,
> not that security or research acceptance criteria passed.

## 1. Create the Python environment

Use Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt -c constraints.txt
Copy-Item .env.example .env
```

## 2. Start LM Studio

- Load the configured model `qwen/qwen3-4b-2507`.
- Start the local OpenAI-compatible server on port `1234`.
- Confirm `config.yaml` uses `provider: lm_studio`, `base_url: http://127.0.0.1:1234/v1` and the exact loaded model identifier.

Check the actual model response:

```powershell
python scripts/check_llm_connection.py
```

Do not continue if this check fails. The runtime deliberately does not silently replace a failed real provider with mock text.

## 3. Prepare the scanner and calibration baselines

```powershell
python data/download_deepset.py
python train_scanner.py
python run_calibration_quick.py 80
```

## 4. Start the interactive application

```powershell
python main.py
```

## 5. Run diagnostics and experiments

```powershell
$env:TRUSTTRACE_STRICT_EVAL = "1"
python -m pytest -q
python scripts/run_final_validation.py  # diagnostic execution only
python scripts/run_ablation_matrix.py   # same scenario manifest across A0-A4
```

The ablation matrix defaults to quick diagnostic mode. Use `python scripts/run_ablation_matrix.py --full --trials 100` for the full configured experiment. The matrix stores stable scenario IDs and a scenario-manifest hash. Review the generated JSON files and document model, dataset, config and code versions before using results in research. The separate deepset held-out benchmark also binds results to the dataset, model and config hashes and rejects incompatible resume attempts.

See `README.md` for architecture, ablation definitions, security limitations and interpretation guidance.
