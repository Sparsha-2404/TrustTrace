# TrustTrace V3 Setup Guide

The authoritative setup and run instructions are maintained in [`README.md`](README.md) and [`README_QUICKSTART.md`](README_QUICKSTART.md).

Recommended environment: Python 3.11, Windows PowerShell, LM Studio local server, model `qwen/qwen3-4b-2507`.

Quick setup:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt -c constraints.txt
Copy-Item .env.example .env
python scripts/check_llm_connection.py
```

Do not start experiments until the real LLM connection check succeeds. See `README.md` for scanner training, calibration, interactive use, A0–A4 experiments and research limitations.
