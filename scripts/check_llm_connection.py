"""Verify that the configured real LLM endpoint can answer a request."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    os.environ.setdefault("TRUSTTRACE_STRICT_EVAL", "1")
    from llm.llm_provider import MockProvider, get_llm_provider

    provider = get_llm_provider(force_reload=True)
    if isinstance(provider, MockProvider):
        raise RuntimeError("MockProvider is not accepted by this connection check.")

    response = provider.generate(
        "Reply with exactly: TRUSTTRACE_LLM_CONNECTION_OK",
        system_prompt="You are performing a connectivity check. Do not add explanation.",
    )
    if not isinstance(response, str) or not response.strip():
        raise RuntimeError("The LLM endpoint returned an empty response.")

    print(f"Provider class: {provider.__class__.__name__}")
    print(f"Configured model: {getattr(provider, 'model', 'unknown')}")
    print("Response:", response.strip()[:300])
    print("LLM endpoint connection: SUCCESS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"LLM endpoint connection: FAILED ({exc})", file=sys.stderr)
        raise SystemExit(1)
