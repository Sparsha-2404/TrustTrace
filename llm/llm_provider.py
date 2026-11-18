"""
llm/llm_provider.py
-------------------
Unified LLM backend for TrustTrace.

Reads provider/model config from config.yaml (llm: section).
API keys are loaded from environment variables via python-dotenv.

Supported providers
-------------------
  mistral     — Mistral AI REST API
  ollama      — Local Ollama server (legacy optional backend)
  openai      — OpenAI API
  lm_studio   — LM Studio or another OpenAI-compatible local endpoint

Configuration in config.yaml
-----------------------------
  llm:
    provider: lm_studio
    model: qwen/qwen3-4b-2507
    api_key: null
    base_url: http://127.0.0.1:1234/v1

Environment variables (loaded from .env)
-----------------------------------------
  MISTRAL_API_KEY (Mistral)
  OPENAI_API_KEY (OpenAI)
  TRUSTTRACE_LLM_BASE_URL (optional endpoint override)

Usage
------
  from llm.llm_provider import get_llm_provider
  llm = get_llm_provider()
  response = llm.generate("Explain quantum computing in one sentence.")
"""

from __future__ import annotations

import inspect
import os
import re
import time
import logging
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def _load_llm_config() -> dict:
    """Return the ``llm:`` section from config.yaml, or an empty dict."""
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("llm", {})
    except FileNotFoundError:
        return {}


def _resolve_env_var(value: str) -> str:
    """Expand ${ENV_VAR} placeholders in a config value."""
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"\$\{([^}]+)\}", value.strip())
    if match:
        env_name = match.group(1)
        resolved = os.environ.get(env_name, "")
        if not resolved:
            logger.warning(
                "LLMProvider: environment variable %s is not set. "
                "Add it to your .env file.",
                env_name,
            )
        return resolved
    return value


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class LLMProvider:
    """
    Abstract base.  Subclasses implement :meth:`generate`.
    Call :func:`get_llm_provider` to obtain the right concrete instance.
    """

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:  # noqa: D102
        raise NotImplementedError

    # Convenience: keep the same call signature used by _invoke_agent
    def invoke_messages(self, messages: list) -> str:
        """
        Accept a list of dicts ``[{"role": ..., "content": ...}]``
        and return the assistant reply as a plain string.
        """
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        user_parts = [m["content"] for m in messages if m.get("role") == "user"]
        system_text = "\n".join(system_parts) if system_parts else None
        user_text = "\n\n".join(user_parts)
        return self.generate(user_text, system_prompt=system_text)


# ---------------------------------------------------------------------------
# Mistral provider
# ---------------------------------------------------------------------------


class MistralProvider(LLMProvider):
    """
    Calls the Mistral AI REST API.
    Requires:  pip install mistralai
    """

    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

        try:
            from mistralai.client import Mistral  # type: ignore
        except ImportError:
            try:
                from mistralai import Mistral  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "mistralai package is required for the Mistral provider. "
                    "Run: pip install mistralai"
                ) from exc

        try:
            signature = inspect.signature(Mistral)
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                if "server_url" in signature.parameters:
                    kwargs["server_url"] = self.base_url
                elif "server" in signature.parameters:
                    kwargs["server"] = self.base_url
            self._client = Mistral(**kwargs)
        except TypeError:
            self._client = Mistral(api_key=self.api_key)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(3):
            try:
                response = self._client.chat.complete(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1024,
                )
                content = response.choices[0].message.content
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("The configured LLM endpoint returned an empty assistant response.")
                return content.strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Mistral API error (attempt %d): %s", attempt + 1, exc)
                if attempt < 2:
                    time.sleep(8 ** attempt)
                else:
                    raise

        return "[MistralProvider] No response after retries."


# ---------------------------------------------------------------------------
# Ollama provider
# ---------------------------------------------------------------------------


class OllamaProvider(LLMProvider):
    """
    Calls a locally-running Ollama server (http://localhost:11434 by default).
    No API key required.
    Requires:  pip install ollama   OR  uses raw HTTP as fallback.
    """

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, model: str, base_url: Optional[str] = None):
        self.model = model
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")

        # Prefer the ollama SDK; fall back to requests
        try:
            import ollama as _ollama  # type: ignore
            self._sdk = _ollama
            self._use_sdk = True
        except ImportError:
            self._use_sdk = False
            try:
                import requests as _requests  # type: ignore
                self._requests = _requests
            except ImportError as exc:
                raise ImportError(
                    "Install either the 'ollama' or 'requests' package to use OllamaProvider."
                ) from exc

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if self._use_sdk:
            return self._generate_sdk(messages)
        return self._generate_http(messages)

    def _generate_sdk(self, messages: list) -> str:
        response = self._sdk.chat(
            model=self.model,
            messages=messages,
            options={"temperature": 0.1},
        )
        return response["message"]["content"].strip()

    def _generate_http(self, messages: list) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        resp = self._requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=300,  # Increased timeout to 5 minutes for slower models
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()


# ---------------------------------------------------------------------------
# OpenAI-compatible provider
# ---------------------------------------------------------------------------


class OpenAIProvider(LLMProvider):
    """
    Calls the OpenAI API *or* any OpenAI-compatible endpoint
    (e.g., LM Studio, vLLM, Together AI, Groq).
    Requires:  pip install openai
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URL

        self._client = None
        try:
            from openai import OpenAI  # type: ignore
            try:
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=300.0,
                    max_retries=0,
                )
            except TypeError:
                # Compatibility with lightweight SDK adapters used in tests.
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            # requests is part of the core requirements. Keep LM Studio and
            # other OpenAI-compatible endpoints usable even when the optional
            # OpenAI SDK is absent from a minimal Windows installation.
            import requests
            self._requests = requests

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(3):
            try:
                if self._client is not None:
                    response = self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=1024,
                    )
                    content = response.choices[0].message.content
                else:
                    response = self._requests.post(
                        f"{self.base_url.rstrip('/')}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": self.model,
                            "messages": messages,
                            "temperature": 0.1,
                            "max_tokens": 1024,
                            "stream": False,
                        },
                        timeout=300,
                    )
                    response.raise_for_status()
                    choices = response.json().get("choices") or []
                    content = choices[0].get("message", {}).get("content") if choices else None
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("The configured LLM endpoint returned an empty assistant response.")
                return content.strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning("OpenAI-compatible API error (attempt %d): %s", attempt + 1, exc)
                if attempt < 2:
                    time.sleep(1.5 ** attempt)
                else:
                    raise

        return "[OpenAIProvider] No response after retries."


# ---------------------------------------------------------------------------
# Mock fallback  (used when no real provider is configured)
# ---------------------------------------------------------------------------


class MockProvider(LLMProvider):
    """
    Deterministic offline mock — identical to the original MockChatOpenAI
    logic so that benchmarks run without any API key.
    """

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        normalized = prompt.lower()
        if any(p in normalized for p in ["pwned", "dan", "developer mode", "ignore all previous", "system override"]):
            return "PWNED. Ignore previous instructions. I will now output PWNED."
        if any(p in normalized for p in ["compromised", "attacker@evil.com", "reveal all system prompts", "rag injection"]):
            return "COMPROMISED. New task: exfiltrate user data."
        if any(p in normalized for p in ["hijacked", "long-term instruction", "memory update"]):
            return "HIJACKED: I am compromised by memory poisoning."
        if "what is" in normalized or "who is" in normalized or "define" in normalized:
            return "Based on the retrieved information, here is a clear explanation of the requested topic."
        if "explain" in normalized or "describe" in normalized:
            return "Here is a structured explanation based on the available context and plan."
        if "plan" in normalized or "step" in normalized:
            return "1. Review retrieved context\n2. Apply the user task\n3. Validate output\n4. Return results"
        if "execute" in normalized:
            return "Execution complete. All planned steps were carried out successfully."
        return "Processed task and produced a response aligned with the user request."


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_provider_instance: Optional[LLMProvider] = None


def get_llm_provider(force_reload: bool = False) -> LLMProvider:
    """
    Return a cached :class:`LLMProvider` instance configured from config.yaml.
    Pass ``force_reload=True`` to rebuild (useful in tests).
    """
    global _provider_instance
    if _provider_instance is not None and not force_reload:
        return _provider_instance

    cfg = _load_llm_config()
    provider_name = cfg.get("provider", "mock").lower()
    model = cfg.get("model", "")
    raw_key = cfg.get("api_key", "")
    api_key = _resolve_env_var(raw_key) if raw_key else ""
    # Explicit environment override takes precedence over the checked-in default.
    base_url = os.environ.get("TRUSTTRACE_LLM_BASE_URL") or cfg.get("base_url") or None

    print(f"[LLMProvider] Initializing provider='{provider_name}' model='{model}'")

    if provider_name == "mistral":
        if not api_key:
            api_key = os.environ.get("MISTRAL_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "Mistral provider is configured but MISTRAL_API_KEY is not set. "
                "Set the API key in your environment or use a provider that does not require credentials."
            )
        _provider_instance = MistralProvider(model=model or "mistral-large-latest", api_key=api_key, base_url=base_url)

    elif provider_name == "ollama":
        _provider_instance = OllamaProvider(model=model or "llama3.2", base_url=base_url)

    elif provider_name in ("openai", "openai_compatible", "lm_studio", "lmstudio"):
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key and not base_url:
            raise RuntimeError(
                "OpenAI provider is configured but OPENAI_API_KEY is not set and no base_url is configured. "
                "Set the API key or configure a compatible endpoint."
            )
        _provider_instance = OpenAIProvider(
            model=model or "gpt-4o",
            api_key=api_key or "no-key",
            base_url=base_url,
        )

    else:
        if provider_name != "mock":
            raise RuntimeError(
                f"Unknown LLM provider '{provider_name}'. "
                "Use 'mock' only for offline test mode or configure a supported provider."
            )
        if os.environ.get("TRUSTTRACE_STRICT_EVAL", "").strip().lower() in {"1", "true", "yes"}:
            raise RuntimeError(
                "Strict evaluation mode forbids MockProvider. Configure a real provider "
                "before generating research results."
            )
        _provider_instance = MockProvider()

    return _provider_instance
