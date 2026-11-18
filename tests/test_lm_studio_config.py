import os
import yaml


def test_project_config_targets_confirmed_lm_studio_model():
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    assert cfg["llm"]["provider"] == "lm_studio"
    assert cfg["llm"]["base_url"] == "http://127.0.0.1:1234/v1"
    assert cfg["llm"]["model"] == "qwen/qwen3-4b-2507"


def test_openai_compatible_provider_alias_accepts_lm_studio(monkeypatch):
    import sys
    import types
    from llm import llm_provider

    captured = {}
    class FakeClient:
        def __init__(self, api_key, base_url):
            captured.update(api_key=api_key, base_url=base_url)
    fake_openai = types.SimpleNamespace(OpenAI=FakeClient)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr(llm_provider, "_load_llm_config", lambda: {
        "provider": "lm_studio",
        "model": "qwen/qwen3-4b-2507",
        "base_url": "http://127.0.0.1:1234/v1",
        "api_key": None,
    })
    llm_provider._provider_instance = None
    try:
        provider = llm_provider.get_llm_provider()
        assert provider.model == "qwen/qwen3-4b-2507"
        assert captured["base_url"] == "http://127.0.0.1:1234/v1"
        assert captured["api_key"] == "no-key"
    finally:
        llm_provider._provider_instance = None
