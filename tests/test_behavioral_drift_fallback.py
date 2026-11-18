import importlib

import numpy as np
import pytest


def test_behavioral_drift_falls_back_when_sentence_transformer_is_unavailable(monkeypatch):
    # This test verifies the non-strict fallback even when the project .env
    # enables strict mode for real research runs.
    monkeypatch.delenv("TRUSTTRACE_STRICT_EVAL", raising=False)
    drift_module = importlib.import_module("drift.behavioral_drift")

    class Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("download failed")

    monkeypatch.setattr(drift_module, "SentenceTransformer", Boom)

    module = drift_module.BehavioralDriftModule()
 
    embeddings = module.embedder.encode(["hello world", "another sample"])
    assert embeddings.shape == (2, 384)
    assert isinstance(module.baselines, dict)
 

def test_behavioral_drift_normalizes_cosine_similarity(monkeypatch):
    monkeypatch.delenv("TRUSTTRACE_STRICT_EVAL", raising=False)
    drift_module = importlib.import_module("drift.behavioral_drift")
 
    class Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("download failed")
 
    monkeypatch.setattr(drift_module, "SentenceTransformer", Boom)
 
    module = drift_module.BehavioralDriftModule()
    module.baselines[drift_module.PROMPT_BASELINE_NAME] = module.embedder.encode(["hello"], normalize_embeddings=True)[0]
 
    _, _, bd_same = module.compute_drift_with_similarity(drift_module.PROMPT_BASELINE_NAME, "hello")
    _, _, bd_different = module.compute_drift_with_similarity(drift_module.PROMPT_BASELINE_NAME, "world")
 
    assert bd_same == pytest.approx(0.0, abs=1e-6)
    assert bd_different == pytest.approx(0.5, abs=1e-6)


def test_strict_mode_refuses_nonsemantic_fallback(monkeypatch):
    drift_module = importlib.import_module("drift.behavioral_drift")

    class Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(drift_module, "SentenceTransformer", Boom)
    monkeypatch.setenv("TRUSTTRACE_STRICT_EVAL", "1")
    with pytest.raises(RuntimeError, match="strict evaluation will not substitute a fallback"):
        drift_module.BehavioralDriftModule()
