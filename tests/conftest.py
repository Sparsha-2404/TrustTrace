"""Isolate unit tests from model downloads and machine-specific .env settings.

Production/research runs still require the real SentenceTransformer model in
strict mode. Unit tests use a deterministic test-only encoder so that trust,
recovery and validation tests do not need network access or a model cache.
"""

import re

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def deterministic_test_embedding_backend(monkeypatch):
    import drift.behavioral_drift as drift_module

    class DeterministicTestEmbedder:
        def __init__(self, model_name=None):
            self.model_name = model_name

        def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
            vectors = []
            for text in texts:
                vector = np.zeros(384, dtype=np.float32)
                for token in re.findall(r"[a-z0-9]+", str(text).lower()):
                    index = sum(ord(char) for char in token) % 384
                    vector[index] += 1.0
                if normalize_embeddings:
                    norm = np.linalg.norm(vector)
                    if norm:
                        vector /= norm
                vectors.append(vector)
            return np.vstack(vectors)

    monkeypatch.setattr(drift_module, "SentenceTransformer", DeterministicTestEmbedder)
