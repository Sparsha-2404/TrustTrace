"""Shared persistent ChromaDB store used by the pipeline and memory manager.

The application uses one persistent database and one collection for all
retrieval/memory operations. Tests may inject an in-memory collection, but the
normal application bootstrap must use this persistent store.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

COLLECTION_NAME = "pipeline_memory"


def default_chroma_path() -> str:
    configured = os.environ.get("TRUSTTRACE_CHROMA_PATH")
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    return str(Path(__file__).resolve().parents[1] / "data" / "chroma_db")


@lru_cache(maxsize=4)
def get_chroma_store(collection_name: str = COLLECTION_NAME, path: str | None = None) -> tuple[Any, Any]:
    """Return the singleton PersistentClient and its shared collection.

    Import and initialization errors intentionally propagate. The caller may
    choose an in-memory test fallback, but production bootstrap must not hide
    a missing/corrupt persistent store.
    """
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "ChromaDB is required for TrustTrace persistent memory. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc

    path = os.path.abspath(os.path.expanduser(path)) if path else default_chroma_path()
    os.makedirs(path, exist_ok=True)
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(name=collection_name)
    return client, collection


def reset_store_cache() -> None:
    """Clear the cached client (primarily for isolated tests)."""
    get_chroma_store.cache_clear()
