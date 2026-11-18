import os
import logging

logger = logging.getLogger(__name__)

def _get_version_file_path(chroma_path: str) -> str:
    return os.path.join(chroma_path, "__chroma_version.txt")

def _store_current_version(chroma_path: str, version: str) -> None:
    version_file = _get_version_file_path(chroma_path)
    temporary = version_file + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(version)
    os.replace(temporary, version_file)

def _load_stored_version(chroma_path: str) -> str | None:
    version_file = _get_version_file_path(chroma_path)
    if os.path.exists(version_file):
        try:
            with open(version_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            logger.warning(f"Failed to read ChromaDB version file: {e}")
    return None

def ensure_chroma_db(chroma_path: str) -> None:
    """Ensure that a usable ChromaDB persistence directory exists.

    If the directory is missing, it will be created. Version mismatches and
    initialization errors are surfaced without moving or resetting user data.
    """
    os.makedirs(chroma_path, exist_ok=True)
    # Check version consistency
    stored_version = _load_stored_version(chroma_path)
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("ChromaDB is required to inspect the persistent database") from exc
    current_version = chromadb.__version__
    version_mismatch = stored_version is not None and stored_version != current_version
    if version_mismatch:
        raise RuntimeError(
            f"ChromaDB version mismatch (stored={stored_version}, current={current_version}). "
            "The database was left untouched. Back it up and follow the documented migration path."
        )
    # Try to instantiate a client to detect corruption. Never move or reset
    # persistent user data automatically on a generic initialization error.
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        _ = client.get_or_create_collection("_health_check")
    except Exception as exc:
        raise RuntimeError(
            f"ChromaDB initialization failed at {chroma_path}; database was not modified."
        ) from exc
    # Store the current version for future checks
    _store_current_version(chroma_path, current_version)
