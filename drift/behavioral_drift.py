import hashlib
import json
import os
import re
import math
import warnings

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - import error or network issue
    SentenceTransformer = None

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASELINE_DIR = os.path.join(os.path.dirname(__file__), "..", "calibration", "baselines")
PROMPT_BASELINE_NAME = "UserPrompt"
BASELINE_MANIFEST_PATH = os.path.join(BASELINE_DIR, "manifest.json")
REQUIRED_BASELINES = ("Retriever", "Planner", "Executor", "Generator", PROMPT_BASELINE_NAME)


class _SimpleEmbedder:
    """Fallback embedder that uses character n-gram hashing when sentence-transformers is unavailable."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        vectors = []
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            tokens = [tok.lower() for tok in re.findall(r"[a-z0-9]+", text)]
            vec = np.zeros(self.dim, dtype=np.float32)
            for token in tokens:
                idx = sum(ord(ch) for ch in token) % self.dim
                vec[idx] += 1.0
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            vectors.append(vec)
        return np.vstack(vectors)


class BehavioralDriftModule:
    """
    Stores per-agent baseline embeddings (collected during calibration).
    Computes BD = 1 - cos(M_t, M_baseline) for every new agent output.
    """

    def __init__(self):
        strict_eval = os.environ.get("TRUSTTRACE_STRICT_EVAL", "").strip().lower() in {"1", "true", "yes"}
        if SentenceTransformer is None:
            if strict_eval:
                raise RuntimeError(
                    "Strict evaluation requires sentence-transformers. Install the project requirements; "
                    "the character-hash fallback is disabled for research runs."
                )
            self.embedder = _SimpleEmbedder()
            self.embedding_backend = "simple_hash_fallback"
            warnings.warn(
                "sentence-transformers is unavailable; using the non-semantic hash fallback. "
                "Do not use these baselines for research claims.", RuntimeWarning
            )
        else:
            try:
                self.embedder = SentenceTransformer(EMBED_MODEL)
                self.embedding_backend = EMBED_MODEL
            except Exception as exc:
                if strict_eval:
                    raise RuntimeError(
                        f"Could not initialise required embedding model {EMBED_MODEL!r}; "
                        "strict evaluation will not substitute a fallback."
                    ) from exc
                self.embedder = _SimpleEmbedder()
                self.embedding_backend = "simple_hash_fallback"
                warnings.warn(
                    f"Could not load {EMBED_MODEL}; using the non-semantic hash fallback. "
                    "Do not use these baselines for research claims.", RuntimeWarning
                )
        self.baselines: dict = {}
        self._drift_cache: dict = {}
        self.embedding_backend = getattr(self, "embedding_backend", EMBED_MODEL)
        os.makedirs(BASELINE_DIR, exist_ok=True)
        self._load_baselines()

    @staticmethod
    def _file_sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_baselines(self):
        if not os.path.isdir(BASELINE_DIR):
            return
        dimensions = set()
        for fname in os.listdir(BASELINE_DIR):
            if not fname.endswith(".json") or fname == "manifest.json":
                continue
            agent = fname[:-5]
            path = os.path.join(BASELINE_DIR, fname)
            with open(path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
            vec = np.asarray(raw, dtype=np.float32)
            if vec.ndim != 1 or vec.size == 0 or not np.isfinite(vec).all():
                raise RuntimeError(f"Invalid behavioral baseline file: {path}")
            norm = float(np.linalg.norm(vec))
            if not math.isfinite(norm) or norm <= 0.0:
                raise RuntimeError(f"Zero/invalid behavioral baseline vector: {path}")
            dimensions.add(int(vec.size))
            self.baselines[agent] = vec / norm
        if len(dimensions) > 1:
            raise RuntimeError(f"Behavioral baseline dimensions disagree: {sorted(dimensions)}")

    def save_baseline(self, agent_name: str, mean_embedding: np.ndarray):
        if not isinstance(agent_name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", agent_name):
            raise ValueError("agent_name must contain only letters, digits and underscores")
        vector = np.asarray(mean_embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("baseline embedding must be a finite non-empty vector")
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError("baseline embedding must have non-zero magnitude")
        vector = vector / norm
        path = os.path.join(BASELINE_DIR, f"{agent_name}.json")
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(vector.tolist(), handle)
        os.replace(temporary, path)
        self.baselines[agent_name] = vector
        self._drift_cache.clear()

    def collect_baseline(self, agent_name: str, outputs: list):
        if not outputs:
            raise ValueError(f"Cannot collect an empty baseline for {agent_name}")
        unique_outputs = sorted({str(output).strip() for output in outputs if str(output).strip()})
        if not unique_outputs:
            raise ValueError(f"No non-empty text available for baseline {agent_name}")
        if len(unique_outputs) < 5:
            print(f"  WARNING: Only {len(unique_outputs)} unique outputs for {agent_name}. Baseline may be unreliable.")

        embeddings = np.asarray(self.embedder.encode(
            unique_outputs,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        ), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(unique_outputs) or not np.isfinite(embeddings).all():
            raise RuntimeError(f"Embedding model returned invalid vectors for baseline {agent_name}")
        mean_emb = embeddings.mean(axis=0)
        norm = float(np.linalg.norm(mean_emb))
        if not math.isfinite(norm) or norm <= 0.0:
            raise RuntimeError(f"Could not construct a non-zero baseline for {agent_name}")
        self.save_baseline(agent_name, mean_emb)

    def finalize_baselines(self, required_names=REQUIRED_BASELINES) -> str:
        """Write a manifest binding baselines to the embedding backend and hashes."""
        missing = [name for name in required_names if name not in self.baselines]
        if missing:
            raise RuntimeError(f"Cannot finalize calibration; missing baselines: {missing}")
        files = {
            f"{name}.json": self._file_sha256(os.path.join(BASELINE_DIR, f"{name}.json"))
            for name in required_names
        }
        dimension = int(next(iter(self.baselines.values())).size)
        manifest = {
            "schema_version": 1,
            "embedding_backend": self.embedding_backend,
            "embedding_dimension": dimension,
            "required_baselines": list(required_names),
            "files_sha256": files,
        }
        temporary = BASELINE_MANIFEST_PATH + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        os.replace(temporary, BASELINE_MANIFEST_PATH)
        return BASELINE_MANIFEST_PATH

    def baselines_valid_for(self, required_names=REQUIRED_BASELINES) -> bool:
        """Verify required baseline files match this embedding backend manifest."""
        if not os.path.isfile(BASELINE_MANIFEST_PATH):
            return False
        try:
            with open(BASELINE_MANIFEST_PATH, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            if manifest.get("schema_version") != 1:
                return False
            if manifest.get("embedding_backend") != self.embedding_backend:
                return False
            if manifest.get("embedding_dimension") != next(iter(self.baselines.values())).size:
                return False
            hashes = manifest.get("files_sha256", {})
            for name in required_names:
                filename = f"{name}.json"
                path = os.path.join(BASELINE_DIR, filename)
                if name not in self.baselines or not os.path.isfile(path):
                    return False
                if hashes.get(filename) != self._file_sha256(path):
                    return False
            return True
        except (OSError, ValueError, TypeError, StopIteration, json.JSONDecodeError):
            return False

    def compute_drift_with_similarity(self, agent_name: str, current_output: str) -> tuple:
        """Returns (baseline_similarity, global_similarity, BD)."""
        current_output = str(current_output or "").strip()
        if not current_output or current_output in {"[]", "[QUARANTINED]"} or current_output.startswith("[BLOCKED]"):
            return 1.0, 1.0, 0.0
 
        cache_key = f"{agent_name}:{hashlib.md5(current_output.encode()).hexdigest()}"
        if cache_key in self._drift_cache:
            cached = self._drift_cache[cache_key]
            return cached["baseline_sim"], cached["global_sim"], cached["bd"]
 
        if agent_name not in self.baselines:
            # User-prompt drift must not fall back to an agent-output baseline.
            if agent_name == PROMPT_BASELINE_NAME:
                centroid = self.get_global_centroid().reshape(1, -1)
                if float(np.linalg.norm(centroid)) == 0.0:
                    if os.environ.get("TRUSTTRACE_STRICT_EVAL", "").strip().lower() in {"1", "true", "yes"}:
                        raise RuntimeError("UserPrompt/global behavioral baseline is missing; recalibrate before strict evaluation")
                    return 1.0, 1.0, 0.0
                M_t = self.embedder.encode([current_output], normalize_embeddings=True)
                global_cos_sim = cosine_similarity(M_t, centroid)[0][0]
                global_similarity = float(global_cos_sim)
                BD = max(0.0, min(1.0, 1.0 - global_similarity))
                return global_similarity, global_similarity, BD
            strict_eval = os.environ.get("TRUSTTRACE_STRICT_EVAL", "").strip().lower() in {"1", "true", "yes"}
            if strict_eval:
                raise RuntimeError(f"Behavioral baseline for {agent_name!r} is missing; recalibrate before strict evaluation")
            agent_name = next((name for name in ("Retriever", "Planner", "Executor", "Generator") if name in self.baselines), None)
        if agent_name not in self.baselines:
            if os.environ.get("TRUSTTRACE_STRICT_EVAL", "").strip().lower() in {"1", "true", "yes"}:
                raise RuntimeError("No agent behavioral baselines are loaded; recalibrate before strict evaluation")
            return 1.0, 1.0, 0.0
 
        M_baseline = self.baselines[agent_name].reshape(1, -1)
        M_t = self.embedder.encode([current_output], normalize_embeddings=True)
        baseline_cos_sim = cosine_similarity(M_t, M_baseline)[0][0]
        baseline_similarity = float(baseline_cos_sim)
        global_centroid = self.get_global_centroid().reshape(1, -1)
        global_cos_sim = cosine_similarity(M_t, global_centroid)[0][0]
        global_similarity = float(global_cos_sim)
        BD = 0.5 * (1.0 - baseline_similarity)
        BD = max(0.0, min(1.0, BD))

        if len(self._drift_cache) < 1000:
            self._drift_cache[cache_key] = {
                "baseline_sim": baseline_similarity,
                "global_sim": global_similarity,
                "bd": BD,
            }
        return baseline_similarity, global_similarity, BD

    def compute_drift(self, agent_name: str, current_output: str) -> float:
        _, _, bd = self.compute_drift_with_similarity(agent_name, current_output)
        return bd

    def has_baseline(self, agent_name: str) -> bool:
        return agent_name in self.baselines

    def get_global_centroid(self) -> np.ndarray:
        if not self.baselines:
            return np.zeros(384)
        # Weight UserPrompt baseline heavily when building the global centroid
        # so semantic/behavioral drift for raw user input tracks prompt space.
        vecs = []
        weights = []
        for name, vec in self.baselines.items():
            vecs.append(vec)
            weights.append(3.0 if name == PROMPT_BASELINE_NAME else 1.0)
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.sum()
        mean_emb = np.average(np.stack(vecs), axis=0, weights=weights)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
        return mean_emb
