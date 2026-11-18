import json
import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

MODEL_PATH = os.path.join(os.path.dirname(__file__), "scanner_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scanner_scaler.pkl")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class InjectionScanner:
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required to initialise InjectionScanner. "
                "Install project dependencies with: python -m pip install sentence-transformers"
            ) from exc
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.clf = None
        self.scaler = None
        self._load_if_exists()

    def _load_if_exists(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            classifier = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            if not hasattr(classifier, "predict_proba") or not hasattr(classifier, "classes_"):
                raise RuntimeError("Saved scanner model is invalid; retrain with python train_scanner.py")
            if set(np.asarray(classifier.classes_).tolist()) != {0, 1}:
                raise RuntimeError("Saved scanner model is not binary; retrain with python train_scanner.py")
            if not hasattr(scaler, "transform") or not hasattr(scaler, "n_features_in_"):
                raise RuntimeError("Saved scanner scaler is invalid; retrain with python train_scanner.py")
            if getattr(classifier, "n_features_in_", None) != scaler.n_features_in_:
                raise RuntimeError("Saved scanner model/scaler feature dimensions disagree; retrain")
            probe = np.asarray(self.embedder.encode(["TrustTrace scanner dimension probe"]), dtype=float)
            if probe.ndim != 2 or probe.shape[1] != scaler.n_features_in_:
                raise RuntimeError(
                    "Saved scanner artifacts do not match the loaded embedding model; "
                    "retrain with python train_scanner.py"
                )
            self.clf, self.scaler = classifier, scaler

    @staticmethod
    def _normalise_label(value):
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            return int(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"0", "benign", "safe", "normal"}:
                return 0
            if normalized in {"1", "injection", "malicious", "unsafe", "attack"}:
                return 1
        raise ValueError(f"Unsupported scanner label: {value!r}; expected binary 0/1 labels")

    def train(self, data_path: str = "data/deepset_injections/train.json"):
        """Train a binary injection classifier from a verified labelled dataset."""
        if not os.path.isfile(data_path):
            raise FileNotFoundError(
                f"Scanner training data not found: {data_path}. "
                "Run python data/download_deepset.py first."
            )
        with open(data_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list) or not data:
            raise ValueError("Scanner dataset must be a non-empty JSON list of records")

        texts, labels = [], []
        for index, row in enumerate(data):
            if not isinstance(row, dict):
                raise ValueError(f"Dataset row {index} must be an object")
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Dataset row {index} has missing/empty text")
            if "label" not in row:
                raise ValueError(f"Dataset row {index} has no label")
            texts.append(text.strip())
            labels.append(self._normalise_label(row["label"]))

        classes, counts = np.unique(labels, return_counts=True)
        if set(classes.tolist()) != {0, 1}:
            raise ValueError(
                "Scanner training requires both benign (0) and injection (1) classes; "
                f"found labels/counts: {dict(zip(classes.tolist(), counts.tolist()))}"
            )
        if min(counts) < 2:
            raise ValueError("Scanner training requires at least two examples per class")

        # Add a small set of deterministic benign queries only after validating
        # the downloaded binary dataset; never replace the real dataset.
        try:
            from experiment.scenarios import get_benign_queries
            seen = set(texts)
            for query in get_benign_queries(20):
                if isinstance(query, str) and query.strip() and query not in seen:
                    texts.append(query)
                    labels.append(0)
                    seen.add(query)
        except (ImportError, AttributeError):
            pass

        print(f"Training scanner on {len(texts)} samples ({sum(labels)} injections)")
        embeddings = self.embedder.encode(texts, batch_size=64, show_progress_bar=True)
        embeddings = np.asarray(embeddings, dtype=float)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(texts) or not np.isfinite(embeddings).all():
            raise RuntimeError("Embedding model returned invalid training vectors")

        scaler = StandardScaler()
        X = scaler.fit_transform(embeddings)
        classifier = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
        classifier.fit(X, labels)
        if set(classifier.classes_.tolist()) != {0, 1}:
            raise RuntimeError("Trained scanner classifier is not binary")

        model_tmp = MODEL_PATH + ".tmp"
        scaler_tmp = SCALER_PATH + ".tmp"
        joblib.dump(classifier, model_tmp)
        joblib.dump(scaler, scaler_tmp)
        os.replace(model_tmp, MODEL_PATH)
        os.replace(scaler_tmp, SCALER_PATH)
        self.clf, self.scaler = classifier, scaler
        print("Scanner trained and saved.")

    def score(self, text: str) -> float:
        """Return the calibrated model probability for the injection class."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if self.clf is None or self.scaler is None:
            raise RuntimeError("Scanner model/scaler not loaded. Train the scanner first.")
        embedding = np.asarray(self.embedder.encode([text]), dtype=float)
        if embedding.ndim != 2 or embedding.shape[0] != 1 or not np.isfinite(embedding).all():
            raise RuntimeError("Embedding model returned an invalid vector")
        X = self.scaler.transform(embedding)
        class_indices = np.where(np.asarray(self.clf.classes_) == 1)[0]
        if len(class_indices) != 1:
            raise RuntimeError("Loaded scanner model has no unique injection class (label 1)")
        probability = float(self.clf.predict_proba(X)[0][int(class_indices[0])])
        if not np.isfinite(probability):
            raise RuntimeError("Scanner returned a non-finite probability")
        return min(1.0, max(0.0, probability))

    def score_and_update(self, event_id: int, text: str, logger) -> float:
        """Score text and write result back to the interaction log."""
        s = self.score(text)
        logger.update_suspicion(event_id, s)
        return s
