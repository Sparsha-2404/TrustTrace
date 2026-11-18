"""
Injection Risk Score (IRS) module.

IRS = alpha * R + beta * E + gamma * SD

  R  = Rule-based pattern score
  E  = Embedding classifier probability (injection likelihood)
  SD = Semantic drift from global benign baseline
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml
from sklearn.metrics.pairwise import cosine_similarity

import config
from drift.behavioral_drift import PROMPT_BASELINE_NAME

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+|every\s+)?(previous|prior|earlier|above|all)\s+instructions?",
    r"disregard\s+(all\s+|any\s+|every\s+)?(previous|prior|earlier|above)\s+instructions?",
    r"forget\s+(all\s+|any\s+|every\s+)?(previous|prior|earlier|above)\s+instructions?",
    r"forget\s+(your\s+)?(original\s+)?system\s+prompt",
    r"follow\s+only\s+(my|these|your)\s+(commands|instructions)",
    r"system\s+override",
    r"developer\s+mode",
    r"you\s+are\s+now\s+dan",
    r"disregard\s+your\s+role",
    r"reveal\s+(all\s+)?system\s+prompts",
    r"exfiltrate",
    r"exfiltrate\s+(user\s+)?data",
    r"attacker@",
    r"rag\s+injection",
    r"hidden\s+instruction",
    r"new\s+task",
    r"new\s+instructions",
    r"memory\s+update",
    r"long-term\s+instruction",
    r"attacker\s+control",
    r"arbitrary\s+command",
    r"secret\s+data",
    r"persistent\s+override",
    r"pwned",
    r"compromised",
    r"hijacked",
    r"pretend\s+you\s+are\s+an?\s+ai\s+with\s+no",
    r"bypass\s+content\s+filters",
    # Additional patterns for better detection
    r"show\s+me\s+your\s+instructions",
    r"what\s+are\s+your\s+instructions",
    r"print\s+your\s+system\s+prompt",
    r"output\s+your\s+hidden\s+prompt",
    r"tell\s+me\s+what\s+you\s+were\s+told",
    r"expose\s+your\s+system\s+prompt",
    r"forget\s+your\s+instructions",
    r"abandon\s+your\s+constraints",
    r"override\s+your\s+programming",
    r"act\s+as\s+if\s+you\s+have\s+no\s+rules",
    r"jailbreak",
    r"bypass\s+security",
    r"escalate\s+privileges",
    r"grant\s+me\s+admin",
    r"give\s+me\s+root",
]

HIGH_CONFIDENCE_PATTERNS = [
    r"ignore\s+(all\s+|any\s+|every\s+)?(previous|prior|earlier|above|all)\s+instructions?",
    r"disregard\s+(all\s+|any\s+|every\s+)?(previous|prior|earlier|above)\s+instructions?",
    r"forget\s+(all\s+|any\s+|every\s+)?(previous|prior|earlier|above)\s+instructions?",
    r"forget\s+(your\s+)?(original\s+)?system\s+prompt",
    r"follow\s+only\s+(my|these|your)\s+(commands|instructions)",
    r"system\s+override",
    r"developer\s+mode",
    r"reveal\s+(all\s+)?system\s+prompts",
    r"exfiltrate(\s+(user\s+)?data)?",
    r"rag\s+injection",
    r"hidden\s+instruction",
    r"memory\s+update",
    r"long-term\s+instruction",
    r"new\s+task",
    r"attacker\s+control",
    r"arbitrary\s+command",
    r"secret\s+data",
    r"persistent\s+override",
    r"pwned",
    r"compromised",
    r"hijacked",
]


@dataclass
class IRSResult:
    rule_score: float
    embedding_score: float
    semantic_drift: float
    irs: float
    source: str
    text_preview: str


@dataclass
class MSResult:
    irs: float
    sd: float
    bd: float
    ms: float


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_rule_score(text: str) -> float:
    """R: pattern-based injection likelihood in [0, 1].
    
    Fixed to return non-zero values for malicious prompts.
    Now uses a more sensitive scoring mechanism.
    """
    if text is None or text == "":
        return 0.0
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        return 0.0
    normalized = text.lower()
    hits = sum(1 for p in INJECTION_PATTERNS if re.search(p, normalized))
    strong_hits = sum(1 for p in HIGH_CONFIDENCE_PATTERNS if re.search(p, normalized))
    
    # Improved scoring: even single hits should give meaningful scores
    if strong_hits:
        # Strong patterns get high scores
        return min(1.0, 0.7 + 0.15 * strong_hits)
    elif hits >= 1:
        # Regular patterns get proportional scores
        # Single hit = 0.3, 2 hits = 0.5, 3+ hits = 0.7+
        if hits == 1:
            return 0.35
        elif hits == 2:
            return 0.50
        else:
            return min(1.0, 0.60 + (hits - 2) * 0.1)
    
    return 0.0


class InjectionRiskAssessor:
    """Computes IRS and MS using methodology equations.
    
    Enhanced to use multiple factors:
    - Rule confidence
    - Semantic similarity
    - Intent confidence
    - Behavioral drift
    - Trust history
    - Propagation risk
    """

    def __init__(self, scanner=None, drift_module=None, embedder=None):
        cfg = _load_config()
        # Weights for different factors
        self.alpha = cfg.get("alpha", 0.25)  # Rule score weight
        self.beta = cfg.get("beta", 0.25)   # Embedding score weight
        self.gamma = cfg.get("gamma", 0.15) # Semantic drift weight
        self.delta = cfg.get("delta_intent", 0.15)  # Intent confidence weight
        self.epsilon = cfg.get("epsilon", 0.10) # Behavioral drift weight
        self.zeta = cfg.get("zeta", 0.10)    # Propagation risk weight
        
        self.lambda1 = cfg.get("lambda1", 0.40)
        self.lambda2 = cfg.get("lambda2", 0.20)
        self.lambda3 = cfg.get("lambda3", 0.20)
        self.lambda4 = cfg.get("lambda4", 0.10)
        self.lambda5 = cfg.get("lambda5", 0.10)

        for name in (
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
            "lambda1", "lambda2", "lambda3", "lambda4", "lambda5",
        ):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"IRS/MS weight {name} must be between 0 and 1") from exc
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"IRS/MS weight {name} must be between 0 and 1")
            setattr(self, name, value)
        
        self.scanner = scanner
        self.drift = drift_module
        self.embedder = embedder
        self._global_benign_centroid: Optional[np.ndarray] = None
        self._embedding_cache: dict = {}

    def _encode_cached(self, text: str):
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        emb = self.embedder.encode([text], batch_size=1, show_progress_bar=False, normalize_embeddings=True)
        if len(self._embedding_cache) < 500:
            self._embedding_cache[cache_key] = emb
        return emb

    def set_global_benign_centroid(self, centroid: np.ndarray) -> None:
        self._global_benign_centroid = centroid

    def _semantic_drift(self, text: str) -> float:
        """SD: drift from UserPrompt baseline or global benign centroid."""
        if self.embedder is None:
            return 0.0

        if self.drift is not None and PROMPT_BASELINE_NAME in self.drift.baselines:
            baseline = self.drift.baselines[PROMPT_BASELINE_NAME].reshape(1, -1)
        elif self._global_benign_centroid is not None:
            baseline = self._global_benign_centroid.reshape(1, -1)
        elif self.drift is not None and self.drift.baselines:
            agent = next(iter(self.drift.baselines))
            baseline = self.drift.baselines[agent].reshape(1, -1)
        else:
            return 0.0

        emb = self._encode_cached(text)
        if emb.ndim != 2 or baseline.ndim != 2 or emb.shape[1] != baseline.shape[1]:
            raise RuntimeError("IRS embedding dimension does not match the configured benign baseline")
        cos_sim = cosine_similarity(emb, baseline)[0][0]
        return max(0.0, min(1.0, 1.0 - float(cos_sim)))

    def _embedding_score(self, text: str) -> float:
        """E: classifier-based injection probability."""
        if self.scanner is not None and self.scanner.clf is not None:
            return self.scanner.score(text)
        # An embedding norm is not a calibrated injection probability. If no
        # trained classifier is available, mark this channel unavailable (0)
        # rather than manufacturing a pseudo-probability from vector magnitude.
        return 0.0

    def compute_irs(
        self, 
        text: str, 
        source: str = "message",
        intent_confidence: float = 0.0,
        behavioral_drift: float = 0.0,
        trust_score: float = 1.0,
        propagation_risk: float = 0.0,
    ) -> IRSResult:
        """Enhanced IRS computation using multiple factors.
        
        IRS = alpha*R + beta*E + gamma*SD + delta*I + epsilon*BD + zeta*PR
        
        Where:
        - R = Rule score
        - E = Embedding score
        - SD = Semantic drift
        - I = Intent confidence
        - BD = Behavioral drift
        - PR = Propagation risk
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        for name, value in (
            ("intent_confidence", intent_confidence),
            ("behavioral_drift", behavioral_drift),
            ("trust_score", trust_score),
            ("propagation_risk", propagation_risk),
        ):
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be between 0 and 1") from exc
            if not 0.0 <= numeric <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        intent_confidence = float(intent_confidence)
        behavioral_drift = float(behavioral_drift)
        trust_score = float(trust_score)
        propagation_risk = float(propagation_risk)

        R = compute_rule_score(text)
        E = self._embedding_score(text)
        SD = self._semantic_drift(text)
        
        # Compute enhanced IRS with all factors
        irs = (
            self.alpha * R + 
            self.beta * E + 
            self.gamma * SD + 
            self.delta * intent_confidence + 
            self.epsilon * behavioral_drift + 
            self.zeta * propagation_risk
        )
        
        # Adjust based on trust score (lower trust = higher IRS)
        trust_penalty = (1.0 - trust_score) * 0.2
        irs = irs + trust_penalty
        
        irs = max(0.0, min(1.0, irs))
        
        if config.DEBUG and not getattr(config, "EXPERIMENT_MODE", False):
            print(f"[IRS TRACE] Payload: {text[:60]}... | R={R:.3f} E={E:.3f} SD={SD:.3f} I={intent_confidence:.3f} BD={behavioral_drift:.3f} PR={propagation_risk:.3f} Trust={trust_score:.3f} => IRS={irs:.3f}")
        
        preview = text[:120].replace("\n", " ") if text else ""
        return IRSResult(
            rule_score=round(R, 4),
            embedding_score=round(E, 4),
            semantic_drift=round(SD, 4),
            irs=round(irs, 4),
            source=source,
            text_preview=preview,
        )

    def compute_ms(
        self, 
        irs: float, 
        sd: float, 
        bd: float,
        intent_confidence: float = 0.0,
        propagation_risk: float = 0.0,
    ) -> MSResult:
        """Enhanced MS computation using multiple factors.
        
        MS = lambda1*IRS + lambda2*SD + lambda3*BD + lambda4*I + lambda5*PR
        """
        values = {
            "irs": irs,
            "sd": sd,
            "bd": bd,
            "intent_confidence": intent_confidence,
            "propagation_risk": propagation_risk,
        }
        for name, value in values.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be between 0 and 1") from exc
            if not 0.0 <= numeric <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
            values[name] = numeric
        irs = values["irs"]
        sd = values["sd"]
        bd = values["bd"]
        intent_confidence = values["intent_confidence"]
        propagation_risk = values["propagation_risk"]
        ms = (
            self.lambda1 * irs
            + self.lambda2 * sd
            + self.lambda3 * bd
            + self.lambda4 * intent_confidence
            + self.lambda5 * propagation_risk
        )
        ms = max(0.0, min(1.0, ms))
        return MSResult(
            irs=round(irs, 4),
            sd=round(sd, 4),
            bd=round(bd, 4),
            ms=round(ms, 4),
        )

    @staticmethod
    def compute_memory_trust(irs: float, agent_trust: float) -> float:
        """MT_i = T_a * (1 - IRS), with validated [0, 1] inputs."""
        try:
            irs = float(irs)
            agent_trust = float(agent_trust)
        except (TypeError, ValueError) as exc:
            raise ValueError("irs and agent_trust must be between 0 and 1") from exc
        if not 0.0 <= irs <= 1.0 or not 0.0 <= agent_trust <= 1.0:
            raise ValueError("irs and agent_trust must be between 0 and 1")
        return round(agent_trust * (1.0 - irs), 4)
