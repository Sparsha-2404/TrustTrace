"""
Decision Engine - Centralized enforcement logic for TrustTrace.

This module separates monitoring from enforcement. Detection modules produce evidence,
and the Decision Engine consumes that evidence to determine execution flow.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set
import time

import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

_DEFAULT_THRESHOLDS = {
    "irs_high": 0.6,
    "irs_medium": 0.4,
    "trust_critical": 0.3,
    "trust_warning": 0.5,
    "bd_critical": 0.5,
    "ms_critical": 0.5,
}


def _load_decision_thresholds() -> Dict[str, float]:
    """Load and validate enforcement thresholds from config.yaml."""
    thresholds = dict(_DEFAULT_THRESHOLDS)
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Unable to load decision thresholds from {_CONFIG_PATH}") from exc

    configured = cfg.get("decision_engine", {}) or {}
    if not isinstance(configured, dict):
        raise ValueError("config.yaml decision_engine section must be a mapping")
    unknown = set(configured) - set(_DEFAULT_THRESHOLDS)
    if unknown:
        raise ValueError(f"Unknown decision_engine threshold(s): {sorted(unknown)}")
    thresholds.update(configured)
    for name, value in thresholds.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"decision_engine.{name} must be between 0 and 1") from exc
        if not 0.0 <= numeric <= 1.0:
            raise ValueError(f"decision_engine.{name} must be between 0 and 1")
        thresholds[name] = numeric
    if thresholds["irs_medium"] > thresholds["irs_high"]:
        raise ValueError("decision_engine.irs_medium cannot exceed irs_high")
    if thresholds["trust_critical"] > thresholds["trust_warning"]:
        raise ValueError("decision_engine.trust_critical cannot exceed trust_warning")
    return thresholds


class DecisionType(Enum):
    """Types of security decisions."""
    EXECUTE = "execute"  # Allow normal execution
    CONTAIN = "contain"  # Stop propagation and quarantine
    RECOVER = "recover"  # Trigger recovery procedures
    REFUSE = "refuse"  # Block with refusal message
    QUARANTINE = "quarantine"  # Isolate specific agent


@dataclass
class Decision:
    """A security decision with detailed reasoning."""
    decision_type: DecisionType
    confidence: float  # 0.0 to 1.0
    reason: str
    evidence: List[str] = field(default_factory=list)
    affected_agents: Set[str] = field(default_factory=set)
    triggered_by: str = ""  # Which detection module triggered this
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


class DecisionEngine:
    """
    Centralized decision engine that consumes evidence from detection modules
    and enforces security decisions.
    
    Pipeline:
    Detection Modules -> Evidence -> Decision Engine -> Decision -> Enforcement
    """

    def __init__(self):
        self.decision_history: List[Decision] = []
        self.current_decision: Optional[Decision] = None
        
        # Adaptive thresholds that adjust based on context — loaded from
        # config.yaml's decision_engine: block (previously hardcoded here
        # and never synced with the calibrated config values).
        self.base_thresholds = _load_decision_thresholds()
        
        # Context-aware adjustments
        self.context_adjustments = {
            "first_turn": 0.0,  # No adjustment on first turn
            "repeated_attack": 0.1,  # Lower threshold for repeated attacks
            "high_trust_history": 0.05,  # Slightly higher threshold for trusted users
        }

    def evaluate(
        self,
        irs: float,
        ms: float,
        bd: float,
        trust_scores: Dict[str, float],
        intent_confidence: float = 0.0,
        propagation_risk: float = 0.0,
        attack_type: Optional[str] = None,
        context: Optional[Dict] = None,
    ) -> Decision:
        """
        Evaluate all evidence and make a security decision.
        
        Args:
            irs: Injection Risk Score
            ms: Message Suspicion
            bd: Behavioral Drift
            trust_scores: Dict of agent trust scores
            intent_confidence: Confidence in malicious intent
            propagation_risk: Risk of attack propagation
            attack_type: Detected attack type
            context: Additional context (turn number, history, etc.)
        
        Returns:
            Decision object with type and reasoning
        """
        for name, value in (
            ("irs", irs),
            ("ms", ms),
            ("bd", bd),
            ("intent_confidence", intent_confidence),
            ("propagation_risk", propagation_risk),
        ):
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be between 0 and 1") from exc
            if not 0.0 <= numeric <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if not isinstance(trust_scores, dict):
            raise TypeError("trust_scores must be a dictionary")
        validated_trust = {}
        for agent, value in trust_scores.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"trust score for {agent!r} must be between 0 and 1") from exc
            if not 0.0 <= numeric <= 1.0:
                raise ValueError(f"trust score for {agent!r} must be between 0 and 1")
            validated_trust[str(agent)] = numeric
        irs, ms, bd = float(irs), float(ms), float(bd)
        intent_confidence, propagation_risk = float(intent_confidence), float(propagation_risk)
        trust_scores = validated_trust
        context = context or {}
        if not isinstance(context, dict):
            raise TypeError("context must be a dictionary")
        evidence = []
        affected_agents = set()
        
        # Get adaptive thresholds based on context
        thresholds = self._get_adaptive_thresholds(context)
        
        # Check for critical trust violations
        min_trust = min(trust_scores.values()) if trust_scores else 1.0
        critical_trust_agents = {
            agent for agent, trust in trust_scores.items()
            if trust < thresholds["trust_critical"]
        }
        
        if critical_trust_agents:
            evidence.append(f"Critical trust violation: {critical_trust_agents}")
            affected_agents.update(critical_trust_agents)
        
        # Check for high IRS
        if irs > thresholds["irs_high"]:
            evidence.append(f"High IRS: {irs:.3f} > {thresholds['irs_high']}")
        
        # Check for high MS
        if ms > thresholds["ms_critical"]:
            evidence.append(f"High message suspicion: {ms:.3f}")
        
        # Check for high behavioral drift
        if bd > thresholds["bd_critical"]:
            evidence.append(f"High behavioral drift: {bd:.3f}")
        
        # Check for high intent confidence
        if intent_confidence > 0.7:
            evidence.append(f"High malicious intent confidence: {intent_confidence:.3f}")
        
        # Check for propagation risk
        if propagation_risk > 0.6:
            evidence.append(f"High propagation risk: {propagation_risk:.3f}")
        
        # Make decision based on evidence
        decision_type, confidence, reason = self._make_decision(
            irs, ms, bd, min_trust, intent_confidence, propagation_risk,
            evidence, thresholds, attack_type
        )
        
        # Determine affected agents
        if not affected_agents:
            affected_agents = {agent for agent, trust in trust_scores.items() 
                             if trust < thresholds["trust_warning"]}
        
        decision = Decision(
            decision_type=decision_type,
            confidence=confidence,
            reason=reason,
            evidence=evidence,
            affected_agents=affected_agents,
            triggered_by="DecisionEngine",
            metadata={
                "irs": irs,
                "ms": ms,
                "bd": bd,
                "min_trust": min_trust,
                "intent_confidence": intent_confidence,
                "propagation_risk": propagation_risk,
                "attack_type": attack_type,
                "thresholds_used": thresholds,
            }
        )
        
        self.current_decision = decision
        self.decision_history.append(decision)
        return decision

    def _get_adaptive_thresholds(self, context: Dict) -> Dict[str, float]:
        """Get thresholds adjusted for current context."""
        thresholds = self.base_thresholds.copy()
        
        # Adjust for turn number (multi-turn attacks)
        turn_number = context.get("turn_number", 1)
        if turn_number > 3:
            # Lower threshold after several turns (more suspicious)
            thresholds["irs_high"] -= 0.1
            thresholds["trust_critical"] += 0.1
        
        # Adjust for attack history
        attack_history = context.get("attack_history", [])
        if len(attack_history) > 2:
            # Lower threshold for repeated attacks
            thresholds["irs_high"] -= self.context_adjustments["repeated_attack"]
        
        # Adjust for trust history
        trust_history = context.get("trust_history", 1.0)
        if trust_history > 0.8:
            # Slightly higher threshold for trusted users
            thresholds["irs_high"] += self.context_adjustments["high_trust_history"]
        
        # Ensure thresholds stay in valid range
        for key in thresholds:
            thresholds[key] = max(0.0, min(1.0, thresholds[key]))
        
        return thresholds

    def _make_decision(
        self,
        irs: float,
        ms: float,
        bd: float,
        min_trust: float,
        intent_confidence: float,
        propagation_risk: float,
        evidence: List[str],
        thresholds: Dict[str, float],
        attack_type: Optional[str],
    ) -> tuple:
        """
        Make the final decision based on all factors.
        
        Returns:
            (decision_type, confidence, reason)
        """
        # Critical violations -> REFUSE or CONTAIN
        if min_trust < thresholds["trust_critical"]:
            if irs > thresholds["irs_high"]:
                return (
                    DecisionType.REFUSE,
                    0.9,
                    f"Critical trust violation ({min_trust:.3f}) with high IRS ({irs:.3f})"
                )
            return (
                DecisionType.CONTAIN,
                0.85,
                f"Critical trust violation ({min_trust:.3f})"
            )
        
        # High IRS with intent -> REFUSE
        if irs > thresholds["irs_high"] and intent_confidence > 0.7:
            return (
                DecisionType.REFUSE,
                0.85,
                f"High IRS ({irs:.3f}) with high malicious intent ({intent_confidence:.3f})"
            )
        
        # High IRS -> CONTAIN
        if irs > thresholds["irs_high"]:
            return (
                DecisionType.CONTAIN,
                0.8,
                f"High IRS ({irs:.3f}) exceeds threshold ({thresholds['irs_high']})"
            )
        
        # High propagation risk -> CONTAIN
        if propagation_risk > 0.7:
            return (
                DecisionType.CONTAIN,
                0.75,
                f"High propagation risk ({propagation_risk:.3f})"
            )
        
        # Medium IRS with corroborating factors -> QUARANTINE (BD alone is not enough)
        if irs > thresholds["irs_medium"] and (
            ms > 0.4 or (bd > thresholds["bd_critical"] and intent_confidence > 0.25)
        ):
            return (
                DecisionType.QUARANTINE,
                0.7,
                f"Medium IRS ({irs:.3f}) with elevated MS/BD"
            )
        
        # Warning level trust -> QUARANTINE specific agents
        if min_trust < thresholds["trust_warning"]:
            return (
                DecisionType.QUARANTINE,
                0.6,
                f"Trust warning level ({min_trust:.3f})"
            )
        
        # Safe to execute
        return (
            DecisionType.EXECUTE,
            1.0 - max(irs, ms, bd),
            "All security checks passed"
        )

    def should_execute_agent(self, agent: str, trust_score: float) -> bool:
        """
        Determine if an agent should execute based on trust score.
        
        Trust-based execution control:
        - Trust > 0.8: Execute normally
        - Trust 0.5-0.8: Execute with extra validation
        - Trust 0.3-0.5: Quarantine agent
        - Trust < 0.3: Terminate pipeline
        """
        if trust_score > 0.8:
            return True
        elif trust_score > 0.5:
            # Execute with extra validation (caller must implement)
            return True
        elif trust_score > 0.3:
            # Quarantine agent
            return False
        else:
            # Terminate pipeline
            return False

    def get_decision_summary(self) -> Dict:
        """Get summary of all decisions made."""
        if not self.decision_history:
            return {"total_decisions": 0}
        
        decision_counts = {}
        for decision in self.decision_history:
            dtype = decision.decision_type.value
            decision_counts[dtype] = decision_counts.get(dtype, 0) + 1
        
        return {
            "total_decisions": len(self.decision_history),
            "decision_counts": decision_counts,
            "current_decision": self.current_decision.decision_type.value if self.current_decision else None,
        }
