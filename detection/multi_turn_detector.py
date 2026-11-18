"""
Multi-turn Detection Module.

Detects attacks that develop over multiple conversation turns.
Maintains conversation trust, attack history, and persistent attacker profiles.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time


@dataclass
class TurnInfo:
    """Information about a single conversation turn."""
    turn_number: int
    user_prompt: str
    irs: float
    ms: float
    trust_before: float
    trust_after: float
    detected_attack: bool
    timestamp: float


@dataclass
class MultiTurnResult:
    """Result of multi-turn analysis."""
    is_multi_turn_attack: bool
    confidence: float
    reason: str
    turn_count: int
    attack_progression: List[float] = field(default_factory=list)
    trust_trend: List[float] = field(default_factory=list)


class MultiTurnDetector:
    """
    Detects multi-turn attacks by analyzing conversation history.
    """
    
    def __init__(self, max_history: int = 10):
        self.conversation_history: List[TurnInfo] = []
        self.max_history = max_history
        self.attacker_profiles: Dict[str, Dict] = {}
    
    def add_turn(
        self,
        user_prompt: str,
        irs: float,
        ms: float,
        trust_before: float,
        trust_after: float,
        detected_attack: bool = False,
    ) -> None:
        """Add information about a conversation turn."""
        turn_number = len(self.conversation_history) + 1
        turn = TurnInfo(
            turn_number=turn_number,
            user_prompt=user_prompt,
            irs=irs,
            ms=ms,
            trust_before=trust_before,
            trust_after=trust_after,
            detected_attack=detected_attack,
            timestamp=time.time()
        )
        
        self.conversation_history.append(turn)
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
    
    def analyze(self) -> MultiTurnResult:
        """
        Analyze the conversation history for multi-turn attack patterns.
        
        Returns:
            MultiTurnResult with analysis result
        """
        if len(self.conversation_history) < 2:
            return MultiTurnResult(
                is_multi_turn_attack=False,
                confidence=0.0,
                reason="Insufficient conversation history",
                turn_count=len(self.conversation_history)
            )
        
        # Check for escalating IRS
        irs_values = [turn.irs for turn in self.conversation_history]
        irs_trend = self._calculate_trend(irs_values)
        
        # Check for declining trust
        trust_values = [turn.trust_after for turn in self.conversation_history]
        trust_trend = self._calculate_trend(trust_values)
        
        # Count detected attacks
        attack_count = sum(1 for turn in self.conversation_history if turn.detected_attack)
        
        # Check for suspicious patterns
        is_attack = False
        confidence = 0.0
        reasons = []
        
        # Escalating IRS is suspicious
        if irs_trend > 0.2:
            is_attack = True
            confidence += 0.3
            reasons.append("Escalating IRS over conversation")
        
        # Declining trust is suspicious
        if trust_trend < -0.2:
            is_attack = True
            confidence += 0.3
            reasons.append("Declining trust over conversation")
        
        # Multiple attacks in history
        if attack_count >= 2:
            is_attack = True
            confidence += 0.4
            reasons.append(f"Multiple attacks detected ({attack_count})")
        
        # High IRS in recent turns
        recent_irs = irs_values[-3:] if len(irs_values) >= 3 else irs_values
        if max(recent_irs) > 0.5:
            is_attack = True
            confidence += 0.2
            reasons.append("High IRS in recent turns")
        
        confidence = min(1.0, confidence)
        reason = "; ".join(reasons) if reasons else "No multi-turn attack patterns detected"
        
        return MultiTurnResult(
            is_multi_turn_attack=is_attack,
            confidence=confidence,
            reason=reason,
            turn_count=len(self.conversation_history),
            attack_progression=irs_values,
            trust_trend=trust_values
        )
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate the trend (slope) of a list of values."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression
        n = len(values)
        x = list(range(n))
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(xi * yi for xi, yi in zip(x, values))
        sum_x2 = sum(xi * xi for xi in x)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def get_conversation_risk(self) -> float:
        """
        Get overall conversation risk score (0.0 to 1.0).
        """
        result = self.analyze()
        if not result.is_multi_turn_attack:
            return 0.0
        
        # Combine confidence with turn count
        turn_factor = min(1.0, result.turn_count / 5.0)
        return result.confidence * (0.5 + 0.5 * turn_factor)
    
    def reset(self) -> None:
        """Reset conversation history."""
        self.conversation_history.clear()
