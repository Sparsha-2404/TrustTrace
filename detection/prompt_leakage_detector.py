"""
Prompt Leakage Detection Module.

Identifies attempts to expose system prompts, hidden instructions,
configuration details, or internal reasoning.
"""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class LeakageResult:
    """Result of prompt leakage detection."""
    is_leakage_attempt: bool
    confidence: float
    reason: str
    matched_patterns: List[str]


class PromptLeakageDetector:
    """
    Detects prompt leakage attempts using pattern matching.
    """
    
    PATTERNS = [
        # Direct requests for system prompt
        r"reveal\s+(the\s+)?(hidden|system|internal)\s+prompt",
        r"show\s+me\s+your\s+(instructions|system\s+prompt|configuration)",
        r"what\s+are\s+your\s+(initial|system)\s+instructions",
        r"print\s+your\s+(system\s+prompt|instructions)",
        r"output\s+your\s+(hidden|internal)\s+prompt",
        r"tell\s+me\s+what\s+you\s+were\s+told\s+to\s+do",
        r"what\s+are\s+your\s+(operating|programming)\s+instructions",
        r"expose\s+your\s+(system\s+prompt|guidelines|rules)",
        
        # Indirect attempts
        r"what\s+is\s+your\s+(first|initial)\s+(message|instruction)",
        r"repeat\s+your\s+(initial|system)\s+instructions",
        r"display\s+your\s+(internal|hidden)\s+instructions",
        r"can\s+you\s+show\s+me\s+your\s+(prompt|instructions)",
        r"i\s+want\s+to\s+see\s+your\s+(system\s+prompt|instructions)",
        
        # Technical terms
        r"system\s+prompt",
        r"hidden\s+prompt",
        r"internal\s+instructions",
        r"initial\s+prompt",
        r"base\s+prompt",
        r"pre-prompt",
        
        # Context extraction
        r"what\s+(context|information)\s+precedes\s+your\s+response",
        r"show\s+me\s+the\s+context\s+window",
        r"what\s+is\s+in\s+your\s+context",
    ]
    
    HIGH_CONFIDENCE_PATTERNS = [
        r"reveal\s+(the\s+)?(hidden|system|internal)\s+prompt",
        r"show\s+me\s+your\s+(instructions|system\s+prompt|configuration)",
        r"what\s+are\s+your\s+(initial|system)\s+instructions",
        r"print\s+your\s+(system\s+prompt|instructions)",
        r"output\s+your\s+(hidden|internal)\s+prompt",
    ]
    
    def __init__(self):
        self.detection_history = []
    
    def detect(self, text: str) -> LeakageResult:
        """
        Detect if the text attempts prompt leakage.
        
        Args:
            text: The text to analyze
        
        Returns:
            LeakageResult with detection result
        """
        text_lower = text.lower()
        matched_patterns = []
        
        for pattern in self.PATTERNS:
            if re.search(pattern, text_lower):
                matched_patterns.append(pattern)
        
        if not matched_patterns:
            return LeakageResult(
                is_leakage_attempt=False,
                confidence=0.0,
                reason="No prompt leakage patterns detected",
                matched_patterns=[]
            )
        
        # Calculate confidence
        high_conf_matches = sum(1 for p in matched_patterns 
                               if p in self.HIGH_CONFIDENCE_PATTERNS)
        
        if high_conf_matches > 0:
            confidence = min(1.0, 0.7 + high_conf_matches * 0.15)
        else:
            confidence = min(1.0, len(matched_patterns) * 0.25)
        
        reason = f"Prompt leakage attempt detected via {len(matched_patterns)} pattern(s)"
        
        result = LeakageResult(
            is_leakage_attempt=True,
            confidence=confidence,
            reason=reason,
            matched_patterns=matched_patterns
        )
        
        self.detection_history.append(result)
        return result
    
    def get_leakage_score(self, text: str) -> float:
        """
        Get a single score for prompt leakage (0.0 to 1.0).
        """
        result = self.detect(text)
        return result.confidence if result.is_leakage_attempt else 0.0
