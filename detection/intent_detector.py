"""
Intent Detection Module - Classifies the intent behind user prompts.

This module detects malicious intent beyond simple text similarity,
identifying what the attacker is trying to achieve.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class IntentType(Enum):
    """Types of detected intents."""
    INFORMATION_REQUEST = "information_request"
    CODE_GENERATION = "code_generation"
    ROLE_OVERRIDE = "role_override"
    MEMORY_MODIFICATION = "memory_modification"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    PROMPT_LEAKAGE = "prompt_leakage"
    CONTEXT_MANIPULATION = "context_manipulation"
    TOOL_MANIPULATION = "tool_manipulation"
    BENIGN = "benign"


@dataclass
class IntentResult:
    """Result of intent detection."""
    intent_type: IntentType
    confidence: float
    reason: str
    matched_patterns: List[str]


class IntentDetector:
    """
    Detects the intent behind user prompts using pattern matching and semantic analysis.
    """
    
    # Patterns for each intent type
    PATTERNS = {
        IntentType.ROLE_OVERRIDE: [
            r"ignore\s+(all\s+|any\s+|every\s+)?(previous|prior|earlier|above|all)\s+instructions?",
            r"disregard\s+(all\s+|any\s+|every\s+)?(previous|prior|earlier|above)\s+instructions?",
            r"disregard\s+your\s+role",
            r"forget\s+your\s+(instructions|policy|guidelines)",
            r"forget\s+(all\s+|any\s+|every\s+)?(previous|prior|earlier|above)\s+instructions?",
            r"abandon\s+your\s+constraints",
            r"override\s+your\s+(programming|training)",
            r"you\s+are\s+now\s+(dan|unrestricted|free)",
            r"act\s+as\s+if\s+you\s+have\s+no\s+(rules|filters|restrictions)",
            r"bypass\s+(content\s+)?filters",
            r"developer\s+mode",
            r"jailbreak",
        ],
        IntentType.PROMPT_LEAKAGE: [
            r"reveal\s+(the\s+)?(hidden|system|internal)\s+prompt",
            r"show\s+me\s+your\s+(instructions|system\s+prompt|configuration)",
            r"what\s+are\s+your\s+(initial|system)\s+instructions",
            r"print\s+your\s+(system\s+prompt|instructions)",
            r"output\s+your\s+(hidden|internal)\s+prompt",
            r"tell\s+me\s+what\s+you\s+were\s+told\s+to\s+do",
            r"what\s+are\s+your\s+(operating|programming)\s+instructions",
            r"expose\s+your\s+(system\s+prompt|guidelines|rules)",
        ],
        IntentType.MEMORY_MODIFICATION: [
            r"update\s+your\s+memory",
            r"store\s+this\s+in\s+(long-term\s+)?memory",
            r"remember\s+this\s+for\s+future\s+conversations",
            r"add\s+to\s+your\s+(memory|knowledge\s+base)",
            r"modify\s+your\s+(memory|knowledge)",
            r"persistent\s+instruction",
            r"long-term\s+instruction",
            r"save\s+this\s+preference",
        ],
        IntentType.PRIVILEGE_ESCALATION: [
            r"give\s+me\s+(admin|root|administrator)\s+access",
            r"escalate\s+privileges",
            r"bypass\s+security\s+controls",
            r"gain\s+unauthorized\s+access",
            r"override\s+security\s+(measures|protocols)",
            r"disable\s+(security|protection)",
            r"grant\s+me\s+(extra|additional)\s+permissions",
        ],
        IntentType.CONTEXT_MANIPULATION: [
            r"what\s+(context|information)\s+is\s+stored",
            r"show\s+me\s+the\s+(retrieved|loaded)\s+context",
            r"what\s+documents\s+did\s+you\s+retrieve",
            r"summarize\s+the\s+(retrieved|knowledge\s+base)\s+documents",
            r"what\s+is\s+in\s+the\s+knowledge\s+base",
            r"list\s+all\s+(stored|saved)\s+information",
        ],
        IntentType.TOOL_MANIPULATION: [
            r"execute\s+the\s+following\s+command",
            r"run\s+this\s+(tool|command|script)",
            r"call\s+the\s+tool",
            r"use\s+the\s+tool\s+to",
            r"invoke\s+the\s+(function|tool)",
            r"shell\s+command",
            r"system\s+command",
        ],
        IntentType.CODE_GENERATION: [
            r"write\s+(code|a\s+script|a\s+program)",
            r"generate\s+(code|a\s+script)",
            r"create\s+(malicious|harmful)\s+code",
            r"help\s+me\s+write\s+(exploit|malware|virus)",
            r"code\s+for\s+(hacking|attacking|exploiting)",
        ],
    }
    
    def __init__(self):
        self.detection_history = []
    
    def detect(self, text: str) -> IntentResult:
        """
        Detect the intent of the given text.
        
        Args:
            text: The text to analyze
        
        Returns:
            IntentResult with detected intent type and confidence
        """
        text_lower = text.lower()
        
        # Check each intent type
        best_intent = IntentType.BENIGN
        best_confidence = 0.0
        best_matches = []
        best_reason = "No malicious intent detected"
        
        for intent_type, patterns in self.PATTERNS.items():
            matches = []
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    matches.append(pattern)
            
            if matches:
                # Calculate confidence based on number of matches
                confidence = min(1.0, len(matches) * 0.3 + 0.4)
                
                if confidence > best_confidence:
                    best_intent = intent_type
                    best_confidence = confidence
                    best_matches = matches
                    best_reason = f"Detected {intent_type.value} intent via {len(matches)} pattern(s)"
        
        result = IntentResult(
            intent_type=best_intent,
            confidence=best_confidence,
            reason=best_reason,
            matched_patterns=best_matches
        )
        
        self.detection_history.append(result)
        return result
    
    def get_malicious_intent_score(self, text: str) -> float:
        """
        Get a single score for malicious intent (0.0 to 1.0).
        
        Returns 0.0 for benign intent, higher values for more malicious intents.
        """
        result = self.detect(text)
        
        if result.intent_type == IntentType.BENIGN:
            return 0.0
        
        # Weight different intents by severity
        severity_weights = {
            IntentType.ROLE_OVERRIDE: 0.9,
            IntentType.PROMPT_LEAKAGE: 0.85,
            IntentType.PRIVILEGE_ESCALATION: 0.95,
            IntentType.MEMORY_MODIFICATION: 0.7,
            IntentType.CONTEXT_MANIPULATION: 0.6,
            IntentType.TOOL_MANIPULATION: 0.75,
            IntentType.CODE_GENERATION: 0.5,
        }
        
        severity = severity_weights.get(result.intent_type, 0.5)
        return result.confidence * severity
