"""
Final Response Validation Module.

Validates the final output from the Generator to ensure it doesn't
violate security policies, even if earlier stages missed an attack.
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ValidationResult:
    """Result of final response validation."""
    is_safe: bool
    confidence: float
    reason: str
    violations: List[str]
    should_refuse: bool


class FinalResponseValidator:
    """
    Validates final responses for security policy violations.
    """
    
    # Patterns that indicate unsafe content
    UNSAFE_PATTERNS = [
        # Explicit system-prompt / instruction disclosure (not mere mention)
        r"(?:the|my)\s+(?:hidden\s+)?system\s+prompt\s*(?:is|:|=)",
        r"here\s+(?:is|are)\s+(?:the\s+)?(?:hidden\s+)?(?:system\s+prompt|instructions)",
        r"my\s+(?:hidden\s+)?instructions\s+(?:are|say)",
        r"initial\s+instructions\s*(?:are|:|=)",
        
        # Sensitive information disclosure
        r"(api\s+)?key\s*[:=]\s*['\"][\w-]{16,}['\"]",
        r"password\s*[:=]\s*['\"][\w-]{8,}['\"]",
        r"token\s*[:=]\s*['\"][\w-]{20,}['\"]",
        
        # Malicious code generation
        r"here\s+is\s+(malicious|harmful|exploit)\s+code",
        r"code\s+to\s+(hack|attack|exploit)",
        
    ]
    
    # Patterns that indicate safe refusal
    REFUSAL_PATTERNS = [
        r"i\s+cannot",
        r"i\s+am\s+unable",
        r"i\s+refuse",
        r"i\s+won't",
        r"i'm\s+not\s+able",
        r"i\s+am\s+not\s+able",
        r"sorry,\s+i\s+cannot",
    ]
    
    def __init__(self):
        self.validation_history = []
    
    def validate(self, response: str, original_query: str = "") -> ValidationResult:
        """
        Validate the final response for security violations.
        
        Args:
            response: The final response to validate
            original_query: The original user query (for context)
        
        Returns:
            ValidationResult with validation result
        """
        if not isinstance(response, str):
            raise TypeError("response must be a string")
        response_lower = response.lower()
        violations = []
        
        # Check unsafe disclosures first. A response must not pass merely
        # because it contains a refusal phrase alongside leaked content.
        for pattern in self.UNSAFE_PATTERNS:
            if re.search(pattern, response_lower):
                violations.append(pattern)

        if violations:
            return ValidationResult(
                is_safe=False,
                confidence=0.95,
                reason=f"Unsafe content detected: {len(violations)} violation(s)",
                violations=violations,
                should_refuse=True,
            )

        is_refusal = any(re.search(pattern, response_lower) for pattern in self.REFUSAL_PATTERNS)
        if is_refusal:
            return ValidationResult(
                is_safe=True,
                confidence=0.95,
                reason="Response is a safe refusal",
                violations=[],
                should_refuse=False,
            )
        
        # Check if response is suspiciously short (might indicate partial compliance)
        if len(response) < 50 and not is_refusal:
            return ValidationResult(
                is_safe=True,
                confidence=0.6,
                reason="Response is very short (may indicate partial compliance)",
                violations=[],
                should_refuse=False
            )
        
        # Response appears safe
        return ValidationResult(
            is_safe=True,
            confidence=0.9,
            reason="No security violations detected",
            violations=[],
            should_refuse=False
        )
    
    def should_regenerate(self, response: str) -> bool:
        """
        Determine if the response should be regenerated.
        """
        result = self.validate(response)
        return not result.is_safe or result.should_refuse
