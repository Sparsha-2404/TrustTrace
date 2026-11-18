"""
Obfuscation Detection Module.

Detects attempts to hide malicious content using various encoding methods.
"""

import base64
import binascii
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ObfuscationResult:
    """Result of obfuscation detection."""
    is_obfuscated: bool
    confidence: float
    method: str
    reason: str
    decoded_sample: Optional[str] = None


class ObfuscationDetector:
    """
    Detects obfuscated content using various encoding methods.
    """
    
    # Patterns for different obfuscation methods
    PATTERNS = {
        "base64": [
            r'[A-Za-z0-9+/]{20,}={0,2}',  # Base64 pattern
            r'base64\s*(decode|encode)',
        ],
        "hex": [
            r'0x[0-9a-fA-F]+',  # Hex pattern
            r'\\x[0-9a-fA-F]{2}',  # Escaped hex
            r'[0-9a-fA-F]{32,}',  # Long hex strings
        ],
        "rot13": [
            r'rot13',
            r'rot-13',
            r'caesar\s+cipher',
        ],
        "unicode": [
            r'\\u[0-9a-fA-F]{4}',  # Unicode escape
            r'\\U[0-9a-fA-F]{8}',  # Extended Unicode
        ],
        "html": [
            r'&#[0-9]+;',  # HTML entities
            r'&[a-z]+;',  # Named HTML entities
        ],
        "markdown": [
            r'\*\*.*\*\*',  # Bold markdown
            r'`[^`]+`',  # Code blocks
        ],
        "invisible": [
            r'[\u200b-\u200d]',  # Zero-width characters
            r'[\u2060-\u2064]',  # Invisible operators
        ],
    }
    
    def __init__(self):
        self.detection_history = []
    
    def detect(self, text: str) -> ObfuscationResult:
        """
        Detect if the text contains obfuscated content.
        
        Args:
            text: The text to analyze
        
        Returns:
            ObfuscationResult with detection result
        """
        text_lower = text.lower()
        detected_methods = []
        
        for method, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    detected_methods.append(method)
                    break
        
        if not detected_methods:
            return ObfuscationResult(
                is_obfuscated=False,
                confidence=0.0,
                method="none",
                reason="No obfuscation patterns detected"
            )
        
        # Calculate confidence based on number of methods
        confidence = min(1.0, len(detected_methods) * 0.3 + 0.4)
        
        # Try to decode if possible
        decoded_sample = None
        if "base64" in detected_methods:
            decoded_sample = self._try_decode_base64(text)
        
        primary_method = detected_methods[0]
        reason = f"Obfuscation detected using {', '.join(detected_methods)}"
        
        result = ObfuscationResult(
            is_obfuscated=True,
            confidence=confidence,
            method=primary_method,
            reason=reason,
            decoded_sample=decoded_sample
        )
        
        self.detection_history.append(result)
        return result
    
    def _try_decode_base64(self, text: str) -> Optional[str]:
        """Try to decode base64 content from text."""
        # Extract potential base64 strings
        base64_pattern = r'[A-Za-z0-9+/]{20,}={0,2}'
        matches = re.findall(base64_pattern, text)
        
        for match in matches:
            try:
                # Try to decode
                decoded = base64.b64decode(match).decode('utf-8', errors='ignore')
                if len(decoded) > 3 and all(ord(c) < 128 for c in decoded):
                    return decoded[:100]  # Return sample
            except:
                continue
        
        return None
    
    def get_obfuscation_score(self, text: str) -> float:
        """
        Get a single score for obfuscation (0.0 to 1.0).
        """
        result = self.detect(text)
        return result.confidence if result.is_obfuscated else 0.0
