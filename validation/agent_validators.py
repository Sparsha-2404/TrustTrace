"""
Agent Validators - Security validation for agent inputs and outputs.

This module provides validation for:
- Planner goal validation (Issue 16)
- Retriever context validation (Issue 17)
- Tool Security validation (Issue 18)
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    confidence: float
    reason: str
    violations: List[str]


class PlannerValidator:
    """
    Validates Planner goals for safety and legality.
    
    ISSUE 16 FIX: Planner should validate goals before planning.
    """
    
    # Malicious goal patterns
    MALICIOUS_GOAL_PATTERNS = [
        r"reveal\s+(the\s+)?(hidden|system|internal)\s+prompt",
        r"expose\s+(your\s+)?(instructions|system\s+prompt)",
        r"bypass\s+(security|content\s+filters)",
        r"override\s+your\s+(role|instructions|constraints)",
        r"escalate\s+privileges",
        r"grant\s+(admin|root|unauthorized)\s+access",
        r"generate\s+(malicious|harmful)\s+code",
        r"help\s+(hack|attack|exploit)",
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"forget\s+your\s+(instructions|policy|guidelines)",
    ]
    
    # Safe goal patterns
    SAFE_GOAL_PATTERNS = [
        r"create\s+a\s+plan",
        r"help\s+me\s+(understand|learn)",
        r"explain\s+(how|why|what)",
        r"summarize",
        r"analyze",
        r"compare",
        r"provide\s+information",
    ]
    
    def __init__(self):
        self.validation_history = []
    
    def validate_goal(self, goal: str, trust_score: float = 1.0) -> ValidationResult:
        """
        Validate a Planner goal for safety.
        
        Args:
            goal: The goal to validate
            trust_score: Current trust score of the Planner
        
        Returns:
            ValidationResult with validation result
        """
        if not isinstance(goal, str) or not goal.strip():
            return ValidationResult(
                is_valid=False,
                confidence=1.0,
                reason="Planner goal must be a non-empty string",
                violations=["Invalid planner goal"],
            )
        try:
            trust_score = float(trust_score)
        except (TypeError, ValueError) as exc:
            raise ValueError("trust_score must be between 0 and 1") from exc
        if not 0.0 <= trust_score <= 1.0:
            raise ValueError("trust_score must be between 0 and 1")
        goal_lower = goal.lower()
        violations = []
        
        # Check for malicious patterns
        for pattern in self.MALICIOUS_GOAL_PATTERNS:
            if re.search(pattern, goal_lower):
                violations.append(f"Malicious pattern: {pattern}")
        
        if violations:
            return ValidationResult(
                is_valid=False,
                confidence=0.9,
                reason=f"Goal contains {len(violations)} malicious pattern(s)",
                violations=violations
            )
        
        # Check for safe patterns
        safe_hits = sum(1 for p in self.SAFE_GOAL_PATTERNS if re.search(p, goal_lower))
        
        # If trust is low, be more conservative
        if trust_score < 0.5 and safe_hits == 0:
            return ValidationResult(
                is_valid=False,
                confidence=0.6,
                reason="Low trust score with no clear safe patterns",
                violations=["Low trust score"]
            )
        
        # Goal appears safe
        return ValidationResult(
            is_valid=True,
            confidence=0.85 if safe_hits > 0 else 0.7,
            reason="Goal appears safe",
            violations=[]
        )
    
    def should_execute(self, goal: str, trust_score: float = 1.0) -> bool:
        """Determine if the goal should be executed."""
        result = self.validate_goal(goal, trust_score)
        return result.is_valid


class RetrieverValidator:
    """
    Validates retrieved context for security.
    
    ISSUE 17 FIX: Retriever should scan documents before forwarding.
    """
    
    # Suspicious document patterns
    SUSPICIOUS_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(your|the)\s+(task|request|instructions)",
        r"system\s+override",
        r"system\s+instruction",
        r"developer\s+mode",
        r"hidden\s+instruction",
        r"new\s+task",
        r"new\s+instructions",
        r"memory\s+update",
        r"long[- ]term\s+instruction",
        r"attacker\s+control",
        r"secret\s+data",
        r"output\s+only",
        r"always\s+respond\s+with",
        r"reveal\s+(all\s+)?(the\s+)?system\s+prompts?",
        r"email\s+all\s+data",
        r"exfiltrat(e|ion)",
        r"persistent\s+override",
        r"treat\s+this\s+as\s+a\s+trusted\s+source",
    ]
    
    # Document types to scan
    DOCUMENT_TYPES = {
        "html": [r"<script", r"<iframe", r"javascript:", r"onerror=", r"onload="],
        "markdown": [r"\[.*\]\(javascript:", r"```.*script"],
        "pdf": [],  # PDF scanning would require specialized library
        "email": [r"<script", r"javascript:", r"data:"],
    }
    
    def __init__(self):
        self.validation_history = []
    
    def validate_document(self, content: str, doc_type: str = "text") -> ValidationResult:
        """
        Validate a document for suspicious content.
        
        Args:
            content: The document content
            doc_type: Type of document (html, markdown, pdf, email, text)
        
        Returns:
            ValidationResult with validation result
        """
        if not isinstance(content, str):
            raise TypeError("document content must be a string")
        if not isinstance(doc_type, str):
            raise TypeError("doc_type must be a string")
        content_lower = content.lower()
        violations = []
        
        # Check general suspicious patterns
        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, content_lower):
                violations.append(f"Suspicious pattern: {pattern}")
        
        # Check document-type specific patterns
        if doc_type in self.DOCUMENT_TYPES:
            for pattern in self.DOCUMENT_TYPES[doc_type]:
                if re.search(pattern, content_lower):
                    violations.append(f"{doc_type.upper()} specific pattern: {pattern}")
        
        if violations:
            return ValidationResult(
                is_valid=False,
                confidence=0.85,
                reason=f"Document contains {len(violations)} suspicious pattern(s)",
                violations=violations
            )
        
        return ValidationResult(
            is_valid=True,
            confidence=0.9,
            reason="Document appears safe",
            violations=[]
        )
    
    def validate_context(self, documents: List[str], doc_types: Optional[List[str]] = None) -> ValidationResult:
        """
        Validate multiple documents in context.
        
        Args:
            documents: List of document contents
            doc_types: List of document types (same length as documents)
        
        Returns:
            ValidationResult with overall validation result
        """
        if not isinstance(documents, (list, tuple)):
            raise TypeError("documents must be a list or tuple")
        if not documents:
            return ValidationResult(
                is_valid=True,
                confidence=1.0,
                reason="No documents to validate",
                violations=[]
            )

        if doc_types is None:
            doc_types = ["text"] * len(documents)
        elif not isinstance(doc_types, (list, tuple)) or len(doc_types) != len(documents):
            raise ValueError("doc_types must contain exactly one type per document")
        
        all_violations = []
        invalid_count = 0
        
        for doc, doc_type in zip(documents, doc_types):
            result = self.validate_document(doc, doc_type)
            if not result.is_valid:
                invalid_count += 1
                all_violations.extend(result.violations)
        
        if invalid_count > 0:
            return ValidationResult(
                is_valid=False,
                confidence=0.8,
                reason=f"{invalid_count} of {len(documents)} documents are suspicious",
                violations=all_violations
            )
        
        return ValidationResult(
            is_valid=True,
            confidence=0.9,
            reason="All documents appear safe",
            violations=[]
        )


class ToolValidator:
    """
    Validates tool requests for security.
    
    ISSUE 18 FIX: Tool requests should be validated before execution.
    """
    
    # Dangerous tool patterns
    DANGEROUS_TOOLS = [
        r"shell\s+command",
        r"system\s+command",
        r"exec\s*\(",
        r"eval\s*\(",
        r"os\.system",
        r"subprocess\.call",
        r"subprocess\.run",
        r"__import__",
        r"open\s*\(",
        r"file\s*\(",
    ]
    
    # Dangerous arguments
    DANGEROUS_ARGUMENTS = [
        r"--force",
        r"--delete",
        r"--remove",
        r"--exec",
        r";\s*rm",
        r";\s*del",
        r">\s*/dev/",
        r"&&\s*rm",
        r"&&\s*del",
    ]
    
    def __init__(self):
        self.validation_history = []
    
    def validate_tool_request(
        self,
        tool_name: str,
        arguments: str,
        trust_score: float = 1.0,
        permissions: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate a tool request for security.
        
        Args:
            tool_name: Name of the tool being requested
            arguments: Arguments to the tool
            trust_score: Current trust score
            permissions: List of available permissions
        
        Returns:
            ValidationResult with validation result
        """
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("tool_name must be a non-empty string")
        if not isinstance(arguments, str):
            raise TypeError("arguments must be a string")
        try:
            trust_score = float(trust_score)
        except (TypeError, ValueError) as exc:
            raise ValueError("trust_score must be between 0 and 1") from exc
        if not 0.0 <= trust_score <= 1.0:
            raise ValueError("trust_score must be between 0 and 1")
        if permissions is not None and not isinstance(permissions, (list, tuple, set)):
            raise TypeError("permissions must be a list, tuple or set")

        violations = []
        tool_lower = tool_name.lower()
        args_lower = arguments.lower()
        
        # Check for dangerous tools
        for pattern in self.DANGEROUS_TOOLS:
            if re.search(pattern, tool_lower) or re.search(pattern, args_lower):
                violations.append(f"Dangerous tool pattern: {pattern}")
        
        # Check for dangerous arguments
        for pattern in self.DANGEROUS_ARGUMENTS:
            if re.search(pattern, args_lower):
                violations.append(f"Dangerous argument pattern: {pattern}")
        
        # Check permissions if provided
        if permissions is not None:
            required_permission = self._get_required_permission(tool_name)
            if required_permission and required_permission not in permissions:
                violations.append(f"Missing required permission: {required_permission}")
        
        # Low trust requires stricter validation
        if trust_score < 0.5:
            if violations:
                return ValidationResult(
                    is_valid=False,
                    confidence=0.95,
                    reason=f"Low trust score with {len(violations)} violation(s)",
                    violations=violations
                )
            else:
                # Even without violations, low trust means extra caution
                return ValidationResult(
                    is_valid=False,
                    confidence=0.6,
                    reason="Low trust score requires manual approval",
                    violations=["Low trust score"]
                )
        
        if violations:
            return ValidationResult(
                is_valid=False,
                confidence=0.85,
                reason=f"Tool request contains {len(violations)} violation(s)",
                violations=violations
            )
        
        return ValidationResult(
            is_valid=True,
            confidence=0.9,
            reason="Tool request appears safe",
            violations=[]
        )
    
    def _get_required_permission(self, tool_name: str) -> Optional[str]:
        """Get the required permission for a tool."""
        tool_lower = tool_name.lower()
        
        if "shell" in tool_lower or "command" in tool_lower:
            return "execute_command"
        elif "file" in tool_lower or "write" in tool_lower:
            return "write_file"
        elif "read" in tool_lower:
            return "read_file"
        elif "network" in tool_lower or "http" in tool_lower:
            return "network_access"
        
        return None
    
    def should_execute(
        self,
        tool_name: str,
        arguments: str,
        trust_score: float = 1.0,
        permissions: Optional[List[str]] = None
    ) -> bool:
        """Determine if the tool should be executed."""
        result = self.validate_tool_request(tool_name, arguments, trust_score, permissions)
        return result.is_valid
