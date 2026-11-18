"""
Decision Engine - Centralized enforcement logic for TrustTrace.

This module separates monitoring from enforcement. Detection modules produce evidence,
and the Decision Engine consumes that evidence to determine execution flow.
"""

from .decision_engine import DecisionEngine, Decision, DecisionType

__all__ = ['DecisionEngine', 'Decision', 'DecisionType']
