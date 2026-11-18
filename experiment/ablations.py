"""Explicit Experiment 3 ablation definitions for TrustTrace Version 3.

The names and included mechanisms follow Table 9 of the Version 3 manuscript.
A2 content checks and A3/A4 trust updates are enforced online between pipeline
agents by BatchTrustTrace._run_live_pipeline. The legacy event replay path is
retained only for recorded episodes and compatibility tests.
"""
from dataclasses import dataclass
from typing import Literal

AblationId = Literal["A0", "A1", "A2", "A3", "A4"]

@dataclass(frozen=True)
class AblationConfig:
    ablation_id: AblationId
    name: str
    entry_gate: bool
    per_hop_checks: bool
    trust_dynamics: bool
    propagation: bool
    recovery: bool

ABLATIONS = {
    "A0": AblationConfig("A0", "No defence", False, False, False, False, False),
    "A1": AblationConfig("A1", "Entry gate only (L0)", True, False, False, False, False),
    "A2": AblationConfig("A2", "Entry gate + per-hop content checks", True, True, False, False, False),
    "A3": AblationConfig("A3", "A2 + trust dynamics (mu=0)", True, True, True, False, True),
    "A4": AblationConfig("A4", "Full TrustTrace (A3 + propagation)", True, True, True, True, True),
}

def get_ablation(ablation_id: str) -> AblationConfig:
    try:
        return ABLATIONS[ablation_id.upper()]
    except (KeyError, AttributeError):
        raise ValueError(f"Unknown ablation {ablation_id!r}; choose one of {', '.join(ABLATIONS)}") from None
