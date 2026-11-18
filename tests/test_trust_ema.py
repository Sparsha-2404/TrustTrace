"""Focused regression tests for the Version 3 EMA trust equation."""

import pytest

from drift.behavioral_drift import BehavioralDriftModule
from graph.propagation_graph import PropagationGraph
from trust.trust_engine import TrustEngine


def make_engine():
    graph = PropagationGraph()
    drift = BehavioralDriftModule()
    return TrustEngine(graph, drift), graph


def test_benign_health_keeps_trust_at_one_when_sender_is_trusted():
    engine, graph = make_engine()
    result = engine.update(
        receiver="Planner", sender="Retriever", ms=0.0, bd=0.0,
        current_output="ordinary benign output", timestamp=1.0,
    )
    assert result.new_trust == pytest.approx(1.0)
    assert graph.get_trust("Planner") == pytest.approx(1.0)


def test_trust_update_matches_ema_equation_before_propagation():
    engine, graph = make_engine()
    engine.trust_scores["Retriever"] = 0.8
    graph.set_trust("Retriever", 0.8)
    engine.trust_scores["Planner"] = 0.9
    graph.set_trust("Planner", 0.9)
    ms, bd = 0.2, 0.4
    rho = engine.rho
    health = 1.0 - (ms + bd) / 2.0
    ema_value = rho * 0.9 + (1.0 - rho) * health
    weight = graph.get_edge_weight("Retriever", "Planner")
    propagation_drop = engine.mu * weight * (1.0 - 0.8)

    result = engine.update(
        receiver="Planner", sender="Retriever", ms=ms, bd=bd,
        current_output="test output", timestamp=2.0,
    )

    assert result.new_trust == pytest.approx(ema_value * (1.0 - propagation_drop))
    assert result.propagation_drop == pytest.approx(propagation_drop)


def test_ms_and_bd_outside_unit_interval_are_rejected():
    engine, _ = make_engine()
    with pytest.raises(ValueError, match=r"normalised to \[0, 1\]"):
        engine.update(
            receiver="Planner", sender="Retriever", ms=1.2, bd=0.0,
            current_output="invalid score", timestamp=3.0,
        )


def test_scenario_origin_labels_follow_threat_model():
    from experiment.scenarios import ATTACK_SCENARIOS_HANDCRAFTED

    origins = {row["type"]: row["origin"] for row in ATTACK_SCENARIOS_HANDCRAFTED}
    assert origins["direct"] == "UserInput"
    assert origins["indirect"] == "Document"
    assert origins["memory"] == "MemoryEntry"
