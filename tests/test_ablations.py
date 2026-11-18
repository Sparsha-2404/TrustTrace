from experiment.ablations import ABLATIONS, get_ablation


def test_ablation_table_matches_manuscript_table_9():
    assert list(ABLATIONS) == ["A0", "A1", "A2", "A3", "A4"]
    assert ABLATIONS["A0"].entry_gate is False
    assert ABLATIONS["A0"].per_hop_checks is False
    assert ABLATIONS["A1"].entry_gate is True
    assert ABLATIONS["A1"].per_hop_checks is False
    assert ABLATIONS["A2"].entry_gate is True
    assert ABLATIONS["A2"].per_hop_checks is True
    assert ABLATIONS["A2"].trust_dynamics is False
    assert ABLATIONS["A3"].trust_dynamics is True
    assert ABLATIONS["A3"].propagation is False
    assert ABLATIONS["A4"].trust_dynamics is True
    assert ABLATIONS["A4"].propagation is True


def test_ablation_ids_are_case_insensitive_and_invalid_values_fail():
    assert get_ablation("a4") == get_ablation("A4")
    try:
        get_ablation("A5")
    except ValueError as exc:
        assert "A0, A1, A2, A3, A4" in str(exc)
    else:
        raise AssertionError("Unknown ablation should raise ValueError")


def test_recovery_only_enabled_for_trust_dynamics_ablations():
    assert not ABLATIONS["A0"].recovery
    assert not ABLATIONS["A1"].recovery
    assert not ABLATIONS["A2"].recovery
    assert ABLATIONS["A3"].recovery
    assert ABLATIONS["A4"].recovery


def test_batch_threshold_does_not_depend_on_ground_truth_attack_type():
    from experiment.batch_processor import BatchTrustTrace

    instance = object.__new__(BatchTrustTrace)
    instance.cfg = {
        "lambda": 0.42,
        "attack_thresholds": {
            "direct": 0.10,
            "indirect": 0.90,
            "memory": 0.20,
        },
    }
    assert instance._lambda_for("direct") == 0.42
    assert instance._lambda_for("indirect") == 0.42
    assert instance._lambda_for("memory") == 0.42
    assert instance._lambda_for("unknown") == 0.42
