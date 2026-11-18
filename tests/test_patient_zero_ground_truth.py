from experiment.scenarios import ATTACK_SCENARIOS_HANDCRAFTED, GROUND_TRUTH_MAP


def test_memory_poisoning_origin_is_not_mislabelled_as_retriever():
    memory_scenarios = [s for s in ATTACK_SCENARIOS_HANDCRAFTED if s["type"] == "memory"]
    assert memory_scenarios
    assert all(s["origin"] == "MemoryEntry" for s in memory_scenarios)
    assert all(s["ground_truth"] == "MemoryStore" for s in memory_scenarios)
    assert GROUND_TRUTH_MAP["memory"] == "MemoryStore"


def test_direct_and_indirect_ground_truth_is_first_pipeline_agent():
    assert GROUND_TRUTH_MAP["direct"] == "Retriever"
    assert GROUND_TRUTH_MAP["indirect"] == "Retriever"
    assert GROUND_TRUTH_MAP["jailbreak"] == "Retriever"


def test_scenario_origin_and_component_ground_truth_are_separate_fields():
    for scenario in ATTACK_SCENARIOS_HANDCRAFTED:
        assert "origin" in scenario
        assert "ground_truth" in scenario
        assert scenario["origin"] != scenario["ground_truth"] or scenario["type"] != "memory"


def test_patient_zero_comparison_baselines_are_deterministic():
    from detector.patient_zero import PatientZeroDetector

    class FakeTrust:
        EXCLUDED_AGENTS = set()
        delta = 0.4
        def get_all_compromised(self):
            return {"Planner", "Retriever", "Executor"}
        def get_compromise_timestamp(self, agent):
            return {"Retriever": 10.0, "Planner": 12.0, "Executor": 11.0}.get(agent)

    class FakeGraph:
        G = None

    detector = PatientZeroDetector(FakeGraph(), FakeTrust())
    assert detector.detect_earliest_flagged() == "Retriever"
    assert detector.detect_stage1_only() == "Retriever"


def test_pcs_handles_missing_compromise_timestamps_without_nan():
    from detector.patient_zero import PatientZeroDetector

    class FakeTrust:
        EXCLUDED_AGENTS = set()
        delta = 0.4
        def get_all_compromised(self):
            return {"Retriever", "Planner"}
        def get_compromise_timestamp(self, agent):
            return None

    class FakeGraph:
        import networkx as nx
        G = nx.DiGraph()
        G.add_edge("Retriever", "Planner")
        first_below_delta = {}
        def backward_traversal(self, node, delta):
            return node

    detector = PatientZeroDetector(FakeGraph(), FakeTrust())
    assert detector._compute_pcs("Retriever", {"Retriever", "Planner"}) == 0.0
    assert detector._compute_pcs("Planner", {"Retriever", "Planner"}) == 0.0
