from eval.metrics import MetricsCollector


def test_missing_patient_zero_prediction_counts_as_incorrect():
    collector = MetricsCollector()
    collector.record_attack_with_source(
        succeeded=False, detected=True, source="handcrafted",
        attack_type="indirect", pz_predicted=None,
        pz_ground_truth="Retriever",
        patient_zero_methods={
            "graph_pcs_backward": None,
            "earliest_flagged": "Retriever",
            "stage1_only": "Planner",
        },
    )
    metrics = collector.compute()
    assert metrics["Patient_Zero_Accuracy"] == 0.0
    methods = metrics["patient_zero_method_comparison"]
    assert methods["graph_pcs_backward"] == {"total": 1, "correct": 0, "accuracy": 0.0}
    assert methods["earliest_flagged"] == {"total": 1, "correct": 1, "accuracy": 1.0}
    assert methods["stage1_only"] == {"total": 1, "correct": 0, "accuracy": 0.0}


def test_availability_is_unknown_without_observed_tasks():
    metrics = MetricsCollector().compute()
    assert metrics["System_Availability"] is None
    assert metrics["Recovery_Time_Mean_s"] is None
    assert metrics["Recovery_Time_Median_s"] is None


def test_attack_detection_confusion_counts_are_independent_of_goal_success():
    collector = MetricsCollector()
    collector.record_attack(succeeded=True, detected=True)
    collector.record_attack(succeeded=False, detected=False)
    metrics = collector.compute()
    assert metrics["TP"] == 1
    assert metrics["FN"] == 1
    assert metrics["ASR"] == 0.5


def test_recovery_rolls_back_explicit_poisoned_fixture_before_cleanup(monkeypatch):
    from recovery.recovery_manager import RecoveryManager

    deleted = []

    class FakeMemory:
        def delete_entry(self, entry_id):
            deleted.append(entry_id)
            return True

    class FakeTrust:
        trust_scores = {"Retriever": 1.0}
        delta = 0.4

        def get_all_compromised(self):
            return set()

        def recover_agent(self, agent):
            return 1.0

    class FakeDetector:
        pass

    manager = RecoveryManager.__new__(RecoveryManager)
    manager.memory = FakeMemory()
    manager.trust = FakeTrust()
    manager.detector = FakeDetector()
    manager.quarantined = set()
    manager.held_tasks = []
    manager.rollback_memory = lambda timestamp: 0
    manager.restore = lambda: (0, 0)
    manager._generate_report = lambda **kwargs: None

    manager.execute_recovery(
        patient_zero="Retriever",
        attack_source="indirect",
        propagation_path=["Retriever", "Planner"],
        compromise_timestamp=1.0,
        run_id="test-recovery",
        memory_entry_ids=["poisoned-fixture"],
    )
    assert deleted == ["poisoned-fixture"]


def test_baseline_reuses_exact_supplied_scenario_manifest(monkeypatch):
    from types import SimpleNamespace
    from experiment import full_pipeline
    from attacks.indirect_injection import INDIRECT_PAYLOADS
    from attacks.memory_poisoning import MEMORY_PAYLOADS

    scenarios = [
        {"scenario_id": "direct-0000", "type": "direct", "payload_index": 0, "query": "q-direct"},
        {"scenario_id": "indirect-0000", "type": "indirect", "payload_index": 0, "query": "q-indirect"},
        {"scenario_id": "memory-0000", "type": "memory", "payload_index": 0, "query": "q-memory"},
    ]
    monkeypatch.setattr(full_pipeline, "InteractionLogger", lambda: object())
    monkeypatch.setattr(full_pipeline, "MemoryManager", lambda collection: SimpleNamespace(collection=collection))
    monkeypatch.setattr(full_pipeline, "knowledge_base", object())
    monkeypatch.setattr(
        full_pipeline, "inject_direct",
        lambda run_pipeline, **kwargs: {"Generator": "safe response"},
    )
    monkeypatch.setattr(full_pipeline, "inject_indirect", lambda *args, **kwargs: "indirect-fixture")
    monkeypatch.setattr(full_pipeline, "inject_memory_poison", lambda *args, **kwargs: "memory-fixture")
    monkeypatch.setattr(full_pipeline, "cleanup_indirect", lambda *args, **kwargs: None)
    monkeypatch.setattr(full_pipeline, "cleanup_memory_poison", lambda *args, **kwargs: None)

    def fake_pipeline(query):
        if query == "q-indirect":
            return {"Retriever": INDIRECT_PAYLOADS[0], "Generator": "safe response"}
        if query == "q-memory":
            return {"Retriever": MEMORY_PAYLOADS[0], "Generator": "safe response"}
        raise AssertionError(query)

    monkeypatch.setattr(full_pipeline, "run_pipeline", fake_pipeline)
    result = full_pipeline.run_baseline_experiments(scenarios)

    assert result["scenario_ids"] == [s["scenario_id"] for s in scenarios]
    assert [item["scenario_id"] for item in result["details"]] == result["scenario_ids"]
    assert result["total_attacks"] == len(scenarios)
    assert result["success_measurement"] == "legacy_final_generator_lexical_marker_proxy"
