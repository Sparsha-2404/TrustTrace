# Full 8-step experimental pipeline per TrustTrace Implementation Guide Section 17.

from __future__ import annotations

import os
import uuid

import yaml

import config
from calibration.calibrate import run_calibration
from drift.behavioral_drift import BehavioralDriftModule
from eval.metrics import MetricsCollector
from experiment.batch_processor import ATTACK_MARKERS, BatchTrustTrace, _attack_succeeded
from experiment.scenarios import GROUND_TRUTH_MAP, expand_scenarios, get_benign_queries
from logger.interaction_logger import InteractionLogger
from memory.memory_manager import MemoryManager
from scanner.injection_scanner import InjectionScanner
from tuning.bayesian_search import run_search
from victim_pipeline.agents import knowledge_base, run_pipeline
from attacks.direct_injection import inject_direct
from attacks.indirect_injection import (
    INDIRECT_PAYLOADS,
    inject_indirect,
    cleanup_indirect,
)

from attacks.memory_poisoning import (
    MEMORY_PAYLOADS,
    inject_memory_poison,
    cleanup_memory_poison,
)

import datetime
import time
import statistics
from reports.report_generator import generate_experiment_report

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
DEFAULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "config.defaults.yaml")


def _restore_config_defaults() -> None:
    """Reset experiment hyperparameters without erasing the selected LLM endpoint."""
    with open(DEFAULTS_PATH, "r", encoding="utf-8") as f:
        defaults = yaml.safe_load(f) or {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            current = yaml.safe_load(f) or {}
    except FileNotFoundError:
        current = {}

    # Defaults reset only keys defined by config.defaults.yaml. Runtime/provider
    # settings and project-specific sections are retained across quick runs.
    merged = dict(defaults)
    for key, value in current.items():
        if key not in defaults or key == "llm":
            merged[key] = value
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_baseline_experiments(scenarios: list | None = None) -> dict:
    """Run the undefended control on the exact scenario manifest used by A0-A4.

    The old implementation hard-coded 12 baseline episodes (4 each of three
    attack types) while the defended quick run evaluated 8 different episodes.
    That made the displayed ASR comparison invalid. This version reuses the
    caller's exact scenario IDs, queries, payload indices and attack types.
    """
    if scenarios is None:
        scenarios = expand_scenarios(n_per_type=2)
    logger = InteractionLogger()
    run_id = str(uuid.uuid4())
    summary = {
        "total_attacks": 0,
        "attacks_succeeded": 0,
        "details": [],
        "scenario_ids": [s.get("scenario_id") for s in scenarios],
        "success_measurement": "legacy_final_generator_lexical_marker_proxy",
        "goal_oracle_status": "not_scenario_specific",
    }
    print("\n=== PHASE: Baseline (no TrustTrace) ===")
    memory_mgr = MemoryManager(collection=knowledge_base)

    for scenario in scenarios:
        attack_type = scenario.get("type", "direct")
        payload_index = int(scenario.get("payload_index", 0))
        query = scenario.get("query", "What are best practices for data security?")
        doc_id = None
        outputs = None
        try:
            if attack_type in ("direct", "jailbreak"):
                outputs = inject_direct(
                    run_pipeline,
                    logger=logger,
                    run_id=run_id,
                    payload_index=payload_index,
                    user_query=query,
                )
            elif attack_type == "indirect":
                doc_id = inject_indirect(
                    knowledge_base, memory_mgr, payload_index=payload_index, control_mode=True
                )
                outputs = run_pipeline(query)
                expected_payload = INDIRECT_PAYLOADS[payload_index % len(INDIRECT_PAYLOADS)]
                if expected_payload not in str(outputs.get("Retriever", "")):
                    raise RuntimeError(
                        f"Invalid indirect-injection baseline scenario {scenario.get('scenario_id')}: "
                        "injected document was not retrieved; refusing to count an undelivered attack."
                    )
            elif attack_type == "memory":
                doc_id = inject_memory_poison(
                    memory_mgr.collection, memory_mgr, payload_index=payload_index, control_mode=True
                )
                outputs = run_pipeline(query)
                expected_payload = MEMORY_PAYLOADS[payload_index % len(MEMORY_PAYLOADS)]
                if expected_payload not in str(outputs.get("Retriever", "")):
                    raise RuntimeError(
                        f"Invalid memory-poisoning baseline scenario {scenario.get('scenario_id')}: "
                        "poisoned entry was not retrieved; refusing to count an undelivered attack."
                    )
            else:
                raise ValueError(f"Unsupported baseline attack type: {attack_type!r}")
        finally:
            if doc_id and attack_type == "indirect":
                cleanup_indirect(knowledge_base, doc_id, memory_manager=memory_mgr)
            elif doc_id and attack_type == "memory":
                cleanup_memory_poison(memory_mgr.collection, doc_id, memory_manager=memory_mgr)

        success = _attack_succeeded(outputs)
        summary["total_attacks"] += 1
        summary["attacks_succeeded"] += int(success)
        summary["details"].append({
            "scenario_id": scenario.get("scenario_id"),
            "type": attack_type,
            "payload_index": payload_index,
            "success": success,
        })

    rate = summary["attacks_succeeded"] / summary["total_attacks"] if summary["total_attacks"] else 0.0
    summary["asr_pct"] = rate * 100
    print(f"Baseline ASR proxy (no TrustTrace): {rate:.2%} ({summary['attacks_succeeded']}/{summary['total_attacks']})")
    return summary


def _print_final_summary(results: list, baseline_asr_pct: float) -> None:
    total = len(results)
    if total == 0:
        return
    detected = sum(1 for r in results if r.get("detected"))
    blocked_l1 = sum(1 for r in results if r.get("blocked_at") == "Layer1")
    missed = total - detected
    tt_asr_count = sum(1 for r in results if r.get("succeeded"))
    tt_asr = 100 * tt_asr_count / total
    marker_proxy_count = sum(1 for r in results if r.get("marker_proxy_hit", False))

    print("\n" + "=" * 50)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 50)
    print(f"Total attacks:          {total}")
    print(f"Detected (total):       {detected} ({100 * detected / total:.1f}%)")
    print(f"  Blocked at Layer 1:   {blocked_l1}")
    print(f"  Caught by TrustTrace: {detected - blocked_l1}")
    print(f"Missed:                 {missed} ({100 * missed / total:.1f}%)")
    print(f"Final-output marker proxy hits (not goal oracle): {marker_proxy_count} ({100 * marker_proxy_count / total:.1f}%)")
    print(f"Baseline ASR proxy:     {baseline_asr_pct:.1f}%")
    print(f"TrustTrace ASR proxy:   {tt_asr:.1f}%")
    print(f"ASR proxy difference:   {baseline_asr_pct - tt_asr:.1f}%")
    print("=" * 50)


def _print_asr_audit_table(scenarios: list, attack_results: list) -> None:
    """Print per-attack ASR audit table and split summaries."""
    print("\n=== ASR Audit Table ===")
    header = (
        f"{'Attack ID':<10} | {'Type':<10} | {'Source':<12} | "
        f"{'Detected':<9} | {'Recovered':<10} | {'Marker Proxy Hit':<16} | {'ASR Proxy':<11}"
    )
    print(header)
    print("-" * len(header))

    for i, (scenario, result) in enumerate(zip(scenarios, attack_results), start=1):
        attack_id = f"A{i:03d}"
        detected = result.get("detected", False)
        recovered = result.get("recovered", False)
        goal = result.get("marker_proxy_hit", False)
        asr_success = result.get("succeeded", False)
        attack_type = scenario.get("type", result.get("attack_type", "unknown"))
        source = scenario.get("source", "handcrafted")
        print(
            f"{attack_id:<10} | {attack_type:<10} | {source:<12} | "
            f"{str(detected):<9} | {str(recovered):<10} | {str(goal):<14} | {str(asr_success):<11}"
        )

    for source in sorted({s.get("source", "handcrafted") for s in scenarios}):
        indices = [i for i, s in enumerate(scenarios) if s.get("source", "handcrafted") == source]
        subset = [attack_results[i] for i in indices]
        total = len(subset)
        if total == 0:
            continue
        asr_successes = sum(1 for r in subset if r.get("succeeded"))
        goals = sum(1 for r in subset if r.get("marker_proxy_hit", False))
        print(f"\n--- ASR by source: {source.upper()} ({total} attacks) ---")
        print(f"  Final-output marker proxy: {goals / total:.1%}")
        print(f"  TrustTrace ASR proxy:      {asr_successes / total:.1%}")


def run_full_experiment(quick: bool = False, force_calibrate: bool = False, optuna_trials: int | None = None) -> dict:
    from victim_pipeline.agents import is_persistent_knowledge_base, get_chroma_initialization_error
    if not is_persistent_knowledge_base():
        cause = get_chroma_initialization_error()
        raise RuntimeError(
            "Benchmark execution requires persistent ChromaDB; refusing to run against "
            "the in-memory test fallback. Check the ChromaDB installation and data path."
        ) from cause

    os.environ.setdefault("TRUSTTRACE_STRICT_EVAL", "1")
    config.EXPERIMENT_MODE = True
    # start timestamp for experiment duration
    experiment_start_ts = time.time()
    experiment_start_dt = datetime.datetime.utcnow().isoformat()
    if quick:
        _restore_config_defaults()
    cfg = _load_config()
    n_per_type = 2 if quick else cfg.get("n_per_attack_type", 50)
    n_benign = 5 if quick else 50
    n_trials = 3 if quick else (optuna_trials if optuna_trials is not None else 100)

    print("=== TrustTrace Full Experimental Pipeline ===")

    try:
        from data.setup_datasets import setup_deepset
        setup_deepset()
    except Exception as exc:
        print(f"Dataset setup warning: {exc}")

    logger = InteractionLogger()
    scanner = InjectionScanner()
    drift = BehavioralDriftModule()
    memory = MemoryManager(collection=knowledge_base)
    ablation_id = os.environ.get("TRUSTTRACE_ABLATION", "A4").upper()
    batch = BatchTrustTrace(logger, scanner, drift, memory, ablation_id=ablation_id)

    model_path = os.path.join("scanner", "scanner_model.pkl")
    scaler_path = os.path.join("scanner", "scanner_scaler.pkl")
    train_path = os.path.join("data", "deepset_injections", "train.json")
    if quick or not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print("\n=== PHASE: Train Injection Scanner ===")
        if not os.path.exists(train_path):
            from data.download_deepset import main as download_main
            download_main()
        scanner.train(train_path)
    else:
        print("Scanner model ready.")

    print("\n=== PHASE: Trust Calibration ===")
    n_cal = 15 if quick else cfg.get("calibration_runs", 80)
    from drift.behavioral_drift import PROMPT_BASELINE_NAME
    need_cal = force_calibrate or not drift.baselines_valid_for(
        ("Retriever", "Planner", "Executor", "Generator", PROMPT_BASELINE_NAME)
    )
    if need_cal:
        run_calibration(logger, drift, n_runs=n_cal)
    else:
        print(f"Calibration baselines ready ({n_cal} runs configured). Use --recalibrate to refresh.")

    if not drift.baselines_valid_for(("Retriever", "Planner", "Executor", "Generator", PROMPT_BASELINE_NAME)):
        raise RuntimeError("Calibration baseline manifest is missing or invalid; refusing to run experiments.")
    # BatchTrustTrace is constructed before calibration to keep its stack
    # reusable; refresh its cached global centroid after new baselines land.
    batch.irs.set_global_benign_centroid(drift.get_global_centroid())

    scenarios = expand_scenarios(n_per_type=n_per_type)
    baseline = run_baseline_experiments(scenarios)
    expected_scenario_ids = [s.get("scenario_id") for s in scenarios]
    if baseline.get("scenario_ids") != expected_scenario_ids:
        raise RuntimeError("Baseline and defended scenario manifests differ; refusing comparative ASR reporting")
    baseline_asr_pct = baseline.get("asr_pct", 0.0)

    print("\n=== PHASE: TrustTrace Attack Experiments ===")
    collector = MetricsCollector()
    attack_results: list = []

    for i, scenario in enumerate(scenarios):
        attack_type = scenario["type"]
        gt_agent = scenario.get("ground_truth") or GROUND_TRUTH_MAP.get(attack_type, "Retriever")
        if config.EXPERIMENT_MODE:
            print(f"  [{i + 1}/{len(scenarios)}]", end="")
        t0 = time.perf_counter()
        result = batch.run_attack_scenario(scenario, enable_recovery=True)
        t1 = time.perf_counter()
        # Record execution time and average trust score after the run
        result["execution_time"] = t1 - t0
        trust_scores = list(batch.trust.trust_scores.values())
        result["avg_trust"] = statistics.mean(trust_scores) if trust_scores else None
        attack_results.append(result)

        collector.record_attack_with_source(
            succeeded=result["succeeded"],
            detected=result["detected"],
            source=scenario.get("source", "handcrafted"),
            attack_type=attack_type,
            pz_predicted=result.get("patient_zero"),
            pz_ground_truth=gt_agent,
            recovery_time_s=result.get("recovery_time_s"),
            chroma_size=result.get("chroma_size", 0),
            retrieval_hits=result.get("retrieval_hits", 0),
            patient_zero_methods=result.get("patient_zero_methods", {}),
        )
    # Immediately generate a comprehensive HTML report after the attack scenarios
    metrics_partial = collector.report()
    # compute aggregated values available from execution
    avg_irs = 0.0
    irs_vals = [r.get("pz_confidence") for r in attack_results if r.get("pz_confidence") is not None]
    if irs_vals:
        avg_irs = sum(irs_vals) / len(irs_vals)
    avg_trust = 0.0
    trust_vals = [r.get("avg_trust") for r in attack_results if r.get("avg_trust") is not None]
    if trust_vals:
        avg_trust = sum(trust_vals) / len(trust_vals)

    report_data = {
        "project_name": "TrustTrace",
        "experiment_date": experiment_start_dt,
        "duration_s": time.time() - experiment_start_ts,
        "llm_provider": _load_config().get("llm", {}).get("provider", "unknown"),
        "dataset_used": os.path.join("data", "deepset_injections"),
        "total_prompts": len(scenarios),
        "author": "TrustTrace Research",
        "version": "3.0-hardened-candidate",
        "ablation_id": ablation_id,
        "attack_success_measurement": (
            "final Generator lexical marker proxy; not a semantic goal-completion oracle"
        ),
        "scenarios": scenarios,
        "scenario_manifest_sha256": __import__("hashlib").sha256(
            yaml.safe_dump(scenarios, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "baseline_scenario_ids": baseline.get("scenario_ids", []),
        "attack_results": attack_results,
        "metrics": metrics_partial,
        "combined_metrics": metrics_partial,
        "baseline": baseline,
        "avg_irs": avg_irs,
        "avg_trust": avg_trust,
        "avg_latency": sum(r.get("execution_time", 0) for r in attack_results) / (len(attack_results) or 1),
        "benign_false_positives": [],
    }

    reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "experiment_report.html")

    _print_asr_audit_table(scenarios, attack_results)
    _print_final_summary(attack_results, baseline_asr_pct)

    print(f"\n=== PHASE: Benign Runs ({n_benign}) ===")
    benign_fps = []
    for i, query in enumerate(get_benign_queries(n_benign)):
        result = batch.run_benign(query)
        collector.record_benign(result["detected"], query=query)
        if result["detected"]:
            benign_fps.append(query)
        if (i + 1) % 10 == 0 or quick:
            print(f"  Benign {i + 1}/{n_benign}: flagged={result['detected']}")

    print("\n=== PHASE: Results ===")
    metrics = collector.report()
    collector.report_split()

    # Generate the final HTML report only after benign episodes are recorded,
    # so its FPR and false-positive list are based on the complete run.
    report_data["metrics"] = metrics
    report_data["combined_metrics"] = metrics
    report_data["benign_false_positives"] = list(benign_fps)
    report_data["research_readiness"] = "diagnostic_not_publication_ready"
    report_data["holdout_evaluated"] = False
    try:
        generate_experiment_report(report_path, report_data)
        print(f"\nGenerated final HTML experiment report: {report_path}")
    except Exception as exc:
        print(f"Failed to write HTML report: {exc}")

    print("\n" + "=" * 50)
    print("FINAL VALIDATION REPORT")
    print("=" * 50)
    print(f"Detection Rate:           {metrics['Detection_Rate']:.2%}")
    print(f"ASR:                      {metrics['ASR']:.2%}")
    print(f"FPR:                      {metrics['FPR']:.2%}")
    print(f"Patient Zero Accuracy:    {metrics['Patient_Zero_Accuracy']:.2%}")
    mean_recovery = "N/A" if metrics["Recovery_Time_Mean_s"] is None else f"{metrics['Recovery_Time_Mean_s']:.3f}s"
    median_recovery = "N/A" if metrics["Recovery_Time_Median_s"] is None else f"{metrics['Recovery_Time_Median_s']:.3f}s"
    print(f"Mean Recovery Time:       {mean_recovery}")
    print(f"Median Recovery Time:     {median_recovery}")
    availability = metrics["System_Availability"]
    availability_text = "N/A (no task observations)" if availability is None else f"{availability:.2%}"
    print(f"Availability (Recovery):  {availability_text}")
    print(f"Chroma Collection Size:   {metrics['Chroma_Collection_Size_Mean']:.1f}")
    print(f"Retrieval Hits (mean):    {metrics['Retrieval_Hits_Mean']:.1f}")
    print(f"Benign False Positives:   {metrics['Benign_False_Positive_Count']}")
    if benign_fps:
        print("  Flagged benign queries:")
        for q in benign_fps[:5]:
            print(f"    - {q[:70]}")
    print("=" * 50)

    if not quick:
        print("\n=== PHASE: Bayesian Hyperparameter Search ===")
        best_params = run_search(
            scenarios,
            [s.get("ground_truth", "Retriever") for s in scenarios],
            batch_factory=lambda: BatchTrustTrace(logger, scanner, drift, memory, ablation_id=ablation_id),
            n_trials=n_trials,
        )
        print("Best params written to config.yaml")
    else:
        best_params = {}

    return {
        "baseline": baseline,
        "attack_results": attack_results,
        "combined_metrics": metrics,
        "best_params": best_params,
        "n_scenarios": len(scenarios),
        "scenarios": scenarios,
        "ablation_id": ablation_id,
        "benign_false_positives": benign_fps,
        "research_readiness": "diagnostic_not_publication_ready",
        "holdout_evaluated": False,
    }
