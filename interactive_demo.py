"""
Interactive TrustTrace Demo

This script provides an interactive mode where users can input any prompt
and TrustTrace will:
1. Run the prompt through the victim pipeline
2. Trace which agents are used
3. Build the propagation graph dynamically
4. Calculate trust scores in real-time
5. Visualize the interaction graph
6. Display trust scores and detection results

This complements the experimental mode in main.py which is used for paper results.
"""

import os
import sys
import time
import uuid
from typing import Dict, List, Optional

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from logger.interaction_logger import InteractionLogger, log_pipeline_run
from scanner.injection_scanner import InjectionScanner
from graph.propagation_graph import PropagationGraph
from drift.behavioral_drift import BehavioralDriftModule
from trust.trust_engine import TrustEngine
from memory.memory_manager import MemoryManager
from detector.patient_zero import PatientZeroDetector
from recovery.recovery_manager import RecoveryManager
from runtime.trusttrace_runtime import TrustTraceRuntime
from victim_pipeline.agents import run_pipeline, knowledge_base
from calibration.calibrate import run_calibration
import yaml


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def initialize_trusttrace():
    """Initialize all TrustTrace components."""
    os.environ.setdefault("TRUSTTRACE_STRICT_EVAL", "1")
    from victim_pipeline.agents import is_persistent_knowledge_base, get_chroma_initialization_error
    if not is_persistent_knowledge_base():
        cause = get_chroma_initialization_error()
        raise RuntimeError(
            "Interactive TrustTrace requires persistent ChromaDB; the in-memory "
            "fallback is reserved for isolated tests."
        ) from cause

    print("=== Initializing TrustTrace Components ===")
    
    cfg = _load_config()
    
    # Initialize components
    logger = InteractionLogger()
    scanner = InjectionScanner()
    graph = PropagationGraph()
    drift = BehavioralDriftModule()
    trust_engine = TrustEngine(graph, drift)
    
    # CRITICAL FIX: Use shared knowledge_base collection from victim pipeline
    # This ensures MemoryManager tracks the same writes that attacks inject
    memory_mgr = MemoryManager(collection=knowledge_base)
    
    detector = PatientZeroDetector(graph, trust_engine)
    recovery_mgr = RecoveryManager(memory_mgr, trust_engine, detector)

    # Ensure the real training dataset exists before training the scanner.
    train_path = os.path.join("data", "deepset_injections", "train.json")
    if not os.path.isfile(train_path):
        from data.download_deepset import main as download_main
        download_main()

    # Train scanner if needed
    model_path = os.path.join("scanner", "scanner_model.pkl")
    scaler_path = os.path.join("scanner", "scanner_scaler.pkl")
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print("Training injection scanner...")
        scanner.train("data/deepset_injections/train.json")
    else:
        print("Scanner model already trained.")
    
    # Check calibration
    required_baselines = ("Retriever", "Planner", "Executor", "Generator", "UserPrompt")
    calib_needed = not drift.baselines_valid_for(required_baselines)
    if calib_needed:
        print("Running calibration phase...")
        run_calibration(logger, drift, n_runs=cfg.get("calibration_runs", 80))
    else:
        print("Calibration baselines already exist.")

    if not drift.baselines_valid_for(required_baselines):
        raise RuntimeError("Calibration did not produce a valid embedding-bound baseline manifest.")

    runtime = TrustTraceRuntime(
        logger=logger,
        scanner=scanner,
        graph=graph,
        drift=drift,
        trust_engine=trust_engine,
        memory=memory_mgr,
        detector=detector,
        recovery=recovery_mgr,
    )
    
    thresholds = {
        "lambda_direct": cfg.get("lambda_direct", 0.6),
        "lambda_indirect": cfg.get("lambda_indirect", 0.55),
        "lambda_memory": cfg.get("lambda_memory", 0.5),
    }
    
    return {
        "logger": logger,
        "scanner": scanner,
        "graph": graph,
        "drift": drift,
        "trust_engine": trust_engine,
        "memory_mgr": memory_mgr,
        "detector": detector,
        "recovery_mgr": recovery_mgr,
        "runtime": runtime,
        "thresholds": thresholds,
        "config": cfg,
    }


def process_user_prompt(
    prompt: str,
    components: Dict,
    attack_type: str = "benign",
) -> Dict:
    """Process a prompt through the same enforced TrustTrace runtime as main.py.

    The previous interactive demo called run_pipeline() first and only scored
    its outputs afterwards. That allowed the LLM to consume unsafe content
    before the security checks ran. This path delegates execution and blocking
    to TrustTraceRuntime.execute() instead.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    runtime = components.get("runtime")
    if runtime is None:
        raise RuntimeError("TrustTrace runtime is not initialized; restart the interactive demo.")

    normalized_type = (attack_type or "benign").lower()
    report = runtime.execute(
        prompt,
        attack_type=None if normalized_type == "benign" else normalized_type,
        auto_detect=(normalized_type == "benign"),
        simulate_attack=False,
    )
    run_id = report.get("run_id", "unknown")
    events = runtime.logger.get_events_for_run(run_id) if run_id != "unknown" else []
    return {
        **report,
        "run_id": run_id,
        "prompt": prompt,
        "attack_type": report.get("attack_type", normalized_type),
        "outputs": report.get("agent_outputs", {"Generator": report.get("final_output", "")}),
        "trust_scores": report.get("final_trust_scores", dict(runtime.trust.trust_scores)),
        "compromised_agents": report.get("compromised", []),
        "patient_zero": report.get("patient_zero"),
        "threshold_exceeded": bool(report.get("detected")),
        "events_processed": len(events),
    }


def print_results(result: Dict):
    """Print formatted results."""
    print("\n" + "=" * 60)
    print("TRUSTRACE ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\nRun ID: {result['run_id']}")
    print(f"Prompt: {result['prompt']}")
    print(f"Attack Type: {result['attack_type']}")
    print(f"Events Processed: {result['events_processed']}")
    
    print("\n--- Agent Outputs ---")
    for agent, output in result['outputs'].items():
        print(f"{agent}: {output[:100]}...")
    
    print("\n--- Trust Scores ---")
    for agent, score in result['trust_scores'].items():
        status = "COMPROMISED" if agent in result['compromised_agents'] else "OK"
        print(f"{agent}: {score:.4f} [{status}]")
    
    if result['compromised_agents']:
        print(f"\n--- Compromise Detected ---")
        print(f"Compromised Agents: {result['compromised_agents']}")
        print(f"Patient Zero: {result['patient_zero']}")
    else:
        print("\n--- No Compromise Detected ---")
    
    if result['threshold_exceeded']:
        print(f"\n--- Threshold Alert ---")
        print(f"Suspicion score exceeded threshold for {result['attack_type']} attack")
    
    print("=" * 60)


def print_graph_summary(graph: PropagationGraph):
    """Print a summary of the propagation graph."""
    print("\n--- Propagation Graph Summary ---")
    print(f"Total Nodes: {len(graph.get_all_nodes())}")
    print(f"Nodes: {graph.get_all_nodes()}")
    
    print("\n--- Edge Weights (Mean Suspicion) ---")
    for node in graph.get_all_nodes():
        predecessors = graph.get_predecessors(node)
        for pred in predecessors:
            weight = graph.get_edge_weight(pred, node)
            if weight > 0:
                print(f"{pred} -> {node}: {weight:.4f}")


def interactive_mode():
    """Run interactive mode where user inputs prompts."""
    print("=" * 60)
    print("TRUSTRACE INTERACTIVE MODE")
    print("=" * 60)
    print("\nThis mode allows you to:")
    print("- Input any prompt and see TrustTrace analyze it")
    print("- View real-time trust scores for each agent")
    print("- See the propagation graph build dynamically")
    print("- Detect compromises and identify Patient Zero")
    print("\nCommands:")
    print("- Type your prompt to analyze it")
    print("- Type 'graph' to see the current graph")
    print("- Type 'reset' to reset trust scores and graph")
    print("- Type 'quit' to exit")
    print("=" * 60)
    
    # Initialize components
    components = initialize_trusttrace()
    
    while True:
        print("\n" + "-" * 60)
        user_input = input("Enter prompt (or command): ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("Exiting interactive mode...")
            break
        
        elif user_input.lower() == 'graph':
            print_graph_summary(components["graph"])
        
        elif user_input.lower() == 'reset':
            print("Resetting trust scores and graph...")
            components["runtime"].reset_state()
            components["graph"] = components["runtime"].graph
            components["trust_engine"] = components["runtime"].trust
            components["detector"] = components["runtime"].detector
            components["recovery_mgr"] = components["runtime"].recovery
            print("Reset complete.")
        
        else:
            # Process the prompt
            result = process_user_prompt(user_input, components, attack_type="benign")
            print_results(result)


def attack_simulation_mode():
    """Run demonstration attacks through TrustTrace's enforced runtime."""
    print("=" * 60)
    print("TRUSTRACE ATTACK SIMULATION MODE")
    print("=" * 60)
    print("1. Direct Injection")
    print("2. Indirect Injection")
    print("3. Memory Poisoning")
    print("4. Benign Query")
    print("Type 'quit' to exit.")

    components = initialize_trusttrace()
    payload_counts = {"direct": 6, "indirect": 4, "memory": 4}
    queries = {
        "direct": "Please summarize the project documentation.",
        "indirect": "Tell me about data security best practices.",
        "memory": "What are the user preferences and memory instructions?",
    }

    while True:
        choice = input("\nSelect attack type (1-4 or quit): ").strip().lower()
        if choice == "quit":
            print("Exiting attack simulation mode...")
            return
        if choice == "4":
            prompt = input("Enter a benign query: ").strip()
            if not prompt:
                print("Please enter a non-empty query.")
                continue
            result = process_user_prompt(prompt, components, attack_type="benign")
            print_results(result)
            continue
        attack_map = {"1": "direct", "2": "indirect", "3": "memory"}
        attack_type = attack_map.get(choice)
        if attack_type is None:
            print("Invalid choice.")
            continue
        payload_index_text = input(f"Select payload index (0-{payload_counts[attack_type] - 1}): ").strip()
        try:
            payload_index = int(payload_index_text)
            if not 0 <= payload_index < payload_counts[attack_type]:
                raise ValueError
        except ValueError:
            print("Invalid payload index.")
            continue

        # Run all synthetic attack insertion and enforcement inside the runtime.
        report = components["runtime"].execute(
            queries[attack_type],
            attack_type=attack_type,
            payload_index=payload_index,
            auto_detect=False,
            simulate_attack=True,
        )
        run_id = report.get("run_id", "unknown")
        events = components["logger"].get_events_for_run(run_id) if run_id != "unknown" else []
        result = {
            **report,
            "run_id": run_id,
            "prompt": queries[attack_type],
            "attack_type": attack_type,
            "outputs": report.get("agent_outputs", {"Generator": report.get("final_output", "")}),
            "trust_scores": report.get("final_trust_scores", dict(components["runtime"].trust.trust_scores)),
            "compromised_agents": report.get("compromised", []),
            "patient_zero": report.get("patient_zero"),
            "threshold_exceeded": bool(report.get("detected")),
            "events_processed": len(events),
        }
        print_results(result)


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("TRUSTRACE INTERACTIVE DEMO")
    print("=" * 60)
    print("\nSelect mode:")
    print("1. Interactive Mode - Input your own prompts")
    print("2. Attack Simulation Mode - Test against known attacks")
    print("3. Exit")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        interactive_mode()
    elif choice == '2':
        attack_simulation_mode()
    else:
        print("Exiting...")


if __name__ == "__main__":
    main()
