"""
Bayesian hyperparameter search (Change 7).
Optuna TPE, 70/30 split, tunes trust-engine parameters.
"""

from __future__ import annotations

import os
import logging
from typing import Callable, List, Optional

import optuna
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_config(params: dict) -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.update(params)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)


def objective(
    trial: optuna.Trial,
    attack_scenarios: list,
    ground_truths: list,
    batch_factory: Callable,
) -> float:
    """Maximise Detection Rate − FPR."""
    # Tune only parameters that are part of the implemented Version 3 model.
    # Legacy w_m/w_b/eta parameters belonged to an obsolete trust equation.
    params = {
        "lambda": trial.suggest_float("lambda", 0.25, 0.6),
        "rho": trial.suggest_float("rho", 0.80, 0.99),
        "mu": trial.suggest_float("mu", 0.0, 0.6),
        "delta": trial.suggest_float("delta", 0.2, 0.6),
        "gamma": trial.suggest_float("gamma", 0.1, 0.6),
        "persistence_window": trial.suggest_int("persistence_window", 1, 6),
    }
    params["k"] = params["persistence_window"]

    _write_config(params)

    from eval.metrics import MetricsCollector
    collector = MetricsCollector()

    for scenario, _truth in zip(attack_scenarios, ground_truths):
        batch = batch_factory()
        try:
            result = batch.run_attack_scenario(scenario, enable_recovery=False)
        except Exception as exc:
            raise RuntimeError(
                f"Hyperparameter trial {trial.number} failed on scenario "
                f"{scenario.get('scenario_id', scenario)!r}; failed trials are not "
                "silently converted into missed detections."
            ) from exc
        if not isinstance(result, dict) or "succeeded" not in result or "detected" not in result:
            raise RuntimeError(
                f"Hyperparameter trial {trial.number} received an invalid result "
                f"for scenario {scenario.get('scenario_id', scenario)!r}."
            )
        collector.record_attack(succeeded=result["succeeded"], detected=result["detected"])

    metrics = collector.compute()
    return metrics["Detection_Rate"] - metrics["FPR"]


def run_search(
    attack_scenarios: list,
    ground_truths: list,
    batch_factory: Callable,
    n_trials: int = 100,
) -> dict:
    """Run Bayesian search on 70% tuning split. Returns best parameters."""
    if len(attack_scenarios) != len(ground_truths):
        raise ValueError("attack_scenarios and ground_truths must have equal lengths")
    if len(attack_scenarios) < 2:
        raise ValueError("At least two scenarios are required for a 70/30 tuning split")

    # Deterministic stratified split avoids the prior order bias caused by
    # scenarios being grouped by attack family. The holdout indices are not
    # used for optimization and are returned for a separate frozen evaluation.
    groups = {}
    for index, scenario in enumerate(attack_scenarios):
        key = (scenario.get("type", "unknown"), scenario.get("source", "handcrafted"))
        groups.setdefault(key, []).append(index)
    tune_indices, holdout_indices = [], []
    for key in sorted(groups):
        indices = groups[key]
        if len(indices) == 1:
            tune_indices.extend(indices)
        else:
            cut = max(1, min(len(indices) - 1, int(len(indices) * 0.7)))
            tune_indices.extend(indices[:cut])
            holdout_indices.extend(indices[cut:])
    if not holdout_indices:
        # A tiny dataset cannot support a stratified holdout; fail rather than
        # silently evaluate on training rows.
        raise ValueError("Stratified split produced no holdout rows; provide more scenarios per stratum")
    tune_indices.sort()
    holdout_indices.sort()
    tune_scenarios = [attack_scenarios[i] for i in tune_indices]
    tune_truths = [ground_truths[i] for i in tune_indices]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: objective(trial, tune_scenarios, tune_truths, batch_factory),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    best["persistence_window"] = best.get("persistence_window", 3)
    best["k"] = best["persistence_window"]
    print(f"\nBest params after {n_trials} trials:")
    for k, v in best.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    _write_config(best)
    best["tuning_split_indices"] = tune_indices
    best["holdout_split_indices"] = holdout_indices
    best["holdout_evaluated"] = False
    return best


def run_tuning(scanner, drift, n_trials: int = 20) -> dict:
    """Backward-compatible wrapper used by legacy scripts."""
    from experiment.batch_processor import BatchTrustTrace
    from experiment.scenarios import ATTACK_SCENARIOS_HANDCRAFTED
    from logger.interaction_logger import InteractionLogger
    from memory.memory_manager import MemoryManager
    from victim_pipeline.agents import knowledge_base

    logger = InteractionLogger()
    memory = MemoryManager(collection=knowledge_base)
    scenarios = ATTACK_SCENARIOS_HANDCRAFTED
    truths = [s.get("ground_truth", "Retriever") for s in scenarios]

    def factory():
        return BatchTrustTrace(logger, scanner, drift, memory)

    return run_search(scenarios, truths, factory, n_trials=n_trials)
