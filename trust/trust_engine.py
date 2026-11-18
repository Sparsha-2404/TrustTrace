from __future__ import annotations

import os
import time
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

import yaml
import config

from drift.behavioral_drift import BehavioralDriftModule
from graph.propagation_graph import PropagationGraph


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


@dataclass
class TrustUpdateResult:
    agent: str
    sender: str
    previous_trust: float
    new_trust: float
    ms: float
    bd: float
    recovery: float
    propagation_drop: float
    weight: float
    compromised: bool


@dataclass
class MultilayerResult:
    agent: str
    compromised: bool
    irs: float
    ms: float
    bd: float
    lambda_threshold: float
    delta: float
    gamma: float
    irs_pass: bool
    ms_pass: bool
    bd_pass: bool
    consecutive: int
    required_k: int


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TrustEngine:
    """
    Base trust update: T_a(t) = rho*T_a(t-1) + (1-rho)*H_a(t),
    where H_a(t) = 1 - (MS + BD)/2. Trust loss is then propagated
    separately over the interaction graph.
    Compromise requires multi-layer: IRS > lambda, MS > delta, BD > gamma (k consecutive).
    """

    EXCLUDED_AGENTS = {"TrustTrace", "System"}

    def __init__(self, graph: PropagationGraph, drift_module: BehavioralDriftModule):
        cfg = _load_config()
        self.graph = graph
        self.drift = drift_module
        self.mu = float(cfg.get("mu", 0.3))
        self.rho = float(cfg.get("rho", 0.95))
        self.delta = float(cfg.get("delta", 0.4))
        self.gamma = float(cfg.get("gamma", 0.25))
        for name, value in (("mu", self.mu), ("rho", self.rho), ("delta", self.delta), ("gamma", self.gamma)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"config.yaml '{name}' must be between 0 and 1")
        # Per-agent persistence can be configured explicitly. The global k is
        # the fallback for agent names not listed in the table.
        self.persistence_table = {
            "Planner": 2,   # Receives from User + Retriever
            "Retriever": 2, # Receives from User
            "Executor": 1,  # Receives from Planner
            "Generator": 1, # Receives from Executor
        }
        configured_persistence = cfg.get("persistence_by_agent", {}) or {}
        if not isinstance(configured_persistence, dict):
            raise ValueError("config.yaml 'persistence_by_agent' must be a mapping")
        for agent, required in configured_persistence.items():
            if isinstance(required, bool):
                raise ValueError(f"persistence_by_agent.{agent} must be a positive integer")
            try:
                required = int(required)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"persistence_by_agent.{agent} must be a positive integer") from exc
            if required < 1:
                raise ValueError(f"persistence_by_agent.{agent} must be a positive integer")
            self.persistence_table[str(agent)] = required
        self.k = int(cfg.get("persistence_window", cfg.get("k", 3)))
        if self.k < 1:
            raise ValueError("config.yaml 'persistence_window' must be at least 1")
        self.trust_scores: Dict[str, float] = {}
        self.multilayer_counts: Dict[str, int] = {}
        self.compromised: Set[str] = set()
        self.compromise_timestamps: Dict[str, float] = {}
        self.recovery_timestamps: Dict[str, float] = {}
        self.last_multilayer: Dict[str, MultilayerResult] = {}
        self.last_update: Optional[TrustUpdateResult] = None
        self._debug = config.DEBUG
        for node in self.graph.get_all_nodes():
            self._init_agent(node)

    def _init_agent(self, name: str) -> None:
        if name in self.EXCLUDED_AGENTS:
            return
        if name not in self.trust_scores:
            self.trust_scores[name] = 1.0
            self.multilayer_counts[name] = 0

    @staticmethod
    def _bound(value: float) -> float:
        return min(1.0, max(0.0, value))

    def update(
        self,
        receiver: str,
        sender: str,
        ms: float,
        bd: float,
        current_output: str,
        timestamp: Optional[float] = None,
    ) -> TrustUpdateResult:
        if not isinstance(receiver, str) or not receiver or not isinstance(sender, str) or not sender:
            raise ValueError("receiver and sender must be non-empty strings")
        if timestamp is not None:
            try:
                timestamp = float(timestamp)
            except (TypeError, ValueError) as exc:
                raise ValueError("timestamp must be a finite non-negative number") from exc
            if not math.isfinite(timestamp) or timestamp < 0.0:
                raise ValueError("timestamp must be a finite non-negative number")
        if receiver in self.EXCLUDED_AGENTS:
            return TrustUpdateResult(
                agent=receiver,
                sender=sender,
                previous_trust=1.0,
                new_trust=1.0,
                ms=ms,
                bd=bd,
                recovery=0.0,
                propagation_drop=0.0,
                weight=0.0,
                compromised=False,
            )
        self._init_agent(receiver)
        self._init_agent(sender)

        T_A = self.trust_scores[sender]
        T_B = self.trust_scores[receiver]
        w_AB = self.graph.get_edge_weight(sender, receiver)

        # Version 3 trust equation: behavioural health is the EMA target.
        # MS and BD are expected to be normalised to [0, 1] by their producers.
        if isinstance(ms, bool) or isinstance(bd, bool):
            raise ValueError("MS and BD must be numeric values in [0, 1]")
        try:
            ms, bd = float(ms), float(bd)
        except (TypeError, ValueError) as exc:
            raise ValueError("MS and BD must be numeric values in [0, 1]") from exc
        if not math.isfinite(ms) or not math.isfinite(bd) or not (0.0 <= ms <= 1.0) or not (0.0 <= bd <= 1.0):
            raise ValueError("MS and BD must be finite and normalised to [0, 1]")
        H = 1.0 - (ms + bd) / 2.0
        ema_contribution = (1.0 - self.rho) * H
        T_new = self.rho * T_B + ema_contribution
        T_new = self._bound(T_new)

        # Keep graph propagation as a distinct operation after the EMA update.
        propagation_drop = self.mu * w_AB * (1.0 - T_A)
        T_new = self._bound(T_new * (1.0 - propagation_drop))
        # Preserve the existing result field for downstream logging compatibility;
        # it now records the positive EMA health contribution, not a recovery action.
        recovery = ema_contribution

        if self._debug:
            print(f"[BD Debug] Agent: {receiver} | BD: {bd:.3f} | Trust Before: {T_B:.3f} | Trust After: {T_new:.3f}")

        self.trust_scores[receiver] = T_new
        self.graph.set_trust(receiver, T_new)

        result = TrustUpdateResult(
            agent=receiver,
            sender=sender,
            previous_trust=T_B,
            new_trust=T_new,
            ms=ms,
            bd=bd,
            recovery=recovery,
            propagation_drop=propagation_drop,
            weight=w_AB,
            compromised=receiver in self.compromised,
        )
        self.last_update = result
        return result

    def evaluate_multilayer(
        self,
        agent: str,
        irs: float,
        ms: float,
        bd: float,
        lambda_threshold: float,
        timestamp: Optional[float] = None,
    ) -> MultilayerResult:
        """
        Compromise only when IRS > lambda AND MS > delta AND BD > gamma for k consecutive evaluations.
        Additionally, the agent's trust score must be below the delta threshold.
        Uses adaptive persistence window per agent.
        """
        if agent in self.EXCLUDED_AGENTS:
            return MultilayerResult(
                agent=agent,
                compromised=False,
                irs=irs,
                ms=ms,
                bd=bd,
                lambda_threshold=lambda_threshold,
                delta=self.delta,
                gamma=self.gamma,
                irs_pass=False,
                ms_pass=False,
                bd_pass=False,
                consecutive=0,
                required_k=self.persistence_table.get(agent, self.k),
            )
        self._init_agent(agent)
        ts = time.time() if timestamp is None else float(timestamp)
        if not math.isfinite(ts) or ts < 0.0:
            raise ValueError("timestamp must be a finite non-negative number")
        validated = {}
        for name, value in (("IRS", irs), ("MS", ms), ("BD", bd), ("lambda_threshold", lambda_threshold)):
            if isinstance(value, bool):
                raise ValueError(f"{name} must be numeric and normalised to [0, 1]")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric and normalised to [0, 1]") from exc
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ValueError(f"{name} must be finite and normalised to [0, 1]")
            validated[name] = number
        irs = validated["IRS"]
        ms = validated["MS"]
        bd = validated["BD"]
        lambda_threshold = validated["lambda_threshold"]

        # Get adaptive k for this agent
        required_k = self.persistence_table.get(agent, self.k)

        irs_pass = irs > lambda_threshold
        ms_pass = ms > self.delta
        bd_pass = bd > self.gamma
        
        # Additional check: trust score must be below threshold
        current_trust = self.trust_scores.get(agent, 1.0)
        trust_below_threshold = current_trust < self.delta
        
        all_pass = irs_pass and ms_pass and bd_pass and trust_below_threshold

        if all_pass:
            self.multilayer_counts[agent] = self.multilayer_counts.get(agent, 0) + 1
            # The persistence window is an actual confirmation requirement.
            # A single suspicious observation must not mark an agent compromised.
            if self.multilayer_counts[agent] >= required_k and agent not in self.compromised:
                self.compromise_timestamps[agent] = ts
                self.compromised.add(agent)
        else:
            # Consecutive confirmation resets on any failed layer. Once an
            # incident is confirmed, keep the compromised state latched until
            # explicit recovery; a transient benign sample must not silently
            # clear containment.
            self.multilayer_counts[agent] = 0

        consecutive = self.multilayer_counts.get(agent, 0)

        result = MultilayerResult(
            agent=agent,
            compromised=agent in self.compromised,
            irs=irs,
            ms=ms,
            bd=bd,
            lambda_threshold=lambda_threshold,
            delta=self.delta,
            gamma=self.gamma,
            irs_pass=irs_pass,
            ms_pass=ms_pass,
            bd_pass=bd_pass,
            consecutive=consecutive,
            required_k=required_k,
        )
        self.last_multilayer[agent] = result

        if (self._debug or config.EXPERIMENT_MODE) and not getattr(config, "SUPPRESS_MULTILAYER", False):
            status = "COMPROMISED" if result.compromised else "healthy"
            failed = []
            if not irs_pass:
                failed.append(f"IRS {irs:.3f} <= lambda {lambda_threshold:.3f}")
            if not ms_pass:
                failed.append(f"MS {ms:.3f} <= delta {self.delta:.3f}")
            if not bd_pass:
                failed.append(f"BD {bd:.3f} <= gamma {self.gamma:.3f}")
            if not trust_below_threshold:
                failed.append(f"Trust {current_trust:.3f} >= delta {self.delta:.3f}")
            fail_msg = "; ".join(failed) if failed else "all layers passed"
            print(
                f"[Multilayer] {agent} | {status} | IRS={irs:.3f} MS={ms:.3f} BD={bd:.3f} Trust={current_trust:.3f} | "
                f"k={consecutive}/{required_k} | {fail_msg}"
            )

        return result

    def compromise(self, agent: str, timestamp: Optional[float] = None) -> None:
        """Explicitly mark agent compromised (multi-layer confirmed)."""
        self._init_agent(agent)
        ts = time.time() if timestamp is None else float(timestamp)
        if not 0.0 <= ts < float("inf"):
            raise ValueError("timestamp must be a finite non-negative number")
        self.compromised.add(agent)
        self.compromise_timestamps[agent] = ts
        self.multilayer_counts[agent] = self.persistence_table.get(agent, self.k)

    def is_compromised(self, agent: str) -> bool:
        return agent in self.compromised

    def get_all_compromised(self) -> Set[str]:
        return set(self.compromised)

    def get_compromise_timestamp(self, agent: str) -> Optional[float]:
        return self.compromise_timestamps.get(agent)

    def get_recovery_timestamp(self, agent: str) -> Optional[float]:
        return self.recovery_timestamps.get(agent)

    def recover_agent(self, agent: str) -> float:
        ts = time.time()
        self.trust_scores[agent] = 1.0
        self.multilayer_counts[agent] = 0
        self.compromised.discard(agent)
        self.compromise_timestamps.pop(agent, None)
        self.recovery_timestamps[agent] = ts
        self.graph.set_trust(agent, 1.0)
        self.graph.reset_compromise_count(agent)
        return 1.0

    def recover_gradually(self, agent: str, kappa: float = 0.1) -> None:
        """Eq (26): T_new_a = min(1, T_a + kappa)"""
        try:
            kappa = float(kappa)
        except (TypeError, ValueError) as exc:
            raise ValueError("kappa must be between 0 and 1") from exc
        if not 0.0 <= kappa <= 1.0:
            raise ValueError("kappa must be between 0 and 1")
        if agent not in self.compromised:
            T = self.trust_scores.get(agent, 1.0)
            T_new = self._bound(T + kappa)
            self.trust_scores[agent] = T_new
            self.graph.set_trust(agent, T_new)
