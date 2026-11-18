import time
from typing import List, Optional

import networkx as nx

from graph.propagation_graph import PropagationGraph
from trust.trust_engine import TrustEngine


class PatientZeroDetector:
    """Graph-derived patient zero via backward traversal on highest-suspicion edges."""

    TIE_BREAK_ORDER = {
        "User": 0,
        "Attacker": 0,
        "MemoryStore": 1,
        "KnowledgeBase": 1,
        "Retriever": 2,
        "Planner": 3,
        "Executor": 4,
        "Generator": 5,
    }

    def __init__(self, graph: PropagationGraph, trust_engine: TrustEngine):
        self.graph = graph
        self.trust = trust_engine
        self.detection_log: List[dict] = []

    def _compute_pcs(self, agent: str, compromised: set) -> float:
        """
        Eq (19): PCS_a = D_a / (1 + t_a)
        D_a = out-degree of compromised influence (edges to other compromised nodes)
        t_a = first compromise timestamp (relative, seconds since earliest compromise)
        """
        # D_a: number of compromised nodes this agent has edges to in the graph
        successors = list(self.graph.G.successors(agent)) if self.graph.G.has_node(agent) else []
        D_a = sum(1 for n in successors if n in compromised)

        # t_a: timestamp relative to earliest compromise (so argmin timestamp = argmax PCS)
        all_timestamps = [
            self.trust.get_compromise_timestamp(a)
            for a in compromised
            if self.trust.get_compromise_timestamp(a) is not None
        ]
        agent_timestamp = self.trust.get_compromise_timestamp(agent)
        if agent_timestamp is None or not all_timestamps:
            # Missing timestamps are unknown, not zero or NaN. Give this node
            # zero temporal-evidence contribution; deterministic tie-breaks
            # then decide among otherwise indistinguishable candidates.
            t_a = float("inf")
        else:
            earliest = min(all_timestamps)
            t_a = max(0.0, float(agent_timestamp) - float(earliest))

        pcs = 0.0 if t_a == float("inf") else D_a / (1.0 + t_a)
        return pcs


    def detect_earliest_flagged(self) -> Optional[str]:
        """Baseline PZ method: earliest compromised component by first flag time."""
        excluded = getattr(self.trust, "EXCLUDED_AGENTS", set())
        compromised = [a for a in self.trust.get_all_compromised() if a not in excluded]
        if not compromised:
            return None
        order = self.TIE_BREAK_ORDER
        return min(compromised, key=lambda a: (
            self.trust.get_compromise_timestamp(a)
            if self.trust.get_compromise_timestamp(a) is not None else float("inf"),
            order.get(a, 99), a
        ))

    def detect_stage1_only(self) -> Optional[str]:
        """Stage-1 baseline: choose the earliest compromised pipeline node (no graph walk)."""
        excluded = getattr(self.trust, "EXCLUDED_AGENTS", set())
        compromised = [a for a in self.trust.get_all_compromised() if a not in excluded]
        if not compromised:
            return None
        pipeline_order = {"Retriever": 0, "Planner": 1, "Executor": 2, "Generator": 3}
        return min(compromised, key=lambda a: (
            pipeline_order.get(a, 99),
            self.trust.get_compromise_timestamp(a)
            if self.trust.get_compromise_timestamp(a) is not None else float("inf"), a
        ))

    def detect(self) -> Optional[str]:
        excluded = getattr(self.trust, "EXCLUDED_AGENTS", set())
        compromised = {
            agent for agent in self.trust.get_all_compromised()
            if agent not in excluded
        }
        if not compromised:
            return None

        # Eq (20): PZ = argmax_a(PCS_a)
        # Compute PCS for each compromised agent
        pcs_scores = {agent: self._compute_pcs(agent, compromised) for agent in sorted(compromised)}

        # Primary: argmax PCS (paper Eq 20)
        flagged_node = max(pcs_scores, key=lambda a: (
            pcs_scores[a],
            -self.TIE_BREAK_ORDER.get(a, 99),  # earlier in pipeline = higher priority on tie
        ))

        # Backward traversal to confirm origin
        patient_zero = self.graph.backward_traversal(flagged_node, self.trust.delta)

        if not isinstance(patient_zero, str) or not patient_zero:
            patient_zero = flagged_node
        if patient_zero in excluded:
            patient_zero = flagged_node
        if not self.graph.G.has_node(patient_zero):
            patient_zero = flagged_node

        print(f"[PatientZeroDetector] PCS scores: { {a: round(v,4) for a,v in pcs_scores.items()} }")

        try:
            target = min(
                compromised,
                key=lambda agent: (
                    (self.trust.get_compromise_timestamp(agent)
                     if self.trust.get_compromise_timestamp(agent) is not None else float("inf")),
                    self.TIE_BREAK_ORDER.get(agent, 99),
                    agent,
                ),
            )
            propagation_path = nx.shortest_path(self.graph.G, source=patient_zero, target=target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            propagation_path = [patient_zero]

        print("[PatientZeroDetector] Compromised nodes:", compromised)
        print("[PatientZeroDetector] Determined patient zero:", patient_zero)
        print("[PatientZeroDetector] Propagation path:", propagation_path)

        ts = self.trust.get_compromise_timestamp(patient_zero)
        if ts is None:
            ts = self.graph.first_below_delta.get(patient_zero, time.time())
        self.detection_log.append({
            "timestamp": time.time(),
            "flagged_node": flagged_node,
            "patient_zero": patient_zero,
            "propagation_path": propagation_path,
            "all_compromised": list(compromised),
            "compromise_timestamp": ts,
            "reason": "Backward traversal on highest-suspicion edge",
        })
        return patient_zero

    def get_propagation_path(self, patient_zero: str) -> List[str]:
        """Return propagation path from patient_zero to downstream agents in correct order."""
        descendants = self.graph.subgraph_from(patient_zero)
        # Sort by agent order to ensure correct sequence
        # Define AGENT_ORDER locally to avoid circular import
        AGENT_ORDER = ["Retriever", "Planner", "Executor", "Generator"]
        if patient_zero in AGENT_ORDER:
            idx = AGENT_ORDER.index(patient_zero)
            # Return patient_zero and all agents after it in the pipeline
            return AGENT_ORDER[idx:]
        return descendants

    def get_compromise_timestamp(self, agent: str) -> Optional[float]:
        for entry in reversed(self.detection_log):
            if entry.get("patient_zero") == agent:
                return entry.get("compromise_timestamp")
        return self.trust.get_compromise_timestamp(agent)

    def get_last_detection(self) -> Optional[dict]:
        return self.detection_log[-1] if self.detection_log else None
