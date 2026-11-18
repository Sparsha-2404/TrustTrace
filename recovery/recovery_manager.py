import json
import os
import re
import time
from typing import List, Set

from detector.patient_zero import PatientZeroDetector
from display import terminal_display as display
from memory.memory_manager import MemoryManager
from trust.trust_engine import TrustEngine


REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "logs", "reports")


class RecoveryManager:
    """Quarantine → rollback → restore with real state changes."""

    def __init__(
        self,
        memory: MemoryManager,
        trust: TrustEngine,
        detector: PatientZeroDetector,
    ):
        self.memory = memory
        self.trust = trust
        self.detector = detector
        self.quarantined: Set[str] = set()
        self.held_tasks: List[dict] = []
        os.makedirs(REPORT_DIR, exist_ok=True)

    def quarantine(self, compromised_agents: Set[str]) -> None:
        agents = set(compromised_agents)
        if not agents:
            return
        # Do not report containment as successful if the memory quarantine
        # operation failed. Propagate the error so the caller can fail closed.
        self.memory.quarantine_entries(agents)
        self.quarantined.update(agents)

    def is_quarantined(self, agent: str) -> bool:
        return agent in self.quarantined

    def hold_task(self, task: dict) -> None:
        self.held_tasks.append(task)

    def rollback_memory(self, compromise_timestamp: float) -> int:
        return self.memory.rollback_after(compromise_timestamp, self.quarantined)

    def restore(self) -> tuple:
        restored_agents = list(self.quarantined)
        for agent in restored_agents:
            restored_trust = self.trust.recover_agent(agent)
            ckpt = self.memory.checkpoint()
            display.show_recovery(ckpt, agent, restored_trust)

        self.quarantined.clear()
        # Held tasks are intentionally NOT replayed automatically. Replaying
        # them through the bare victim pipeline would bypass TrustTrace's live
        # validators, trust gates, and audit callbacks. Keep them queued for an
        # explicit, separately revalidated release workflow.
        held_count = len(self.held_tasks)
        replayed_count = 0
        return held_count, replayed_count

    def release_held_tasks_for_review(self) -> list:
        """Return and clear held tasks for an external human-reviewed workflow."""
        tasks = list(self.held_tasks)
        self.held_tasks.clear()
        return tasks

    def execute_recovery(
        self,
        patient_zero: str,
        attack_source: str,
        propagation_path: list,
        compromise_timestamp: float,
        run_id: str,
        memory_entry_ids: list | None = None,
    ) -> float:
        start = time.perf_counter()
        compromised = self.trust.get_all_compromised()
        self.quarantine(compromised)
        rolled_back = self.rollback_memory(compromise_timestamp)
        # Explicit attack fixtures are known poisoned records. Roll them back
        # while still present; caller cleanup is an idempotent safety net.
        for entry_id in memory_entry_ids or []:
            if not self.memory.delete_entry(entry_id):
                raise RuntimeError(f"Recovery could not confirm rollback of poisoned entry {entry_id}")
            rolled_back += 1
        held_count, replayed_count = self.restore()
        if not self.recovery_complete(self.trust) or self.quarantined:
            raise RuntimeError("Recovery did not restore all agent trust scores and clear quarantine")
        recovery_time = time.perf_counter() - start
        self._generate_report(
            run_id=run_id,
            attack_source=attack_source,
            patient_zero=patient_zero,
            propagation_path=propagation_path,
            compromised_agents=list(compromised),
            held_tasks=held_count,
            replayed_tasks=replayed_count,
            rolled_back_count=rolled_back,
            recovery_time_s=recovery_time,
            compromise_timestamp=compromise_timestamp,
        )
        return recovery_time

    def recovery_complete(self, trust_engine: TrustEngine) -> bool:
        return all(
            trust_engine.trust_scores.get(agent, 1.0) > trust_engine.delta
            for agent in trust_engine.trust_scores
        )

    def _generate_report(self, **kwargs) -> None:
        run_id = str(kwargs.get("run_id", ""))
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
            raise ValueError("run_id must contain only letters, digits, underscores or hyphens")
        os.makedirs(REPORT_DIR, exist_ok=True)
        report_path = os.path.join(REPORT_DIR, f"incident_{run_id}.json")
        temporary_path = report_path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(kwargs, handle, indent=2, default=str)
        os.replace(temporary_path, report_path)
