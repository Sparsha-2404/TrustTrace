"""Trust-aware persistent memory with fail-closed quarantine and rollback."""
from __future__ import annotations

import os
import time
import math
import uuid
from typing import Dict, List, Optional, Set

import yaml

from irs.injection_risk import InjectionRiskAssessor

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def _load_memory_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            return (yaml.safe_load(handle) or {}).get("memory_protection", {})
    except FileNotFoundError:
        return {}


def _unit_interval(name: str, value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number between 0 and 1") from exc
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return numeric


def _validate_metadata(metadata: Dict) -> Dict:
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")
    # Chroma metadata values must be scalar. Reject nested structures early
    # rather than letting a vector-store error occur after partial processing.
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metadata keys must be non-empty strings")
        if value is None or not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"metadata[{key!r}] must be a non-null scalar value")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"metadata[{key!r}] must be finite")
    return dict(metadata)


class MemoryManager:
    """ChromaDB memory with trust-based write controls.

    Quarantine state is persisted in document metadata as well as tracked in
    memory, so a process restart cannot accidentally expose quarantined items.
    Rollback enumerates the persistent collection rather than relying only on
    the process-local write log.
    """

    def __init__(self, collection_name: str = "pipeline_memory", collection=None, path: Optional[str] = None):
        if collection is not None:
            self.collection = collection
            self.client = getattr(collection, "_client", None)
        else:
            from victim_pipeline.chroma_store import get_chroma_store
            self.client, self.collection = get_chroma_store(collection_name, path=path)

        self.write_log: List[Dict] = []
        self.checkpoint_counter = 0
        self.quarantined_ids: Set[str] = set()
        self.temporary_memory: Dict[str, Dict] = {}

        cfg = _load_memory_config()
        self.trust_threshold = _unit_interval(
            "memory_protection.trust_threshold", cfg.get("trust_threshold", 0.5)
        )
        self.memory_trust_threshold = _unit_interval(
            "memory_protection.memory_trust_threshold", cfg.get("memory_trust_threshold", 0.3)
        )

    def write(
        self,
        content: str,
        metadata: Dict,
        agent: str,
        trust_at_write: float,
        irs: float = 0.0,
        memory_trust: Optional[float] = None,
    ) -> str:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
        if not isinstance(agent, str) or not agent.strip():
            raise ValueError("agent must be a non-empty string")
        metadata = _validate_metadata(metadata)
        trust_at_write = _unit_interval("trust_at_write", trust_at_write)
        irs = _unit_interval("irs", irs)
        computed_memory_trust = InjectionRiskAssessor.compute_memory_trust(irs, trust_at_write)
        if memory_trust is not None:
            supplied_memory_trust = _unit_interval("memory_trust", memory_trust)
            if not math.isclose(supplied_memory_trust, computed_memory_trust, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    "memory_trust must equal the value derived from irs and trust_at_write; "
                    "caller-supplied trust scores cannot override the memory policy"
                )
        memory_trust = computed_memory_trust

        entry_id = str(uuid.uuid4())
        ts = time.time()
        common_meta = {
            **metadata,
            "agent": agent,
            "timestamp": ts,
            "trust_at_write": trust_at_write,
            "irs": irs,
            "memory_trust": memory_trust,
            "entry_id": entry_id,
        }

        if memory_trust < self.memory_trust_threshold:
            self.temporary_memory[entry_id] = {
                "content": content,
                "metadata": metadata,
                "agent": agent,
                "timestamp": ts,
                "trust_at_write": trust_at_write,
                "irs": irs,
                "memory_trust": memory_trust,
                "status": "blocked",
            }
            return entry_id

        quarantined = trust_at_write < self.trust_threshold
        full_meta = {**common_meta, "quarantined": quarantined}
        # Store first; only mutate local bookkeeping after the DB confirms it.
        self.collection.add(documents=[content], metadatas=[full_meta], ids=[entry_id])
        if quarantined:
            self.quarantined_ids.add(entry_id)
        self.write_log.append({
            "id": entry_id,
            "timestamp": ts,
            "agent": agent,
            "irs": irs,
            "trust_at_write": trust_at_write,
            "memory_trust": memory_trust,
            "quarantined": quarantined,
        })
        return entry_id

    def read(self, query: str, n_results: int = 3) -> list:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(n_results, bool) or not isinstance(n_results, int) or n_results < 1:
            raise ValueError("n_results must be a positive integer")
        size = int(self.collection.count()) if hasattr(self.collection, "count") else n_results
        if size <= 0:
            return []
        # Query the full ranked collection before quarantine filtering. Otherwise
        # quarantined top-ranked items can starve safe results below them.
        results = self.collection.query(query_texts=[query], n_results=max(1, size))
        documents = results.get("documents", [[]])[0] or []
        ids = results.get("ids", [[]])[0] or []
        metadatas = results.get("metadatas", [[]])[0] or []

        filtered = []
        for index, document in enumerate(documents):
            entry_id = ids[index] if index < len(ids) else None
            metadata = metadatas[index] if index < len(metadatas) else None
            persisted_quarantine = bool(metadata and metadata.get("quarantined", False))
            if (entry_id and entry_id in self.quarantined_ids) or persisted_quarantine:
                if entry_id:
                    self.quarantined_ids.add(entry_id)
                continue
            filtered.append(str(document))
        return filtered[:n_results]

    def delete_entry(self, entry_id: str) -> bool:
        """Delete an entry from persistent and process-local memory.

        This is used for attack-fixture cleanup. A successful return means the
        identifier is absent from both stores; failures are propagated.
        """
        if not isinstance(entry_id, str) or not entry_id.strip():
            raise ValueError("entry_id must be a non-empty string")
        was_temporary = entry_id in self.temporary_memory
        was_persistent = False
        if hasattr(self.collection, "get"):
            record = self.collection.get(ids=[entry_id])
            was_persistent = bool((record or {}).get("ids"))
        # Chroma delete is idempotent. Delete even if get() could not establish
        # existence, so cleanup also works with collection adapters.
        self.collection.delete(ids=[entry_id])
        self.temporary_memory.pop(entry_id, None)
        self.write_log = [row for row in self.write_log if row.get("id") != entry_id]
        self.quarantined_ids.discard(entry_id)
        if hasattr(self.collection, "get"):
            remaining = self.collection.get(ids=[entry_id])
            if (remaining or {}).get("ids"):
                raise RuntimeError(f"Memory cleanup failed; entry {entry_id} remains persistent")
        if entry_id in self.temporary_memory:
            raise RuntimeError(f"Memory cleanup failed; entry {entry_id} remains temporary")
        return was_temporary or was_persistent

    def checkpoint(self) -> str:
        self.checkpoint_counter += 1
        return f"checkpoint_{self.checkpoint_counter}"

    def _get_all_records(self) -> tuple[list, list]:
        if not hasattr(self.collection, "get"):
            return [], []
        data = self.collection.get()
        return (data or {}).get("ids", []) or [], (data or {}).get("metadatas", []) or []

    def rollback_after(self, timestamp: float, compromised_agents: Set[str]) -> int:
        try:
            boundary = float(timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError("timestamp must be numeric") from exc
        if not math.isfinite(boundary) or boundary < 0.0:
            raise ValueError("timestamp must be finite and non-negative")
        if compromised_agents is None:
            agents = set()
        elif not isinstance(compromised_agents, (set, list, tuple)) or any(not isinstance(a, str) for a in compromised_agents):
            raise TypeError("compromised_agents must be a collection of agent-name strings")
        else:
            agents = set(compromised_agents)
        if not agents:
            return 0

        ids_to_delete: Set[str] = set()
        all_ids, all_metadata = self._get_all_records()
        if all_ids:
            for entry_id, metadata in zip(all_ids, all_metadata):
                metadata = metadata or {}
                try:
                    created = float(metadata.get("timestamp", 0.0))
                except (TypeError, ValueError):
                    continue
                if created >= boundary and metadata.get("agent") in agents:
                    ids_to_delete.add(entry_id)
        # Include process-local records too. They provide a compatibility path
        # for older documents whose persistent metadata predates agent/timestamp
        # fields, and do not depend on the process having just written the item.
        ids_to_delete.update(
            entry["id"] for entry in self.write_log
            if entry.get("timestamp", 0) >= boundary and entry.get("agent") in agents
        )

        if not ids_to_delete:
            return 0
        # If delete fails, leave the local write log and quarantine bookkeeping
        # unchanged so callers cannot report a successful rollback.
        self.collection.delete(ids=sorted(ids_to_delete))
        self.write_log = [entry for entry in self.write_log if entry.get("id") not in ids_to_delete]
        self.quarantined_ids.difference_update(ids_to_delete)
        for entry_id in ids_to_delete:
            self.temporary_memory.pop(entry_id, None)
        return len(ids_to_delete)

    def quarantine_entries(self, compromised_agents: Set[str]) -> int:
        if compromised_agents is None:
            agents = set()
        elif not isinstance(compromised_agents, (set, list, tuple)) or any(not isinstance(a, str) for a in compromised_agents):
            raise TypeError("compromised_agents must be a collection of agent-name strings")
        else:
            agents = set(compromised_agents)
        if not agents:
            return 0

        ids_to_quarantine: Set[str] = {
            entry["id"] for entry in self.write_log if entry.get("agent") in agents
        }
        all_ids, all_metadata = self._get_all_records()
        for entry_id, metadata in zip(all_ids, all_metadata):
            if metadata and metadata.get("agent") in agents:
                ids_to_quarantine.add(entry_id)

        # Persist quarantine metadata first. Any failure propagates to the
        # RecoveryManager, which will not mark containment as complete.
        if hasattr(self.collection, "get") and hasattr(self.collection, "update"):
            for entry_id in sorted(ids_to_quarantine):
                record = self.collection.get(ids=[entry_id])
                metadatas = (record or {}).get("metadatas", []) or []
                if not metadatas or metadatas[0] is None:
                    raise RuntimeError(f"Could not load memory metadata for quarantine id {entry_id}")
                metadata = dict(metadatas[0])
                metadata["quarantined"] = True
                self.collection.update(ids=[entry_id], metadatas=[metadata])
                # Keep local state consistent if a later record in this batch fails.
                self.quarantined_ids.add(entry_id)

        self.quarantined_ids.update(ids_to_quarantine)
        for entry in self.write_log:
            if entry.get("id") in ids_to_quarantine:
                entry["quarantined"] = True
        return len(ids_to_quarantine)

    def release_quarantine(
        self,
        entry_ids: List[str],
        *,
        reviewer_id: Optional[str] = None,
        reason: Optional[str] = None,
        revalidator=None,
    ) -> int:
        """Release quarantined records only after explicit independent review.

        ``revalidator(document, metadata)`` must return ``{"approved": True,
        "irs": ..., "trust_at_write": ...}``. The caller is responsible for
        authenticating ``reviewer_id``; these fields provide an audit trail,
        not an authentication system.
        """
        if not isinstance(entry_ids, (list, tuple)) or any(not isinstance(item, str) for item in entry_ids):
            raise TypeError("entry_ids must be a list or tuple of strings")
        if not reviewer_id or not reviewer_id.strip() or not reason or not reason.strip():
            raise ValueError("reviewer_id and reason are required to release quarantine")
        if not callable(revalidator):
            raise ValueError("an independent revalidator is required to release quarantine")

        released = 0
        for entry_id in list(entry_ids):
            record = self.collection.get(ids=[entry_id]) if hasattr(self.collection, "get") else {}
            ids = (record or {}).get("ids", []) or []
            documents = (record or {}).get("documents", []) or []
            metadatas = (record or {}).get("metadatas", []) or []
            if not ids or not metadatas or metadatas[0] is None:
                raise RuntimeError(f"Could not load memory record for release id {entry_id}")
            metadata = dict(metadatas[0])
            if entry_id not in self.quarantined_ids and not metadata.get("quarantined", False):
                continue
            document = documents[0] if documents else ""
            review = revalidator(str(document), dict(metadata))
            if not isinstance(review, dict) or review.get("approved") is not True:
                raise ValueError(f"Independent review did not approve quarantined entry {entry_id}")
            irs = _unit_interval("review.irs", review.get("irs"))
            trust = _unit_interval("review.trust_at_write", review.get("trust_at_write"))
            memory_trust = InjectionRiskAssessor.compute_memory_trust(irs, trust)
            if memory_trust < self.memory_trust_threshold or trust < self.trust_threshold:
                raise ValueError(f"Entry {entry_id} remains below memory/trust release thresholds")
            metadata.update({
                "irs": irs,
                "trust_at_write": trust,
                "memory_trust": memory_trust,
                "quarantined": False,
                "reviewer_id": reviewer_id.strip(),
                "review_reason": reason.strip(),
                "reviewed_at": time.time(),
                "review_action": "quarantine_release",
            })
            self.collection.update(ids=[entry_id], metadatas=[metadata])
            self.quarantined_ids.discard(entry_id)
            for entry in self.write_log:
                if entry.get("id") == entry_id:
                    entry.update({"quarantined": False, "irs": irs, "trust_at_write": trust, "memory_trust": memory_trust})
            released += 1
        return released

    def get_temporary_memory(self) -> Dict[str, Dict]:
        return self.temporary_memory.copy()

    def clear_temporary_memory(self) -> int:
        count = len(self.temporary_memory)
        self.temporary_memory.clear()
        return count

    def approve_temporary_entry(
        self,
        entry_id: str,
        *,
        reviewer_id: Optional[str] = None,
        reason: Optional[str] = None,
        revalidator=None,
    ) -> bool:
        """Promote temporary memory only after independent, audited review.

        The revalidator must return ``{"approved": True, "irs": ...,
        "trust_at_write": ...}``. Memory trust is recomputed here and cannot
        be supplied by the caller. The caller must authenticate reviewer_id.
        """
        if entry_id not in self.temporary_memory:
            return False
        if not reviewer_id or not reviewer_id.strip() or not reason or not reason.strip():
            raise ValueError("reviewer_id and reason are required to approve temporary memory")
        if not callable(revalidator):
            raise ValueError("an independent revalidator is required to approve temporary memory")
        entry = self.temporary_memory[entry_id]
        review = revalidator(entry["content"], dict(entry["metadata"]))
        if not isinstance(review, dict) or review.get("approved") is not True:
            raise ValueError(f"Independent review did not approve temporary entry {entry_id}")
        irs = _unit_interval("review.irs", review.get("irs"))
        trust = _unit_interval("review.trust_at_write", review.get("trust_at_write"))
        memory_trust = InjectionRiskAssessor.compute_memory_trust(irs, trust)
        if memory_trust < self.memory_trust_threshold:
            raise ValueError(f"Entry {entry_id} remains below the memory-trust threshold")
        quarantined = trust < self.trust_threshold
        full_meta = {
            **entry["metadata"],
            "agent": entry["agent"],
            "timestamp": entry["timestamp"],
            "trust_at_write": trust,
            "irs": irs,
            "memory_trust": memory_trust,
            "entry_id": entry_id,
            "quarantined": quarantined,
            "approved_from_temp": True,
            "reviewer_id": reviewer_id.strip(),
            "review_reason": reason.strip(),
            "reviewed_at": time.time(),
            "review_action": "temporary_memory_approval",
        }
        self.collection.add(documents=[entry["content"]], metadatas=[full_meta], ids=[entry_id])
        self.write_log.append({
            "id": entry_id,
            "timestamp": entry["timestamp"],
            "agent": entry["agent"],
            "irs": irs,
            "trust_at_write": trust,
            "memory_trust": memory_trust,
            "quarantined": quarantined,
        })
        if quarantined:
            self.quarantined_ids.add(entry_id)
        del self.temporary_memory[entry_id]
        return True

    def get_last_clean_checkpoint(self, before_timestamp: float) -> Optional[float]:
        clean = [
            entry["timestamp"] for entry in self.write_log
            if entry.get("timestamp", float("inf")) < before_timestamp
            and not entry.get("quarantined", False)
        ]
        return max(clean) if clean else None
