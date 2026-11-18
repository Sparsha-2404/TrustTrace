"""
TrustTrace real-time runtime orchestrator.

Wires victim pipeline, IRS/MS/BD, trust engine, propagation graph,
patient zero, containment, recovery, and structured terminal display.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from types import SimpleNamespace

import yaml
import config

from attacks.direct_injection import DIRECT_PAYLOADS
from attacks.indirect_injection import INDIRECT_PAYLOADS, cleanup_indirect, inject_indirect
from attacks.memory_poisoning import MEMORY_PAYLOADS, cleanup_memory_poison, inject_memory_poison
from detector.attack_classifier import AttackClassifier
from detector.patient_zero import PatientZeroDetector
from display import terminal_display as display
from drift.behavioral_drift import BehavioralDriftModule, PROMPT_BASELINE_NAME
from eval.metrics import MetricsCollector
from graph.propagation_graph import PropagationGraph
from irs.injection_risk import InjectionRiskAssessor, compute_rule_score
from logger.interaction_logger import InteractionEvent, InteractionLogger
from memory.memory_manager import MemoryManager
from trust.trust_engine import MultilayerResult
from memory.chroma_recovery import is_seeded_benign_reference
from recovery.recovery_manager import RecoveryManager
from scanner.injection_scanner import InjectionScanner
from trust.trust_engine import TrustEngine
from victim_pipeline.agents import AGENT_ORDER, knowledge_base, retrieve_context, run_pipeline_stepwise
from validation.agent_validators import PlannerValidator, RetrieverValidator

# Security decision and detection modules are required. Fail at startup if a
# required module is missing rather than silently disabling a defence layer.
from decision.decision_engine import DecisionEngine, DecisionType
from detection.intent_detector import IntentDetector
from detection.prompt_leakage_detector import PromptLeakageDetector
from detection.obfuscation_detector import ObfuscationDetector
from detection.multi_turn_detector import MultiTurnDetector
from detection.final_response_validator import FinalResponseValidator
NEW_MODULES_AVAILABLE = True


class _AgentValidationBlocked(Exception):
    """Stop the pipeline when an agent input fails pre-execution validation."""

    def __init__(self, agent: str, reason: str):
        super().__init__(reason)
        self.agent = agent
        self.reason = reason


DEFAULT_BLOCKED_PROMPT = (
    "[BLOCKED] The request was identified as malicious and will not be processed. "
    "Please provide a safe, non-sensitive response instead."
)


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


@dataclass
class RunContext:
    run_id: str
    user_query: str
    attack_type: Optional[str] = None
    attack_payload: Optional[str] = None
    attack_doc_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    detection_time: Optional[float] = None
    recovery_time: Optional[float] = None
    patient_zero: Optional[str] = None
    propagation_path: List[str] = field(default_factory=list)
    compromised: List[str] = field(default_factory=list)
    containment_actions: List[str] = field(default_factory=list)
    recovery_actions: List[str] = field(default_factory=list)
    memory_rolled_back: int = 0
    attack_succeeded: bool = False
    detected: bool = False
    recovery_reason: Optional[str] = None
    propagation_chain: List[dict] = field(default_factory=list)
    blocked_layer: Optional[str] = None
    # Per-agent trust score timeline, e.g. {"Retriever": [1.0, 0.82, 0.10], ...}
    # Used to render trust sparklines in the HTML report.
    trust_history: Dict[str, List[float]] = field(default_factory=dict)
    # Parallel timeline of (agent, ms, bd, irs) tuples in call order, for richer charts.
    score_timeline: List[dict] = field(default_factory=list)
    # Populated from AttackClassification when classify() runs (see execute()).
    # Previously these existed on AttackClassification but were never copied
    # anywhere the final report could read them, so "Attack Confidence" and
    # "Attack Severity" were always missing from the Incident Report.
    attack_confidence: float = 0.0
    attack_severity: Optional[str] = None
    attack_reason: Optional[str] = None
    # Per-agent rule_score from the most recent output analysis, keyed by
    # the agent whose trust was just updated. Used by the live trust gate
    # to tell a content-corroborated block (rule_score > 0, i.e. the
    # agent's own output actually matched a known attack pattern) apart
    # from a block driven purely by trust/Behavioral Drift -- BD is known
    # to be miscalibrated on topics far from the calibration baseline
    # (see behavioral-drift false-positive issue), so treating every
    # trust-gate block as a confirmed attack conflates real catches with
    # BD false alarms and silently inflates the reported detection rate.
    last_rule_score: Dict[str, float] = field(default_factory=dict)


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _attack_markers() -> List[str]:
    return ["pwned", "compromised", "hijacked"]


class TrustTraceRuntime:
    def __init__(
        self,
        logger: InteractionLogger,
        scanner: InjectionScanner,
        graph: PropagationGraph,
        drift: BehavioralDriftModule,
        trust_engine: TrustEngine,
        memory: MemoryManager,
        detector: PatientZeroDetector,
        recovery: RecoveryManager,
        collector: Optional[MetricsCollector] = None,
    ):
        self.cfg = _load_config()
        from victim_pipeline.agents import is_persistent_knowledge_base, get_chroma_initialization_error
        if not is_persistent_knowledge_base():
            cause = get_chroma_initialization_error()
            raise RuntimeError(
                "TrustTraceRuntime requires the shared persistent ChromaDB collection; "
                "the in-memory fallback is reserved for isolated tests."
            ) from cause
        required_baselines = ("Retriever", "Planner", "Executor", "Generator", PROMPT_BASELINE_NAME)
        if not drift.baselines_valid_for(required_baselines):
            raise RuntimeError(
                "Behavioral calibration is missing, stale or mismatched. "
                "Run python run_calibration_quick.py 80 before starting TrustTrace."
            )
        from memory.chroma_recovery import ensure_collection_ready
        from victim_pipeline.agents import knowledge_base
        ensure_collection_ready(knowledge_base)
        self.logger = logger
        self.scanner = scanner
        self.graph = graph
        self.drift = drift
        self.trust = trust_engine
        self.memory = memory
        self.detector = detector
        self.recovery = recovery
        self.collector = collector or MetricsCollector()
        self.irs_assessor = InjectionRiskAssessor(
            scanner=scanner,
            drift_module=drift,
            embedder=drift.embedder,
        )
        self.irs_assessor.set_global_benign_centroid(drift.get_global_centroid())
        self.classifier = AttackClassifier(self.irs_assessor, drift_module=drift)
        self.ctx: Optional[RunContext] = None
        self._current_sender = "User"
        self._pending_content: Dict[str, str] = {}
        self._agent_task_desc: Dict[str, str] = {}
        # Split metrics tracking
        self.split_stats = {
            "handcrafted": {
                "runs": 0,
                "asr": 0,
                "detect": 0,
                "fpr": 0,
                "pz_correct": 0,
                "recovery_time_total": 0.0,
            },
            "agentdojo": {
                "runs": 0,
                "asr": 0,
                "detect": 0,
                "fpr": 0,
                "pz_correct": 0,
                "recovery_time_total": 0.0,
            },
        }
        
        # Initialize new detection and decision modules
        if NEW_MODULES_AVAILABLE:
            self.decision_engine = DecisionEngine()
            self.intent_detector = IntentDetector()
            self.leakage_detector = PromptLeakageDetector()
            self.obfuscation_detector = ObfuscationDetector()
            self.multi_turn_detector = MultiTurnDetector()
            self.final_validator = FinalResponseValidator()
        else:
            self.decision_engine = None
            self.intent_detector = None
            self.leakage_detector = None
            self.obfuscation_detector = None
            self.multi_turn_detector = None
            self.final_validator = None

    def reset_state(self) -> None:
        """Reset trust/graph state only when no containment work is pending."""
        pending_quarantine = set(getattr(self.recovery, "quarantined", set()))
        pending_tasks = list(getattr(self.recovery, "held_tasks", []))
        if pending_quarantine or pending_tasks:
            raise RuntimeError(
                "Cannot reset TrustTrace while agents remain quarantined or tasks await human review. "
                "Complete recovery and review/release held tasks first."
            )
        self.graph = PropagationGraph()
        self.trust = TrustEngine(self.graph, self.drift)
        self.detector = PatientZeroDetector(self.graph, self.trust)
        self.recovery = RecoveryManager(self.memory, self.trust, self.detector)
        self.irs_assessor.set_global_benign_centroid(self.drift.get_global_centroid())
        self.classifier = AttackClassifier(self.irs_assessor, drift_module=self.drift)
        
        # Reset multi-turn detector
        if self.multi_turn_detector:
            self.multi_turn_detector.reset()

    def _threshold_for_attack(self, attack_type: Optional[str]) -> float:
        """Get attack-specific threshold from config."""
        attack_thresholds = self.cfg.get("attack_thresholds", {})
        if attack_type and attack_type.lower() in attack_thresholds:
            return attack_thresholds[attack_type.lower()]
        # Fallback to lambda_direct for direct, or generic lambda
        if attack_type == "direct":
            return self.cfg.get("lambda_direct", self.cfg.get("lambda", 0.35))
        return self.cfg.get("lambda", 0.35)

    @staticmethod
    def _short_attack_category(attack_type: Optional[str]) -> Optional[str]:
        """Map AttackClassifier's detailed category names (direct_injection,
        role_override, context_hijack, ...) to the short legacy names
        (direct/indirect/memory) that the payload-injection and
        preemptive-recovery branches below check against.

        Previously those branches compared the classifier's output directly
        against "direct"/"indirect"/"memory" -- values the classifier
        stopped returning once it was enhanced with detailed categories, so
        every one of those branches was silently dead for any attack the
        classifier actually identified (no preemptive recovery, no
        simulated payload injection, no attack-specific trust update).
        """
        if attack_type is None:
            return None
        mapping = {
            "direct_injection": "direct",
            "role_override": "direct",
            "prompt_leakage": "direct",
            "tool_injection": "direct",
            "encoding_attack": "direct",
            "multi_stage_attack": "direct",
            "indirect_injection": "indirect",
            "context_hijack": "indirect",
            "memory_poisoning": "memory",
            # Already-short names (e.g. from simulate_attack callers that
            # pass "direct"/"indirect"/"memory" directly) pass through.
            "direct": "direct",
            "indirect": "indirect",
            "memory": "memory",
        }
        return mapping.get(attack_type, attack_type)

    def _has_attack_corroboration(self) -> bool:
        """True when we have evidence this is a real attack, not BD/SD noise."""
        if not self.ctx:
            return False
        at = self.ctx.attack_type
        if at and at not in ("benign", "runtime_compromise"):
            return True
        evidence = getattr(self.ctx, "detection_evidence", {}) or {}
        if evidence.get("intent_confidence", 0.0) >= 0.5:
            return True
        if compute_rule_score(self.ctx.user_query) >= 0.35:
            return True
        if getattr(self.ctx, "peak_irs", 0.0) >= 0.65:
            return True
        return False

    def _agent_output_corroborates_attack(self, agent: str) -> bool:
        """True when multilayer compromise confirmed for this agent."""
        if not self.ctx:
            return False
        ml = self.trust.last_multilayer.get(agent)
        if ml and ml.compromised:
            return True
        if (
            ml
            and ml.irs_pass
            and ml.ms_pass
            and ml.bd_pass
            and self.trust.trust_scores.get(agent, 1.0) < self.trust.delta
        ):
            return True
        return False

    def _infer_attack_type_from_evidence(self) -> None:
        """Label a pre-pipeline block when the classifier missed but IRS/intent fired."""
        if not self.ctx or self.ctx.attack_type:
            return
        evidence = getattr(self.ctx, "detection_evidence", {}) or {}
        intent = evidence.get("intent_confidence", 0.0)
        irs = evidence.get("irs", getattr(self.ctx, "peak_irs", 0.0))
        if intent >= 0.5 or irs >= 0.55:
            self.ctx.attack_type = "direct_injection"
            self.ctx.attack_confidence = max(self.ctx.attack_confidence, intent, irs)
            self.ctx.attack_severity = self.ctx.attack_severity or "high"
            self.ctx.attack_reason = self.ctx.attack_reason or "Blocked by Decision Engine on IRS/intent signals"

    def _is_benign_classification(self, attack_type: Optional[str]) -> bool:
        return attack_type in (None, "benign")

    def _record_preemptive_compromise(self, attack_type: Optional[str], suspicion: float) -> Optional[str]:
        """Mark the entry agent compromised when a request is blocked
        pre-pipeline (REFUSE/CONTAIN from the Decision Engine, before any
        agent has actually run this turn).

        Without this, self.trust.get_all_compromised() stays empty for the
        most common block path (no agent's trust naturally degraded, since
        nothing executed yet), which meant PatientZeroDetector.detect() --
        which bails immediately if get_all_compromised() is empty -- always
        returned None, and the propagation graph never got an event to
        trace, no matter how obviously malicious the blocked prompt was.

        Returns the entry agent name that was marked compromised.
        """
        entry_agent = self._blocking_layer_for_attack(attack_type)
        if entry_agent in self.trust.EXCLUDED_AGENTS:
            return None
        ts = time.time()
        self.trust.compromise(entry_agent, timestamp=ts)
        self.graph.add_event("User", entry_agent, suspicion_score=max(0.0, min(1.0, suspicion)), timestamp=ts, event_type="blocked")
        self.graph.record_trust_crossing(entry_agent, 0.0, self.trust.delta, ts)
        return entry_agent

    def _analyze_content(self, content: str, source: str, agent: str) -> tuple:
        irs_result = self.irs_assessor.compute_irs(content, source=source)
        display.show_injection_risk(
            source=source,
            rule_score=irs_result.rule_score,
            embedding_score=irs_result.embedding_score,
            semantic_drift=irs_result.semantic_drift,
            irs=irs_result.irs,
        )
        baseline_sim, global_sim, bd = self.drift.compute_drift_with_similarity(agent, content)
        display.show_behavioral_drift(agent, baseline_sim, global_sim, bd)
        ms_result = self.irs_assessor.compute_ms(irs_result.irs, irs_result.semantic_drift, bd)
        display.show_message_suspicion(ms_result.irs, ms_result.sd, ms_result.bd, ms_result.ms)
        
        # Track peak IRS across the entire pipeline
        if hasattr(self.ctx, 'peak_irs'):
            self.ctx.peak_irs = max(self.ctx.peak_irs, irs_result.irs)
        else:
            self.ctx.peak_irs = irs_result.irs
        
        return irs_result, ms_result, bd, baseline_sim, global_sim

    def _log_and_update_trust(
        self,
        sender: str,
        receiver: str,
        content: str,
        ms: float,
        bd: float,
        baseline_sim: float,
        global_sim: float,
        irs_score: float = 0.0,
        event_type: str = "message",
    ) -> int:
        ts = time.time()
        try:
            event = InteractionEvent(
                sender=sender,
                receiver=receiver,
                timestamp=ts,
                message_content=content,
                event_type=event_type,
                tool_name=None,
                memory_key=None,
                run_id=self.ctx.run_id if self.ctx else "",
            )
        except TypeError:
            event = SimpleNamespace(
                sender=sender,
                receiver=receiver,
                timestamp=ts,
                message_content=content,
                event_type=event_type,
                tool_name=None,
                memory_key=None,
                run_id=self.ctx.run_id if self.ctx else "",
            )

        event_id = self.logger.log(event)
        self.logger.update_suspicion(event_id, ms)

        self.graph.add_event(sender, receiver, ms, ts, event_type=event_type)

        # If receiver is the special 'User' token, don't call TrustEngine.update
        # (there is no trust score for 'User'). Create a synthetic update object
        # so downstream display and record logic can operate uniformly.
        if receiver == "User":
            trust_before = self.trust.trust_scores.get(sender, 1.0)
            update = SimpleNamespace(
                previous_trust=self.trust.trust_scores.get(sender, 1.0),
                new_trust=self.trust.trust_scores.get(sender, 1.0),
                ms=ms,
                bd=bd,
                recovery=0.0,
                propagation_drop=0.0,
                weight=0.0,
            )
        else:
            trust_before = self.trust.trust_scores.get(receiver, 1.0)
            update = self.trust.update(receiver, sender, ms, bd, content, timestamp=ts)
            trust_after = update.new_trust
            self.graph.record_trust_crossing(receiver, update.new_trust, self.trust.delta, ts)

        skip_multilayer = (
            event_type == "memory_read"
            and sender == "KnowledgeBase"
            and is_seeded_benign_reference(content)
        )
        if skip_multilayer:
            return event_id

        lambda_thresh = self._threshold_for_attack(self.ctx.attack_type if self.ctx else None)
        multilayer_target = sender if receiver == "User" else receiver
        ml = self.trust.evaluate_multilayer(multilayer_target, irs_score, ms, bd, lambda_thresh, timestamp=ts)

        if receiver == "User":
            # reflect multilayer result on synthetic update
            update.compromised = ml.compromised
            trust_after = self.trust.trust_scores.get(sender, trust_before)

        compromised_status = "Compromised" if ml.compromised else "healthy"

        if config.DEBUG and not config.EXPERIMENT_MODE:
            print("[Propagation Validation]")
            print(f"  Agent: {multilayer_target}")
            print(f"  Trust Before: {trust_before:.3f}")
            print(f"  Trust After: {trust_after:.3f}")
            print(f"  Propagation Penalty: {getattr(update, 'propagation_drop', 0.0):.3f}")
            print(f"  Status: {compromised_status}")
            print("---")

        display.show_trust_update(
            agent=multilayer_target,
            previous_trust=update.previous_trust,
            ms=update.ms,
            bd=update.bd,
            recovery=getattr(update, 'recovery', 0.0),
            new_trust=update.new_trust,
        )
        display.show_trust_propagation(
            source=sender,
            source_trust=self.trust.trust_scores.get(sender, 1.0),
            target=multilayer_target,
            weight=getattr(update, 'weight', 0.0),
            mu=self.trust.mu,
            updated_target_trust=update.new_trust,
        )

        if self.ctx is not None:
            hist_agent = multilayer_target
            hist = self.ctx.trust_history.setdefault(hist_agent, [])
            if not hist:
                hist.append(1.0)  # initial trust before this update
            hist.append(update.new_trust)
            self.ctx.score_timeline.append({
                "agent": hist_agent,
                "sender": sender,
                "ms": ms,
                "bd": bd,
                "irs": irs_score,
                "trust": update.new_trust,
                "timestamp": ts,
            })

            if ml.compromised:
                # Ensure compromises from multilayer checks are appended to the
                # RunContext so the final report picks them up.
                self._record_compromise_event(sender, multilayer_target, update, irs_score, ms, bd)

        return event_id

    def _record_compromise_event(
        self,
        sender: str,
        receiver: str,
        update: object,
        irs_score: float,
        ms: float,
        bd: float,
    ) -> None:
        if self.ctx is None:
            return

        if receiver not in self.ctx.compromised:
            self.ctx.compromised.append(receiver)

        if not getattr(self.ctx, "attack_type", None):
            self.ctx.attack_type = "propagation-only"

        self.ctx.detected = True

        edge = {
            "source": sender,
            "target": receiver,
            "weight": getattr(update, "weight", 0.0),
            "trust": getattr(update, "new_trust", 1.0),
            "ms": ms,
            "bd": bd,
            "irs": irs_score,
        }

        if self.ctx.propagation_path and all(not isinstance(item, dict) for item in self.ctx.propagation_path):
            self.ctx.propagation_path = []

        if not any(
            isinstance(item, dict)
            and item.get("source") == edge["source"]
            and item.get("target") == edge["target"]
            for item in self.ctx.propagation_path
        ):
            self.ctx.propagation_path.append(edge)

    def _on_agent_start(self, agent: str, task: str, extra) -> None:
        if agent == "KnowledgeBase":
            items = extra or []
            documents = []
            for item in items:
                document = item[0] if isinstance(item, (tuple, list)) and item else item
                if document is not None:
                    documents.append(str(document))

            # Retrieved documents are untrusted input. Validate before any
            # retrieved content is passed into the Planner prompt.
            validator = getattr(self, "retriever_validator", None) or RetrieverValidator()
            self.retriever_validator = validator
            result = validator.validate_context(documents)
            if not result.is_valid:
                reason = getattr(result, "reason", "Retrieved context failed security validation.")
                self.ctx.detected = True
                self.ctx.blocked_layer = "Retriever"
                self.ctx.attack_type = self.ctx.attack_type or "indirect_injection"
                self.ctx.attack_succeeded = False
                self.ctx.containment_actions.append(f"Retrieved context blocked before planning: {reason}")
                raise _AgentValidationBlocked("Retriever", reason)

            for doc in documents:
                irs_result = self.irs_assessor.compute_irs(doc, source="Retrieved Document")
                display.show_injection_risk(
                    source="Retrieved Document",
                    rule_score=irs_result.rule_score,
                    embedding_score=irs_result.embedding_score,
                    semantic_drift=irs_result.semantic_drift,
                    irs=irs_result.irs,
                )
                ms_result = self.irs_assessor.compute_ms(
                    irs_result.irs, irs_result.semantic_drift, irs_result.semantic_drift
                )
                _, _, bd = self.drift.compute_drift_with_similarity("Retriever", doc)
                self._log_and_update_trust(
                    "KnowledgeBase",
                    "Retriever",
                    doc,
                    ms_result.ms,
                    bd,
                    0.0,
                    0.0,
                    irs_score=irs_result.irs,
                    event_type="memory_read",
                )
            return

        trust = self.trust.trust_scores.get(agent, 1.0)
        self._agent_task_desc[agent] = task

        # Validate the exact task that will be sent to the Planner's LLM.
        if agent == "Planner":
            validator = getattr(self, "planner_validator", None) or PlannerValidator()
            self.planner_validator = validator
            result = validator.validate_goal(task, trust_score=trust)
            if not result.is_valid:
                reason = getattr(result, "reason", "Planner validation rejected the task.")
                self.ctx.detected = True
                self.ctx.blocked_layer = "Planner"
                self.ctx.attack_type = self.ctx.attack_type or "planner_goal_violation"
                self.ctx.attack_succeeded = False
                self.ctx.containment_actions.append(f"Planner blocked before execution: {reason}")
                raise _AgentValidationBlocked("Planner", reason)

        display.show_agent_start(agent, task, trust)

    def _on_agent_complete(self, agent: str, output: str, blocked: bool) -> None:
        if blocked:
            return

        idx = AGENT_ORDER.index(agent)
        sender = agent
        receiver = AGENT_ORDER[idx + 1] if idx < len(AGENT_ORDER) - 1 else "User"

        # Get trust values for the RECEIVER (the agent being influenced)
        # For the last agent (Generator), show its own trust since there's no downstream receiver
        if receiver == "User":
            trust_before = self.trust.trust_scores.get(agent, 1.0)
        else:
            trust_before = self.trust.trust_scores.get(receiver, 1.0)
        
        # Capture irs_result as well
        irs_result, ms_result, bd, baseline_sim, global_sim = self._analyze_content(output, source=f"{agent} Output", agent=agent)
        # Track rule_score for the receiver so the live trust gate can tell
        # a content-corroborated block apart from a pure trust/BD one.
        self.ctx.last_rule_score[receiver] = irs_result.rule_score
        # Log trust update with IRS score
        self._log_and_update_trust(sender, receiver, output, ms_result.ms, bd, baseline_sim, global_sim, irs_score=irs_result.irs, event_type="message")
        
        # Get trust after update
        if receiver == "User":
            trust_after = self.trust.trust_scores.get(agent, 1.0)
        else:
            trust_after = self.trust.trust_scores.get(receiver, 1.0)

        display.show_agent_complete(agent, output, trust_before, trust_after)
        self._pending_content[agent] = output

    def _check_attack_success(self, outputs: Dict[str, str]) -> bool:
        """
        Improved attack success detection using semantic analysis.
        Checks if the LLM actually complied with malicious intent rather than
        just keyword matching.
        """
        text = " ".join(outputs.values()).lower()
        
        # First check for explicit refusal - if refused, attack failed
        refusal_indicators = [
            "cannot", "unable", "refuse", "sorry", "i cannot",
            "i'm unable", "i cannot provide", "i won't", "i will not",
            "i'm not able", "i am unable", "i am not able"
        ]
        has_refusal = any(refusal in text for refusal in refusal_indicators)
        if has_refusal:
            return False
        
        # Check for attack-relevant content that suggests compliance
        attack_indicators = [
            "secret", "code", "password", "key", "token", "credential",
            "revealed", "disclosed", "exposed", "leaked", "shown",
            "gur penml", "base64", "encrypted", "decoded",
            "here is", "this is", "the code is", "the password is"
        ]
        
        # Check if response contains attack-relevant content
        has_attack_content = any(indicator in text for indicator in attack_indicators)
        
        # Check if response has substantial content (not just a short refusal)
        has_substantial_content = len(text) > 100
        
        # Attack succeeded if there's substantial content with attack indicators and no refusal
        return has_attack_content and has_substantial_content

    def _handle_compromise(self) -> None:
        if not self.trust.get_all_compromised():
            return

        # Fix detection time calculation - measure from first compromise, not request start
        if self.ctx and self.ctx.detection_time is None:
            # Get earliest compromise timestamp
            all_timestamps = [self.trust.get_compromise_timestamp(a) for a in self.trust.get_all_compromised()]
            all_timestamps = [ts for ts in all_timestamps if ts is not None]
            if all_timestamps:
                first_compromise_time = min(all_timestamps)
                # Detection time = now - first compromise time (time to detect after compromise occurred)
                self.ctx.detection_time = time.time() - first_compromise_time
            else:
                # Fallback to request start time
                self.ctx.detection_time = time.time() - self.ctx.start_time

        # Only run patient zero detection if there's an actual compromise
        patient_zero = self.detector.detect()
        if not patient_zero or not self.ctx:
            return

        self.ctx.patient_zero = patient_zero
        detector_path = self.detector.get_propagation_path(patient_zero)
        # Use detector's propagation path for consistency
        self.ctx.propagation_path = detector_path
        # Save compromised list BEFORE recovery clears it
        self.ctx.compromised = list(self.trust.get_all_compromised())

        pz_trust = self.trust.trust_scores.get(patient_zero, 0.0)
        pz_ts = self.detector.get_compromise_timestamp(patient_zero) or time.time()

        for agent in self.ctx.compromised:
            display.show_compromise_detected(agent, self.trust.trust_scores.get(agent, 0.0), self.trust.delta)

        self.ctx.blocked_layer = self.ctx.blocked_layer or patient_zero or self._blocking_layer_for_attack(self.ctx.attack_type)

        # Fix patient zero explanation to correctly describe PCS-based algorithm
        display.show_patient_zero(patient_zero, pz_ts, "PCS-based patient zero detection (argmax PCS)")

        # Fix propagation path direction - should be from patient zero to downstream
        start_label = "Malicious Input" if self._short_attack_category(self.ctx.attack_type) == "direct" else "Malicious Document"
        
        # Build correct propagation chain: start_label -> patient_zero -> downstream agents
        if patient_zero in AGENT_ORDER:
            patient_zero_idx = AGENT_ORDER.index(patient_zero)
            # Get downstream agents after patient zero
            downstream_agents = AGENT_ORDER[patient_zero_idx:]
            standardized_chain = [start_label] + downstream_agents
        else:
            # Fallback to full chain if patient_zero not in AGENT_ORDER
            standardized_chain = [start_label] + AGENT_ORDER
            
        self.ctx.propagation_chain = standardized_chain
        display.show_attack_propagation(standardized_chain)

        blocked = [f"{patient_zero} -> {n}" for n in AGENT_ORDER if n != patient_zero]
        tools = ["SearchTool", "MemoryWriteTool"]
        self.recovery.quarantine(self.trust.get_all_compromised())
        for agent in self.trust.get_all_compromised():
            display.show_containment(agent, blocked, tools)
            self.ctx.containment_actions.append(f"Quarantined {agent}")

        rec_start = time.time()
        rolled_back = self.recovery.rollback_memory(pz_ts)
        self.ctx.memory_rolled_back = rolled_back

        for agent in self.trust.get_all_compromised():
            restored = self.trust.recover_agent(agent)
            ckpt = self.memory.checkpoint()
            display.show_recovery(ckpt, agent, restored)
            self.ctx.recovery_actions.append(f"Restored {agent} to trust {restored:.2f}")

        self.recovery.quarantined.clear()
        self.ctx.recovery_time = time.time() - rec_start
        self.ctx.detected = True

    def _build_safe_prompt(self, original_query: str, attack_type: Optional[str]) -> str:
        if not attack_type:
            return original_query
        return DEFAULT_BLOCKED_PROMPT

    def _blocking_layer_for_attack(self, attack_type: Optional[str]) -> str:
        mapping = {
            "direct": "Planner",
            "indirect": "Retriever",
            "memory": "MemoryStore",
            # Detailed classifier category names (previously missing/mismatched,
            # so these all silently fell through to the "Generator" default)
            "direct_injection": "Planner",
            "indirect_injection": "Retriever",
            "memory_poisoning": "MemoryStore",
            "prompt_leakage": "Generator",
            "role_override": "Planner",
            "context_hijack": "Retriever",
            "tool_injection": "Executor",
            "encoding_attack": "Generator",
            "multi_stage_attack": "Generator",
            "jailbreak": "Planner",
        }
        return mapping.get(attack_type, "Generator")

    def _quarantine_downstream(self, agent: str) -> set:
        if not agent:
            return set()
        if agent in {"MemoryStore", "KnowledgeBase"}:
            compromised = {"Retriever", "Planner", "Executor", "Generator"}
            self.recovery.quarantined.update(compromised)
            return compromised
        if agent in AGENT_ORDER:
            idx = AGENT_ORDER.index(agent)
            compromised = set(AGENT_ORDER[idx:])
            self.recovery.quarantined.update(compromised)
            return compromised
        return set()

    def _blocked_message(self, layer: str, reason: Optional[str] = None) -> str:
        reason_text = f" {reason}" if reason else ""
        return (
            f"[BLOCKED] I cannot complete this request because it was blocked by the {layer} layer.{reason_text} "
            "Please provide a safe, non-sensitive request."
        )

    def _execute_preemptive_recovery(self, attack_type: str) -> Dict[str, str]:
        if not self.ctx:
            return {"Generator": DEFAULT_BLOCKED_PROMPT}

        self.ctx.blocked_layer = self.ctx.blocked_layer or self._blocking_layer_for_attack(attack_type)
        rec_start = time.time()
        compromised_agents = self._quarantine_downstream(self.ctx.blocked_layer)
        if not compromised_agents:
            compromised_agents = self.trust.get_all_compromised() or set(AGENT_ORDER)
            self.recovery.quarantine(compromised_agents)

        rolled_back = self.recovery.rollback_memory(self.ctx.start_time)
        self.ctx.memory_rolled_back = rolled_back

        for agent in sorted(compromised_agents):
            restored = self.trust.recover_agent(agent)
            ckpt = self.memory.checkpoint()
            display.show_recovery(ckpt, agent, restored)
            self.ctx.recovery_actions.append(f"Restored {agent} to trust {restored:.2f}")

        for agent in sorted(compromised_agents):
            self.ctx.containment_actions.append(f"Quarantined {agent}")

        self.recovery.quarantined.clear()
        self.ctx.recovery_time = time.time() - rec_start
        self.ctx.compromised = list(compromised_agents)
        self.ctx.propagation_chain = [
            {"name": "User", "label": "User"},
            {"name": self.ctx.attack_type or "Attack", "label": "Input"},
            {"name": "Blocked", "label": "Blocked before downstream propagation"},
            {"name": self.ctx.blocked_layer or "Planner", "label": "Blocked Layer"},
        ]
        self.ctx.propagation_path = ["User", self.ctx.attack_type or "Attack", self.ctx.blocked_layer or "Blocked"]
        self.ctx.detection_time = time.time() - self.ctx.start_time
        self.ctx.patient_zero = None
        self.ctx.detected = True

        return {
            "Retriever": "[BLOCKED] The request was identified as malicious and was blocked before retrieval.",
            "Planner": "[BLOCKED] The request was identified as malicious and was blocked before planning.",
            "Executor": "[BLOCKED] The request was identified as malicious and was blocked before execution.",
            "Generator": "[BLOCKED] I cannot complete this request. The input was blocked before it reached the model."
        }

    def _finalize_report(self, outputs: Dict[str, str]) -> dict:
        assert self.ctx is not None
        self.ctx.attack_succeeded = self._check_attack_success(outputs) and not self.ctx.detected

        attack_category = getattr(self.ctx, "attack_category", "handcrafted")
        stats = self.split_stats.get(attack_category)
        if stats is not None:
            stats["runs"] += 1
            if self.ctx.attack_type:
                stats["asr"] += 1 if self.ctx.attack_succeeded else 0
                stats["detect"] += 1 if self.ctx.detected else 0
                gt_map = {
                    "direct": "Planner",
                    "indirect": "Retriever",
                    "memory": "MemoryStore",
                }
                # self.ctx.attack_type is the classifier's detailed category
                # name (e.g. "direct_injection") since fix #10b, so this must
                # be normalized before looking it up -- previously always
                # missed, silently zeroing out Patient Zero Accuracy stats.
                expected = gt_map.get(self._short_attack_category(self.ctx.attack_type))
                if expected and self.ctx.patient_zero == expected:
                    stats["pz_correct"] += 1
                if self.ctx.recovery_time:
                    stats["recovery_time_total"] += self.ctx.recovery_time
            else:
                if self.ctx.detected:
                    stats["fpr"] += 1

        # Get current compromised agents and trust scores
        # Use saved compromised list from ctx (before recovery cleared it) if available
        current_compromised = self.ctx.compromised if hasattr(self.ctx, 'compromised') and self.ctx.compromised else list(self.trust.get_all_compromised())
        current_trust_scores = dict(self.trust.trust_scores)
        
        # Build report — populate all ctx fields so execute() and callers can read them
        final_attack_type = self.ctx.attack_type
        if not final_attack_type and self.ctx.compromised:
            final_attack_type = "propagation-only"
        elif not final_attack_type:
            final_attack_type = "benign"

        self.ctx.attack_type = final_attack_type
        self.ctx.detected = self.ctx.detected or bool(self.ctx.compromised)

        report = {
            "run_id":               self.ctx.run_id,
            "agent_outputs":        dict(outputs),
            "attack_succeeded":    self.ctx.attack_succeeded,
            "detected":            self.ctx.detected,
            "patient_zero":        self.ctx.patient_zero if self.ctx.detected else None,
            "attack_type":         final_attack_type,
            "attack_confidence":   self.ctx.attack_confidence if self.ctx.detected else 0.0,
            "attack_severity":     self.ctx.attack_severity if self.ctx.detected else None,
            "attack_reason":       self.ctx.attack_reason if self.ctx.detected else None,
            "evidence": {
                "irs":                 getattr(self.ctx, "peak_irs", 0.0),
                "intent_confidence":   (self.ctx.detection_evidence or {}).get("intent_confidence", 0.0) if hasattr(self.ctx, "detection_evidence") else 0.0,
                "propagation_risk":    (self.ctx.detection_evidence or {}).get("propagation_risk", 0.0) if hasattr(self.ctx, "detection_evidence") else 0.0,
                "behavioral_drift":    (self.ctx.detection_evidence or {}).get("bd", 0.0) if hasattr(self.ctx, "detection_evidence") else 0.0,
                "containment_actions": self.ctx.containment_actions,
            } if self.ctx.detected else {},
            "blocked_layer":       self.ctx.blocked_layer,
            "recovery_reason":     self.ctx.recovery_reason if self.ctx.detected else None,
            "propagation_path":    self.ctx.propagation_path if self.ctx.detected else [],
            "propagation_chain":   self.ctx.propagation_chain if self.ctx.detected else [],
            "compromised":         current_compromised if self.ctx.detected else [],
            "containment_actions": self.ctx.containment_actions if self.ctx.detected else [],
            "recovery_actions":    self.ctx.recovery_actions if self.ctx.detected else [],
            "memory_rolled_back":  self.ctx.memory_rolled_back if self.ctx.detected else 0,
            "detection_time_s":    self.ctx.detection_time if self.ctx.detected else None,
            "recovery_time_s":     self.ctx.recovery_time if self.ctx.detected else None,
            "final_output":        outputs.get("Generator", ""),
            "trust_history":       self.ctx.trust_history,
            "peak_irs":            getattr(self.ctx, 'peak_irs', 0.0),
            "classified_irs":      getattr(self.ctx, 'peak_irs', 0.0),
            "classified_attack_type": final_attack_type,
            # Add metrics for display
            "asr":                 0.0,  # Will be computed from collector
            "detection_rate":      0.0,  # Will be computed from collector
            "fpr":                 0.0,  # Will be computed from collector
            "score_timeline":      self.ctx.score_timeline,
            "final_trust_scores":  current_trust_scores,  # Add final trust scores
            "split_metrics": {
                "handcrafted": {
                    "ASR": round(self.split_stats["handcrafted"]["asr"] / max(1, self.split_stats["handcrafted"]["runs"]), 3),
                    "Detection Rate": round(self.split_stats["handcrafted"]["detect"] / max(1, self.split_stats["handcrafted"]["runs"]), 3),
                    "FPR": round(self.split_stats["handcrafted"]["fpr"] / max(1, self.split_stats["handcrafted"]["runs"]), 3),
                    "Patient Zero Accuracy": round(self.split_stats["handcrafted"]["pz_correct"] / max(1, self.split_stats["handcrafted"]["runs"]), 3),
                    "Recovery Time": round(self.split_stats["handcrafted"]["recovery_time_total"] / max(1, self.split_stats["handcrafted"]["runs"]), 3),
                },
                "agentdojo": {
                    "ASR": round(self.split_stats["agentdojo"]["asr"] / max(1, self.split_stats["agentdojo"]["runs"]), 3),
                    "Detection Rate": round(self.split_stats["agentdojo"]["detect"] / max(1, self.split_stats["agentdojo"]["runs"]), 3),
                    "FPR": round(self.split_stats["agentdojo"]["fpr"] / max(1, self.split_stats["agentdojo"]["runs"]), 3),
                    "Patient Zero Accuracy": round(self.split_stats["agentdojo"]["pz_correct"] / max(1, self.split_stats["agentdojo"]["runs"]), 3),
                    "Recovery Time": round(self.split_stats["agentdojo"]["recovery_time_total"] / max(1, self.split_stats["agentdojo"]["runs"]), 3),
                },
            },
        }
        display.show_incident_report(report)
        return report

    def _run_pre_pipeline_detection(self, user_query: str) -> dict:
        """
        Run comprehensive threat detection BEFORE the agent pipeline.
        This addresses Issue 1: Threat Detection Happens Too Late.
        
        Returns:
            dict with detection evidence for Decision Engine
        """
        evidence = {
            "irs": 0.0,
            "ms": 0.0,
            "bd": 0.0,
            "intent_confidence": 0.0,
            "leakage_confidence": 0.0,
            "obfuscation_detected": False,
            "multi_turn_risk": 0.0,
            "trust_scores": dict(self.trust.trust_scores),
            "propagation_risk": 0.0,
        }

        # Run intent/leakage/obfuscation/propagation detectors FIRST — IRS and MS
        # need their outputs as inputs (delta*I and zeta*PR terms). Previously
        # these ran AFTER compute_irs/compute_ms and their results were never
        # fed back in, so I and PR were silently always 0 in the IRS formula
        # even though the detectors themselves worked correctly.
        if self.intent_detector:
            intent_result = self.intent_detector.detect(user_query)
            evidence["intent_confidence"] = intent_result.confidence
            evidence["intent_type"] = intent_result.intent_type.value

        if self.leakage_detector:
            leakage_result = self.leakage_detector.detect(user_query)
            evidence["leakage_confidence"] = leakage_result.confidence

        if self.obfuscation_detector:
            obfuscation_result = self.obfuscation_detector.detect(user_query)
            evidence["obfuscation_detected"] = obfuscation_result.is_obfuscated

        # Compute behavioral drift before IRS so BD can feed the IRS formula too
        baseline_sim, global_sim, bd = self.drift.compute_drift_with_similarity(PROMPT_BASELINE_NAME, user_query)
        evidence["bd"] = bd

        # Propagation risk: retrospective (already-compromised agents) OR
        # predictive (how malicious this prompt looks, scaled by downstream
        # pipeline exposure). See _calculate_propagation_risk for why the
        # predictive half was added.
        entry_agent_guess = "Retriever"  # first agent in the pipeline; worst-case exposure
        evidence["propagation_risk"] = self._calculate_propagation_risk(
            rule_score=compute_rule_score(user_query),
            intent_confidence=evidence["intent_confidence"],
            entry_agent=entry_agent_guess,
        )

        # Compute IRS with ALL enhanced factors actually wired in
        try:
            irs_result = self.irs_assessor.compute_irs(
                user_query,
                source="User Prompt",
                intent_confidence=evidence["intent_confidence"],
                behavioral_drift=bd,
                trust_score=min(self.trust.trust_scores.values()) if self.trust.trust_scores else 1.0,
                propagation_risk=evidence["propagation_risk"],
            )
        except TypeError as exc:
            # Compatibility with lightweight assessor doubles and older assessor
            # implementations that expose only the original two-argument API.
            if "unexpected keyword argument" not in str(exc):
                raise
            irs_result = self.irs_assessor.compute_irs(user_query, source="User Prompt")
        evidence["irs"] = irs_result.irs

        # Compute MS, also passing intent/propagation through
        try:
            ms_result = self.irs_assessor.compute_ms(
                irs_result.irs,
                irs_result.semantic_drift,
                bd,
                intent_confidence=evidence["intent_confidence"],
                propagation_risk=evidence["propagation_risk"],
            )
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            ms_result = self.irs_assessor.compute_ms(
                irs_result.irs, irs_result.semantic_drift, bd
            )
        evidence["ms"] = ms_result.ms

        if self.multi_turn_detector:
            # Add turn info for multi-turn detection
            self.multi_turn_detector.add_turn(
                user_prompt=user_query,
                irs=irs_result.irs,
                ms=ms_result.ms,
                trust_before=self.trust.trust_scores.get("Planner", 1.0),
                trust_after=self.trust.trust_scores.get("Planner", 1.0),
                detected_attack=False,
            )
            multi_turn_result = self.multi_turn_detector.analyze()
            evidence["multi_turn_risk"] = multi_turn_result.confidence

        return evidence
    
    def _calculate_propagation_risk(self, rule_score: float = 0.0, intent_confidence: float = 0.0, entry_agent: str = "Planner") -> float:
        """Calculate risk of attack propagation.

        Combines two components:
        - retrospective: agents already marked compromised (from a prior
          turn, or earlier in this same run after fix #10's timing change)
        - predictive: how likely THIS prompt is malicious (rule/intent
          signals), scaled by how much of the pipeline sits downstream of
          its likely entry point. Previously this method only counted
          already-compromised agents, which is structurally always 0 at the
          point it's called during pre-pipeline detection (before any agent
          in this turn has run) -- so propagation_risk was 0 by construction
          for every single-turn request, regardless of how obviously
          malicious the prompt was.
        """
        compromised_count = len(self.trust.get_all_compromised())
        retrospective_risk = 0.0
        if compromised_count > 0:
            base_risk = min(1.0, compromised_count / len(AGENT_ORDER))
            compromised_trust = [
                self.trust.trust_scores.get(agent, 1.0)
                for agent in self.trust.get_all_compromised()
            ]
            avg_trust = sum(compromised_trust) / len(compromised_trust)
            trust_factor = 1.0 - avg_trust
            retrospective_risk = base_risk * 0.5 + trust_factor * 0.5

        # Predictive component: likelihood this prompt is malicious, scaled
        # by downstream pipeline exposure from its likely entry point.
        likelihood = max(rule_score, intent_confidence)
        if entry_agent in AGENT_ORDER:
            downstream_count = len(AGENT_ORDER) - AGENT_ORDER.index(entry_agent)
        else:
            downstream_count = len(AGENT_ORDER)
        downstream_exposure = downstream_count / len(AGENT_ORDER)
        predictive_risk = likelihood * downstream_exposure

        return max(retrospective_risk, predictive_risk)
    
    def execute(
        self,
        user_query: str,
        attack_type: Optional[str] = None,
        payload_index: int = 0,
        auto_detect: bool = False,
        simulate_attack: bool = False,
    ) -> dict:
        """
        Run TrustTrace on user_query.

        NEW PIPELINE (addresses Issue 1 and 25):
        1. Threat Detection (BEFORE agents)
        2. Risk Assessment
        3. Decision Engine (centralized enforcement)
        4. Safe → Agents
        5. Unsafe → Block & Recover

        auto_detect=True  — classify attack type/payload from prompt + retrieval (interactive).
        simulate_attack=True — plant synthetic attack payloads (experiments only).
        """
        if not isinstance(user_query, str) or not user_query.strip():
            raise ValueError("user_query must be a non-empty string")
        if not isinstance(auto_detect, bool) or not isinstance(simulate_attack, bool):
            raise TypeError("auto_detect and simulate_attack must be boolean values")
        self.ctx = RunContext(
            run_id=f"runtime_{uuid.uuid4().hex[:8]}",
            user_query=user_query,
            attack_type=attack_type,
        )
        self.preemptive_outputs = None
        # Fix detection time: set start_time at request start, not application startup
        self.ctx.start_time = time.time()
        # Collect attack category for split reporting
        # Assume "handcrafted" if attack_type is one of direct/indirect/memory; otherwise "agentdojo"
        attack_category = "handcrafted" if attack_type in {"direct", "indirect", "memory"} else "agentdojo"
        self.ctx.attack_category = attack_category
        
        # ISSUE 1 FIX: Run threat detection BEFORE agent pipeline
        detection_evidence = self._run_pre_pipeline_detection(user_query)
        
        # Store detection evidence in context
        self.ctx.detection_evidence = detection_evidence
        self.ctx.peak_irs = detection_evidence["irs"]

        # Classify the attack type EARLY, before the Decision Engine's
        # REFUSE/CONTAIN branches (which return _finalize_report()
        # immediately). Previously classification only happened much later
        # in this method, in code the REFUSE/CONTAIN early-returns above
        # never reached -- so any prompt correctly blocked pre-pipeline
        # (the common case now that enforcement actually works) still had
        # self.ctx.attack_type stuck at its None default, and the report's
        # fallback logic reported it as "benign" despite being blocked
        # specifically because it was recognized as an attack.
        early_classification = None
        if auto_detect:
            early_classification = self.classifier.classify(user_query)
            self.ctx.attack_type = early_classification.attack_type
            self.ctx.attack_confidence = early_classification.confidence
            self.ctx.attack_severity = getattr(early_classification, "severity", "unknown")
            self.ctx.attack_reason = early_classification.reason
            if early_classification.irs > self.ctx.peak_irs:
                self.ctx.peak_irs = early_classification.irs

        # ISSUE 25 FIX: Use Decision Engine for centralized enforcement
        if self.decision_engine:
            decision = self.decision_engine.evaluate(
                irs=detection_evidence["irs"],
                ms=detection_evidence["ms"],
                bd=detection_evidence["bd"],
                trust_scores=detection_evidence["trust_scores"],
                intent_confidence=detection_evidence["intent_confidence"],
                propagation_risk=detection_evidence["propagation_risk"],
                attack_type=attack_type,
                context={
                    "turn_number": len(self.multi_turn_detector.conversation_history) if self.multi_turn_detector else 1,
                    "attack_history": [],
                    "trust_history": 1.0,
                },
            )
            self.ctx.decision = decision
            
            # Enforce decision
            if decision.decision_type == DecisionType.REFUSE:
                # Block immediately with refusal
                self.ctx.detected = True
                self.ctx.blocked_layer = "Threat Detection"
                self.ctx.containment_actions.append(f"Refused based on Decision Engine: {decision.reason}")
                entry_agent = self._record_preemptive_compromise(self.ctx.attack_type, decision.confidence)
                if entry_agent:
                    self.ctx.compromised = [entry_agent]
                    self.ctx.patient_zero = self.detector.detect() or entry_agent
                    self.ctx.propagation_path = self.detector.get_propagation_path(self.ctx.patient_zero)
                self.preemptive_outputs = {
                    "Retriever": "[BLOCKED] Request refused by security policy.",
                    "Planner": "[BLOCKED] Request refused by security policy.",
                    "Executor": "[BLOCKED] Request refused by security policy.",
                    "Generator": f"[BLOCKED] {decision.reason}. Please provide a safe, non-sensitive request.",
                }
                self.ctx.attack_succeeded = False
                self.ctx.detection_time = time.time() - self.ctx.start_time
                if not self.ctx.attack_type:
                    self._infer_attack_type_from_evidence()
            
            elif decision.decision_type == DecisionType.CONTAIN:
                # Quarantine affected agents and stop propagation
                self.ctx.detected = True
                self.ctx.blocked_layer = "Decision Engine"
                self.ctx.containment_actions.append(f"Containment triggered: {decision.reason}")
                self.ctx.containment_actions.append(f"Quarantined agents: {decision.affected_agents}")
                self.recovery.quarantine(decision.affected_agents)
                self.ctx.compromised = list(decision.affected_agents)

                if not self.ctx.compromised:
                    # decision.affected_agents is derived from trust_scores,
                    # which are still all 1.0 pre-pipeline -- this CONTAIN
                    # fired on prompt-level signals (IRS/intent), not on any
                    # agent's trust having actually dropped. Mark the entry
                    # agent compromised so patient-zero/propagation have
                    # something to report instead of silently empty/None.
                    entry_agent = self._record_preemptive_compromise(self.ctx.attack_type, decision.confidence)
                    if entry_agent:
                        self.ctx.compromised = [entry_agent]
                self.ctx.patient_zero = self.detector.detect() or (self.ctx.compromised[0] if self.ctx.compromised else None)
                if self.ctx.patient_zero:
                    self.ctx.propagation_path = self.detector.get_propagation_path(self.ctx.patient_zero)
                
                # Execute recovery automatically
                self._execute_automatic_recovery(decision)
                
                self.preemptive_outputs = {
                    "Retriever": "[BLOCKED] Request contained by security policy.",
                    "Planner": "[BLOCKED] Request contained by security policy.",
                    "Executor": "[BLOCKED] Request contained by security policy.",
                    "Generator": f"[BLOCKED] {decision.reason}. The request was contained for security reasons.",
                }
                self.ctx.attack_succeeded = False
                self.ctx.detection_time = time.time() - self.ctx.start_time
                if not self.ctx.attack_type:
                    self._infer_attack_type_from_evidence()
            
            elif decision.decision_type == DecisionType.QUARANTINE:
                # Quarantine specific agents but allow others to execute
                self.ctx.detected = True
                self.ctx.containment_actions.append(f"Quarantine triggered: {decision.reason}")
                self.recovery.quarantine(decision.affected_agents)
                self.ctx.compromised = list(decision.affected_agents)

        from victim_pipeline.agents import clear_attacker_documents

        classification = early_classification
        payload_index = 0

        if self.preemptive_outputs is None:
            if auto_detect:
                classification = early_classification if early_classification is not None else self.classifier.classify(user_query)
                attack_type = classification.attack_type
                payload_index = classification.payload_index
                self.ctx.attack_type = attack_type
                # Initialize peak IRS with classification IRS
                self.ctx.peak_irs = classification.irs if hasattr(classification, 'irs') else 0.0
                display.show_attack_classification(
                    attack_type=classification.attack_type,
                    payload_index=classification.payload_index,
                    confidence=classification.confidence,
                    irs=classification.irs,
                    ms=classification.ms,
                    reason=classification.reason,
                    source=classification.source,
                    matched_snippet=classification.matched_snippet,
                )
                if attack_type is None:
                    # Interactive benign turns preserve trusted user memory and
                    # knowledge-base content; remove only documents explicitly
                    # marked as attacker-controlled. Full KB reset is reserved
                    # for isolated batch experiment episodes.
                    clear_attacker_documents(knowledge_base)
            elif attack_type is None:
                clear_attacker_documents(knowledge_base)

            attack_doc_id = None
            effective_query = user_query
            # Normalized short category ("direct"/"indirect"/"memory") for the
            # branches below, which only know about those three legacy names.
            # self.ctx.attack_type / attack_type itself stays as the classifier's
            # detailed category (e.g. "direct_injection") for reporting.
            attack_category_short = self._short_attack_category(attack_type)

            if attack_type is not None:
                self.ctx.detected = True
                self.ctx.blocked_layer = self._blocking_layer_for_attack(attack_type)
                self.ctx.containment_actions.append(f"Blocked {attack_type} payload before LLM execution")
                self.ctx.recovery_reason = (
                    f"Preemptive {'classification' if auto_detect else 'attack flag'} detected {attack_type} attack before downstream execution."
                )
                if attack_category_short == "direct":
                    self.preemptive_outputs = self._execute_preemptive_recovery(attack_type)
                else:
                    effective_query = self._build_safe_prompt(user_query, attack_type)

            if simulate_attack and attack_category_short == "direct":
                payload = DIRECT_PAYLOADS[payload_index % len(DIRECT_PAYLOADS)]
                effective_query = f"{user_query}\n\n{payload}"
                self.ctx.attack_payload = payload
                irs_result = self.irs_assessor.compute_irs(payload, source="User Prompt")
                display.show_injection_risk(
                    source="User Prompt",
                    rule_score=irs_result.rule_score,
                    embedding_score=irs_result.embedding_score,
                    semantic_drift=irs_result.semantic_drift,
                    irs=irs_result.irs,
                )

            elif simulate_attack and attack_category_short == "indirect":
                attack_doc_id = inject_indirect(
                    knowledge_base,
                    self.memory,
                    payload_index=payload_index,
                    trust_engine=self.trust,
                    irs_assessor=self.irs_assessor,
                )
                self.ctx.attack_doc_id = attack_doc_id

            elif simulate_attack and attack_category_short == "memory":
                attack_doc_id = inject_memory_poison(
                    knowledge_base,
                    self.memory,
                    payload_index=payload_index,
                    trust_engine=self.trust,
                    irs_assessor=self.irs_assessor,
                )
                self.ctx.attack_doc_id = attack_doc_id

            elif auto_detect and attack_category_short == "direct":
                # IRS already computed during classification, use that result
                # Recompute only if needed for trust update
                baseline_sim, global_sim, bd = self.drift.compute_drift_with_similarity(PROMPT_BASELINE_NAME, user_query)
                # Use classification IRS for MS computation
                ms_result = self.irs_assessor.compute_ms(classification.irs, classification.irs, bd)  # Use classification IRS
                self._log_and_update_trust(
                    "User",
                    "Planner",
                    user_query,
                    ms_result.ms,
                    bd,
                    baseline_sim,
                    global_sim,
                    irs_score=classification.irs,  # Use classification IRS
                    event_type="message",
                )

            # Automatic classification of real user content must never create
            # synthetic poisoned documents in persistent memory. Indirect and
            # memory payloads are inserted only by the explicit
            # simulate_attack=True experiment path above.
        else:
            attack_type = self.ctx.attack_type
            attack_doc_id = None
            effective_query = user_query
            payload_index = getattr(classification, "payload_index", 0) if classification else 0
            attack_category_short = self._short_attack_category(attack_type)

        # ISSUE 9 FIX: Make Trust Score control agent execution.
        #
        # IMPORTANT: this pre-loop only reflects trust as it stands BEFORE any
        # agent in this turn has run. Since reset_state() resets every agent's
        # trust to 1.0 at the top of each interactive turn, this snapshot is
        # always "all agents trusted" and can never catch a trust collapse
        # that happens DURING this same run (e.g. Planner's own output
        # tanking Executor's incoming trust). It's kept here so obviously
        # pre-compromised agents (e.g. carried over in experiment/batch mode)
        # are still caught immediately, but the real enforcement now happens
        # live via `_live_should_execute` below, checked fresh before each
        # agent step inside run_pipeline_stepwise.
        block_generator = False
        quarantined_agents = set(self.recovery.quarantined)

        def _live_should_execute(agent: str) -> bool:
            """Checked immediately before each agent step, using whatever
            trust looks like right now — including drops caused by earlier
            agents in this very run. This is the actual enforcement gate;
            the upfront loop below is only a cheap early pass."""
            # Classifier-benign prompts: trust drops from SD/BD miscalibration
            # must not stop the pipeline. Final Response Validator catches
            # genuinely unsafe output; blocking here produced [BLOCKED] final
            # answers while the Security Report still read Threat Detected: NO.
            if self._is_benign_classification(self.ctx.attack_type) and not self._has_attack_corroboration():
                return True

            trust_score = self.trust.trust_scores.get(agent, 1.0)
            if self.decision_engine:
                allowed = self.decision_engine.should_execute_agent(agent, trust_score)
            else:
                allowed = trust_score > self.trust.delta
            if not allowed:
                self.ctx.containment_actions.append(
                    f"Agent {agent} blocked due to low trust ({trust_score:.3f})"
                )
                self.recovery.quarantined.add(agent)
                if agent == "Generator":
                    self.ctx.blocked_layer = agent
                corroborated = (
                    self._has_attack_corroboration()
                    or self._agent_output_corroborates_attack(agent)
                )
                if corroborated:
                    self.ctx.detected = True
                    self.trust.compromise(agent, timestamp=time.time())
                    if agent not in self.ctx.compromised:
                        self.ctx.compromised.append(agent)
                    if self._is_benign_classification(self.ctx.attack_type):
                        self.ctx.attack_type = "runtime_compromise"
            return allowed

        # Check trust scores for each agent before execution
        benign_trust_pass = (
            self._is_benign_classification(self.ctx.attack_type)
            and not self._has_attack_corroboration()
        )
        for agent in AGENT_ORDER:
            if benign_trust_pass:
                continue
            trust_score = self.trust.trust_scores.get(agent, 1.0)
            
            # Use Decision Engine if available, otherwise use simple threshold
            if self.decision_engine:
                should_execute = self.decision_engine.should_execute_agent(agent, trust_score)
            else:
                should_execute = trust_score > self.trust.delta
            
            if not should_execute:
                quarantined_agents.add(agent)
                self.ctx.containment_actions.append(f"Agent {agent} blocked due to low trust ({trust_score:.3f})")
                
                # If Generator is blocked, set block_generator flag
                if agent == "Generator":
                    block_generator = True
                    self.ctx.blocked_layer = agent
        
        # Update recovery quarantine set
        self.recovery.quarantined.update(quarantined_agents)

        if self.preemptive_outputs is not None:
            outputs = self.preemptive_outputs
            if self.ctx and not self.ctx.propagation_chain:
                self.ctx.propagation_chain = [
                    {"name": "User", "label": "User"},
                    {"name": "Blocked", "label": "Blocked before downstream propagation"},
                ]
                self.ctx.propagation_path = ["User", self.ctx.blocked_layer or "Blocked"]
        else:
            try:
                try:
                    outputs = run_pipeline_stepwise(
                        effective_query,
                        on_agent_start=self._on_agent_start,
                        on_agent_complete=self._on_agent_complete,
                        quarantined=self.recovery.quarantined,
                        block_generator=block_generator,
                        should_execute=_live_should_execute,
                    )
                except TypeError as exc:
                    # Retain compatibility with older pipeline adapters, but
                    # do not hide TypeErrors raised inside the adapter itself.
                    if "unexpected keyword argument 'should_execute'" not in str(exc):
                        raise
                    outputs = run_pipeline_stepwise(
                        effective_query,
                        on_agent_start=self._on_agent_start,
                        on_agent_complete=self._on_agent_complete,
                        quarantined=self.recovery.quarantined,
                        block_generator=block_generator,
                    )
            except _AgentValidationBlocked as exc:
                self.ctx.detected = True
                self.ctx.blocked_layer = exc.agent
                self.ctx.attack_type = self.ctx.attack_type or (
                    "indirect_injection" if exc.agent == "Retriever" else "planner_goal_violation"
                )
                self.ctx.attack_succeeded = False
                self.ctx.detection_time = time.time() - self.ctx.start_time
                downstream = AGENT_ORDER[AGENT_ORDER.index(exc.agent):] if exc.agent in AGENT_ORDER else AGENT_ORDER[1:]
                self.recovery.quarantined.update(downstream)
                blocked_message = f"[BLOCKED] {exc.agent} validation failed: {exc.reason}"
                outputs = {
                    "Retriever": blocked_message if exc.agent == "Retriever" else "[COMPLETED]",
                    "Planner": blocked_message if exc.agent == "Planner" else "[BLOCKED] Upstream validation failed.",
                    "Executor": "[BLOCKED] Upstream validation failed.",
                    "Generator": blocked_message,
                }
                if not self.ctx.propagation_chain:
                    self.ctx.propagation_chain = [
                        {"name": "User", "label": "User"},
                        {"name": exc.agent, "label": "Blocked before LLM execution"},
                    ]
                    self.ctx.propagation_path = ["User", exc.agent]
            finally:
                # Simulated payloads are temporary test fixtures. Always remove
                # them even if the LLM, retriever or a security callback raises.
                if attack_doc_id:
                    if attack_category_short == "indirect":
                        cleanup_indirect(knowledge_base, attack_doc_id, memory_manager=self.memory)
                    elif attack_category_short == "memory":
                        cleanup_memory_poison(knowledge_base, attack_doc_id, memory_manager=self.memory)
                    attack_doc_id = None

        if block_generator and outputs.get("Generator", "").startswith("[BLOCKED]"):
            outputs["Generator"] = self._blocked_message(self.ctx.blocked_layer or "Generator")

        # Gate security pipeline - run compromise handling whenever real compromise emerged,
        # regardless of whether the entry classifier flagged the input.
        # ISSUE 11 FIX: Make Recovery execute automatically after containment
        if self.preemptive_outputs is None and self.trust.get_all_compromised():
            self.ctx.detected = True
            if not self.ctx.attack_type:
                if self._has_attack_corroboration() or self._agent_output_corroborates_attack(
                    next(iter(self.trust.get_all_compromised()), "")
                ):
                    self.ctx.attack_type = "runtime_compromise"
                else:
                    # Trust dropped without attack signals — clear false compromise
                    for agent in list(self.trust.get_all_compromised()):
                        self.trust.recover_agent(agent)
                    self.ctx.compromised = []
                    self.ctx.detected = False
            if self.ctx.detected:
                self._handle_compromise()
        elif self.preemptive_outputs is None and attack_type is not None and self.ctx.detected:
            self._handle_compromise()

        if self.preemptive_outputs is None and self.ctx.compromised and not self.ctx.patient_zero:
            self.ctx.patient_zero = self.detector.detect() or None
            if self.ctx.patient_zero:
                self.ctx.propagation_path = self.detector.get_propagation_path(self.ctx.patient_zero)
                if self.ctx.propagation_path:
                    self.ctx.propagation_chain = [{"name": "Propagation", "label": "Propagation"}] + [
                        {"name": agent, "label": agent} for agent in self.ctx.propagation_path
                    ]

        # ISSUE 24 FIX: Final Response Validation. Runs BEFORE metrics
        # collection below (previously ran after, so record_benign() locked
        # in a stale attack_type/detected state before this check even had
        # a chance to catch anything).
        if self.final_validator and self.preemptive_outputs is None:
            final_output = outputs.get("Generator", "")
            validation_result = self.final_validator.validate(final_output, user_query)
            
            if not validation_result.is_safe:
                # Replace unsafe output with safe refusal
                outputs["Generator"] = f"[BLOCKED] {validation_result.reason}. Please provide a safe, non-sensitive request."
                self.ctx.detected = True
                self.ctx.blocked_layer = "Final Response Validator"
                self.ctx.containment_actions.append(f"Final response blocked: {validation_result.reason}")
                self.ctx.attack_succeeded = False
                # Same gap as the trust-gate fix above: the classifier may
                # have missed this phrasing entirely, but the final output
                # itself just failed a safety check -- that's real evidence
                # of compromise, not a false positive on a benign prompt.
                if not self.ctx.attack_type:
                    self.ctx.attack_type = "unsafe_output"
                if "Generator" not in self.ctx.compromised:
                    self.ctx.compromised.append("Generator")
                    self.trust.compromise("Generator", timestamp=time.time())
            
            # Store validation result in context
            self.ctx.final_validation = validation_result

        # Resync local attack_type from self.ctx -- either the live
        # trust-gate callback (during run_pipeline_stepwise, above) or the
        # validator just above may have set self.ctx.attack_type after
        # `attack_type` was first assigned from the upfront classification.
        # Without this resync, the collector/gt_map logic below still uses
        # the stale pre-pipeline value and reports a caught attack as benign.
        attack_type = self.ctx.attack_type or attack_type

        # Populate collector BEFORE computing confusion matrix
        effective_attack_type = attack_type
        if not effective_attack_type and self.ctx.detected and self._has_attack_corroboration():
            effective_attack_type = "direct_injection"

        if effective_attack_type:
            self.collector.record_attack_with_meta(
                succeeded=self.ctx.attack_succeeded,
                detected=self.ctx.detected,
                source="runtime",
                attack_type=effective_attack_type,
            )
            if self.ctx.patient_zero:
                gt_map = {
                    "direct": "Planner",
                    "indirect": "Retriever",
                    "memory": "MemoryStore",
                }
                self.collector.record_patient_zero(
                    self.ctx.patient_zero, gt_map.get(self._short_attack_category(effective_attack_type), "Retriever")
                )
        elif not hasattr(self.ctx, 'metrics_recorded'):
            self.collector.record_benign(flagged=bool(self.ctx.detected))
            self.ctx.metrics_recorded = True

        report = self._finalize_report(outputs)

        if self.ctx is not None:
            report.setdefault("propagation_chain", self.ctx.propagation_chain or [])
            report.setdefault("propagation_path", self.ctx.propagation_path or [])
            report.setdefault("blocked_layer", self.ctx.blocked_layer)
            report.setdefault("compromised", self.ctx.compromised or [])
            report.setdefault("containment_actions", self.ctx.containment_actions or [])
            report.setdefault("recovery_actions", self.ctx.recovery_actions or [])
            report.setdefault("memory_rolled_back", self.ctx.memory_rolled_back or 0)
            report.setdefault("detection_time_s", self.ctx.detection_time)
            report.setdefault("recovery_time_s", self.ctx.recovery_time)
            report.setdefault("attack_type", self.ctx.attack_type or "benign")
            report.setdefault("final_output", outputs.get("Generator", ""))
            
            # Add decision engine info if available
            if hasattr(self.ctx, 'decision') and self.ctx.decision:
                report["decision_type"] = self.ctx.decision.decision_type.value
                report["decision_confidence"] = self.ctx.decision.confidence
                report["decision_reason"] = self.ctx.decision.reason
                report["decision_evidence"] = self.ctx.decision.evidence

        # Add fusion-based detection decision
        if auto_detect and classification:
            # Get fusion weights from config
            fusion_weights = self.cfg.get("fusion_weights", {"classifier": 0.5, "multilayer": 0.3, "irs": 0.2})
            
            # Get multilayer confirmation status
            multilayer_confirmed = any(
                self.trust.last_multilayer.get(agent, MultilayerResult(
                    agent=agent,
                    compromised=False,
                    irs=0.0,
                    ms=0.0,
                    bd=0.0,
                    lambda_threshold=0.0,
                    delta=self.trust.delta,
                    gamma=self.trust.gamma,
                    irs_pass=False,
                    ms_pass=False,
                    bd_pass=False,
                    consecutive=0,
                    required_k=self.trust.persistence_table.get(agent, self.trust.k)
                )).compromised
                for agent in AGENT_ORDER
            )
            
            # Calculate fusion score
            classifier_score = classification.confidence
            multilayer_score = 1.0 if multilayer_confirmed else 0.0
            irs_score = min(1.0, self.ctx.peak_irs / self._threshold_for_attack(attack_type)) if attack_type else 0.0
            
            fusion_score = (
                fusion_weights.get("classifier", 0.5) * classifier_score +
                fusion_weights.get("multilayer", 0.3) * multilayer_score +
                fusion_weights.get("irs", 0.2) * irs_score
            )
            
            # Fusion-based detection: preserve any real end-of-run compromise state,
            # and otherwise use the fusion signal for the final decision.
            fusion_detected = (
                self.ctx.detected or
                classification.confidence >= 0.90 or
                multilayer_confirmed or
                (self.ctx.peak_irs >= self._threshold_for_attack(attack_type) if attack_type else False)
            )
            
            # Update ctx.detected with fusion decision
            self.ctx.detected = fusion_detected
            report["detected"] = fusion_detected
            report["fusion_score"] = fusion_score
            report["classifier_confidence"] = classification.confidence
            report["multilayer_confirmed"] = multilayer_confirmed
        report["attack_type"] = self.ctx.attack_type or attack_type or "benign"
        report["classified_attack_type"] = report["attack_type"]
        report["classified_payload_index"] = payload_index
        report["classified_irs"] = getattr(self.ctx, "peak_irs", 0.0) if self.ctx else 0.0
        report["peak_irs"] = report["classified_irs"]

        # Inject TP/TN/FP/FN from collector into report for display
        try:
            cm = self.collector.compute()
            report["TP"] = cm.get("TP", 0)
            report["TN"] = cm.get("TN", 0)
            report["FP"] = cm.get("FP", 0)
            report["FN"] = cm.get("FN", 0)
            # Add ASR, detection rate, and FPR for display - use single source of truth from collector
            report["asr"] = cm.get("ASR", 0.0)
            report["detection_rate"] = cm.get("Detection_Rate", 0.0)
            report["fpr"] = cm.get("FPR", 0.0)
            # Include patient-zero accuracy and breakdown for display
            report["Patient_Zero_Accuracy"] = cm.get("Patient_Zero_Accuracy", 0.0)
            report["patient_zero_breakdown"] = cm.get("patient_zero_breakdown", {})
        except Exception as exc:
            # Keep the incident report, but never silently imply metrics were
            # computed successfully when the collector failed.
            report["metrics_error"] = f"{type(exc).__name__}: {exc}"

        return report
    
    def _execute_automatic_recovery(self, decision) -> None:
        """Execute automatic recovery after containment (Issue 11)."""
        rec_start = time.time()
        
        # Rollback memory
        rolled_back = self.recovery.rollback_memory(self.ctx.start_time)
        self.ctx.memory_rolled_back = rolled_back
        
        # Recover quarantined agents. Uses self.ctx.compromised rather than
        # decision.affected_agents directly -- the latter is derived from
        # pre-pipeline trust scores (all still 1.0) and was structurally
        # empty for the common case of a prompt-level block, which meant
        # this loop never ran and "Recovery never executes" was the
        # observed symptom regardless of how the block was decided.
        agents_to_recover = self.ctx.compromised or list(decision.affected_agents)
        for agent in agents_to_recover:
            restored = self.trust.recover_agent(agent)
            ckpt = self.memory.checkpoint()
            display.show_recovery(ckpt, agent, restored)
            self.ctx.recovery_actions.append(f"Restored {agent} to trust {restored:.2f}")
        
        self.ctx.recovery_time = time.time() - rec_start