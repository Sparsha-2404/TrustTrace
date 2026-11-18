import json
import os
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EvalResults:
    """Data class to hold evaluation results.
    
    ISSUE 21 FIX: Enhanced with additional metrics fields.
    """
    total_attacks: int = 0
    attacks_succeeded: int = 0
    attacks_detected: int = 0
    benign_total: int = 0
    benign_flagged: int = 0
    pz_total: int = 0
    pz_correct: int = 0
    recovery_times: List[float] = field(default_factory=list)
    tasks_during_recovery: int = 0
    tasks_continued_during_recovery: int = 0
    raw_records: List[dict] = field(default_factory=list)
    chroma_sizes: List[int] = field(default_factory=list)
    retrieval_hits: List[int] = field(default_factory=list)
    benign_false_positives: List[str] = field(default_factory=list)
    
    # ISSUE 21 FIX: New metric fields
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    detection_delays: List[float] = field(default_factory=list)
    trust_drops: List[float] = field(default_factory=list)
    propagation_lengths: List[int] = field(default_factory=list)
    containment_times: List[float] = field(default_factory=list)


class MetricsCollector:
    """
    Collects data during experiments and computes all metrics.

    ISSUE 21 FIX: Enhanced metrics:
    1. ASR — Attack Success Rate
    2. Detection Rate
    3. False Positive Rate (FPR)
    4. Patient Zero Identification Accuracy
    5. Recovery Time
    6. System Availability During Recovery
    7. Precision
    8. Recall
    9. F1 Score
    10. Specificity
    11. Sensitivity
    12. Detection Delay
    13. Containment Time
    14. Average Trust Drop
    15. Propagation Length
    """

    def __init__(self):
        self.results = EvalResults()
        self.attack_success = {}
        self._records: List[dict] = []
        self.patient_zero_methods: Dict[str, Dict[str, int]] = {}

    def record_attack(self, succeeded: bool, detected: bool):
        """Record an attack outcome and detector classification separately.

        Every row is an attack-positive episode: detected => TP, otherwise FN.
        ``succeeded`` currently reflects a final-output lexical marker proxy;
        it is not a validated scenario-specific attack-goal oracle.
        """
        self.results.total_attacks += 1
        self.results.attacks_succeeded += int(bool(succeeded))
        if detected:
            self.results.attacks_detected += 1
            self.results.true_positives += 1
        else:
            self.results.false_negatives += 1

    def record_attack_with_meta(
        self,
        succeeded: bool,
        detected: bool,
        source: str,
        attack_type: str,
        patient_zero_correct: bool = False,
        recovery_time_s: Optional[float] = None,
        chroma_size: int = 0,
        retrieval_hits: int = 0,
    ):
        """Extended version that stores source tag and patient zero correctness for split reporting."""
        self.record_attack(succeeded, detected)
        if recovery_time_s is not None and recovery_time_s > 0:
            self.results.recovery_times.append(recovery_time_s)
        if chroma_size:
            self.results.chroma_sizes.append(chroma_size)
        if retrieval_hits:
            self.results.retrieval_hits.append(retrieval_hits)
        self.results.raw_records.append({
            "succeeded": succeeded,
            "detected": detected,
            "source": source,
            "type": attack_type,
            "patient_zero_correct": patient_zero_correct,
            "recovery_time_s": recovery_time_s,
        })
        if attack_type not in self.attack_success:
            self.attack_success[attack_type] = {}
        src_dict = self.attack_success[attack_type]
        if source not in src_dict:
            src_dict[source] = {"total": 0, "succeeded": 0, "detected": 0}
        src_dict[source]["total"] += 1
        if succeeded:
            src_dict[source]["succeeded"] += 1
        if detected:
            src_dict[source]["detected"] += 1

    def ingest(self, record: dict):
        """Replay a single raw record into this collector's counters."""
        self.record_attack(succeeded=record["succeeded"], detected=record["detected"])

    def record_benign(self, flagged: bool, query: str = ""):
        """Call for every benign (clean) interaction."""
        self.results.benign_total += 1
        if flagged:
            self.results.benign_flagged += 1
            self.results.false_positives += 1
            if query:
                self.results.benign_false_positives.append(query)
        else:
            self.results.true_negatives += 1

    def record_patient_zero(self, predicted: Optional[str], ground_truth: str):
        """
        Ground truth = the agent that first received the injection.
        Predicted = what backward_traversal() returned.
        """
        self.results.pz_total += 1
        if predicted is not None and predicted == ground_truth:
            self.results.pz_correct += 1

    def record_recovery(self, start_time: float, end_time: float,
                         tasks_assigned: int, tasks_completed: int):
        """Record a recovery event's timing and task continuity."""
        if end_time < start_time:
            raise ValueError("recovery end_time must not precede start_time")
        if (isinstance(tasks_assigned, bool) or isinstance(tasks_completed, bool)
                or not isinstance(tasks_assigned, int) or not isinstance(tasks_completed, int)
                or tasks_assigned < 0 or tasks_completed < 0
                or tasks_completed > tasks_assigned):
            raise ValueError("recovery task counts must be integers with 0 <= completed <= assigned")
        self.results.recovery_times.append(end_time - start_time)
        self.results.tasks_during_recovery += tasks_assigned
        self.results.tasks_continued_during_recovery += tasks_completed
    
    # ISSUE 21 FIX: New recording methods for enhanced metrics
    def record_detection_delay(self, delay: float):
        """Record detection delay in seconds."""
        self.results.detection_delays.append(delay)
    
    def record_trust_drop(self, drop: float):
        """Record trust drop amount."""
        self.results.trust_drops.append(drop)
    
    def record_propagation_length(self, length: int):
        """Record propagation chain length."""
        self.results.propagation_lengths.append(length)
    
    def record_containment_time(self, time: float):
        """Record containment time in seconds."""
        self.results.containment_times.append(time)

    def compute(self) -> dict:
        """Compute all metrics from collected data and expose raw data for visualisation.

        ISSUE 21 FIX: Enhanced with precision, recall, F1, specificity, sensitivity, etc.
        ISSUE 22 FIX: Fixed ASR calculation.
        
        ASR currently counts legacy final-output marker-proxy hits / attack episodes.
        It is not a validated semantic or scenario-specific goal-success rate.
        """
        r = self.results

        asr = r.attacks_succeeded / r.total_attacks if r.total_attacks else 0.0
        dr = r.attacks_detected / r.total_attacks if r.total_attacks else 0.0
        fpr = r.benign_flagged / r.benign_total if r.benign_total else 0.0
        pz_acc = r.pz_correct / r.pz_total if r.pz_total else 0.0

        # No recovery observations means unknown/not measured, not zero seconds.
        rt_mean = statistics.mean(r.recovery_times) if r.recovery_times else None
        rt_median = statistics.median(r.recovery_times) if r.recovery_times else None
        rt_std = statistics.stdev(r.recovery_times) if len(r.recovery_times) > 1 else (0.0 if r.recovery_times else None)

        # No observed tasks means availability is unknown, not 100%.
        avail = (r.tasks_continued_during_recovery / r.tasks_during_recovery
                 if r.tasks_during_recovery else None)

        pz_breakdown = {}
        for rec in r.raw_records:
            if not rec.get("patient_zero_ground_truth"):
                continue
            atype = rec.get("type")
            if atype not in pz_breakdown:
                pz_breakdown[atype] = {"total": 0, "correct": 0}
            pz_breakdown[atype]["total"] += 1
            if rec.get("patient_zero_correct"):
                pz_breakdown[atype]["correct"] += 1
        pz_breakdown_simple = {k: (v["correct"] / v["total"] if v["total"] else 0.0) for k, v in pz_breakdown.items()}

        chroma_mean = statistics.mean(r.chroma_sizes) if r.chroma_sizes else 0
        retrieval_mean = statistics.mean(r.retrieval_hits) if r.retrieval_hits else 0

        # One operating point is not an ROC curve and cannot support AUC.
        roc = None
        roc_operating_point = {"tpr": dr, "fpr": fpr}

        # ISSUE 21 FIX: Use the new TP/TN/FP/FN fields
        tp = r.true_positives
        tn = r.true_negatives
        fp = r.false_positives
        fn = r.false_negatives
        
        # ISSUE 21 FIX: Additional metrics
        # Precision = TP / (TP + FP)
        precision = tp / max(1, tp + fp)
        
        # Recall (Sensitivity) = TP / (TP + FN)
        recall = tp / max(1, tp + fn)
        
        # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * (precision * recall) / max(0.001, precision + recall)
        
        # Specificity = TN / (TN + FP)
        specificity = tn / max(1, tn + fp)
        
        # Sensitivity (same as Recall)
        sensitivity = recall
        
        # Average Detection Delay
        avg_detection_delay = statistics.mean(r.detection_delays) if r.detection_delays else 0.0
        
        # Average Containment Time
        avg_containment_time = statistics.mean(r.containment_times) if r.containment_times else 0.0
        
        # Average Trust Drop
        avg_trust_drop = statistics.mean(r.trust_drops) if r.trust_drops else 0.0
        
        # Average Propagation Length
        avg_propagation_length = statistics.mean(r.propagation_lengths) if r.propagation_lengths else 0

        return {
            "ASR": round(asr, 4),
            "Detection_Rate": round(dr, 4),
            "FPR": round(fpr, 4),
            "Patient_Zero_Accuracy": round(pz_acc, 4),
            "Recovery_Time_Mean_s": round(rt_mean, 3) if rt_mean is not None else None,
            "Recovery_Time_Median_s": round(rt_median, 3) if rt_median is not None else None,
            "Recovery_Time_Std_s": round(rt_std, 3) if rt_std is not None else None,
            "System_Availability": round(avail, 4) if avail is not None else None,
            "Chroma_Collection_Size_Mean": round(chroma_mean, 1),
            "Retrieval_Hits_Mean": round(retrieval_mean, 1),
            "Benign_False_Positives": r.benign_false_positives,
            "Benign_False_Positive_Count": len(r.benign_false_positives),
            "recovery_times": r.recovery_times,
            "patient_zero_breakdown": pz_breakdown_simple,
            "patient_zero_method_comparison": {
                name: {
                    "total": counts["total"],
                    "correct": counts["correct"],
                    "accuracy": counts["correct"] / counts["total"] if counts["total"] else None,
                }
                for name, counts in self.patient_zero_methods.items()
            },
            "roc": roc,
            "roc_operating_point": roc_operating_point,
            "roc_note": "A single operating point is reported; ROC curve/AUC is not calculated.",
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "total_attacks": r.total_attacks,
            "total_benign": r.benign_total,
            # ISSUE 21 FIX: New metrics
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "Specificity": round(specificity, 4),
            "Sensitivity": round(sensitivity, 4),
            "Avg_Detection_Delay": round(avg_detection_delay, 4),
            "Avg_Containment_Time": round(avg_containment_time, 4),
            "Avg_Trust_Drop": round(avg_trust_drop, 4),
            "Avg_Propagation_Length": round(avg_propagation_length, 2),
        }

    def report(self):
        """Print formatted metrics report."""
        metrics = self.compute()
        print("\n=== TrustTrace Evaluation Results ===")
        print(f"  Attack Success Rate (ASR):          {metrics['ASR']:.2%}")
        print(f"  Detection Rate:                     {metrics['Detection_Rate']:.2%}")
        print(f"  False Positive Rate (FPR):          {metrics['FPR']:.2%}")
        print(f"  Patient Zero Accuracy:              {metrics['Patient_Zero_Accuracy']:.2%}")
        mean_recovery = "N/A (no recovery observations)" if metrics["Recovery_Time_Mean_s"] is None else f"{metrics['Recovery_Time_Mean_s']:.2f}s"
        median_recovery = "N/A (no recovery observations)" if metrics["Recovery_Time_Median_s"] is None else f"{metrics['Recovery_Time_Median_s']:.2f}s"
        print(f"  Mean Recovery Time:                 {mean_recovery}")
        print(f"  Median Recovery Time:               {median_recovery}")
        availability = metrics["System_Availability"]
        availability_text = "N/A (no task observations)" if availability is None else f"{availability:.2%}"
        print(f"  System Availability During Recovery:{availability_text}")
        print(f"  Chroma Collection Size (mean):       {metrics['Chroma_Collection_Size_Mean']:.1f}")
        print(f"  Retrieval Hits (mean):              {metrics['Retrieval_Hits_Mean']:.1f}")
        print(f"  Benign False Positives:             {metrics['Benign_False_Positive_Count']}")
        return metrics

    def record_attack_with_source(
        self,
        succeeded: bool,
        detected: bool,
        source: str,
        attack_type: str,
        pz_predicted: str = None,
        pz_ground_truth: str = None,
        recovery_time_s: Optional[float] = None,
        chroma_size: int = 0,
        retrieval_hits: int = 0,
        patient_zero_methods: Optional[dict] = None,
    ):
        """Record attack with source tag and method-wise Patient Zero outcomes."""
        self.record_attack(succeeded, detected)
        if recovery_time_s is not None and recovery_time_s > 0:
            self.results.recovery_times.append(recovery_time_s)
        if chroma_size:
            self.results.chroma_sizes.append(chroma_size)
        if retrieval_hits:
            self.results.retrieval_hits.append(retrieval_hits)
        pz_correct = None
        method_predictions = {}
        if pz_ground_truth:
            # Missing prediction is a failed identification, not an excluded row.
            self.record_patient_zero(pz_predicted, pz_ground_truth)
            pz_correct = bool(pz_predicted == pz_ground_truth)
            method_predictions = patient_zero_methods or {
                "graph_pcs_backward": pz_predicted,
                "earliest_flagged": None,
                "stage1_only": None,
            }
            for method, prediction in method_predictions.items():
                counts = self.patient_zero_methods.setdefault(method, {"total": 0, "correct": 0})
                counts["total"] += 1
                counts["correct"] += int(prediction is not None and prediction == pz_ground_truth)

        record = {
            "succeeded": bool(succeeded),
            "detected": bool(detected),
            "source": source,
            "type": attack_type,
            "patient_zero_correct": pz_correct,
            "patient_zero_predicted": pz_predicted,
            "patient_zero_ground_truth": pz_ground_truth,
            "patient_zero_methods": method_predictions,
            "recovery_time_s": recovery_time_s,
        }
        self.results.raw_records.append(record)
        self._records.append({
            "succeeded": bool(succeeded),
            "detected": bool(detected),
            "source": source,
            "type": attack_type,
            "pz_correct": pz_correct,
        })

    def report_split(self):
        """Print metrics split by handcrafted vs AgentDojo."""
        if not hasattr(self, "_records") or not self._records:
            print("  No split records available.")
            return

        sources = set(r["source"] for r in self._records)

        for source in sorted(sources):
            subset = [r for r in self._records if r["source"] == source]
            total = len(subset)
            detected = sum(1 for r in subset if r["detected"])
            succeeded = sum(1 for r in subset if r["succeeded"])
            pz_records = [r for r in subset if r["pz_correct"] is not None]
            pz_correct = sum(1 for r in pz_records if r["pz_correct"])

            print(f"\n--- {source.upper()} ({total} attacks) ---")
            print(f"  Detection Rate: {detected / total:.1%}")
            print(f"  ASR:            {succeeded / total:.1%}")
            if pz_records:
                print(f"  PZ Accuracy:    {pz_correct / len(pz_records):.1%}")
