"""Run a split-safe TrustTrace evaluation on deepset/prompt-injections.

By default this evaluates ONLY the 116-example held-out test split. The
runtime's scanner is trained from train.json by bootstrap_system; the test
examples are not concatenated into the training set here.

A fresh run overwrites the split-specific output, preventing accidental mixing
with stale/failed attempts. Use --resume only to continue the same run.
Set TRUSTTRACE_STRICT_EVAL=1 to forbid mock LLM responses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


DATA_DIR = Path("data/deepset_injections")
REPORT_DIR = Path("reports")


def _normalize_label(raw):
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"benign", "legit", "legitimate", "0", "false", "safe"}:
            return False
        if value in {"attack", "malicious", "injection", "1", "true", "unsafe"}:
            return True
        raise ValueError(f"Unrecognised dataset label: {raw!r}")
    return bool(int(raw))


def load_items(split: str):
    """Load exactly one official split, preserving its local index."""
    path = DATA_DIR / f"{split}.json"
    with path.open(encoding="utf-8") as handle:
        raw_items = json.load(handle)
    items = []
    for source_index, item in enumerate(raw_items):
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid/missing text in {path}, row {source_index}")
        items.append({
            "index": source_index,
            "text": text,
            "ground_truth_attack": _normalize_label(item.get("label")),
        })
    return items


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_results(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    """95% Wilson score interval; returns proportions, not percentages."""
    if total == 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def compute_summary(results_path: Path, summary_path: Path, split: str, expected_n: int, partial: bool):
    rows = _read_results(results_path)
    source_items = load_items(split)
    if len(source_items) != expected_n:
        raise ValueError(
            f"Dataset size changed during evaluation: expected {expected_n}, now found {len(source_items)}."
        )
    expected_labels = {item["index"]: item["ground_truth_attack"] for item in source_items}
    dataset_path = DATA_DIR / f"{split}.json"
    dataset_sha256 = _sha256_file(dataset_path)
    indices = [row.get("index") for row in rows]
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise ValueError("Every result row must contain an integer prompt index.")
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate prompt indices found; do not report this run.")
    invalid_indices = [index for index in indices if index not in expected_labels]
    if invalid_indices:
        raise ValueError(f"Result contains indices outside the official {split} split: {invalid_indices[:10]}")
    for row in rows:
        if row.get("split") != split:
            raise ValueError(f"Result row split mismatch: expected {split!r}, got {row.get('split')!r}.")
        if row.get("dataset_sha256") != dataset_sha256:
            raise ValueError("Result row dataset hash does not match the current official split; refusing to score mixed/stale data.")
        index = row["index"]
        if not isinstance(row.get("ground_truth_attack"), bool):
            raise ValueError(f"Result row {index} has a non-boolean ground_truth_attack value.")
        if row["ground_truth_attack"] != expected_labels[index]:
            raise ValueError(f"Ground-truth label mismatch for {split} index {index}; refusing to score tampered results.")
        if not row.get("error") and not isinstance(row.get("detected"), bool):
            raise ValueError(f"Result row {index} has a non-boolean detected value.")

    errors = [row for row in rows if row.get("error")]
    good = [row for row in rows if not row.get("error")]
    if errors:
        raise RuntimeError(
            f"Run contains {len(errors)} errored prompts. Summary was not accepted as final. "
            f"Inspect {results_path}, fix the cause, and rerun into a fresh output file."
        )
    if not partial and set(indices) != set(range(expected_n)):
        missing = sorted(set(range(expected_n)) - set(indices))
        raise RuntimeError(
            f"Full evaluation does not cover the complete {split} split; missing indices: {missing[:10]}"
        )
    if not partial and len(good) != expected_n:
        raise RuntimeError(f"Expected {expected_n} evaluated prompts for split={split}, got {len(good)}.")
    if not good:
        raise RuntimeError("No successful prompt records; cannot calculate metrics.")

    tp = tn = fp = fn = 0
    layer_tp = {}
    for row in good:
        gt = bool(row["ground_truth_attack"])
        detected = bool(row["detected"])
        if gt and detected:
            tp += 1
            layer = row.get("blocked_layer")
            if layer:
                layer_tp[layer] = layer_tp.get(layer, 0) + 1
        elif gt and not detected:
            fn += 1
        elif not gt and detected:
            fp += 1
        else:
            tn += 1

    n = tp + tn + fp + fn
    precision_den = tp + fp
    recall_den = tp + fn
    fpr_den = fp + tn
    accuracy = (tp + tn) / n
    precision = tp / precision_den if precision_den else 0.0
    recall = tp / recall_den if recall_den else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / fpr_den if fpr_den else 0.0

    providers = sorted({str(row.get("provider", "")) for row in good})
    models = sorted({str(row.get("model", "")) for row in good})
    config_hashes = sorted({str(row.get("config_sha256", "")) for row in good})
    if len(providers) != 1 or len(models) != 1 or len(config_hashes) != 1 or not config_hashes[0]:
        raise ValueError("Results must use exactly one recorded provider, model and config hash.")

    summary = {
        "split": split,
        "dataset_sha256": dataset_sha256,
        "provider": providers[0],
        "model": models[0],
        "config_sha256": config_hashes[0],
        "partial_run": bool(partial),
        "n_expected_in_split": expected_n,
        "n_evaluated": n,
        "n_errors": 0,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": round(accuracy, 6),
        "accuracy_95_ci_wilson": wilson_interval(tp + tn, n),
        "precision": round(precision, 6),
        "precision_95_ci_wilson": wilson_interval(tp, precision_den),
        "recall": round(recall, 6),
        "recall_95_ci_wilson": wilson_interval(tp, recall_den),
        "f1": round(f1, 6),
        "false_positive_rate": round(fpr, 6),
        "false_positive_rate_95_ci_wilson": wilson_interval(fp, fpr_den),
        "true_positive_blocked_layer_counts": layer_tp,
        "true_positive_layer_counts_sum": sum(layer_tp.values()),
        "goal_based_attack_success_rate": None,
        "goal_based_attack_success_rate_note": (
            "Not calculated on this prompt-detection benchmark. Use explicit scenario success predicates "
            "in the multi-agent episode experiments; do not substitute 1 - recall."
        ),
    }
    if sum(layer_tp.values()) > tp:
        raise AssertionError("True-positive interception counts cannot exceed TP.")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with temporary_summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    os.replace(temporary_summary, summary_path)
    print(json.dumps(summary, indent=2))
    print(f"Summary written to {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "test"), default="test",
                        help="Evaluate one split only; final headline results must use test.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an existing output file. Do not use with a new protocol/config.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run a partial smoke test only; partial results cannot be reported as final.")
    parser.add_argument("--output", default=None, help="Optional JSONL output path.")
    parser.add_argument("--summary", default=None, help="Optional summary JSON path.")
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.split == "train":
        print("WARNING: train split selected; treat results as diagnostic only, not held-out performance.")

    os.environ.setdefault("TRUSTTRACE_STRICT_EVAL", "1")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else REPORT_DIR / f"deepset_{args.split}_results.jsonl"
    summary_path = Path(args.summary) if args.summary else REPORT_DIR / f"deepset_{args.split}_summary.json"
    items = load_items(args.split)
    expected_n = len(items)
    partial = args.limit is not None and args.limit < expected_n
    if args.limit is not None:
        items = items[:args.limit]

    existing_rows = []
    if args.resume:
        if not output_path.exists():
            raise FileNotFoundError(f"Cannot resume: {output_path} does not exist.")
        existing_rows = _read_results(output_path)
        if any(row.get("split") != args.split for row in existing_rows):
            raise ValueError("Existing output contains a different split; choose a fresh --output path.")
        done = {row["index"] for row in existing_rows}
        print(f"Resuming {output_path}: {len(done)} prompt indices already present.")
        mode = "a"
    else:
        # Fresh runs never append to stale records from prior attempts.
        mode = "w"
        done = set()

    print(f"Loading {args.split} split: {expected_n} examples; this run targets {len(items)}.")
    # Import runtime only after argument/data validation so split checks can run
    # in lightweight environments without ChromaDB or an LLM backend installed.
    from main import bootstrap_system, _load_config
    runtime = bootstrap_system(_load_config(), force_calibrate=False)
    llm_cfg = (_load_config() or {}).get("llm", {})
    provider = str(llm_cfg.get("provider", "unknown"))
    model = str(llm_cfg.get("model", "unknown"))
    config_path = Path("config.yaml")
    config_sha256 = _sha256_file(config_path)
    dataset_sha256 = _sha256_file(DATA_DIR / f"{args.split}.json")
    if args.resume:
        for row in existing_rows:
            if row.get("provider") != provider or row.get("model") != model:
                raise ValueError("Cannot resume with a different LLM provider/model; choose a fresh output file.")
            if row.get("config_sha256") != config_sha256:
                raise ValueError("Cannot resume with a different config.yaml; choose a fresh output file.")
            if row.get("dataset_sha256") != dataset_sha256:
                raise ValueError("Cannot resume with a changed dataset split; choose a fresh output file.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    processed_this_run = 0
    with output_path.open(mode, encoding="utf-8") as results_file:
        for item in items:
            i = item["index"]
            if i in done:
                continue
            runtime.reset_state()
            try:
                report = runtime.execute(item["text"], auto_detect=True)
                # In strict evidence mode, reject mock markers wherever visible.
                serialised = json.dumps(report, default=str)
                if os.environ.get("TRUSTTRACE_STRICT_EVAL") == "1" and "[mock:" in serialised.lower():
                    raise RuntimeError("Mock output detected in report; result rejected.")
                row = {
                    "split": args.split,
                    "index": i,
                    "ground_truth_attack": item["ground_truth_attack"],
                    "detected": bool(report.get("detected")),
                    "attack_type": report.get("attack_type"),
                    "blocked_layer": report.get("blocked_layer"),
                    "patient_zero": report.get("patient_zero"),
                    "text_preview": item["text"][:120],
                    "provider": provider,
                    "model": model,
                    "config_sha256": config_sha256,
                    "dataset_sha256": dataset_sha256,
                }
            except Exception as exc:
                row = {
                    "split": args.split,
                    "index": i,
                    "ground_truth_attack": item["ground_truth_attack"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "text_preview": item["text"][:120],
                    "provider": provider,
                    "model": model,
                    "config_sha256": config_sha256,
                    "dataset_sha256": dataset_sha256,
                }
            results_file.write(json.dumps(row) + "\n")
            results_file.flush()
            processed_this_run += 1
            if processed_this_run % 10 == 0:
                elapsed = time.perf_counter() - start_time
                rate = processed_this_run / elapsed if elapsed else 0.0
                remaining = len(items) - len(done) - processed_this_run
                print(f"[{processed_this_run}] {rate:.2f} prompts/s; approx remaining {max(0, remaining)}")

    # A failed/partial run must not be presented as final results.
    compute_summary(output_path, summary_path, args.split, expected_n, partial)


if __name__ == "__main__":
    main()
