"""
TrustTrace HTML Report Generator
=================================

Generates a single self-contained HTML file after a run (interactive or
experiment) containing:

  1. A confusion-matrix heatmap (TP / FP / FN / TN) with Precision,
     Recall, F1, and Accuracy computed automatically.
  2. Per-agent trust-score sparklines showing how each agent's trust
     evolved over the course of the run (from RunContext.trust_history).
  3. A compact run summary (attack type, detection, patient zero,
     propagation chain, recovery).

No external libraries or internet access are required — everything is
plain HTML + inline SVG + inline CSS, so the file opens directly in any
browser on the machine TrustTrace is running on.

Usage
-----
    from eval_html_report import generate_html_report

    html_path = generate_html_report(report, out_dir="reports")
    print(f"Report written to {html_path}")

`report` is the dict returned by `TrustTraceRuntime.execute(...)` /
`TrustTraceRuntime._finalize_report(...)`. The generator is defensive
about missing keys so it works whether the report came from interactive
mode, validation mode, or a single experiment run.
"""

from __future__ import annotations

import html
import os
import time
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
# Colour palette (kept inline so the file has zero external dependencies)
# --------------------------------------------------------------------------- #
_COLORS = {
    "bg": "#0f1117",
    "card": "#181b24",
    "border": "#2a2e3a",
    "text": "#e6e8ee",
    "muted": "#9aa1b1",
    "good": "#3ddc84",     # TP / TN — correct
    "bad": "#ff5c5c",      # FN — missed attack (worst case)
    "warn": "#ffb84d",     # FP — false alarm
    "accent": "#5b8cff",
    "trust_high": "#3ddc84",
    "trust_mid": "#ffb84d",
    "trust_low": "#ff5c5c",
}


def _trust_color(value: float) -> str:
    if value >= 0.7:
        return _COLORS["trust_high"]
    if value >= 0.4:
        return _COLORS["trust_mid"]
    return _COLORS["trust_low"]


def _safe(text) -> str:
    return html.escape(str(text)) if text is not None else "—"


# --------------------------------------------------------------------------- #
# Confusion matrix heatmap
# --------------------------------------------------------------------------- #
def _confusion_matrix_svg(tp: int, fp: int, fn: int, tn: int) -> str:
    """Render a 2x2 confusion-matrix heatmap as inline SVG.

    Layout:
                 Predicted: Attack   Predicted: Benign
    Actual: Attack     TP                  FN
    Actual: Benign     FP                  TN
    """
    total = max(1, tp + fp + fn + tn)
    cells = [
        ("TP", tp, _COLORS["good"], 0, 0),
        ("FN", fn, _COLORS["bad"], 1, 0),
        ("FP", fp, _COLORS["warn"], 0, 1),
        ("TN", tn, _COLORS["good"], 1, 1),
    ]

    cell_w, cell_h = 160, 110
    label_pad = 110
    width = label_pad + cell_w * 2 + 20
    height = label_pad + cell_h * 2 + 20

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;max-width:480px;height:auto;font-family:Segoe UI,Arial,sans-serif;">'
    ]

    # Column headers
    svg_parts.append(
        f'<text x="{label_pad + cell_w/2}" y="22" text-anchor="middle" '
        f'fill="{_COLORS["muted"]}" font-size="13" font-weight="600">Predicted: Attack</text>'
    )
    svg_parts.append(
        f'<text x="{label_pad + cell_w*1.5}" y="22" text-anchor="middle" '
        f'fill="{_COLORS["muted"]}" font-size="13" font-weight="600">Predicted: Benign</text>'
    )

    # Row headers
    svg_parts.append(
        f'<text x="10" y="{label_pad + cell_h/2}" fill="{_COLORS["muted"]}" '
        f'font-size="13" font-weight="600">Actual:</text>'
    )
    svg_parts.append(
        f'<text x="10" y="{label_pad + cell_h/2 + 16}" fill="{_COLORS["muted"]}" '
        f'font-size="13" font-weight="600">Attack</text>'
    )
    svg_parts.append(
        f'<text x="10" y="{label_pad + cell_h*1.5}" fill="{_COLORS["muted"]}" '
        f'font-size="13" font-weight="600">Actual:</text>'
    )
    svg_parts.append(
        f'<text x="10" y="{label_pad + cell_h*1.5 + 16}" fill="{_COLORS["muted"]}" '
        f'font-size="13" font-weight="600">Benign</text>'
    )

    for label, value, color, col, row in cells:
        x = label_pad + col * cell_w
        y = 30 + row * cell_h
        pct = value / total * 100
        # Opacity scales with share of total so heavier cells read "hotter"
        opacity = 0.25 + 0.65 * min(1.0, value / total * 2)
        svg_parts.append(
            f'<rect x="{x}" y="{y}" width="{cell_w - 8}" height="{cell_h - 8}" '
            f'rx="10" fill="{color}" fill-opacity="{opacity:.2f}" '
            f'stroke="{color}" stroke-width="1.5"/>'
        )
        svg_parts.append(
            f'<text x="{x + (cell_w-8)/2}" y="{y + (cell_h-8)/2 - 4}" '
            f'text-anchor="middle" fill="{_COLORS["text"]}" font-size="26" '
            f'font-weight="700">{value}</text>'
        )
        svg_parts.append(
            f'<text x="{x + (cell_w-8)/2}" y="{y + (cell_h-8)/2 + 18}" '
            f'text-anchor="middle" fill="{_COLORS["muted"]}" font-size="12">'
            f'{label} · {pct:.1f}%</text>'
        )

    svg_parts.append("</svg>")
    return "".join(svg_parts)


def _confusion_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "fpr": fpr,
    }


def _metric_chip(label: str, value: float) -> str:
    pct = value * 100
    return (
        f'<div class="chip">'
        f'<div class="chip-label">{_safe(label)}</div>'
        f'<div class="chip-value">{pct:.1f}%</div>'
        f"</div>"
    )


# --------------------------------------------------------------------------- #
# Trust sparklines
# --------------------------------------------------------------------------- #
def _sparkline_svg(agent: str, values: List[float], delta_threshold: float = 0.4) -> str:
    """Render a single agent's trust trajectory as an SVG line chart."""
    width, height = 480, 90
    pad_l, pad_r, pad_t, pad_b = 8, 8, 14, 18

    if not values:
        values = [1.0]

    n = len(values)
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def x_at(i: int) -> float:
        return pad_l + (i / max(1, n - 1)) * plot_w

    def y_at(v: float) -> float:
        v = max(0.0, min(1.0, v))
        return pad_t + (1.0 - v) * plot_h

    points = " ".join(f"{x_at(i):.1f},{y_at(v):.1f}" for i, v in enumerate(values))
    final_value = values[-1]
    line_color = _trust_color(final_value)

    # Threshold line (delta) — anything dipping below this is "compromised"
    thresh_y = y_at(delta_threshold)

    dots = []
    for i, v in enumerate(values):
        dots.append(
            f'<circle cx="{x_at(i):.1f}" cy="{y_at(v):.1f}" r="3" '
            f'fill="{_trust_color(v)}" />'
        )

    svg = (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;max-width:480px;height:auto;font-family:Segoe UI,Arial,sans-serif;">'
        f'<line x1="{pad_l}" y1="{thresh_y:.1f}" x2="{width - pad_r}" y2="{thresh_y:.1f}" '
        f'stroke="{_COLORS["muted"]}" stroke-width="1" stroke-dasharray="4,4" opacity="0.6"/>'
        f'<text x="{width - pad_r}" y="{thresh_y - 4:.1f}" text-anchor="end" '
        f'fill="{_COLORS["muted"]}" font-size="10">δ = {delta_threshold:.2f}</text>'
        f'<polyline points="{points}" fill="none" stroke="{line_color}" '
        f'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
        + "".join(dots)
        + f'<text x="{pad_l}" y="{height - 4}" fill="{_COLORS["muted"]}" font-size="11">start</text>'
        f'<text x="{width - pad_r}" y="{height - 4}" text-anchor="end" '
        f'fill="{line_color}" font-size="11" font-weight="600">{final_value:.2f}</text>'
        f"</svg>"
    )
    return svg


# --------------------------------------------------------------------------- #
# Top-level report assembly
# --------------------------------------------------------------------------- #
def _propagation_chain_str(prop_chain: List) -> str:
    if not prop_chain:
        return "—"
    parts = []
    for e in prop_chain:
        if isinstance(e, dict):
            parts.append(e.get("name") or e.get("agent") or e.get("label") or str(e))
        else:
            parts.append(str(e))
    return " → ".join(parts)


def generate_html_report(
    report: dict,
    delta_threshold: float = 0.4,
    out_dir: str = "reports",
    filename: Optional[str] = None,
) -> str:
    """Build the HTML report file and return its path.

    `report` is the dict returned by TrustTraceRuntime.execute(). Missing
    keys are tolerated — this works for interactive, validation, and
    experiment-mode reports alike.
    """
    tp = int(report.get("TP", 0) or 0)
    tn = int(report.get("TN", 0) or 0)
    fp = int(report.get("FP", 0) or 0)
    fn = int(report.get("FN", 0) or 0)

    metrics = _confusion_metrics(tp, fp, fn, tn)
    cm_svg = _confusion_matrix_svg(tp, fp, fn, tn)

    trust_history: Dict[str, List[float]] = report.get("trust_history") or {}
    sparkline_blocks = []
    # Stable agent order if present in the pipeline convention
    preferred_order = ["Retriever", "Planner", "Executor", "Generator", "MemoryStore"]
    ordered_agents = [a for a in preferred_order if a in trust_history]
    ordered_agents += [a for a in trust_history if a not in ordered_agents]

    for agent in ordered_agents:
        values = trust_history[agent]
        final_val = values[-1] if values else 1.0
        status = "Compromised" if final_val < delta_threshold else "Healthy"
        status_color = _trust_color(final_val)
        sparkline_blocks.append(
            f'<div class="spark-card">'
            f'<div class="spark-head">'
            f'<span class="spark-agent">{_safe(agent)}</span>'
            f'<span class="spark-status" style="color:{status_color}">{status}</span>'
            f"</div>"
            f"{_sparkline_svg(agent, values, delta_threshold)}"
            f"</div>"
        )

    if not sparkline_blocks:
        sparkline_blocks.append(
            '<div class="muted-note">No per-agent trust history was recorded for this run.</div>'
        )

    attack_type = (report.get("classified_attack_type") or report.get("attack_type") or "benign")
    detected = bool(report.get("detected", False))
    attack_succeeded = bool(report.get("attack_succeeded", False))
    blocked_layer = report.get("blocked_layer") or "N/A"
    patient_zero = report.get("patient_zero") or "N/A"
    chain_str = _propagation_chain_str(report.get("propagation_chain", []))
    recovery_time = report.get("recovery_time_s")
    recovered = recovery_time is not None
    final_output = report.get("final_output", "")
    attack_confidence = report.get("attack_confidence", 0.0)
    attack_severity = report.get("attack_severity") or "N/A"
    evidence = report.get("evidence") or {}
    pz_accuracy = report.get("Patient_Zero_Accuracy")
    pz_accuracy_line = (
        f'<div class="kv"><span class="label">Patient Zero Accuracy</span>{pz_accuracy * 100:.1f}%</div>'
        if pz_accuracy is not None
        else ""
    )
    
    # Additional fields for enhanced report
    detection_time = report.get("detection_time_s")
    classified_irs = report.get("classified_irs", 0.0)
    classified_ms = report.get("classified_ms", 0.0)
    classified_reason = report.get("classified_reason", "N/A")
    containment_actions = report.get("containment_actions", [])
    recovery_actions = report.get("recovery_actions", [])
    compromised = report.get("compromised", [])
    final_trust_scores = report.get("final_trust_scores", {})
    asr = report.get("asr", 0.0)
    detection_rate = report.get("detection_rate", 0.0)
    fpr = report.get("fpr", 0.0)

    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TrustTrace Security Report</title>
<style>
  :root {{
    --bg: {_COLORS["bg"]};
    --card: {_COLORS["card"]};
    --border: {_COLORS["border"]};
    --text: {_COLORS["text"]};
    --muted: {_COLORS["muted"]};
    --accent: {_COLORS["accent"]};
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 32px 24px 60px;
    background: var(--bg);
    color: var(--text);
    font-family: "Segoe UI", Arial, sans-serif;
  }}
  .wrap {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .subtitle {{ color: var(--muted); font-size: 13px; margin-bottom: 28px; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
  }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
  }}
  .card h2 {{
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
    margin: 0 0 14px;
  }}
  .status-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }}
  .badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
  }}
  .badge.detect-yes {{ background: rgba(255,92,92,0.15); color: {_COLORS["bad"]}; }}
  .badge.detect-no {{ background: rgba(61,220,132,0.15); color: {_COLORS["good"]}; }}
  .badge.succ-yes {{ background: rgba(255,92,92,0.15); color: {_COLORS["bad"]}; }}
  .badge.succ-no {{ background: rgba(61,220,132,0.15); color: {_COLORS["good"]}; }}
  .kv {{ font-size: 14px; margin: 6px 0; color: var(--text); }}
  .kv span.label {{ color: var(--muted); display: inline-block; min-width: 120px; }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }}
  .chip {{
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 8px 12px;
    min-width: 90px;
    text-align: center;
  }}
  .chip-label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.03em; }}
  .chip-value {{ font-size: 18px; font-weight: 700; margin-top: 2px; }}
  .section-title {{
    font-size: 15px;
    font-weight: 600;
    margin: 36px 0 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}
  .spark-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 16px;
  }}
  .spark-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px 16px 8px;
  }}
  .spark-head {{
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    margin-bottom: 4px;
  }}
  .spark-agent {{ font-weight: 600; }}
  .spark-status {{ font-weight: 600; }}
  .muted-note {{ color: var(--muted); font-size: 13px; }}
  .cm-wrap {{ display: flex; flex-wrap: wrap; gap: 24px; align-items: center; }}
  .final-output {{
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 13px;
    white-space: pre-wrap;
    max-height: 220px;
    overflow-y: auto;
    color: var(--text);
  }}
  .table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .table th {{ text-align: left; padding: 10px; border-bottom: 1px solid var(--border); color: var(--muted); font-weight: 600; }}
  .table td {{ padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.05); }}
  .table tr:last-child td {{ border-bottom: none; }}
  .metric-value {{ font-weight: 700; }}
  .timeline-item {{ padding: 8px 0; border-left: 2px solid var(--border); padding-left: 12px; margin-bottom: 8px; }}
  .timeline-item::before {{ content: ''; display: inline-block; width: 8px; height: 8px; background: var(--accent); border-radius: 50%; margin-left: -17px; margin-right: 8px; }}
  footer {{ margin-top: 40px; color: var(--muted); font-size: 12px; text-align: center; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>TrustTrace Security Report</h1>
  <div class="subtitle">Generated {generated_at}</div>

  <div class="grid">
    <div class="card">
      <h2>Detection Result</h2>
      <div class="status-row">
        <span class="badge {'detect-yes' if detected else 'detect-no'}">
          {'DETECTED' if detected else 'NOT DETECTED'}
        </span>
        <span class="badge {'succ-yes' if attack_succeeded else 'succ-no'}">
          {'ATTACK SUCCEEDED' if attack_succeeded else 'ATTACK BLOCKED'}
        </span>
      </div>
      <div class="kv"><span class="label">Attack Type</span>{_safe(str(attack_type).upper())}</div>
      <div class="kv"><span class="label">Attack Confidence</span>{attack_confidence:.2f}</div>
      <div class="kv"><span class="label">Attack Severity</span>{_safe(attack_severity)}</div>
      <div class="kv"><span class="label">Blocked Layer</span>{_safe(blocked_layer)}</div>
      <div class="kv"><span class="label">Patient Zero</span>{_safe(patient_zero)}</div>
      <div class="kv"><span class="label">Detection Time</span>{f'{detection_time:.3f}s' if detection_time else 'N/A'}</div>
      <div class="kv"><span class="label">Recovery</span>{'Triggered' if recovered else 'Not triggered'}{f' ({recovery_time:.3f}s)' if recovery_time else ''}</div>
      <div class="kv"><span class="label">Detection Reason</span>{_safe(classified_reason)}</div>
    </div>

    <div class="card">
      <h2>Attack Classification</h2>
      <div class="kv"><span class="label">IRS Score</span>{classified_irs:.3f}</div>
      <div class="kv"><span class="label">MS Score</span>{classified_ms:.3f}</div>
      {pz_accuracy_line}
    </div>

    <div class="card">
      <h2>Detection Evidence</h2>
      <div class="kv"><span class="label">IRS</span>{evidence.get('irs', 0):.3f}</div>
      <div class="kv"><span class="label">Intent Confidence</span>{evidence.get('intent_confidence', 0):.3f}</div>
      <div class="kv"><span class="label">Propagation Risk</span>{evidence.get('propagation_risk', 0):.3f}</div>
      <div class="kv"><span class="label">Behavioral Drift</span>{evidence.get('behavioral_drift', 0):.3f}</div>
    </div>

    <div class="card">
      <h2>Propagation Chain</h2>
      <div class="kv" style="font-size:13px;line-height:1.6;">{_safe(chain_str)}</div>
    </div>
  </div>

  {f'<div class="section-title">Compromised Components</div><div class="card"><div class="kv">{", ".join(_safe(c) for c in compromised) if compromised else "None"}</div></div>' if compromised else ''}

  {f'<div class="section-title">Containment Actions</div><div class="card"><div class="kv">{", ".join(_safe(a) for a in containment_actions) if containment_actions else "None"}</div></div>' if containment_actions else ''}

  {f'<div class="section-title">Recovery Actions</div><div class="card"><div class="kv">{", ".join(_safe(a) for a in recovery_actions) if recovery_actions else "None"}</div></div>' if recovery_actions else ''}

  {f'<div class="section-title">Final Trust Scores</div><div class="card"><table class="table"><tr><th>Agent</th><th>Trust Score</th></tr>' + ''.join(f'<tr><td>{_safe(agent)}</td><td class="metric-value">{score:.3f}</td></tr>' for agent, score in final_trust_scores.items() if agent in ["Retriever", "Planner", "Executor", "Generator"]) + '</table></div>' if final_trust_scores else ''}

  {f'<div class="section-title">Session Metrics</div><div class="card"><div class="kv"><span class="label">ASR</span>{asr:.2%}</div><div class="kv"><span class="label">Detection Rate</span>{detection_rate:.2%}</div><div class="kv"><span class="label">FPR</span>{fpr:.2%}</div></div>' if attack_type != "benign" else ''}

  <div class="section-title">Confusion Matrix (cumulative, this session)</div>
  <div class="card">
    <div class="cm-wrap">
      <div>{cm_svg}</div>
      <div class="chips">
        {_metric_chip("Precision", metrics["precision"])}
        {_metric_chip("Recall", metrics["recall"])}
        {_metric_chip("F1 Score", metrics["f1"])}
        {_metric_chip("Accuracy", metrics["accuracy"])}
        {_metric_chip("FPR", metrics["fpr"])}
      </div>
    </div>
  </div>

  <div class="section-title">Per-Agent Trust Trajectory</div>
  <div class="spark-grid">
    {''.join(sparkline_blocks)}
  </div>

  {f'<div class="section-title">Final Response</div><div class="card"><div class="final-output">{_safe(final_output)}</div></div>' if final_output else ''}

  <footer>TrustTrace — Real-Time Trust Propagation Monitoring &amp; Attack Recovery</footer>
</div>
</body>
</html>"""

    os.makedirs(out_dir, exist_ok=True)
    if filename is None:
        filename = f"trusttrace_report_{int(time.time())}.html"
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return out_path
