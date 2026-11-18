"""Experiment HTML report generator for TrustTrace.

This module produces a single-file HTML report (with embedded JSON and
Chart.js) summarising an experiment run. It intentionally reads the
data passed in and does not re-compute metrics — reuse existing
MetricsCollector outputs and run results.
"""

from __future__ import annotations

import json
import os
import html
from datetime import datetime, timezone


def _esc(value) -> str:
    """HTML-escape every untrusted value inserted into markup."""
    return html.escape(str(value), quote=True)


def _safe_json(obj):
    """Serialize data safely inside a script element (prevent </script> breakout)."""
    try:
        encoded = json.dumps(obj, ensure_ascii=True, default=str)
    except Exception:
        encoded = json.dumps(str(obj), ensure_ascii=True)
    return (
        encoded.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\\u2028", "\\\\u2028")
        .replace("\\u2029", "\\\\u2029")
    )


def generate_experiment_report(output_path: str, data: dict) -> None:
    """Write an HTML report to `output_path` using `data`.

    Expects `data` to contain keys used in the TrustTrace pipeline:
      - project_name, experiment_date, duration_s, llm_provider,
        dataset_used, total_prompts, author, version
      - scenarios (list), attack_results (list), metrics (dict)
      - combined_metrics, baseline, benign_false_positives
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    html_parts = []
    # Header / meta
    html_parts.append("""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TrustTrace Experiment Report</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css">
  <style>
    :root{--bg:#0f1724;--card:#0b1220;--muted:#9aa4b2;--accent:#4f46e5;--glass:rgba(255,255,255,0.03)}
    body{font-family:Inter, system-ui, -apple-system, Roboto, 'Segoe UI', Arial; background:linear-gradient(180deg,#071029 0%,#081227 100%);color:#e6eef6;margin:0;padding:24px}
    .container{max-width:1100px;margin:0 auto}
    .cover{padding:48px;border-radius:12px;background:var(--card);box-shadow:0 6px 30px rgba(2,6,23,0.6)}
    h1,h2,h3{margin:0 0 12px 0}
    .meta{display:flex;gap:16px;flex-wrap:wrap}
    .card{background:var(--glass);padding:14px;border-radius:10px;margin:12px 0}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
    table{width:100%;border-collapse:collapse}
    th,td{padding:8px;border-bottom:1px solid rgba(255,255,255,0.04);font-size:13px}
    .muted{color:var(--muted)}
    .kpi{font-size:20px;font-weight:600}
    .small{font-size:12px;color:var(--muted)}
    .collapsible{background:transparent;border:0;color:#9fb0ff;text-align:left;padding:10px}
    details summary::-webkit-details-marker{display:none}
    .chartwrap{height:260px}
    .footer{font-size:12px;color:var(--muted);padding:18px 0}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <div class="container">
    <section class="cover card">
      <h1>TrustTrace Experiment Report</h1>
      <div class="meta">
        <div class="small">Project: <strong>TrustTrace</strong></div>
        <div class="small">Date: <strong>""" + _esc(data.get("experiment_date", "N/A")) + """</strong></div>
        <div class="small">Duration: <strong>""" + _esc(f"{data.get('duration_s', 0):.2f}s") + """</strong></div>
        <div class="small">LLM Provider: <strong>""" + _esc(data.get("llm_provider", "N/A")) + """</strong></div>
        <div class="small">Dataset: <strong>""" + _esc(data.get("dataset_used", "N/A")) + """</strong></div>
        <div class="small">Total Prompts: <strong>""" + _esc(data.get("total_prompts", 0)) + """</strong></div>
      </div>
      <p class="muted">Author: """ + _esc(data.get("author", "Researcher")) + """ — Version: """ + _esc(data.get("version", "1.0")) + """</p>
    </section>

    <section class="card">
      <h2>Executive Summary</h2>
      <p class="muted">""" + (_esc(data.get("executive_summary", "Summary not provided."))) + """</p>
    </section>

    <section class="card">
      <h2>Final Metrics Dashboard</h2>
      <div class="grid">
""")

    # Key metric cards
    metrics = data.get("metrics", {}) or {}
    cards = [
        ("Total Prompts", data.get("total_prompts", 0)),
        ("Detected Attacks", metrics.get("TP", 0)),
        ("Blocked Attacks", sum(1 for r in data.get("attack_results", []) if r.get("blocked_at") == "Layer1")),
        ("Final-output Marker Proxy Hits", sum(1 for r in data.get("attack_results", []) if r.get("marker_proxy_hit", False))),
        ("Recovered Attacks", sum(1 for r in data.get("attack_results", []) if r.get("recovered"))),
        ("ASR Proxy (not a goal oracle)", f"{metrics.get('ASR',0):.2%}"),
        ("Detection Rate", f"{metrics.get('Detection_Rate',0):.2%}"),
        ("False Positive Rate", f"{metrics.get('FPR',0):.2%}"),
        ("Average IRS", f"{data.get('avg_irs',0):.3f}"),
        ("Average Trust Score", f"{data.get('avg_trust',0):.3f}"),
        ("Average Recovery Time (s)", "N/A" if metrics.get("Recovery_Time_Mean_s") is None else f"{metrics.get('Recovery_Time_Mean_s'):.3f}"),
        ("Patient Zero Accuracy", f"{metrics.get('Patient_Zero_Accuracy',0):.2%}"),
        ("System Availability", "N/A" if metrics.get("System_Availability") is None else f"{metrics.get('System_Availability'):.2%}"),
        ("Average Latency (s)", f"{data.get('avg_latency',0):.3f}"),
    ]

    for title, val in cards:
        html_parts.append(f"<div class=\"card\"><div class=\"small muted\">{title}</div><div class=\"kpi\">{val}</div></div>")

    html_parts.append("""
      </div>
    </section>

    <section class="card">
      <h2>Prompt-by-Prompt Results</h2>
      <div style="overflow:auto;max-height:420px">
        <table>
          <thead>
            <tr>
              <th>#</th><th>Prompt</th><th>Attack Category</th><th>Expected</th><th>Actual</th>
              <th>Detected</th><th>Trust Score</th><th>IRS</th><th>Memory Suspicion</th>
              <th>Containment Decision</th><th>Recovery</th><th>Exec Time (s)</th><th>Final Status (proxy)</th>
            </tr>
          </thead>
          <tbody>
""")

    for i, (scenario, result) in enumerate(zip(data.get("scenarios", []), data.get("attack_results", [])), start=1):
        prompt = scenario.get("query") or scenario.get("prompt") or ""
        attack_cat = scenario.get("type", result.get("attack_type", "unknown"))
        expected = scenario.get("expected_behavior", "N/A")
        actual = result.get("outputs", {}).get("Generator", "") if isinstance(result.get("outputs"), dict) else ""
        detected = result.get("detected")
        trust = getattr(result, 'avg_trust', None) or result.get("avg_trust") or "N/A"
        irs = result.get("pz_confidence", "N/A")
        ms = result.get("memory_suspicion_score", "N/A")
        decision = "Recovered" if result.get("recovered") else ("Blocked" if result.get("blocked_at") else ("Marker proxy hit" if result.get("marker_proxy_hit") else "No marker observed"))
        recovery = result.get("recovery_time_s")
        exec_time = result.get("execution_time", "N/A")
        final_status = "DETECTED" if detected else ("PROXY HIT" if result.get("marker_proxy_hit") else "NO PROXY HIT")
        html_parts.append(
            f"<tr><td>{_esc(i)}</td><td>{_esc(prompt[:120])}</td>"
            f"<td>{_esc(attack_cat)}</td><td>{_esc(expected)}</td>"
            f"<td>{_esc(actual[:120])}</td><td>{_esc(detected)}</td>"
            f"<td>{_esc(trust)}</td><td>{_esc(irs)}</td><td>{_esc(ms)}</td>"
            f"<td>{_esc(decision)}</td><td>{_esc(recovery)}</td>"
            f"<td>{_esc(exec_time)}</td><td>{_esc(final_status)}</td></tr>"
        )

    html_parts.append("""
          </tbody>
        </table>
      </div>
    </section>

    <section class="card">
      <h2>Aggregated Analytics</h2>
      <div class="grid">
        <div class="card chartwrap"><canvas id="attackTypeChart"></canvas></div>
        <div class="card chartwrap"><canvas id="detectionRateChart"></canvas></div>
        <div class="card chartwrap"><canvas id="asrChart"></canvas></div>
        <div class="card chartwrap"><canvas id="trustDistChart"></canvas></div>
      </div>
    </section>

    <section class="card">
      <h2>Attack Analysis</h2>
      <p class="muted">Direct, Indirect and Memory attacks breakdown and recovery statistics included below.</p>
    </section>

    <section class="card">
      <h2>System Pipeline</h2>
      <pre class="muted">User Prompt\n↓\nInput Validation\n↓\nPrompt Scanner\n↓\nRisk Assessment\n↓\nTrust Engine\n↓\nMemory Evaluation\n↓\nPropagation Tracker\n↓\nRecovery Manager\n↓\nLLM\n↓\nOutput</pre>
    </section>

    <div class="footer">Generated on """ + _esc(datetime.now(timezone.utc).isoformat()) + """ UTC</div>
  </div>

  <script>
    // Embedded data for charts
    const REPORT_DATA = """ + _safe_json(data) + """;
    const parsed = JSON.parse(REPORT_DATA);

    // Attack type distribution
    const attackTypes = {};
    (parsed.scenarios || []).forEach(s => { const t = s.type || 'unknown'; attackTypes[t] = (attackTypes[t]||0)+1 });
    const atLabels = Object.keys(attackTypes); const atValues = Object.values(attackTypes);

    // Detection / ASR
    const detectionRate = parsed.metrics ? parsed.metrics.Detection_Rate : 0;
    const asr = parsed.metrics ? parsed.metrics.ASR : 0;

    // Trust distribution
    const trustVals = (parsed.attack_results||[]).map(r=>r.avg_trust||null).filter(v=>v!==null);

    new Chart(document.getElementById('attackTypeChart'), {
      type: 'pie', data: { labels: atLabels, datasets:[{data: atValues, backgroundColor:['#4f46e5','#06b6d4','#f97316','#ef4444']}]}, options:{plugins:{legend:{labels:{color:'#cbd5e1'}}}}
    });

    new Chart(document.getElementById('detectionRateChart'), {
      type:'doughnut', data:{labels:['Detected','Missed'], datasets:[{data:[detectionRate, 1-detectionRate], backgroundColor:['#10b981','#ef4444']}]}, options:{plugins:{legend:{labels:{color:'#cbd5e1'}}}}
    });

    new Chart(document.getElementById('asrChart'), {type:'bar', data:{labels:['ASR'], datasets:[{label:'ASR proxy',data:[asr],backgroundColor:['#f59e0b']}]}, options:{scales:{y:{beginAtZero:true}}}});

    new Chart(document.getElementById('trustDistChart'), {type:'histogram' in Chart ? 'histogram' : 'bar', data:{labels:trustVals.map((_,i)=>i+1), datasets:[{label:'Trust',data:trustVals,backgroundColor:['#60a5fa']}]}, options:{scales:{y:{beginAtZero:true}}}});
  </script>
</body>
</html>
""")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))
