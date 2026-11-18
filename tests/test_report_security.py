from reports.report_generator import generate_experiment_report, _safe_json


def test_report_escapes_untrusted_prompt_and_output(tmp_path):
    target = tmp_path / "report.html"
    payload = {
        "project_name": "TrustTrace",
        "experiment_date": "today",
        "duration_s": 1.0,
        "llm_provider": "local",
        "dataset_used": "test",
        "total_prompts": 1,
        "author": "researcher",
        "version": "3",
        "executive_summary": "<script>alert(1)</script>",
        "scenarios": [{"query": "<img src=x onerror=alert(1)>", "type": "direct"}],
        "attack_results": [{
            "outputs": {"Generator": "</td><script>alert(2)</script>"},
            "succeeded": False,
            "detected": True,
        }],
        "metrics": {},
        "combined_metrics": {},
        "baseline": {},
        "benign_false_positives": [],
    }
    generate_experiment_report(str(target), payload)
    rendered = target.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in rendered
    assert "<img src=x onerror=alert(1)>" not in rendered
    assert "</td><script>alert(2)</script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_embedded_json_cannot_close_script_element():
    encoded = _safe_json({"payload": "</script><script>alert(1)</script>"})
    assert "</script>" not in encoded.lower()
