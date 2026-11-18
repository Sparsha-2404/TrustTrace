import importlib.util
import json
from pathlib import Path


def _load_setup_module():
    path = Path(__file__).resolve().parents[1] / "setup.py"
    spec = importlib.util.spec_from_file_location("trusttrace_setup_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_setup_structure_check_does_not_require_generated_logs_directory(tmp_path, monkeypatch):
    setup = _load_setup_module()
    required = [
        "victim_pipeline", "logger", "scanner", "graph", "drift", "trust",
        "memory", "detector", "recovery", "calibration", "attacks", "eval",
        "tuning", "data",
    ]
    for directory in required:
        (tmp_path / directory).mkdir()
    monkeypatch.chdir(tmp_path)
    assert setup.check_project_structure() is True


def test_setup_dataset_check_validates_both_bundled_splits(tmp_path, monkeypatch):
    setup = _load_setup_module()
    data_dir = tmp_path / "data" / "deepset_injections"
    data_dir.mkdir(parents=True)
    valid = json.dumps([{"text": "sample", "label": 0}])
    (data_dir / "train.json").write_text(valid, encoding="utf-8")
    (data_dir / "test.json").write_text(valid, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert setup.download_deepset_dataset() is True
    (data_dir / "test.json").write_text("[]", encoding="utf-8")
    assert setup.download_deepset_dataset() is False
