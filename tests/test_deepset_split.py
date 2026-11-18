"""Regression tests ensuring the benchmark loader never concatenates splits."""
from scripts.run_deepset_full_benchmark import load_items


def test_train_and_test_are_loaded_as_separate_official_splits():
    train = load_items("train")
    test = load_items("test")
    assert len(train) == 546
    assert len(test) == 116
    assert [row["index"] for row in train] == list(range(546))
    assert [row["index"] for row in test] == list(range(116))
    assert all(isinstance(row["ground_truth_attack"], bool) for row in train + test)


def _write_result(path, index, label, detected=False, dataset_sha256=None):
    import hashlib
    import json
    if dataset_sha256 is None:
        dataset_sha256 = hashlib.sha256((path.parent / "data" / "test.json").read_bytes()).hexdigest()
    path.write_text(
        json.dumps({
            "split": "test",
            "index": index,
            "ground_truth_attack": label,
            "detected": detected,
            "provider": "lm_studio",
            "model": "test-model",
            "config_sha256": "test-config-hash",
            "dataset_sha256": dataset_sha256,
        }) + "\n",
        encoding="utf-8",
    )


def test_summary_rejects_ground_truth_label_tampering(tmp_path, monkeypatch):
    import json
    import pytest
    import scripts.run_deepset_full_benchmark as benchmark

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "test.json").write_text(
        json.dumps([{"text": "benign", "label": 0}]), encoding="utf-8"
    )
    monkeypatch.setattr(benchmark, "DATA_DIR", data_dir)
    results = tmp_path / "results.jsonl"
    _write_result(results, 0, True)

    with pytest.raises(ValueError, match="Ground-truth label mismatch"):
        benchmark.compute_summary(results, tmp_path / "nested" / "summary.json", "test", 1, False)


def test_summary_rejects_out_of_range_prompt_indices(tmp_path, monkeypatch):
    import json
    import pytest
    import scripts.run_deepset_full_benchmark as benchmark

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "test.json").write_text(
        json.dumps([{"text": "benign", "label": 0}]), encoding="utf-8"
    )
    monkeypatch.setattr(benchmark, "DATA_DIR", data_dir)
    results = tmp_path / "results.jsonl"
    _write_result(results, 7, False)

    with pytest.raises(ValueError, match="outside the official"):
        benchmark.compute_summary(results, tmp_path / "summary.json", "test", 1, True)


def test_full_summary_requires_complete_official_index_coverage(tmp_path, monkeypatch):
    import json
    import pytest
    import scripts.run_deepset_full_benchmark as benchmark

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "test.json").write_text(
        json.dumps([
            {"text": "benign one", "label": 0},
            {"text": "attack two", "label": 1},
        ]), encoding="utf-8"
    )
    monkeypatch.setattr(benchmark, "DATA_DIR", data_dir)
    results = tmp_path / "results.jsonl"
    _write_result(results, 0, False)

    with pytest.raises(RuntimeError, match="complete test split"):
        benchmark.compute_summary(results, tmp_path / "summary.json", "test", 2, False)


def test_summary_accepts_complete_hash_bound_split_and_writes_atomically(tmp_path, monkeypatch):
    import json
    import scripts.run_deepset_full_benchmark as benchmark

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dataset_file = data_dir / "test.json"
    dataset_file.write_text(
        json.dumps([
            {"text": "benign", "label": 0},
            {"text": "attack", "label": 1},
        ]), encoding="utf-8"
    )
    monkeypatch.setattr(benchmark, "DATA_DIR", data_dir)
    results = tmp_path / "results.jsonl"
    _write_result(results, 0, False, detected=False)
    with results.open("a", encoding="utf-8") as handle:
        import hashlib
        handle.write(json.dumps({
            "split": "test",
            "index": 1,
            "ground_truth_attack": True,
            "detected": True,
            "provider": "lm_studio",
            "model": "test-model",
            "config_sha256": "test-config-hash",
            "dataset_sha256": hashlib.sha256(dataset_file.read_bytes()).hexdigest(),
        }) + "\n")

    summary_path = tmp_path / "nested" / "summary.json"
    summary = benchmark.compute_summary(results, summary_path, "test", 2, False)
    assert summary["n_evaluated"] == 2
    assert summary["TP"] == 1
    assert summary["TN"] == 1
    assert summary_path.is_file()
    assert not summary_path.with_suffix(".json.tmp").exists()
