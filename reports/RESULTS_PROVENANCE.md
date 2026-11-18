# Results provenance

No experiment result in this source package is certified as publication-ready. Historical legacy run outputs were intentionally excluded from the clean source deliverable because they contained incomplete/error records and could be confused with current results.

Generate fresh outputs with `scripts/run_ablation_matrix.py` or `scripts/run_deepset_full_benchmark.py`, under `TRUSTTRACE_STRICT_EVAL=1`, using the held-out test split and a recorded model/configuration. Quick runs are diagnostics only. The deepset benchmark now binds output rows to the dataset, provider/model and configuration hashes and rejects incompatible resume attempts.
