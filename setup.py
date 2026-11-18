#!/usr/bin/env python3
"""
TrustTrace Setup & Initialization Script

This script handles:
1. Verifying environment setup
2. Downloading Phase 1 datasets (deepset)
3. Training the injection scanner
4. Running quick calibration test
"""

import os
import sys
import json
from pathlib import Path

def check_environment():
    """Verify Python environment is properly configured."""
    print("=" * 60)
    print("🔍 ENVIRONMENT CHECK")
    print("=" * 60)
    
    version_ok = sys.version_info[:2] == (3, 11)
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    print(f"{'✓' if version_ok else '✗'} Python version: {sys.version.split()[0]} (Python 3.11 required)")
    print(f"✓ Working directory: {os.getcwd()}")
    print(f"{'✓' if in_venv else '!'} Virtual environment active: {in_venv} (recommended)")
    return version_ok


def check_dependencies():
    """Verify required packages are installed."""
    print("\n" + "=" * 60)
    print("📦 DEPENDENCY CHECK")
    print("=" * 60)
    
    required_packages = [
        "chromadb",
        "networkx",
        "sentence_transformers",
        "sklearn",
        "optuna",
        "datasets",
        "openai",
        "numpy",
        "pandas",
        "yaml",
        "pyarrow",
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (MISSING)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed!")
    return True


def check_project_structure():
    """Verify project folder structure."""
    print("\n" + "=" * 60)
    print("📁 PROJECT STRUCTURE CHECK")
    print("=" * 60)
    
    required_dirs = [
        "victim_pipeline",
        "logger",
        "scanner",
        "graph",
        "drift",
        "trust",
        "memory",
        "detector",
        "recovery",
        "calibration",
        "attacks",
        "eval",
        "tuning",
        "data",
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ (MISSING)")
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"\n⚠️  Missing directories: {', '.join(missing_dirs)}")
        return False
    
    print("\n✓ Project structure complete!")
    return True


def download_deepset_dataset():
    """Verify the bundled train/test splits or use the canonical atomic downloader."""
    print("\n" + "=" * 60)
    print("📥 VERIFYING DEEPSET DATASET")
    print("=" * 60)

    train_path = Path("data/deepset_injections/train.json")
    test_path = Path("data/deepset_injections/test.json")
    if train_path.exists() and test_path.exists():
        try:
            train_rows = json.loads(train_path.read_text(encoding="utf-8"))
            test_rows = json.loads(test_path.read_text(encoding="utf-8"))
            if not isinstance(train_rows, list) or not train_rows or not isinstance(test_rows, list) or not test_rows:
                raise ValueError("train/test splits must be non-empty JSON arrays")
            for split_name, rows in (("train", train_rows), ("test", test_rows)):
                for index, row in enumerate(rows):
                    if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip() or "label" not in row:
                        raise ValueError(f"Invalid {split_name} row {index}")
            print(f"✓ Dataset verified: {len(train_rows)} train / {len(test_rows)} test samples")
            return True
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"✗ Existing dataset is invalid: {exc}")
            return False

    try:
        from data.download_deepset import main as download_main
        download_main()
        return train_path.is_file() and test_path.is_file()
    except Exception as exc:
        print(f"✗ Download failed: {exc}")
        return False


def verify_core_modules():
    """Verify core TrustTrace modules load correctly."""
    print("\n" + "=" * 60)
    print("✓ CORE MODULE VERIFICATION")
    print("=" * 60)
    
    modules = {
        "Victim Pipeline": "victim_pipeline.agents",
        "Logger": "logger.interaction_logger",
        "Scanner": "scanner.injection_scanner",
        "Graph": "graph.propagation_graph",
        "Drift": "drift.behavioral_drift",
        "Trust Engine": "trust.trust_engine",
        "Memory Manager": "memory.memory_manager",
        "Patient Zero": "detector.patient_zero",
        "Recovery": "recovery.recovery_manager",
        "Calibration": "calibration.calibrate",
        "Metrics": "eval.metrics",
        "Bayesian Search": "tuning.bayesian_search",
    }
    
    failed = []
    for name, module_path in modules.items():
        try:
            __import__(module_path)
            print(f"✓ {name}")
        except Exception as e:
            print(f"✗ {name}: {str(e)[:50]}")
            failed.append((name, e))
    
    if failed:
        print(f"\n⚠️  {len(failed)} modules failed to import")
        return False
    
    print("\n✓ All core modules imported successfully!")
    return True


def main():
    """Run all setup checks and initialization."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  TRUSTTRACE SETUP & INITIALIZATION SCRIPT  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = {
        "Environment": check_environment(),
        "Dependencies": check_dependencies(),
        "Structure": check_project_structure(),
        "Modules": verify_core_modules(),
        "Dataset": download_deepset_dataset(),
    }
    
    print("\n" + "=" * 60)
    print("📊 SETUP SUMMARY")
    print("=" * 60)
    
    for check, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ SETUP COMPLETE - Ready to run TrustTrace!")
        print("\nNext steps:")
        print("1. Train scanner: python -c \"from scanner.injection_scanner import InjectionScanner; s=InjectionScanner(); s.train()\"")
        print("2. Run calibration: python -c \"from calibration.calibrate import run_calibration; ...\"")
        print("3. Run pipeline: python main.py")
    else:
        print("✗ SETUP INCOMPLETE - Fix errors above before proceeding")
        return 1
    
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
