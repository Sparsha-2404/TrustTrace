"""Download the real deepset/prompt-injections dataset.

No synthetic fallback is written: a fabricated or single-class dataset would
produce invalid scanner training and contaminate reported research results.
"""
from __future__ import annotations

import json
import os


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required to download deepset/prompt-injections. "
            "Install project requirements, then retry. No synthetic substitute was created."
        ) from exc

    print("Downloading deepset/prompt-injections from Hugging Face...")
    dataset = load_dataset("deepset/prompt-injections")
    if "train" not in dataset or "test" not in dataset:
        raise RuntimeError("Downloaded dataset does not contain both train and test splits.")

    output_dir = os.path.join("data", "deepset_injections")
    os.makedirs(output_dir, exist_ok=True)
    for split_name in ("train", "test"):
        rows = list(dataset[split_name])
        if not rows:
            raise RuntimeError(f"Downloaded dataset split {split_name!r} is empty.")
        target = os.path.join(output_dir, f"{split_name}.json")
        temporary = target + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, target)
        print(f"Saved {len(rows)} {split_name} samples to {target}")


if __name__ == "__main__":
    main()
