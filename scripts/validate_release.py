#!/usr/bin/env python3
"""Validate aggregate receipts and reject accidental experiment artifacts."""

from __future__ import annotations

import json
import math
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".npy", ".npz", ".parquet", ".h5", ".h5ad", ".csv"
}


def close(actual: float, expected: float, tolerance: float = 5e-10) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"metric mismatch: computed={actual}, stored={expected}")


def check_members(receipt: dict, block: str = "members") -> None:
    members = receipt[block]
    for metric in ("accuracy", "macro_f1", "macro_auroc"):
        values = [float(item["test"][metric]) for item in members]
        close(statistics.mean(values), float(receipt["mean"][metric]))
        close(statistics.stdev(values), float(receipt["sample_sd"][metric]))


def main() -> None:
    adni = json.loads(
        (ROOT / "results/adni_direct_three_seed_mean_v2.json").read_text()
    )
    check_members(adni)
    if adni["aggregation"]["probability_ensemble"]:
        raise AssertionError("ADNI receipt must use direct three-seed means")

    adni_matched = json.loads(
        (ROOT / "results/adni_matched_updated_cerd_v4.json").read_text()
    )
    if adni_matched["aggregation"]["probability_ensemble"]:
        raise AssertionError("matched ADNI receipt must use direct three-seed means")
    for method, metrics in adni_matched["methods"].items():
        for metric in ("accuracy", "macro_f1", "macro_auroc"):
            values = metrics[metric]
            if len(values) != 3 or not all(0.0 <= float(value) <= 1.0 for value in values):
                raise AssertionError(f"invalid matched ADNI values: {method}/{metric}")

    abcd = json.loads(
        (ROOT / "results/abcd_adhd_presentation3_feature_refinement_v5.json").read_text()
    )
    for section in (
        "formal_seed_specific_checkpoint_evaluation",
        "secondary_fixed_epoch_refit",
    ):
        block = abcd[section]
        values = block["members"]
        stored_mean = (
            block["mean"] if "mean" in block else {
                key: block["mean_sample_sd"]["test"][key][0]
                for key in ("accuracy", "macro_f1", "macro_auroc")
            }
        )
        stored_sd = (
            block["sample_sd"] if "sample_sd" in block else {
                key: block["mean_sample_sd"]["test"][key][1]
                for key in ("accuracy", "macro_f1", "macro_auroc")
            }
        )
        for metric in ("accuracy", "macro_f1", "macro_auroc"):
            metric_values = [float(item["test"][metric]) for item in values]
            close(statistics.mean(metric_values), float(stored_mean[metric]))
            close(statistics.stdev(metric_values), float(stored_sd[metric]))

    matched = json.loads(
        (ROOT / "results/abcd_adhd_presentation3_snp_missing15_v4.json").read_text()
    )
    if matched["aggregation"]["probability_ensemble"]:
        raise AssertionError("matched ABCD receipt must use direct three-seed means")
    for method, method_block in matched["results"].items():
        for split in ("validation", "test"):
            aggregate = (
                method_block["mean_sample_sd"][split]
                if "mean_sample_sd" in method_block
                else method_block[f"{split}_mean_sample_sd"]
            )
            for metric in ("accuracy", "macro_f1", "macro_auroc"):
                values = [
                    float(seed_block[split][metric])
                    for seed_block in method_block["seeds"].values()
                ]
                expected_mean, expected_sd = aggregate[metric]
                close(statistics.mean(values), float(expected_mean), tolerance=1e-9)
                close(statistics.stdev(values), float(expected_sd), tolerance=1e-9)

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    forbidden = [path for path in tracked if Path(path).suffix.lower() in FORBIDDEN_SUFFIXES]
    if forbidden:
        raise AssertionError(f"forbidden raw/binary artifacts are tracked: {forbidden}")

    stale_markers = ("62.83", "70.13", "status3_real_generated")
    text_files = [path for path in tracked if Path(path).suffix in {".md", ".json", ".tex"}]
    for relative in text_files:
        content = (ROOT / relative).read_text(errors="replace")
        for marker in stale_markers:
            if marker in content:
                raise AssertionError(f"stale marker {marker!r} found in {relative}")

    print("release validation passed")


if __name__ == "__main__":
    main()
