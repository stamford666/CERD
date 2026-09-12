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


def main() -> None:
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

    current = json.loads(
        (ROOT / "results/abcd_current3_n3000_missing15_formal_v1.json").read_text()
    )
    if current.get("schema") != "abcd-current3-n3000-missing15-full-rerun-v1":
        raise AssertionError("unexpected current ABCD result schema")
    if "no ensemble" not in current.get("aggregation", ""):
        raise AssertionError("current ABCD receipt must use direct three-seed means")
    if set(current["methods"]) != {
        "our_moe", "flex_moe", "i2moe", "moepp_corrected",
        "anymod", "agdic", "acadiff",
    }:
        raise AssertionError("current ABCD receipt does not contain all seven methods")
    for method, block in current["methods"].items():
        for metric in ("accuracy", "macro_f1", "macro_auroc"):
            values = [float(item[metric]) for item in block["seed_metrics"]]
            close(statistics.mean(values), float(block["aggregate"][metric]["mean"]))
            close(statistics.stdev(values), float(block["aggregate"][metric]["sd"]))
        if any(int(row["epochs_completed"]) != 100 for row in block["training"]):
            raise AssertionError(f"incomplete 100-epoch run: {method}")

    protocol = json.loads(
        (ROOT / "results/abcd_current3_n3000_missing15_protocol_v1.json").read_text()
    )
    if protocol.get("status") != "complete":
        raise AssertionError("current ABCD campaign is not complete")
    if protocol["training_only_augmentation"].get("class_weight_power") != 0.75:
        raise AssertionError("unexpected current ABCD class-weight rule")

    modality = json.loads(
        (ROOT / "results/cerd_three_seed_modality_audit_v2.json").read_text()
    )
    if modality.get("schema") != "cerd-three-seed-modality-audit-v2":
        raise AssertionError("unexpected modality-audit schema")
    for dataset in ("adni", "abcd"):
        relevance = sum(
            float(row["decision_change_relevance_percent"]["mean"])
            for row in modality[dataset]["modalities"]
        )
        drop_share = sum(
            float(row["accuracy_drop_share_percent"]["mean"])
            for row in modality[dataset]["modalities"]
        )
        close(relevance, 100.0, tolerance=1e-8)
        close(drop_share, 100.0, tolerance=1e-8)
        if modality[dataset]["faithfulness_correspondence"]["pearson_r"] < 0.9:
            raise AssertionError(f"weak modality correspondence: {dataset}")

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    forbidden = [path for path in tracked if Path(path).suffix.lower() in FORBIDDEN_SUFFIXES]
    if forbidden:
        raise AssertionError(f"forbidden raw/binary artifacts are tracked: {forbidden}")

    stale_markers = ("62.83", "70.13", "status3_real_generated")
    text_files = [
        path
        for path in tracked
        if (ROOT / path).is_file() and Path(path).suffix in {".md", ".json", ".tex"}
    ]
    for relative in text_files:
        content = (ROOT / relative).read_text(errors="replace")
        for marker in stale_markers:
            if marker in content:
                raise AssertionError(f"stale marker {marker!r} found in {relative}")

    print("release validation passed")


if __name__ == "__main__":
    main()
