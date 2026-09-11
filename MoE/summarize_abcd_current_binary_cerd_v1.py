#!/usr/bin/env python3
"""Select a binary CERD configuration from validation and summarize test once available."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("accuracy", "macro_f1", "macro_auroc")
SEEDS = (31, 32, 33)


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "sd": float(array.std(ddof=1))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    candidates = []
    for directory in sorted((root / "validation").iterdir()):
        payloads = [
            json.loads((directory / "abcd" / f"our_moe_seed{seed}.json").read_text())
            for seed in SEEDS
        ]
        aggregate = {
            metric: stats([100.0 * float(payload["validation"][metric]) for payload in payloads])
            for metric in METRICS
        }
        candidates.append({"name": directory.name, "validation": aggregate})
    candidates.sort(
        key=lambda item: (
            item["validation"]["macro_f1"]["mean"],
            item["validation"]["accuracy"]["mean"],
            item["validation"]["macro_auroc"]["mean"],
        ),
        reverse=True,
    )
    selected = candidates[0]["name"]
    selection = {
        "schema": "abcd-current-binary-cerd-selection-v1",
        "selection_scope": "validation only",
        "rule": "highest three-seed arithmetic mean validation Macro-F1; then Accuracy and Macro-AUROC",
        "selected_candidate": selected,
        "candidates": candidates,
    }
    (root / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print("selected", selected)
    for row in candidates:
        print(
            row["name"],
            *(f'{row["validation"][metric]["mean"]:.2f}' for metric in METRICS),
        )

    formal_root = root / "formal" / "abcd"
    formal_files = [formal_root / f"our_moe_seed{seed}.json" for seed in SEEDS]
    if not all(path.exists() for path in formal_files):
        return
    payloads = [json.loads(path.read_text()) for path in formal_files]
    output = {
        **selection,
        "aggregation": "arithmetic mean and sample standard deviation across independently trained seeds; no ensemble",
        "seed_metrics": [
            {
                "seed": seed,
                "best_epoch": int(payload["training"]["best_epoch"]),
                **{metric: 100.0 * float(payload["test"][metric]) for metric in METRICS},
            }
            for seed, payload in zip(SEEDS, payloads)
        ],
    }
    output["test"] = {
        metric: stats([record[metric] for record in output["seed_metrics"]])
        for metric in METRICS
    }
    output["test_subsets"] = {
        subset: {
            metric: stats(
                [
                    100.0 * float(payload["test_subsets"][subset][metric])
                    for payload in payloads
                ]
            )
            for metric in METRICS
        }
        for subset in ("complete", "missing")
    }
    output["test_subset_sizes"] = {
        subset: int(payloads[0]["test_subsets"][subset]["n"])
        for subset in ("complete", "missing")
    }
    (root / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    print("test", *(f'{output["test"][metric]["mean"]:.2f}±{output["test"][metric]["sd"]:.2f}' for metric in METRICS))


if __name__ == "__main__":
    main()
