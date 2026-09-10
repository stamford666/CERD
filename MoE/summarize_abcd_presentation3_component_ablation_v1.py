#!/usr/bin/env python3
"""Aggregate the matched presentation3 component ablations without ensembling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARMS = (
    ("Full CERD", None),
    ("Dense backbone", "dense_backbone"),
    ("w/o multigranular decomposition", "no_multigranular_decomposition"),
    ("Joint branch only", "joint_branch_only"),
    ("w/o conditional completion", "no_completion"),
    ("w/o provenance encoding", "no_provenance"),
    ("Uniform branch weights", "uniform_branch_weights"),
)
METRICS = ("accuracy", "macro_f1", "macro_auroc")
SEEDS = (31, 32, 33)
HERE = Path(__file__).resolve().parent


def source_files(root: Path, arm: str | None) -> list[Path]:
    if arm is None:
        base = HERE / "abcd_adhd_presentation3_snp_missing15_formal_v4" / "abcd"
    else:
        base = root / arm / "formal" / "abcd"
    return [base / f"our_moe_seed{seed}.json" for seed in SEEDS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    records = []
    for display, arm in ARMS:
        files = source_files(root, arm)
        payloads = [json.loads(path.read_text()) for path in files]
        values = {metric: [100.0 * float(payload["test"][metric]) for payload in payloads] for metric in METRICS}
        records.append(
            {
                "name": display,
                "arm": arm or "full",
                "source_files": [str(path) for path in files],
                "seed_metrics": [
                    {"seed": seed, **{metric: values[metric][index] for metric in METRICS}}
                    for index, seed in enumerate(SEEDS)
                ],
                "aggregate": {
                    metric: {
                        "mean": float(np.mean(series)),
                        "sd": float(np.std(series, ddof=1)),
                    }
                    for metric, series in values.items()
                },
            }
        )
    output = {
        "schema": "abcd-presentation3-component-ablation-v1",
        "endpoint": "low-symptom control vs predominantly inattentive ADHD vs ADHD with hyperactive/impulsive component",
        "aggregation": "arithmetic mean and sample standard deviation across seeds 31/32/33; no probability ensemble",
        "selection": "each arm independently selected by validation Macro-F1; test evaluated once after strict validation replay",
        "records": records,
    }
    (root / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    lines = [
        "# ABCD presentation3 matched component ablation",
        "",
        output["aggregation"] + ". " + output["selection"] + ".",
        "",
        "| Configuration | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |",
        "|---|---:|---:|---:|",
    ]
    for record in records:
        cells = []
        for metric in METRICS:
            item = record["aggregate"][metric]
            cells.append(f'{item["mean"]:.2f} ± {item["sd"]:.2f}')
        lines.append(f'| {record["name"]} | ' + " | ".join(cells) + " |")
    (root / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
