#!/usr/bin/env python3
"""Aggregate matched ABCD current-ADHD binary component controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARMS = (
    ("Full CERD", None),
    ("Dense FFN", "dense_backbone"),
    ("Single expert (E=1)", "single_expert"),
    ("w/o multigranular decomposition", "no_multigranular_decomposition"),
    ("w/o conditional completion", "no_completion"),
    ("w/o provenance encoding", "no_provenance"),
    ("Uniform branch weights", "uniform_branch_weights"),
)
METRICS = ("accuracy", "macro_f1", "macro_auroc")
SEEDS = (31, 32, 33)
HERE = Path(__file__).resolve().parent


def source_files(root: Path, arm: str | None) -> list[Path]:
    if arm is None:
        base = HERE / "abcd_current_binary_cerd_v1" / "formal" / "abcd"
        return [base / f"our_moe_seed{seed}.json" for seed in SEEDS]
    else:
        base = root / arm / "formal" / "abcd"
        files = []
        for seed in SEEDS:
            matches = sorted(base.glob(f"our_moe_seed{seed}*.json"))
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected one formal result for {arm} seed {seed}, found {matches}"
                )
            files.append(matches[0])
        return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    records = []
    for display, arm in ARMS:
        files = source_files(root, arm)
        payloads = [json.loads(path.read_text()) for path in files]
        values = {
            metric: [100.0 * float(payload["test"][metric]) for payload in payloads]
            for metric in METRICS
        }
        records.append(
            {
                "name": display,
                "arm": arm or "full",
                "seed_metrics": [
                    {
                        "seed": seed,
                        **{
                            metric: values[metric][index]
                            for metric in METRICS
                        },
                    }
                    for index, seed in enumerate(SEEDS)
                ],
                "aggregate": {
                    metric: {
                        "mean": float(np.mean(series)),
                        "sd": float(np.std(series, ddof=1)),
                    }
                    for metric, series in values.items()
                },
                "missing_subset": {
                    metric: {
                        "mean": float(
                            np.mean(
                                [
                                    100.0
                                    * float(payload["test_subsets"]["missing"][metric])
                                    for payload in payloads
                                ]
                            )
                        ),
                        "sd": float(
                            np.std(
                                [
                                    100.0
                                    * float(payload["test_subsets"]["missing"][metric])
                                    for payload in payloads
                                ],
                                ddof=1,
                            )
                        ),
                    }
                    for metric in METRICS
                },
                "missing_subset_n": int(
                    payloads[0]["test_subsets"]["missing"]["n"]
                ),
            }
        )
    output = {
        "schema": "abcd-current-binary-component-ablation-v1",
        "endpoint": "strict low-symptom non-ADHD control vs current full ADHD",
        "aggregation": "arithmetic mean and sample standard deviation across seeds 31/32/33; no probability ensemble",
        "selection": "the full configuration was selected on validation only; every control changes only its named component, selects its checkpoint by the same validation rule, and is evaluated once on test after strict replay",
        "records": records,
    }
    (root / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    lines = [
        "# ABCD current-ADHD binary matched component controls",
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
    lines.extend(
        [
            "",
            "## Originally incomplete test participants (n=45)",
            "",
            "| Configuration | Accuracy (%) | Macro-F1 (%) | Macro-AUROC (%) |",
            "|---|---:|---:|---:|",
        ]
    )
    for record in records:
        cells = []
        for metric in METRICS:
            item = record["missing_subset"][metric]
            cells.append(f'{item["mean"]:.2f} ± {item["sd"]:.2f}')
        lines.append(f'| {record["name"]} | ' + " | ".join(cells) + " |")
    (root / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
