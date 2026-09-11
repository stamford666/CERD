#!/usr/bin/env python3
"""Summarize binary ABCD baseline metrics as direct three-seed means."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODELS = ("flex_moe", "i2moe", "moepp_corrected", "anymod", "agdic", "acadiff")
SEEDS = (31, 32, 33)
METRICS = ("accuracy", "macro_f1", "macro_auroc")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    output = {
        "schema": "abcd-current-binary-baselines-v1",
        "aggregation": "arithmetic mean and sample standard deviation across seeds 31/32/33; no probability ensemble",
        "methods": {},
    }
    for model in MODELS:
        payloads = [
            json.loads((root / "formal" / model / "abcd" / f"{model}_seed{seed}.json").read_text())
            for seed in SEEDS
        ]
        records = [
            {"seed": seed, **{metric: 100.0 * float(payload["test"][metric]) for metric in METRICS}}
            for seed, payload in zip(SEEDS, payloads)
        ]
        output["methods"][model] = {
            "seed_metrics": records,
            "aggregate": {
                metric: {
                    "mean": float(np.mean([record[metric] for record in records])),
                    "sd": float(np.std([record[metric] for record in records], ddof=1)),
                }
                for metric in METRICS
            },
            "test_subsets": {
                subset: {
                    metric: {
                        "mean": float(
                            np.mean(
                                [
                                    100.0
                                    * float(payload["test_subsets"][subset][metric])
                                    for payload in payloads
                                ]
                            )
                        ),
                        "sd": float(
                            np.std(
                                [
                                    100.0
                                    * float(payload["test_subsets"][subset][metric])
                                    for payload in payloads
                                ],
                                ddof=1,
                            )
                        ),
                    }
                    for metric in METRICS
                }
                for subset in ("complete", "missing")
            },
            "test_subset_sizes": {
                subset: int(payloads[0]["test_subsets"][subset]["n"])
                for subset in ("complete", "missing")
            },
        }
    (root / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    for model, block in output["methods"].items():
        print(model, *(f'{block["aggregate"][metric]["mean"]:.2f}±{block["aggregate"][metric]["sd"]:.2f}' for metric in METRICS))


if __name__ == "__main__":
    main()
