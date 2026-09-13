#!/usr/bin/env python3
"""Replace CERD in a matched aggregate receipt from frozen formal evaluations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


METRICS = ("accuracy", "macro_f1", "macro_auroc")
SEEDS = (31, 32, 33)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument("--cerd-formal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schema", required=True)
    args = parser.parse_args()

    result = json.loads(args.base_results.read_text())
    payloads = [
        json.loads(
            (args.cerd_formal_root / "abcd" / f"our_moe_seed{seed}.json").read_text()
        )
        for seed in SEEDS
    ]
    seed_metrics = []
    for seed, payload in zip(SEEDS, payloads):
        row = {"seed": seed}
        row.update(
            {metric: 100.0 * float(payload["test"][metric]) for metric in METRICS}
        )
        seed_metrics.append(row)
    result["schema"] = args.schema
    result["completed_at"] = datetime.now(timezone.utc).isoformat()
    result["methods"]["our_moe"] = {
        "seed_metrics": seed_metrics,
        "aggregate": {
            metric: {
                "mean": float(np.mean([row[metric] for row in seed_metrics])),
                "sd": float(np.std([row[metric] for row in seed_metrics], ddof=1)),
            }
            for metric in METRICS
        },
        "training": [
            {
                "seed": seed,
                "epochs_completed": int(payload["training"]["epochs_completed"]),
                "best_epoch": int(payload["training"]["best_epoch"]),
            }
            for seed, payload in zip(SEEDS, payloads)
        ],
        "test_subsets": {
            subset: {
                "n": int(payloads[0]["test_subsets"][subset]["n"]),
                **{
                    metric: {
                        "mean": float(
                            np.mean(
                                [
                                    100.0 * float(p["test_subsets"][subset][metric])
                                    for p in payloads
                                ]
                            )
                        ),
                        "sd": float(
                            np.std(
                                [
                                    100.0 * float(p["test_subsets"][subset][metric])
                                    for p in payloads
                                ],
                                ddof=1,
                            )
                        ),
                    }
                    for metric in METRICS
                },
            }
            for subset in ("complete", "missing")
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
