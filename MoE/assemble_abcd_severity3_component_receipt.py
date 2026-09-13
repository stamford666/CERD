#!/usr/bin/env python3
"""Assemble matched ABCD severity3 component controls for the final CERD."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-results", type=Path, required=True)
    parser.add_argument("--matched-ablations", type=Path, required=True)
    parser.add_argument("--reliability-ablation", type=Path)
    parser.add_argument("--capacity-controls", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    main_results = json.loads(args.main_results.read_text())
    matched = json.loads(args.matched_ablations.read_text())
    capacity = json.loads(args.capacity_controls.read_text())

    full_source = main_results["methods"]["our_moe"]
    training_by_seed = {
        int(row["seed"]): row for row in full_source["training"]
    }
    full = {
        "seed_metrics": [
            {
                **row,
                "best_epoch": int(training_by_seed[int(row["seed"])]["best_epoch"]),
                "epochs_completed": int(
                    training_by_seed[int(row["seed"])]["epochs_completed"]
                ),
            }
            for row in full_source["seed_metrics"]
        ],
        "aggregate": full_source["aggregate"],
        "test_subsets": full_source["test_subsets"],
    }

    arms = {
        "full": full,
        **{
            name: block
            for name, block in matched["arms"].items()
            if name != "uniform_weights"
        },
    }
    if "no_reliability" not in arms:
        if args.reliability_ablation is None:
            raise ValueError(
                "matched ablations lack no_reliability; provide --reliability-ablation"
            )
        reliability = json.loads(args.reliability_ablation.read_text())
        arms["no_reliability"] = {
            key: reliability[key]
            for key in ("seed_metrics", "aggregate", "test_subsets")
        }
    for name in ("dense_full", "single_expert"):
        arms[name] = capacity["arms"][name]

    receipt = {
        "schema": "abcd-severity3-e4k2-component-suite-v2",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "dataset_manifest_sha256": hashlib.sha256(
            args.manifest.read_bytes()
        ).hexdigest(),
        "aggregation": (
            "arithmetic mean and sample standard deviation across seeds 31/32/33; "
            "no ensemble"
        ),
        "selection": (
            "best validation Macro-F1 after 100 complete epochs; test evaluated once"
        ),
        "model_configuration": {
            "full_and_one_factor_ablations": "four experts with top-2 routing",
            "no_reliability": "all usable modality reliabilities fixed to one; confidence and branch priors retained",
            "dense_full": "MoE feed-forward block replaced by a capacity-matched dense FFN",
            "single_expert": "one expert with top-1 routing",
        },
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
