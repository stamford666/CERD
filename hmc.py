#!/usr/bin/env python3
"""Combine frozen CERD probability files with hierarchical median consensus."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cerd.ensemble import hierarchical_median_consensus
from cerd.metrics import metric_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        action="append",
        required=True,
        help="Comma-separated NPZ files for one family; repeat for each family.",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    families = []
    reference_labels = None
    for family_spec in args.family:
        seeds = []
        for raw_path in family_spec.split(","):
            path = Path(raw_path).expanduser().resolve()
            with np.load(path, allow_pickle=False) as artifact:
                labels = np.asarray(artifact["labels"], dtype=np.int64)
                probabilities = np.asarray(artifact["probabilities"], dtype=np.float64)
            if reference_labels is None:
                reference_labels = labels
            elif not np.array_equal(labels, reference_labels):
                raise ValueError(f"label/order mismatch: {path}")
            seeds.append(probabilities)
        families.append(np.stack(seeds))
    values = np.stack(families)
    consensus = hierarchical_median_consensus(values)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, labels=reference_labels, probabilities=consensus)
    print(metric_bundle(reference_labels, consensus))


if __name__ == "__main__":
    main()
