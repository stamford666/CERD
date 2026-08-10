"""Parameter-free probability aggregation for CERD model families."""

from __future__ import annotations

import numpy as np


def hierarchical_median_consensus(probabilities: np.ndarray) -> np.ndarray:
    """Aggregate ``[families, seeds, samples, classes]`` probabilities.

    The frozen protocol takes a coordinate-wise median across seeds, then a
    coordinate-wise median across families, and performs exactly one final
    row-wise simplex normalization.
    """

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError("expected [families, seeds, samples, classes]")
    if not np.isfinite(values).all() or bool((values < 0).any()):
        raise ValueError("probabilities must be finite and non-negative")
    family_medians = np.median(values, axis=1)
    consensus = np.median(family_medians, axis=0)
    row_sum = consensus.sum(axis=1, keepdims=True)
    if bool((row_sum <= 0).any()):
        raise ValueError("every consensus row must have positive mass")
    return consensus / row_sum


def seed_median_consensus(probabilities: np.ndarray) -> np.ndarray:
    """Coordinate-wise median across seeds followed by one normalization."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("expected [seeds, samples, classes]")
    return hierarchical_median_consensus(values[None, ...])
