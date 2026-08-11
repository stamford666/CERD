"""Metrics that work for both ADNI multiclass and ABCD binary tasks."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def metric_bundle(labels, probabilities, predictions=None) -> dict[str, float | int]:
    """Return a common metric bundle for both ABCD and ADNI."""

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if predictions is None:
        predictions = probabilities.argmax(axis=1)
    predictions = np.asarray(predictions, dtype=np.int64)
    result = {
        "n": int(labels.size),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "macro_auroc": classification_auc(labels, probabilities),
        "macro_auprc": classification_average_precision(labels, probabilities),
    }
    if probabilities.shape[1] == 2:
        matrix = confusion_matrix(labels, predictions, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        result.update(
            {
                # For a rare binary endpoint, AP is defined for the positive
                # disease class.  Keep ``macro_auprc`` as a compatibility
                # alias while exposing the unambiguous manuscript name.
                "positive_auprc": result["macro_auprc"],
                "positive_f1": float(
                    f1_score(labels, predictions, pos_label=1, zero_division=0)
                ),
                "sensitivity": float(tp / max(tp + fn, 1)),
                "specificity": float(tn / max(tn + fp, 1)),
                "mcc": float(matthews_corrcoef(labels, predictions)),
            }
        )
    return result


def classification_auc(labels, probabilities) -> float:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2:
        raise ValueError(f"Expected a [samples, classes] probability matrix, got {probabilities.shape}")
    if probabilities.shape[1] == 2:
        return float(roc_auc_score(labels, probabilities[:, 1]))
    return float(roc_auc_score(labels, probabilities, multi_class="ovr"))


def classification_average_precision(labels, probabilities) -> float:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if probabilities.shape[1] == 2:
        return float(average_precision_score(labels, probabilities[:, 1]))
    one_hot = np.eye(probabilities.shape[1], dtype=np.float32)[labels]
    return float(average_precision_score(one_hot, probabilities, average="macro"))


def tune_binary_threshold(labels, probabilities) -> tuple[float, float]:
    """Tune on validation data only, maximizing macro-F1 deterministically."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.shape != (labels.size, 2):
        raise ValueError("binary threshold tuning expects labels [N] and probabilities [N, 2]")
    if not np.isfinite(probabilities).all():
        raise ValueError("binary probabilities must be finite")
    positive = probabilities[:, 1]
    candidates = np.unique(np.concatenate(([0.0], positive, [1.0])))
    scores = np.asarray(
        [
            f1_score(
                labels,
                positive >= threshold,
                average="macro",
                zero_division=0,
            )
            for threshold in candidates
        ]
    )
    best = np.flatnonzero(scores == scores.max())
    # Prefer the most conservative threshold when scores tie on rare outcomes.
    index = int(best[-1])
    return float(candidates[index]), float(scores[index])


def binary_predictions_at_threshold(probabilities, threshold: float) -> np.ndarray:
    """Apply a fixed, validation-derived positive-class threshold."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    threshold = float(threshold)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("binary prediction expects a probability matrix with two columns")
    if not np.isfinite(probabilities).all() or not np.isfinite(threshold):
        raise ValueError("binary probabilities and threshold must be finite")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("binary probability threshold must be in [0, 1]")
    return (probabilities[:, 1] >= threshold).astype(np.int64)
