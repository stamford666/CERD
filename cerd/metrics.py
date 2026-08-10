"""Metrics that work for both ADNI multiclass and ABCD binary tasks."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def metric_bundle(labels, probabilities) -> dict[str, float | int]:
    """Return the six metrics used for both ABCD and ADNI."""

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    return {
        "n": int(labels.size),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "macro_auroc": classification_auc(labels, probabilities),
        "macro_auprc": classification_average_precision(labels, probabilities),
    }


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
    labels = np.asarray(labels)
    positive = np.asarray(probabilities)[:, 1]
    candidates = np.unique(np.concatenate(([0.0], positive, [1.0])))
    scores = np.asarray([f1_score(labels, positive >= threshold, average="macro") for threshold in candidates])
    best = np.flatnonzero(scores == scores.max())
    # Prefer the most conservative threshold when scores tie on rare outcomes.
    index = int(best[-1])
    return float(candidates[index]), float(scores[index])


def calibrated_binary_predictions(val_labels, val_probabilities, test_probabilities):
    """Calibrate a positive selection rate on validation, then rank test cases.

    Rare-event probability scales can shift even when ranking remains stable.
    A validation-selected rate is consequently more robust than transferring an
    absolute probability threshold. No test labels are used.
    """
    labels = np.asarray(val_labels)
    val_scores = np.asarray(val_probabilities)[:, 1]
    test_scores = np.asarray(test_probabilities)[:, 1]
    prevalence = max(float(labels.mean()), 1.0 / max(len(labels), 1))
    multipliers = np.asarray([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    rates = np.clip(prevalence * multipliers, 1.0 / len(labels), 0.20)

    def top_rate(scores, rate):
        count = max(1, int(round(len(scores) * float(rate))))
        selected = np.argpartition(scores, -count)[-count:]
        result = np.zeros(len(scores), dtype=np.int64)
        result[selected] = 1
        return result

    validation_scores = np.asarray([
        f1_score(labels, top_rate(val_scores, rate), average="macro") for rate in rates
    ])
    best = np.flatnonzero(validation_scores == validation_scores.max())
    index = int(best[0])
    rate = float(rates[index])
    return top_rate(test_scores, rate), rate, float(validation_scores[index])
