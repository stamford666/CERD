"""Metrics that work for both ADNI multiclass and ABCD binary tasks."""

import os
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


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


def save_binary_prediction_dump(
    method,
    seed,
    val_labels,
    val_probabilities,
    test_labels,
    test_probabilities,
    test_predictions,
    selection_rate,
):
    """Persist one formal ABCD seed when an output directory is requested.

    The opt-in environment variable keeps every baseline's original command
    line and all ADNI behavior unchanged.
    """

    raw_output = os.environ.get("ABCD_PREDICTION_DUMP_DIR")
    if not raw_output:
        return None
    output = Path(raw_output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{str(method).lower()}_seed{int(seed)}.npz"
    np.savez_compressed(
        path,
        method=np.asarray(str(method)),
        seed=np.asarray(int(seed), dtype=np.int64),
        experiment_tag=np.asarray(os.environ.get("ABCD_EXPERIMENT_TAG", "")),
        validation_labels=np.asarray(val_labels, dtype=np.int64),
        validation_probabilities=np.asarray(val_probabilities, dtype=np.float64),
        test_labels=np.asarray(test_labels, dtype=np.int64),
        test_probabilities=np.asarray(test_probabilities, dtype=np.float64),
        test_predictions=np.asarray(test_predictions, dtype=np.int64),
        selection_rate=np.asarray(float(selection_rate), dtype=np.float64),
    )
    print(f"[ABCD prediction dump] {path}")
    return path
