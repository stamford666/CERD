import numpy as np

from cerd.metrics import (
    binary_predictions_at_threshold,
    metric_bundle,
    tune_binary_threshold,
)


def test_metric_bundle_perfect_predictions():
    labels = np.array([0, 1, 2, 0, 1, 2])
    probabilities = np.eye(3)[labels] * 0.9 + 0.1 / 3.0
    metrics = metric_bundle(labels, probabilities)
    assert metrics["n"] == 6
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_auroc", "macro_auprc"):
        assert metrics[key] == 1.0


def test_binary_threshold_is_selected_on_validation_and_transferred_unchanged():
    validation_labels = np.array([0, 0, 1, 1])
    validation_probabilities = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]]
    )
    threshold, score = tune_binary_threshold(
        validation_labels,
        validation_probabilities,
    )
    assert threshold == 0.7
    assert score == 1.0

    test_probabilities = np.array([[0.31, 0.69], [0.29, 0.71], [0.05, 0.95]])
    predictions = binary_predictions_at_threshold(test_probabilities, threshold)
    assert predictions.tolist() == [0, 1, 1]


def test_binary_metric_bundle_accepts_validation_threshold_predictions():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array(
        [[0.6, 0.4], [0.55, 0.45], [0.52, 0.48], [0.1, 0.9]]
    )
    predictions = binary_predictions_at_threshold(probabilities, 0.43)
    metrics = metric_bundle(labels, probabilities, predictions)
    assert metrics["positive_auprc"] == metrics["macro_auprc"]
    assert metrics["positive_f1"] == 0.8
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 0.5
