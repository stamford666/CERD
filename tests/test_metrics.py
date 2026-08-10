import numpy as np

from cerd.metrics import metric_bundle


def test_metric_bundle_perfect_predictions():
    labels = np.array([0, 1, 2, 0, 1, 2])
    probabilities = np.eye(3)[labels] * 0.9 + 0.1 / 3.0
    metrics = metric_bundle(labels, probabilities)
    assert metrics["n"] == 6
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_auroc", "macro_auprc"):
        assert metrics[key] == 1.0
