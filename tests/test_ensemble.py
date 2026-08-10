import numpy as np

from cerd.ensemble import hierarchical_median_consensus, seed_median_consensus


def test_seed_median_is_normalized_and_robust_to_one_outlier():
    probabilities = np.array(
        [
            [[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]],
            [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        ]
    )
    result = seed_median_consensus(probabilities)
    np.testing.assert_allclose(result.sum(axis=1), 1.0)
    assert result.argmax(axis=1).tolist() == [0, 1]


def test_hierarchical_median_shape():
    values = np.full((4, 3, 5, 3), 1.0 / 3.0)
    result = hierarchical_median_consensus(values)
    assert result.shape == (5, 3)
    np.testing.assert_allclose(result, 1.0 / 3.0)
