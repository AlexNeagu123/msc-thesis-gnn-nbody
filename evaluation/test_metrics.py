"""Tests for evaluation/metrics.py."""

import numpy as np

from evaluation.metrics import compute_rollout_mse, min_pairwise_distances


def test_min_pairwise_distances_supports_four_particles() -> None:
    """Minimum distance works for N > 3."""
    positions = np.array(
        [
            [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [10.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 3.0]],
        ]
    )

    distances = min_pairwise_distances(positions)

    assert np.allclose(distances, [3.0, 1.0])


def test_compute_rollout_mse_tracks_nonfinite_rollouts() -> None:
    """Non-finite rollouts are excluded from means but counted."""
    true = np.zeros((2, 3, 4, 5))
    predicted = np.zeros_like(true)
    predicted[1, 2, 0, 0] = np.nan

    mse = compute_rollout_mse(true, predicted)

    assert mse.per_trajectory.shape == (2, 3)
    assert np.isnan(mse.per_trajectory[1, 2])
    assert mse.mean[2] == 0.0
    assert mse.median[2] == 0.0
    assert mse.finite_fraction[2] == 0.5
