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

    assert mse.state.per_trajectory.shape == (2, 3)
    assert np.isnan(mse.state.per_trajectory[1, 2])
    assert np.isnan(mse.position.per_trajectory[1, 2])
    assert mse.state.mean[2] == 0.0
    assert mse.state.median[2] == 0.0
    assert mse.state.finite_fraction[2] == 0.5


def test_compute_rollout_mse_splits_position_and_velocity() -> None:
    """Rollout MSE exposes position-only and velocity-only errors."""
    true = np.zeros((1, 2, 1, 5))
    predicted = np.zeros_like(true)
    predicted[0, 1, 0, 0] = 2.0
    predicted[0, 1, 0, 2] = 4.0

    mse = compute_rollout_mse(true, predicted)

    assert mse.position.per_trajectory[0, 1] == 2.0
    assert mse.velocity.per_trajectory[0, 1] == 8.0
    assert mse.state.per_trajectory[0, 1] == 5.0
