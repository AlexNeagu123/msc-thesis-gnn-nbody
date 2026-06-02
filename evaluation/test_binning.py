"""Tests for evaluation/_binning.py."""

import numpy as np
import pytest

from evaluation._binning import (
    expand_trajectory_mask_to_transitions,
    trajectory_masks,
)


def test_trajectory_masks_basic_correctness() -> None:
    """Each mask selects the trajectories with the matching bin id."""
    bin_id = np.array([0, 1, 0, 2, 1])
    masks = trajectory_masks(bin_id, n_bins=3)

    assert len(masks) == 3
    np.testing.assert_array_equal(masks[0], [True, False, True, False, False])
    np.testing.assert_array_equal(masks[1], [False, True, False, False, True])
    np.testing.assert_array_equal(masks[2], [False, False, False, True, False])


def test_trajectory_masks_empty_bin_returns_all_false() -> None:
    """Bins with no trajectories yield an all-False mask of the right length."""
    bin_id = np.array([0, 0, 1, 1])
    masks = trajectory_masks(bin_id, n_bins=4)

    assert len(masks) == 4
    assert masks[2].shape == bin_id.shape
    assert not masks[2].any()
    assert not masks[3].any()


def test_trajectory_masks_zero_n_bins() -> None:
    """n_bins=0 returns an empty list (no bins to mask)."""
    bin_id = np.array([0, 1, 2])
    assert trajectory_masks(bin_id, n_bins=0) == []


def test_trajectory_masks_n_bins_exceeds_max_id() -> None:
    """Trailing all-False masks appear when n_bins > max(bin_id) + 1."""
    bin_id = np.array([0, 1])
    masks = trajectory_masks(bin_id, n_bins=5)

    assert len(masks) == 5
    for i in (2, 3, 4):
        assert not masks[i].any()


def test_trajectory_masks_rejects_negative_n_bins() -> None:
    """n_bins < 0 is invalid."""
    with pytest.raises(ValueError, match="n_bins must be >= 0"):
        trajectory_masks(np.array([0]), n_bins=-1)


def test_trajectory_masks_rejects_non_1d_input() -> None:
    """Two-dimensional bin_id arrays are rejected."""
    with pytest.raises(ValueError, match="bin_id must be 1-D"):
        trajectory_masks(np.zeros((2, 3), dtype=int), n_bins=1)


def test_expand_preserves_dataset_order() -> None:
    """Expansion repeats each entry T times in row-major dataset order."""
    mask = np.array([True, False, True, False])
    expanded = expand_trajectory_mask_to_transitions(mask, n_transitions=3)

    np.testing.assert_array_equal(
        expanded,
        [True, True, True, False, False, False, True, True, True, False, False, False],
    )
    assert expanded.shape == (mask.shape[0] * 3,)


def test_expand_does_not_tile() -> None:
    """Repeat-vs-tile guard: expansion must NOT be [t0,t1,...,t0,t1,...]."""
    mask = np.array([True, False])
    expanded = expand_trajectory_mask_to_transitions(mask, n_transitions=2)

    np.testing.assert_array_equal(expanded, [True, True, False, False])
    # guard against tiling, which would give [True, False, True, False]
    assert not np.array_equal(expanded, [True, False, True, False])


def test_expand_zero_transitions() -> None:
    """n_transitions=0 produces an empty mask regardless of input length."""
    mask = np.array([True, False, True])
    expanded = expand_trajectory_mask_to_transitions(mask, n_transitions=0)
    assert expanded.shape == (0,)


def test_expand_empty_input() -> None:
    """Empty per-trajectory mask yields an empty expansion."""
    mask = np.array([], dtype=bool)
    expanded = expand_trajectory_mask_to_transitions(mask, n_transitions=5)
    assert expanded.shape == (0,)


def test_expand_rejects_negative_n_transitions() -> None:
    """n_transitions < 0 is invalid."""
    with pytest.raises(ValueError, match="n_transitions must be >= 0"):
        expand_trajectory_mask_to_transitions(np.array([True]), n_transitions=-1)


def test_expand_rejects_non_1d_input() -> None:
    """Two-dimensional masks are rejected."""
    with pytest.raises(ValueError, match="mask must be 1-D"):
        expand_trajectory_mask_to_transitions(np.zeros((2, 3), dtype=bool), n_transitions=1)
