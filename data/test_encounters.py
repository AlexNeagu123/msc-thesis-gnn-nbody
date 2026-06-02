"""Tests for data/encounters.py."""

from __future__ import annotations

import numpy as np
import pytest

from data._types import EncounterBin
from data.encounters import (
    DEFAULT_ENCOUNTER_BINS,
    assign_encounter_bin,
    min_pairwise_distance_over_time,
    target_counts_from_distribution,
)


# min_pairwise_distance_over_time
def _states(*frames: np.ndarray) -> np.ndarray:
    """Stack a sequence of (n_particles, 2) frames into a (T, N, 2) array."""
    return np.stack(frames, axis=0).astype(np.float64)


def test_min_pairwise_distance_two_particles_stationary() -> None:
    """Two coincident particles register distance zero."""
    frame = np.array([[0.0, 0.0], [0.0, 0.0]])
    states = _states(frame, frame, frame)

    assert min_pairwise_distance_over_time(states) == 0.0


def test_min_pairwise_distance_two_particles_constant_separation() -> None:
    """Stationary 3-4-5 triangle separation is recovered exactly."""
    frame = np.array([[0.0, 0.0], [3.0, 4.0]])
    states = _states(frame, frame)

    assert min_pairwise_distance_over_time(states) == pytest.approx(5.0)


def test_min_pairwise_distance_picks_closest_frame() -> None:
    """The function reduces over time to the closest single approach."""
    far = np.array([[0.0, 0.0], [5.0, 0.0]])
    close = np.array([[0.0, 0.0], [1.0, 0.0]])
    states = _states(far, far, close, far)

    assert min_pairwise_distance_over_time(states) == pytest.approx(1.0)


def test_min_pairwise_distance_three_particles_picks_tightest_pair() -> None:
    """With one tight pair and one loose pair, the tight pair wins."""
    frame_a = np.array([[0.0, 0.0], [0.5, 0.0], [3.0, 3.0]])
    frame_b = np.array([[0.1, 0.0], [0.6, 0.0], [3.0, 3.0]])
    states = _states(frame_a, frame_b)

    assert min_pairwise_distance_over_time(states) == pytest.approx(0.5)


def test_min_pairwise_distance_uses_only_first_two_coords() -> None:
    """Velocity / mass columns past index 1 do not affect the spatial distance."""
    frame = np.array([[0.0, 0.0, 9.9, 9.9, 1.0], [3.0, 4.0, -1.1, -1.1, 1.0]])
    states = _states(frame, frame)

    assert min_pairwise_distance_over_time(states) == pytest.approx(5.0)


def test_min_pairwise_distance_rejects_two_dimensional_input() -> None:
    """A 2-D array (no time axis) is malformed and rejected."""
    with pytest.raises(ValueError, match="states must be 3D"):
        min_pairwise_distance_over_time(np.zeros((3, 2)))


def test_min_pairwise_distance_rejects_single_particle() -> None:
    """One particle has no pair, so the call cannot succeed."""
    with pytest.raises(ValueError, match="at least 2 particles"):
        min_pairwise_distance_over_time(np.zeros((4, 1, 2)))


def test_min_pairwise_distance_rejects_one_dimensional_coords() -> None:
    """Need at least 2 spatial coordinates to define a Euclidean distance."""
    with pytest.raises(ValueError, match="at least 2 spatial coords"):
        min_pairwise_distance_over_time(np.zeros((4, 3, 1)))


def test_min_pairwise_distance_rejects_zero_steps() -> None:
    """A zero-length time axis cannot produce a min and is rejected explicitly."""
    with pytest.raises(ValueError, match="at least 1 timestep"):
        min_pairwise_distance_over_time(np.zeros((0, 3, 2)))


# assign_encounter_bin
@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (0.0, "extreme"),
        (0.005, "extreme"),
        (0.009999, "extreme"),
        (0.010, "very_close"),
        (0.015, "very_close"),
        (0.020, "close"),
        (0.049999, "close"),
        (0.050, "moderate"),
        (0.099999, "moderate"),
        (0.100, "mild"),
        (0.199999, "mild"),
        (0.200, "smooth"),
        (1.0, "smooth"),
        (1e10, "smooth"),
    ],
)
def test_assign_encounter_bin_table(distance: float, expected: str) -> None:
    """Boundary table: half-open semantics put exact boundaries in the upper bin."""
    assert assign_encounter_bin(distance, DEFAULT_ENCOUNTER_BINS) == expected


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_assign_encounter_bin_rejects_non_finite(bad: float) -> None:
    """NaN and infinities cannot be placed under [lo, hi); reject explicitly."""
    with pytest.raises(ValueError, match="distance must be finite"):
        assign_encounter_bin(bad, DEFAULT_ENCOUNTER_BINS)


def test_assign_encounter_bin_rejects_value_outside_all_bins() -> None:
    """A negative distance falls outside every default bin; raise rather than guess."""
    with pytest.raises(ValueError, match="does not fall in any bin"):
        assign_encounter_bin(-0.001, DEFAULT_ENCOUNTER_BINS)


def test_assign_encounter_bin_raises_when_top_sentinel_missing() -> None:
    """Custom bins without an inf-top entry surface the misuse loudly."""
    bounded = (
        EncounterBin("low", 0.0, 0.5),
        EncounterBin("mid", 0.5, 1.0),
    )
    with pytest.raises(ValueError, match="does not fall in any bin"):
        assign_encounter_bin(2.0, bounded)


# target_counts_from_distribution
_TRAINING_DISTRIBUTION: dict[str, float] = {
    "extreme": 10,
    "very_close": 15,
    "close": 25,
    "moderate": 25,
    "mild": 15,
    "smooth": 10,
}


def test_target_counts_exact_division() -> None:
    """Total chosen so per-bin shares are exact integers; no rounding needed."""
    counts = target_counts_from_distribution(10000, _TRAINING_DISTRIBUTION)

    assert counts == {
        "extreme": 1000,
        "very_close": 1500,
        "close": 2500,
        "moderate": 2500,
        "mild": 1500,
        "smooth": 1000,
    }
    assert sum(counts.values()) == 10000


def test_target_counts_largest_remainder_with_ties() -> None:
    """Remainder=2 across four tied 0.5 fractions: insertion order picks the first two."""
    counts = target_counts_from_distribution(10, _TRAINING_DISTRIBUTION)

    # raw shares: 1.0, 1.5, 2.5, 2.5, 1.5, 1.0
    # floors:     1,   1,   2,   2,   1,   1   (sum 8, remainder 2)
    # fractions:  0.0, 0.5, 0.5, 0.5, 0.5, 0.0
    # tie-break by insertion order -> very_close, close get +1
    assert counts == {
        "extreme": 1,
        "very_close": 2,
        "close": 3,
        "moderate": 2,
        "mild": 1,
        "smooth": 1,
    }
    assert sum(counts.values()) == 10


def test_target_counts_equal_split_with_remainder() -> None:
    """Six bins with equal weights, total=10: 4 bins get 2, last 2 get 1."""
    equal = {name: 1.0 for name in _TRAINING_DISTRIBUTION}
    counts = target_counts_from_distribution(10, equal)

    assert sum(counts.values()) == 10
    # raw share = 10/6 = 1.667; floors = 1 each (sum 6, remainder 4)
    # fractional ~ 0.667 for all six -> first 4 by insertion order get +1
    expected_keys = list(equal.keys())
    for k in expected_keys[:4]:
        assert counts[k] == 2
    for k in expected_keys[4:]:
        assert counts[k] == 1


def test_target_counts_total_zero_returns_all_zeros() -> None:
    """Asking for zero items distributes none to anyone."""
    counts = target_counts_from_distribution(0, _TRAINING_DISTRIBUTION)

    assert counts == {name: 0 for name in _TRAINING_DISTRIBUTION}


def test_target_counts_single_bin_collects_total() -> None:
    """A one-entry distribution always sends all items to that bin."""
    counts = target_counts_from_distribution(7, {"only": 3.0})

    assert counts == {"only": 7}


def test_target_counts_zero_weight_bin_gets_nothing() -> None:
    """A zero weight maps to zero count; remainder lands in non-zero bins."""
    counts = target_counts_from_distribution(5, {"a": 0.0, "b": 1.0, "c": 1.0})

    assert counts["a"] == 0
    assert counts["b"] + counts["c"] == 5


def test_target_counts_is_deterministic_across_repeat_calls() -> None:
    """Identical inputs always return identical outputs (no random tie-break)."""
    first = target_counts_from_distribution(10, _TRAINING_DISTRIBUTION)
    second = target_counts_from_distribution(10, _TRAINING_DISTRIBUTION)

    assert first == second


def test_target_counts_rejects_negative_total() -> None:
    """A negative count request is meaningless and rejected."""
    with pytest.raises(ValueError, match="total must be >= 0"):
        target_counts_from_distribution(-1, {"a": 1.0})


def test_target_counts_rejects_empty_distribution() -> None:
    """An empty dict has nowhere to assign items; reject."""
    with pytest.raises(ValueError, match="distribution must be non-empty"):
        target_counts_from_distribution(10, {})


def test_target_counts_rejects_negative_weight() -> None:
    """Negative weights have no meaning under largest-remainder; reject."""
    with pytest.raises(ValueError, match="weights must be non-negative"):
        target_counts_from_distribution(10, {"a": 1.0, "b": -0.5})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_target_counts_rejects_non_finite_weight(bad: float) -> None:
    """NaN / +-inf weights cannot be normalised; reject before any rounding work."""
    with pytest.raises(ValueError, match="weights must be finite"):
        target_counts_from_distribution(10, {"a": 1.0, "b": bad})


def test_target_counts_rejects_zero_total_weight() -> None:
    """All-zero weights cannot be normalised; reject explicitly."""
    with pytest.raises(ValueError, match=r"weights must sum to > 0"):
        target_counts_from_distribution(10, {"a": 0.0, "b": 0.0})


@pytest.mark.parametrize("total", [0, 1, 10, 99, 100, 1234, 9999])
def test_target_counts_sum_always_equals_total(total: int) -> None:
    """Largest-remainder must round to exactly `total`, never off by one."""
    counts = target_counts_from_distribution(total, _TRAINING_DISTRIBUTION)

    assert sum(counts.values()) == total
    assert all(c >= 0 for c in counts.values())
