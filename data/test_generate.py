"""Tests for data/generate.py."""

import numpy as np
import pytest

from data._types import SimulationParams
from data.generate import generate_trajectory


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a seeded random generator."""
    return np.random.default_rng(42)


@pytest.fixture
def default_params() -> SimulationParams:
    """Return default simulation parameters for testing."""
    return SimulationParams(
        n_particles=3,
        t_end=1.0,
        dt=0.05,
        G=1.0,
        mass=1.0,
        min_distance=0.001,
        max_position=10.0,
        pos_scale=1.0,
        vel_scale=0.5,
    )


def test_trajectory_shape(
    default_params: SimulationParams,
    rng: np.random.Generator,
) -> None:
    """Output arrays have the expected shape."""
    result = generate_trajectory(default_params, rng)
    assert result is not None

    states, energies = result
    assert states.shape == (20, 3, 5)
    assert energies.shape == (20,)


def test_trajectory_state_columns(
    default_params: SimulationParams,
    rng: np.random.Generator,
) -> None:
    """State columns are [x, y, vx, vy, m] — all values should be finite."""
    result = generate_trajectory(default_params, rng)
    assert result is not None

    states, _ = result
    assert np.all(np.isfinite(states))


def test_energy_conservation(
    default_params: SimulationParams,
    rng: np.random.Generator,
) -> None:
    """IAS15 should conserve energy to near machine precision."""
    result = generate_trajectory(default_params, rng)
    assert result is not None

    _, energies = result
    rel_error = np.abs((energies - energies[0]) / energies[0])
    assert rel_error.max() < 1e-8


def test_center_of_mass_near_zero(
    default_params: SimulationParams,
    rng: np.random.Generator,
) -> None:
    """Center of mass should stay near the origin (we subtract it at init)."""
    result = generate_trajectory(default_params, rng)
    assert result is not None

    states, _ = result
    com_positions = states[:, :, :2].mean(axis=1)
    assert np.abs(com_positions).max() < 1e-8


def test_close_encounter_filtering() -> None:
    """Trajectory with particles starting very close should be rejected."""
    strict_params = SimulationParams(
        n_particles=3,
        t_end=1.0,
        dt=0.05,
        G=1.0,
        mass=1.0,
        min_distance=1.0,
        max_position=10.0,
        pos_scale=1.0,
        vel_scale=0.5,
    )

    rejected = 0
    for seed in range(50):
        r = np.random.default_rng(seed)
        result = generate_trajectory(strict_params, r)
        if result is None:
            rejected += 1

    assert rejected > 0, "no trajectories were rejected despite strict min_distance"


def test_different_seeds_produce_different_trajectories(
    default_params: SimulationParams,
) -> None:
    """Different seeds should give different trajectories."""
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)

    result1 = generate_trajectory(default_params, rng1)
    result2 = generate_trajectory(default_params, rng2)

    assert result1 is not None or result2 is not None

    if result1 is not None and result2 is not None:
        states1, _ = result1
        states2, _ = result2
        assert not np.allclose(states1, states2)
