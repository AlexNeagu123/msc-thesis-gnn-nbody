"""Tests for data/generate.py."""

import numpy as np
import pytest

from data.generate import generate_trajectory


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a seeded random generator."""
    return np.random.default_rng(42)


def test_trajectory_shape(rng: np.random.Generator) -> None:
    """Output arrays have the expected shape."""
    result = generate_trajectory(n_particles=3, t_end=1.0, dt=0.05, rng=rng)
    assert result is not None

    states, energies = result
    assert states.shape == (20, 3, 4)
    assert energies.shape == (20,)


def test_trajectory_state_columns(rng: np.random.Generator) -> None:
    """State columns are [x, y, vx, vy] — positions and velocities should be finite."""
    result = generate_trajectory(n_particles=3, t_end=1.0, dt=0.05, rng=rng)
    assert result is not None

    states, _ = result
    assert np.all(np.isfinite(states))


def test_energy_conservation(rng: np.random.Generator) -> None:
    """IAS15 should conserve energy to near machine precision."""
    result = generate_trajectory(n_particles=3, t_end=1.0, dt=0.05, rng=rng)
    assert result is not None

    _, energies = result
    rel_error = np.abs((energies - energies[0]) / energies[0])
    assert rel_error.max() < 1e-8


def test_center_of_mass_near_zero(rng: np.random.Generator) -> None:
    """Center of mass should stay near the origin (we subtract it at init)."""
    result = generate_trajectory(n_particles=3, t_end=1.0, dt=0.05, rng=rng)
    assert result is not None

    states, _ = result
    com_positions = states[:, :, :2].mean(axis=1)
    assert np.abs(com_positions).max() < 1e-8


def test_close_encounter_filtering() -> None:
    """Trajectory with particles starting very close should be rejected."""
    rejected = 0
    for seed in range(50):
        r = np.random.default_rng(seed)
        result = generate_trajectory(
            n_particles=3,
            t_end=1.0,
            dt=0.05,
            min_distance=1.0,
            rng=r,
        )
        if result is None:
            rejected += 1

    assert rejected > 0, "no trajectories were rejected despite strict min_distance"


def test_different_seeds_produce_different_trajectories() -> None:
    """Different seeds should give different trajectories."""
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)

    result1 = generate_trajectory(n_particles=3, t_end=1.0, dt=0.05, rng=rng1)
    result2 = generate_trajectory(n_particles=3, t_end=1.0, dt=0.05, rng=rng2)

    assert result1 is not None or result2 is not None

    if result1 is not None and result2 is not None:
        states1, _ = result1
        states2, _ = result2
        assert not np.allclose(states1, states2)
