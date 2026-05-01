"""Tests for data/_io.py.

Round-trip a Trajectories bundle through write_trajectories /
read_trajectories and confirm states, energies, and metadata survive.
"""

from pathlib import Path

import numpy as np

from data._io import read_trajectories, write_trajectories
from data._types import Trajectories, TrajectoryMetadata


def _example_trajectories() -> Trajectories:
    """Build a small Trajectories with realistic-looking metadata."""
    rng = np.random.default_rng(42)
    states = rng.normal(size=(2, 5, 3, 5)).astype(np.float64)
    energies = rng.normal(size=(2, 5)).astype(np.float64)
    metadata = TrajectoryMetadata(
        n_trajectories=2,
        n_particles=3,
        n_steps=5,
        t_end=1.0,
        dt=0.2,
        G=1.0,
        mass=1.0,
        min_distance=0.001,
        pos_scale=1.0,
        vel_scale=0.5,
        seed=42,
        rejection_rate=0.25,
    )
    return Trajectories(states=states, energies=energies, metadata=metadata)


def test_round_trip_preserves_arrays(tmp_path: Path) -> None:
    """States and energies survive write -> read byte-equal."""
    original = _example_trajectories()
    path = tmp_path / "round.h5"

    write_trajectories(path, original)
    loaded = read_trajectories(path)

    assert np.array_equal(loaded.states, original.states)
    assert np.array_equal(loaded.energies, original.energies)


def test_round_trip_preserves_metadata(tmp_path: Path) -> None:
    """All TrajectoryMetadata fields survive write -> read."""
    original = _example_trajectories()
    path = tmp_path / "round.h5"

    write_trajectories(path, original)
    loaded = read_trajectories(path)

    assert loaded.metadata == original.metadata


def test_read_tolerates_missing_metadata(tmp_path: Path) -> None:
    """Files written without metadata (test fixtures, legacy) read cleanly."""
    rng = np.random.default_rng(0)
    bare = Trajectories(
        states=rng.normal(size=(1, 3, 3, 5)),
        energies=rng.normal(size=(1, 3)),
        metadata=None,
    )
    path = tmp_path / "bare.h5"

    write_trajectories(path, bare)
    loaded = read_trajectories(path)

    assert loaded.metadata is None
    assert np.array_equal(loaded.states, bare.states)


def test_write_creates_parent_directory(tmp_path: Path) -> None:
    """write_trajectories creates intermediate directories if missing."""
    nested = tmp_path / "a" / "b" / "c.h5"
    write_trajectories(nested, _example_trajectories())
    assert nested.exists()
