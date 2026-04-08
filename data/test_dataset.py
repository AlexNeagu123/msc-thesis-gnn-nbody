"""Tests for data/dataset.py."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from data.dataset import NBodyDataset


@pytest.fixture
def sample_h5(tmp_path: Path) -> tuple[str, np.ndarray]:
    """Create a small HDF5 file with known data for testing."""
    n_traj, n_steps, n_particles = 3, 10, 3
    rng = np.random.default_rng(42)

    trajectories = rng.normal(size=(n_traj, n_steps, n_particles, 5))
    energies = rng.normal(size=(n_traj, n_steps))

    path = str(tmp_path / "test.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("trajectories", data=trajectories)
        f.create_dataset("energies", data=energies)

    return path, trajectories


def test_dataset_length(sample_h5: tuple[str, np.ndarray]) -> None:
    """Total samples = n_trajectories * (n_steps - 1)."""
    path, trajectories = sample_h5
    ds = NBodyDataset(path)

    n_traj, n_steps, _, _ = trajectories.shape
    assert len(ds) == n_traj * (n_steps - 1)


def test_dataset_shapes(sample_h5: tuple[str, np.ndarray]) -> None:
    """Each sample returns two tensors of shape (n_particles, 5)."""
    path, _ = sample_h5
    ds = NBodyDataset(path)

    state_t, state_next = ds[0]
    assert state_t.shape == (3, 5)
    assert state_next.shape == (3, 5)


def test_dataset_dtype(sample_h5: tuple[str, np.ndarray]) -> None:
    """Tensors should be float32."""
    path, _ = sample_h5
    ds = NBodyDataset(path)

    state_t, state_next = ds[0]
    assert state_t.dtype == torch.float32
    assert state_next.dtype == torch.float32


def test_consecutive_pairs(sample_h5: tuple[str, np.ndarray]) -> None:
    """State_t+1 from sample i should match state_t from sample i+1 within same trajectory."""
    path, trajectories = sample_h5
    ds = NBodyDataset(path)

    n_steps = trajectories.shape[1]
    for i in range(n_steps - 2):
        _, target_i = ds[i]
        input_next, _ = ds[i + 1]
        assert torch.allclose(target_i, input_next)


def test_trajectory_boundary(sample_h5: tuple[str, np.ndarray]) -> None:
    """Samples at trajectory boundaries should NOT be consecutive."""
    path, trajectories = sample_h5
    ds = NBodyDataset(path)

    n_steps = trajectories.shape[1]
    steps_per_traj = n_steps - 1

    _, target_last = ds[steps_per_traj - 1]
    input_next_traj, _ = ds[steps_per_traj]
    assert not torch.allclose(target_last, input_next_traj)


def test_values_match_source(sample_h5: tuple[str, np.ndarray]) -> None:
    """Dataset values should match the original HDF5 data."""
    path, trajectories = sample_h5
    ds = NBodyDataset(path)

    state_t, state_next = ds[0]
    expected_t = torch.from_numpy(trajectories[0, 0]).float()
    expected_next = torch.from_numpy(trajectories[0, 1]).float()

    assert torch.allclose(state_t, expected_t)
    assert torch.allclose(state_next, expected_next)
