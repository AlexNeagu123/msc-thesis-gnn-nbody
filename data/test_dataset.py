"""Tests for data/dataset.py."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from data.dataset import NBodyDataset, TrajectoryWindowDataset


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


def test_dataset_can_use_trajectory_prefix(sample_h5: tuple[str, np.ndarray]) -> None:
    """Training subset runs can use a deterministic prefix of the train file."""
    path, trajectories = sample_h5
    ds = NBodyDataset(path, n_trajectories=2)

    assert ds.n_trajectories == 2
    assert len(ds) == 2 * (trajectories.shape[1] - 1)

    state_t, _ = ds[0]
    expected_t = torch.from_numpy(trajectories[0, 0]).float()
    assert torch.allclose(state_t, expected_t)


def test_dataset_rejects_too_many_requested_trajectories(
    sample_h5: tuple[str, np.ndarray],
) -> None:
    """Prefix subset size should not silently exceed the available HDF5 data."""
    path, trajectories = sample_h5

    with pytest.raises(ValueError, match=r"requested .* but only"):
        NBodyDataset(path, n_trajectories=trajectories.shape[0] + 1)


def test_dataset_rejects_non_positive_trajectory_count(
    sample_h5: tuple[str, np.ndarray],
) -> None:
    """A zero-sized prefix subset is invalid."""
    path, _ = sample_h5

    with pytest.raises(ValueError, match="must be positive"):
        NBodyDataset(path, n_trajectories=0)


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


def test_window_length(sample_h5: tuple[str, np.ndarray]) -> None:
    """Total windows = n_trajectories * (n_frames - horizon)."""
    path, trajectories = sample_h5
    horizon = 3
    ds = TrajectoryWindowDataset(path, horizon=horizon)

    n_traj, n_frames, _, _ = trajectories.shape
    assert len(ds) == n_traj * (n_frames - horizon)


def test_window_input_shape(sample_h5: tuple[str, np.ndarray]) -> None:
    """Each input is a single state of shape (n_particles, 5)."""
    path, _ = sample_h5
    ds = TrajectoryWindowDataset(path, horizon=3)

    state_t, _ = ds[0]
    assert state_t.shape == (3, 5)


def test_window_target_shape(sample_h5: tuple[str, np.ndarray]) -> None:
    """Each target is a window of shape (horizon, n_particles, 5)."""
    path, _ = sample_h5
    horizon = 3
    ds = TrajectoryWindowDataset(path, horizon=horizon)

    _, target = ds[0]
    assert target.shape == (horizon, 3, 5)


def test_window_first_sample_matches_source(sample_h5: tuple[str, np.ndarray]) -> None:
    """First sample is trajectory[0, 0] -> trajectory[0, 1:horizon+1]."""
    path, trajectories = sample_h5
    horizon = 3
    ds = TrajectoryWindowDataset(path, horizon=horizon)

    state_t, target = ds[0]
    expected_t = torch.from_numpy(trajectories[0, 0]).float()
    expected_target = torch.from_numpy(trajectories[0, 1 : 1 + horizon]).float()

    assert torch.allclose(state_t, expected_t)
    assert torch.allclose(target, expected_target)


def test_window_does_not_cross_trajectory(sample_h5: tuple[str, np.ndarray]) -> None:
    """Last window of trajectory i ends inside trajectory i; next sample starts trajectory i+1."""
    path, trajectories = sample_h5
    horizon = 3
    n_frames = trajectories.shape[1]
    windows_per_traj = n_frames - horizon
    ds = TrajectoryWindowDataset(path, horizon=horizon)

    last_idx = windows_per_traj - 1
    state_t, target = ds[last_idx]

    expected_t = torch.from_numpy(trajectories[0, last_idx]).float()
    expected_target = torch.from_numpy(
        trajectories[0, last_idx + 1 : last_idx + 1 + horizon]
    ).float()
    assert torch.allclose(state_t, expected_t)
    assert torch.allclose(target, expected_target)

    state_t_next, _ = ds[windows_per_traj]
    expected_t_next = torch.from_numpy(trajectories[1, 0]).float()
    assert torch.allclose(state_t_next, expected_t_next)


def test_window_n_trajectories_slicing(sample_h5: tuple[str, np.ndarray]) -> None:
    """Prefix-subset slicing yields a deterministic prefix of trajectories."""
    path, trajectories = sample_h5
    horizon = 3
    ds = TrajectoryWindowDataset(path, horizon=horizon, n_trajectories=2)

    n_frames = trajectories.shape[1]
    assert ds.n_trajectories == 2
    assert len(ds) == 2 * (n_frames - horizon)

    state_t, _ = ds[0]
    expected_t = torch.from_numpy(trajectories[0, 0]).float()
    assert torch.allclose(state_t, expected_t)


def test_window_horizon_zero_raises(sample_h5: tuple[str, np.ndarray]) -> None:
    """Horizon < 1 is invalid."""
    path, _ = sample_h5
    with pytest.raises(ValueError, match="horizon must be"):
        TrajectoryWindowDataset(path, horizon=0)


def test_window_horizon_negative_raises(sample_h5: tuple[str, np.ndarray]) -> None:
    """Negative horizons are invalid."""
    path, _ = sample_h5
    with pytest.raises(ValueError, match="horizon must be"):
        TrajectoryWindowDataset(path, horizon=-1)


def test_window_horizon_exceeds_frames_raises(sample_h5: tuple[str, np.ndarray]) -> None:
    """Horizon >= n_frames leaves no valid window starts."""
    path, trajectories = sample_h5
    n_frames = trajectories.shape[1]
    with pytest.raises(ValueError, match="horizon must be"):
        TrajectoryWindowDataset(path, horizon=n_frames)


def test_window_horizon_at_max_is_valid(sample_h5: tuple[str, np.ndarray]) -> None:
    """Horizon = n_frames - 1 yields exactly one window per trajectory."""
    path, trajectories = sample_h5
    n_traj, n_frames, _, _ = trajectories.shape
    horizon = n_frames - 1
    ds = TrajectoryWindowDataset(path, horizon=horizon)

    assert len(ds) == n_traj


def test_window_dtype(sample_h5: tuple[str, np.ndarray]) -> None:
    """Tensors should be float32, like NBodyDataset."""
    path, _ = sample_h5
    ds = TrajectoryWindowDataset(path, horizon=3)
    state_t, target = ds[0]

    assert state_t.dtype == torch.float32
    assert target.dtype == torch.float32


def test_window_rejects_non_positive_trajectory_count(
    sample_h5: tuple[str, np.ndarray],
) -> None:
    """n_trajectories <= 0 is invalid."""
    path, _ = sample_h5
    with pytest.raises(ValueError, match="must be positive"):
        TrajectoryWindowDataset(path, horizon=3, n_trajectories=0)


def test_window_rejects_too_many_requested_trajectories(
    sample_h5: tuple[str, np.ndarray],
) -> None:
    """Asking for more trajectories than the file holds raises."""
    path, trajectories = sample_h5
    with pytest.raises(ValueError, match=r"requested .* but only"):
        TrajectoryWindowDataset(path, horizon=3, n_trajectories=trajectories.shape[0] + 1)
