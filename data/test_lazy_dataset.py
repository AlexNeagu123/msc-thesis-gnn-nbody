"""Lazy-window equivalence, boundaries and storage-complexity tests."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data.dataset import NBodyDataset, TrajectoryWindowDataset
from data.lazy_dataset import LazyTrajectoryWindowDataset


@pytest.fixture
def trajectories(tmp_path: Path) -> Path:
    """Write uniquely identifiable frames for three independent simulations."""
    path = tmp_path / "states.h5"
    with h5py.File(path, "w") as handle:
        handle["trajectories"] = np.arange(3 * 9 * 3 * 5, dtype=np.float64).reshape(3, 9, 3, 5)
    return path


@pytest.mark.parametrize("horizon", [1, 2, 5, 8])
@pytest.mark.parametrize("n_trajectories", [None, 1, 2])
def test_matches_legacy_windows(
    trajectories: Path, horizon: int, n_trajectories: int | None
) -> None:
    """Every input and target is unchanged, including trajectory boundaries."""
    lazy = LazyTrajectoryWindowDataset(str(trajectories), horizon, n_trajectories)
    legacy = (
        NBodyDataset(str(trajectories), n_trajectories)
        if horizon == 1
        else TrajectoryWindowDataset(str(trajectories), horizon, n_trajectories)
    )
    assert len(lazy) == len(legacy)
    assert (
        lazy.states.untyped_storage().nbytes() == lazy.states.numel() * lazy.states.element_size()
    )
    torch.testing.assert_close(lazy.inputs, legacy.inputs, rtol=0, atol=0)
    for index in range(len(lazy)):
        for actual, expected in zip(lazy[index], legacy[index], strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(lazy[-1], lazy[len(lazy) - 1], strict=True):
        assert torch.equal(actual, expected)
    with pytest.raises(IndexError):
        lazy[len(lazy)]
    with pytest.raises(IndexError):
        lazy[-len(lazy) - 1]


def test_storage_shared_across_horizons(
    trajectories: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing horizon allocates no expanded target tensor and does not reopen HDF5."""
    first = LazyTrajectoryWindowDataset(str(trajectories), 1)
    monkeypatch.setattr(
        "data.lazy_dataset.read_states", lambda *args: pytest.fail("unexpected reread")
    )
    for horizon in (2, 5, 8):
        changed = first.with_horizon(horizon)
        assert changed.states is first.states
        assert changed.states.numel() == 3 * 9 * 3 * 5
        x, future = changed[0]
        assert x.untyped_storage().data_ptr() == first.states.untyped_storage().data_ptr()
        assert future.untyped_storage().data_ptr() == first.states.untyped_storage().data_ptr()
        batch_x, batch_future = next(iter(DataLoader(changed, batch_size=2)))
        assert batch_x.shape == (2, 3, 5)
        assert batch_future.shape == (2, horizon, 3, 5)
        assert changed.locate_target(future[0]) == "trajectory 0, step 0"
        assert not hasattr(changed, "targets")


@pytest.mark.parametrize("horizon", [0, -1, 9, True, 1.5])
def test_invalid_horizon(trajectories: Path, horizon: int) -> None:
    """Invalid lengths cannot silently drop boundaries or produce empty windows."""
    with pytest.raises(ValueError, match="horizon"):
        LazyTrajectoryWindowDataset(str(trajectories), horizon)


@pytest.mark.parametrize("count", [0, -1, 4, True])
def test_invalid_count(trajectories: Path, count: int) -> None:
    """Trajectory limits must be actual positive available counts."""
    with pytest.raises(ValueError, match="n_trajectories"):
        LazyTrajectoryWindowDataset(str(trajectories), 1, count)
