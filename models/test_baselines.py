"""Tests for models/baselines.py."""

from pathlib import Path

import h5py
import numpy as np
import torch

from models.baselines import (
    ConstantVelocityBaseline,
    MeanStateBaseline,
    MeanVelocityBaseline,
    PersistenceBaseline,
)


def _make_state(batch: int = 4, n_particles: int = 3) -> torch.Tensor:
    """Create a random state tensor [x, y, vx, vy, mass] with mass=1."""
    torch.manual_seed(0)
    pos_vel = torch.randn(batch, n_particles, 4)
    mass = torch.ones(batch, n_particles, 1)
    return torch.cat([pos_vel, mass], dim=-1)


def _write_train_h5(path: Path, *, seed: int = 0, shape: tuple = (4, 10, 3, 5)) -> np.ndarray:
    """Write a small HDF5 trajectory file and return the underlying array."""
    rng = np.random.default_rng(seed)
    trajectories = rng.normal(size=shape).astype(np.float32)
    trajectories[..., 4] = 1.0
    with h5py.File(path, "w") as f:
        f.create_dataset("trajectories", data=trajectories)
    return trajectories


def test_persistence_returns_identical_state() -> None:
    """PersistenceBaseline returns the exact input state."""
    model = PersistenceBaseline()
    state = _make_state()
    out = model(state)

    assert out.shape == state.shape
    assert torch.allclose(out, state)


def test_persistence_does_not_share_storage() -> None:
    """Output is detached from input storage so callers can mutate freely."""
    model = PersistenceBaseline()
    state = _make_state()
    out = model(state)
    out.add_(1.0)

    assert not torch.allclose(out, state)


def test_constant_velocity_position_update() -> None:
    """x_{t+dt} = x_t + dt * v_t."""
    dt = 0.05
    model = ConstantVelocityBaseline(dt=dt)
    state = _make_state()
    out = model(state)

    expected_pos = state[..., :2] + dt * state[..., 2:4]
    assert torch.allclose(out[..., :2], expected_pos)


def test_constant_velocity_preserves_velocity() -> None:
    """v_{t+dt} = v_t."""
    model = ConstantVelocityBaseline(dt=0.05)
    state = _make_state()
    out = model(state)

    assert torch.allclose(out[..., 2:4], state[..., 2:4])


def test_mean_velocity_fit_matches_dataset_mean(tmp_path: Path) -> None:
    """fit() recovers the global mean velocity over particles, frames, trajectories."""
    train_path = tmp_path / "train.h5"
    trajectories = _write_train_h5(train_path)

    model = MeanVelocityBaseline.fit(str(train_path), dt=0.05)

    expected = torch.from_numpy(trajectories[..., 2:4].reshape(-1, 2).mean(axis=0)).float()
    assert model.v_mean.shape == (2,)
    assert torch.allclose(model.v_mean, expected)


def test_mean_velocity_forward_uses_fitted_mean(tmp_path: Path) -> None:
    """Forward output velocities equal v_mean and positions advance by dt*v_mean."""
    train_path = tmp_path / "train.h5"
    _write_train_h5(train_path)

    dt = 0.05
    model = MeanVelocityBaseline.fit(str(train_path), dt=dt)
    state = _make_state()
    out = model(state)

    expected_v = model.v_mean.expand_as(out[..., 2:4])
    expected_pos = state[..., :2] + dt * expected_v

    assert torch.allclose(out[..., 2:4], expected_v)
    assert torch.allclose(out[..., :2], expected_pos)


def test_mean_velocity_rejects_wrong_shape() -> None:
    """Constructing with a non-(2,) v_mean raises."""
    import pytest

    with pytest.raises(ValueError, match="v_mean must have shape"):
        MeanVelocityBaseline(dt=0.05, v_mean=torch.zeros(3))


def test_mean_state_fit_matches_dataset_mean(tmp_path: Path) -> None:
    """fit() recovers the global mean position over particles, frames, trajectories."""
    train_path = tmp_path / "train.h5"
    trajectories = _write_train_h5(train_path)

    model = MeanStateBaseline.fit(str(train_path))

    expected = torch.from_numpy(trajectories[..., :2].reshape(-1, 2).mean(axis=0)).float()
    assert model.x_mean.shape == (2,)
    assert torch.allclose(model.x_mean, expected)


def test_mean_state_forward_snaps_to_centre(tmp_path: Path) -> None:
    """Forward output positions equal x_mean and velocities are zero."""
    train_path = tmp_path / "train.h5"
    _write_train_h5(train_path)

    model = MeanStateBaseline.fit(str(train_path))
    state = _make_state()
    out = model(state)

    expected_pos = model.x_mean.expand_as(out[..., :2])
    assert torch.allclose(out[..., :2], expected_pos)
    assert torch.allclose(out[..., 2:4], torch.zeros_like(out[..., 2:4]))


def test_mean_state_rejects_wrong_shape() -> None:
    """Constructing with a non-(2,) x_mean raises."""
    import pytest

    with pytest.raises(ValueError, match="x_mean must have shape"):
        MeanStateBaseline(x_mean=torch.zeros(3))


def test_all_baselines_preserve_mass() -> None:
    """Every baseline passes the mass column through untouched."""
    state = _make_state()
    state[..., 4] = torch.tensor([1.0, 2.0, 3.0]).expand(state.shape[0], -1)

    models = [
        PersistenceBaseline(),
        ConstantVelocityBaseline(dt=0.05),
        MeanVelocityBaseline(dt=0.05, v_mean=torch.tensor([0.1, 0.2])),
        MeanStateBaseline(x_mean=torch.tensor([0.0, 0.0])),
    ]
    for model in models:
        out = model(state)
        assert torch.allclose(out[..., 4:], state[..., 4:]), type(model).__name__


def test_all_baselines_preserve_shape() -> None:
    """Output shape equals input shape across batch and particle counts."""
    state = _make_state(batch=2, n_particles=5)

    models = [
        PersistenceBaseline(),
        ConstantVelocityBaseline(dt=0.05),
        MeanVelocityBaseline(dt=0.05, v_mean=torch.tensor([0.1, 0.2])),
        MeanStateBaseline(x_mean=torch.tensor([0.0, 0.0])),
    ]
    for model in models:
        out = model(state)
        assert out.shape == state.shape, type(model).__name__


def test_baselines_have_no_trainable_parameters() -> None:
    """Baselines are deterministic, so they must expose zero trainable parameters."""
    models = [
        PersistenceBaseline(),
        ConstantVelocityBaseline(dt=0.05),
        MeanVelocityBaseline(dt=0.05, v_mean=torch.tensor([0.1, 0.2])),
        MeanStateBaseline(x_mean=torch.tensor([0.0, 0.0])),
    ]
    for model in models:
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == 0, type(model).__name__
