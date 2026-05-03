"""Deterministic, non-learning baselines for the n-body next-state task.

These models implement the same `forward(state) -> state` contract as the
trained EGNN/HGNN so they can be plugged into the existing rollout and
metric pipeline without special casing. Mass is always passed through
unchanged.
"""

from __future__ import annotations

import h5py
import torch
from torch import Tensor, nn


class PersistenceBaseline(nn.Module):
    """Predicts the input state unchanged."""

    def forward(self, state: Tensor) -> Tensor:
        """Return a clone of the input state."""
        return state.clone()


class ConstantVelocityBaseline(nn.Module):
    """Free motion: x_{t+dt} = x_t + dt * v_t, v and m unchanged."""

    def __init__(self, dt: float) -> None:
        """Store the integration step as a buffer."""
        super().__init__()
        self.register_buffer("dt", torch.tensor(float(dt)))

    def forward(self, state: Tensor) -> Tensor:
        """Advance positions by dt * v; pass v and m through."""
        pos = state[..., :2]
        vel = state[..., 2:4]
        mass = state[..., 4:]
        pos_new = pos + self.dt * vel
        return torch.cat([pos_new, vel, mass], dim=-1)


class MeanVelocityBaseline(nn.Module):
    """Replaces every particle's velocity with the fitted global mean."""

    def __init__(self, dt: float, v_mean: Tensor) -> None:
        """Store dt and the (2,) mean velocity vector as buffers."""
        super().__init__()
        if v_mean.shape != (2,):
            msg = f"v_mean must have shape (2,), got {tuple(v_mean.shape)}"
            raise ValueError(msg)
        self.register_buffer("dt", torch.tensor(float(dt)))
        self.register_buffer("v_mean", v_mean.detach().clone().float())

    @classmethod
    def fit(cls, train_path: str, dt: float) -> MeanVelocityBaseline:
        """Compute v_mean as the global mean over trajectories x frames x particles."""
        with h5py.File(train_path, "r") as f:
            trajectories = f["trajectories"][:]
        velocities = torch.from_numpy(trajectories[..., 2:4]).float()
        v_mean = velocities.reshape(-1, 2).mean(dim=0)
        return cls(dt=dt, v_mean=v_mean)

    def forward(self, state: Tensor) -> Tensor:
        """Set v to v_mean, advance pos by dt * v_mean, pass m through."""
        pos = state[..., :2]
        mass = state[..., 4:]
        v_new = self.v_mean.expand_as(pos)
        pos_new = pos + self.dt * v_new
        return torch.cat([pos_new, v_new, mass], dim=-1)


class MeanStateBaseline(nn.Module):
    """Snaps every particle to the fitted dataset centre with zero velocity."""

    def __init__(self, x_mean: Tensor) -> None:
        """Store the (2,) mean position as a buffer."""
        super().__init__()
        if x_mean.shape != (2,):
            msg = f"x_mean must have shape (2,), got {tuple(x_mean.shape)}"
            raise ValueError(msg)
        self.register_buffer("x_mean", x_mean.detach().clone().float())

    @classmethod
    def fit(cls, train_path: str) -> MeanStateBaseline:
        """Compute x_mean as the global mean over trajectories x frames x particles."""
        with h5py.File(train_path, "r") as f:
            trajectories = f["trajectories"][:]
        positions = torch.from_numpy(trajectories[..., :2]).float()
        x_mean = positions.reshape(-1, 2).mean(dim=0)
        return cls(x_mean=x_mean)

    def forward(self, state: Tensor) -> Tensor:
        """Snap pos to x_mean, set v to zero, pass m through."""
        pos = state[..., :2]
        mass = state[..., 4:]
        x_new = self.x_mean.expand_as(pos)
        v_new = torch.zeros_like(pos)
        return torch.cat([x_new, v_new, mass], dim=-1)
