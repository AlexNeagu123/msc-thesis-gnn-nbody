"""PyTorch Datasets wrapping the HDF5 trajectory files.

References:
    - EGNN data loading pattern: https://github.com/vgsatorras/egnn
"""

from pathlib import Path

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset

from data._io import read_states


class NBodyDataset(Dataset):
    """Load HDF5 trajectories as consecutive state pairs."""

    def __init__(self, path: str, n_trajectories: int | None = None) -> None:
        """Load trajectories and pair consecutive states; n_trajectories takes a prefix."""
        trajectories = read_states(Path(path))

        if n_trajectories is not None:
            available = trajectories.shape[0]
            if n_trajectories <= 0:
                msg = f"n_trajectories must be positive, got {n_trajectories}"
                raise ValueError(msg)
            if n_trajectories > available:
                msg = (
                    f"requested {n_trajectories} trajectories from {path}, "
                    f"but only {available} are available"
                )
                raise ValueError(msg)
            trajectories = trajectories[:n_trajectories]

        self.n_trajectories, n_steps, n_particles, state_dim = trajectories.shape
        self.steps_per_traj = n_steps - 1

        self.inputs = trajectories[:, :-1].reshape(-1, n_particles, state_dim)
        self.targets = trajectories[:, 1:].reshape(-1, n_particles, state_dim)

        self.inputs = torch.from_numpy(self.inputs).float()
        self.targets = torch.from_numpy(self.targets).float()

    def __len__(self) -> int:
        """Return total number of state-transition pairs."""
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (state_t, state_t+dt), each of shape (n_particles, 5)."""
        return self.inputs[idx], self.targets[idx]


class TrajectoryWindowDataset(Dataset):
    """Load HDF5 trajectories as (state_t, next `horizon` states) pairs for rollout training.

    Windows never cross trajectory boundaries.
    """

    def __init__(
        self,
        path: str,
        horizon: int,
        n_trajectories: int | None = None,
    ) -> None:
        """Load trajectories and build size-`horizon` sliding windows (1 <= horizon < n_frames)."""
        if horizon < 1:
            msg = f"horizon must be >= 1, got {horizon}"
            raise ValueError(msg)

        trajectories = read_states(Path(path))

        if n_trajectories is not None:
            available = trajectories.shape[0]
            if n_trajectories <= 0:
                msg = f"n_trajectories must be positive, got {n_trajectories}"
                raise ValueError(msg)
            if n_trajectories > available:
                msg = (
                    f"requested {n_trajectories} trajectories from {path}, "
                    f"but only {available} are available"
                )
                raise ValueError(msg)
            trajectories = trajectories[:n_trajectories]

        n_traj, n_frames, n_particles, state_dim = trajectories.shape

        if horizon >= n_frames:
            msg = f"horizon must be < n_frames ({n_frames}), got {horizon}"
            raise ValueError(msg)

        self.n_trajectories = n_traj
        self.horizon = horizon
        self.windows_per_traj = n_frames - horizon

        # inputs: state_t for valid window starts in each trajectory
        inputs = trajectories[:, : self.windows_per_traj]
        inputs = inputs.reshape(-1, n_particles, state_dim)

        # targets: sliding window of size `horizon` over trajectories[:, 1:]
        # window starting at t holds trajectories[:, t+1 : t+1+horizon]
        future = trajectories[:, 1:]
        # sliding_window_view appends the window axis last; move it to axis 2
        windows = sliding_window_view(future, window_shape=horizon, axis=1)
        windows = np.moveaxis(windows, -1, 2).copy()
        targets = windows.reshape(-1, horizon, n_particles, state_dim)

        self.inputs = torch.from_numpy(inputs).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self) -> int:
        """Return the total number of windows across trajectories."""
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (state_t, future_window) of shapes (N, 5) and (horizon, N, 5)."""
        return self.inputs[idx], self.targets[idx]
