"""PyTorch Dataset wrapping the HDF5 trajectory files.

References:
    - HDF5 layout owned by data/_io.py
    - EGNN data loading pattern: https://github.com/vgsatorras/egnn
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from data._io import read_trajectories


class NBodyDataset(Dataset):
    """Load HDF5 trajectories as consecutive state pairs."""

    def __init__(self, path: str, n_trajectories: int | None = None) -> None:
        """Load trajectories from HDF5 and create consecutive state pairs.

        Args:
            path: HDF5 file path.
            n_trajectories: if set, use only the first N trajectories. Used by
                data-scaling experiments to slice nested subsets from a larger
                generated file.
        """
        trajectories = read_trajectories(Path(path)).states

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
