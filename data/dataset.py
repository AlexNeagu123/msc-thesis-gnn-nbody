"""PyTorch Dataset wrapping the HDF5 trajectory files.

References:
    - HDF5 layout defined in data/generate.py
    - EGNN data loading pattern: https://github.com/vgsatorras/egnn
"""

import h5py
import torch
from torch.utils.data import Dataset


class NBodyDataset(Dataset):
    """Load HDF5 trajectories as consecutive state pairs."""

    def __init__(self, path: str) -> None:
        """Load trajectories from HDF5 and create consecutive state pairs."""
        with h5py.File(path, "r") as f:
            trajectories = f["trajectories"][:]

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
