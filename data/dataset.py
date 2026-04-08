"""PyTorch Dataset wrapping the HDF5 trajectory files.

Each sample is a (state_t, state_t+1) pair where state has shape (n_particles, 5).
Model-agnostic: returns raw [x, y, vx, vy] per particle, no graph construction.

References:
    - HDF5 layout defined in data/generate.py
    - EGNN data loading pattern: https://github.com/vgsatorras/egnn
"""

import h5py
import torch
from torch.utils.data import Dataset


class NBodyDataset(Dataset):
    """Loads trajectories from an HDF5 file and returns consecutive state pairs.

    The HDF5 file contains:
        /trajectories — (n_trajectories, n_steps, n_particles, 5)
        /energies     — (n_trajectories, n_steps)

    This dataset flattens all trajectories into (state_t, state_t+1) pairs.
    Total samples = n_trajectories * (n_steps - 1).
    """

    def __init__(self, path: str) -> None:
        """Load trajectories from HDF5 and create consecutive state pairs."""
        with h5py.File(path, "r") as f:
            trajectories = f["trajectories"][:]

        _n_traj, _n_steps, n_particles, state_dim = trajectories.shape

        # consecutive pairs: input is step t, target is step t+1
        self.inputs = trajectories[:, :-1].reshape(-1, n_particles, state_dim)
        self.targets = trajectories[:, 1:].reshape(-1, n_particles, state_dim)

        # convert to float32 tensors
        self.inputs = torch.from_numpy(self.inputs).float()
        self.targets = torch.from_numpy(self.targets).float()

    def __len__(self) -> int:
        """Return total number of state-transition pairs."""
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (state_t, state_t+dt), each of shape (n_particles, 5)."""
        return self.inputs[idx], self.targets[idx]
