"""
PyTorch Dataset wrapping the HDF5 trajectory files.

Each sample is a (state_t, state_t+1) pair where state has shape (n_particles, 4).
"""

import h5py
import torch
from torch.utils.data import Dataset


class NBodyDataset(Dataset):
    """
    Loads trajectories from an HDF5 file and returns consecutive state pairs.

    Args:
        path: path to HDF5 file produced by generate.py
        n_particles: number of particles (used to validate loaded data)
    """

    def __init__(self, path: str, n_particles: int) -> None:
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (state_t, state_t+dt), each of shape (n_particles, 4)."""
        pass
