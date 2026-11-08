"""Memory-bounded trajectory windows for the canonical HGNN training path."""

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from data._io import read_states


class LazyTrajectoryWindowDataset(Dataset):
    """Keep one float32 trajectory tensor; slice only the requested target window.

    Memory is O(trajectories*frames), not O(trajectories*frames*horizon).
    Horizon-one samples retain the existing (state, next_state) interface.
    """

    def __init__(
        self,
        path: str,
        horizon: int,
        n_trajectories: int | None = None,
        *,
        states: Tensor | None = None,
    ) -> None:
        """Read trajectories once, or share an existing tensor across curriculum stages."""
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError("horizon must be an integer >= 1")
        if states is None:
            states = torch.from_numpy(read_states(Path(path))).float()
            if n_trajectories is not None:
                if (
                    not isinstance(n_trajectories, int)
                    or isinstance(n_trajectories, bool)
                    or not 1 <= n_trajectories <= len(states)
                ):
                    raise ValueError("n_trajectories must be a positive available trajectory count")
                if n_trajectories < len(states):
                    states = states[:n_trajectories].clone()
        elif n_trajectories is not None:
            raise ValueError("n_trajectories cannot accompany shared states")
        if states.ndim != 4 or not len(states) or horizon >= states.shape[1]:
            raise ValueError("expected nonempty trajectories and horizon < n_frames")
        self.states = states
        self.path = path
        self.horizon = horizon
        self.n_trajectories = len(states)
        self.windows_per_traj = states.shape[1] - horizon

    @property
    def inputs(self) -> Tensor:
        """Flatten valid starts for existing normalization code (at most O(data) copy)."""
        return self.states[:, : self.windows_per_traj].reshape(-1, *self.states.shape[2:])

    def with_horizon(self, horizon: int) -> "LazyTrajectoryWindowDataset":
        """Change the window length without rereading or duplicating trajectory storage."""
        return type(self)(self.path, horizon, states=self.states)

    def __len__(self) -> int:
        """Number of valid starts, never crossing a simulation boundary."""
        return self.n_trajectories * self.windows_per_traj

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return views; DataLoader collates only the windows needed for this batch."""
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        trajectory, start = divmod(idx, self.windows_per_traj)
        future = self.states[trajectory, start + 1 : start + 1 + self.horizon]
        return self.states[trajectory, start], future[0] if self.horizon == 1 else future

    def locate_target(self, target: Tensor) -> str:
        """Locate a first-step diagnostic target without materializing rollout windows."""
        matches = (self.states[:, 1 : self.windows_per_traj + 1] == target.cpu()).all(dim=(-1, -2))
        indices = matches.nonzero()
        if len(indices) == 0:
            return "unknown (no match)"
        trajectory, start = indices[0].tolist()
        return f"trajectory {trajectory}, step {start}"
