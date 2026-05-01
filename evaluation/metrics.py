"""Pure numeric metric computations.

Plotting and animation helpers live in evaluation/plots.py so this module
stays free of matplotlib/IPython dependencies.

References:
    - Energy: T = 0.5 * sum(m * v^2), V = -sum(G * m_i * m_j / r_ij)
    - Rollout: autoregressive single-step prediction fed back as input.
"""

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from data.dataset import NBodyDataset
from evaluation._types import RolloutMSE


def compute_energy(
    states: npt.NDArray[np.floating],
    G: float = 1.0,
) -> npt.NDArray[np.floating]:
    """Compute total physical energy at each step."""
    pos = states[..., :2]
    vel = states[..., 2:4]
    mass = states[..., 4]

    kinetic = 0.5 * (mass * (vel**2).sum(axis=-1)).sum(axis=-1)

    n_particles = states.shape[1]
    potential = np.zeros(len(states))
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            dx = pos[:, i] - pos[:, j]
            r = np.sqrt((dx**2).sum(axis=-1))
            potential -= G * mass[:, i] * mass[:, j] / r

    return kinetic + potential


def rollout(
    model: nn.Module,
    initial_state: Tensor,
    n_steps: int,
    device: torch.device,
) -> npt.NDArray[np.floating]:
    """Autoregressively predict `n_steps` from one initial state."""
    states = [initial_state.cpu()]
    current = initial_state.unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(n_steps):
            pred = model(current)
            states.append(pred.squeeze(0).cpu())
            current = pred

    return torch.stack(states).numpy()


def run_all_rollouts(
    model: nn.Module,
    test_traj: npt.NDArray[np.floating],
    device: torch.device,
) -> npt.NDArray[np.floating]:
    """Run autoregressive rollout on every test trajectory."""
    n_traj = test_traj.shape[0]
    n_steps = test_traj.shape[1] - 1

    predicted = []
    for i in range(n_traj):
        initial = torch.from_numpy(test_traj[i, 0]).float()
        pred = rollout(model, initial, n_steps, device)
        predicted.append(pred)

    return np.array(predicted)


def compute_single_step_metrics(
    model: nn.Module,
    test_path: str,
    device: torch.device,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Compute per-sample single-step losses and minimum pairwise distances."""
    test_set = NBodyDataset(test_path)
    loader = DataLoader(test_set, batch_size=256, shuffle=False)

    sample_losses = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            preds = model(inputs)
            diff = (preds.cpu() - targets) ** 2
            per_sample = diff[..., :4].mean(dim=(1, 2))
            sample_losses.append(per_sample)

    sample_losses = torch.cat(sample_losses).numpy()

    positions = test_set.inputs.numpy()[..., :2]
    min_distances = min_pairwise_distances(positions)

    return sample_losses, min_distances


def min_pairwise_distances(positions: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Return the closest particle-pair distance for each sample."""
    n_particles = positions.shape[1]
    distances = []

    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            delta = positions[:, i] - positions[:, j]
            distances.append(np.sqrt((delta**2).sum(axis=-1)))

    return np.minimum.reduce(distances)


def compute_rollout_mse(
    test_traj: npt.NDArray[np.floating],
    predicted: npt.NDArray[np.floating],
) -> RolloutMSE:
    """Compute rollout MSE while preserving non-finite divergence."""
    diff_state = predicted[..., :4] - test_traj[..., :4]
    with np.errstate(over="ignore", invalid="ignore"):
        per_trajectory = (diff_state**2).mean(axis=(2, 3))

    finite = np.where(np.isfinite(per_trajectory), per_trajectory, np.nan)
    return RolloutMSE(
        per_trajectory=per_trajectory,
        mean=np.nanmean(finite, axis=0),
        median=np.nanmedian(finite, axis=0),
        std=np.nanstd(finite, axis=0),
        finite_fraction=np.isfinite(per_trajectory).mean(axis=0),
    )
