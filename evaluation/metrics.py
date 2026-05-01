"""Shared evaluation utilities.

References:
    - Energy: T = 0.5 * sum(m * v^2), V = -sum(G * m_i * m_j / r_ij)
    - Rollout: autoregressive single-step prediction fed back as input.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from data.dataset import NBodyDataset

COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]


@dataclass
class RolloutMSE:
    """Rollout MSE summary with divergence visibility."""

    per_trajectory: npt.NDArray[np.floating]
    mean: npt.NDArray[np.floating]
    median: npt.NDArray[np.floating]
    std: npt.NDArray[np.floating]
    finite_fraction: npt.NDArray[np.floating]


# --- computation ---


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


def _color(index: int) -> str:
    """Return a stable plot color for a particle index."""
    return COLORS[index % len(COLORS)]


# --- plotting ---


def plot_trajectories(
    test_traj: npt.NDArray[np.floating],
    predicted: npt.NDArray[np.floating],
    traj_indices: list[int],
    model_name: str = "EGNN",
) -> None:
    """Plot side-by-side trajectory comparison."""
    from matplotlib import pyplot as plt

    _fig, axes = plt.subplots(len(traj_indices), 2, figsize=(12, 6 * len(traj_indices)))
    axes = np.atleast_2d(axes)

    for row, idx in enumerate(traj_indices):
        true = test_traj[idx]
        pred = predicted[idx]
        n_particles = true.shape[1]

        for col, (data, title) in enumerate([
            (true, "ground truth"),
            (pred, f"{model_name} predicted"),
        ]):
            ax = axes[row, col]
            for p in range(n_particles):
                ax.plot(
                    data[:, p, 0],
                    data[:, p, 1],
                    color=_color(p),
                    alpha=0.7,
                    linewidth=0.8,
                )
                ax.scatter(
                    data[0, p, 0],
                    data[0, p, 1],
                    color=_color(p),
                    marker="o",
                    s=30,
                    zorder=5,
                )
                ax.scatter(
                    data[-1, p, 0],
                    data[-1, p, 1],
                    color=_color(p),
                    marker="x",
                    s=30,
                    zorder=5,
                )
            ax.set_title(f"trajectory {idx} - {title}")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def animate_trajectories(
    test_traj: npt.NDArray[np.floating],
    predicted: npt.NDArray[np.floating],
    traj_indices: list[int],
    model_name: str = "EGNN",
) -> None:
    """Show side-by-side animated rollouts."""
    for traj_idx in traj_indices:
        _animate_single(test_traj[traj_idx], predicted[traj_idx], traj_idx, model_name)


def _animate_single(
    true: npt.NDArray[np.floating],
    pred: npt.NDArray[np.floating],
    traj_idx: int,
    model_name: str,
) -> None:
    """Animate one trajectory comparison."""
    from IPython.display import HTML, display
    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation

    pad = 0.5
    n_particles = true.shape[1]
    fig, (ax_true, ax_pred) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, title in [
        (ax_true, true, "ground truth"),
        (ax_pred, pred, f"{model_name} predicted"),
    ]:
        finite = np.isfinite(data[:, :, :2]).all(axis=-1)
        finite_pos = data[:, :, :2][finite]
        if len(finite_pos) == 0:
            finite_pos = np.array([[-1.0, -1.0], [1.0, 1.0]])
        xmin, xmax = finite_pos[:, 0].min() - pad, finite_pos[:, 0].max() + pad
        ymin, ymax = finite_pos[:, 1].min() - pad, finite_pos[:, 1].max() + pad
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"trajectory {traj_idx} - {title}")

    true_trails = [
        ax_true.plot([], [], color=_color(p), alpha=0.4, linewidth=0.8)[0]
        for p in range(n_particles)
    ]
    true_dots = [
        ax_true.plot([], [], "o", color=_color(p), markersize=8)[0] for p in range(n_particles)
    ]
    pred_trails = [
        ax_pred.plot([], [], color=_color(p), alpha=0.4, linewidth=0.8)[0]
        for p in range(n_particles)
    ]
    pred_dots = [
        ax_pred.plot([], [], "o", color=_color(p), markersize=8)[0] for p in range(n_particles)
    ]

    artists = true_trails + true_dots + pred_trails + pred_dots

    def _init(artists: list = artists) -> list:
        for a in artists:
            a.set_data([], [])
        return artists

    def _update(
        frame: int,
        true: npt.NDArray = true,
        pred: npt.NDArray = pred,
        true_trails: list = true_trails,
        true_dots: list = true_dots,
        pred_trails: list = pred_trails,
        pred_dots: list = pred_dots,
    ) -> list:
        for p in range(n_particles):
            true_trails[p].set_data(true[: frame + 1, p, 0], true[: frame + 1, p, 1])
            true_dots[p].set_data([true[frame, p, 0]], [true[frame, p, 1]])
            pred_trails[p].set_data(pred[: frame + 1, p, 0], pred[: frame + 1, p, 1])
            pred_dots[p].set_data([pred[frame, p, 0]], [pred[frame, p, 1]])
        return true_trails + true_dots + pred_trails + pred_dots

    anim = FuncAnimation(
        fig,
        _update,
        init_func=_init,
        frames=len(true),
        interval=50,
        blit=True,
    )
    plt.close(fig)
    display(HTML(anim.to_jshtml()))


def plot_energy(
    test_traj: npt.NDArray[np.floating],
    predicted: npt.NDArray[np.floating],
    traj_indices: list[int],
    model_name: str = "EGNN",
) -> None:
    """Plot physical energy drift."""
    from matplotlib import pyplot as plt

    n_traj = test_traj.shape[0]
    _fig, axes = plt.subplots(len(traj_indices), 1, figsize=(10, 4 * len(traj_indices)))
    axes = np.atleast_1d(axes)

    for row, idx in enumerate(traj_indices):
        ax = axes[row]
        ground_truth_energy = compute_energy(test_traj[idx])
        pred_energy = compute_energy(predicted[idx])

        steps = np.arange(len(ground_truth_energy))
        ax.plot(steps, ground_truth_energy, label="ground truth", color="tab:blue", linewidth=1.0)
        ax.plot(
            steps,
            pred_energy,
            label=model_name,
            color="tab:red",
            linewidth=1.0,
            alpha=0.8,
        )

        ax.set_xlabel("step")
        ax.set_ylabel("total energy")
        ax.set_title(f"trajectory {idx}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("relative energy drift at final step (|E_pred(T) - E_pred(0)| / |E_pred(0)|):")
    for idx in range(n_traj):
        pred_e = compute_energy(predicted[idx])
        with np.errstate(divide="ignore", invalid="ignore"):
            drift = abs((pred_e[-1] - pred_e[0]) / pred_e[0])
        print(f"  trajectory {idx:2d}: {drift:.4f}")


def plot_rollout_mse(
    test_traj: npt.NDArray[np.floating],
    predicted: npt.NDArray[np.floating],
) -> None:
    """Plot rollout MSE vs time step, preserving divergence information."""
    from matplotlib import pyplot as plt

    n_steps = test_traj.shape[1] - 1
    mse = compute_rollout_mse(test_traj, predicted)
    steps = np.arange(n_steps + 1)

    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(steps, mse.mean, label="mean finite MSE", color="tab:red", linewidth=1.0)
    ax.plot(steps, mse.median, label="median finite MSE", color="tab:blue", linewidth=1.0)
    ax.fill_between(steps, mse.mean - mse.std, mse.mean + mse.std, color="tab:red", alpha=0.2)
    ax.set_xlabel("step")
    ax.set_ylabel("MSE")
    ax.set_title("rollout MSE vs time step (linear)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps[1:], mse.mean[1:], label="mean finite MSE", color="tab:red", linewidth=1.0)
    ax.plot(
        steps[1:],
        mse.median[1:],
        label="median finite MSE",
        color="tab:blue",
        linewidth=1.0,
    )
    ax.fill_between(
        steps[1:],
        np.maximum(mse.mean[1:] - mse.std[1:], 1e-10),
        mse.mean[1:] + mse.std[1:],
        color="tab:red",
        alpha=0.2,
    )
    ax.set_xlabel("step")
    ax.set_ylabel("MSE")
    ax.set_title("rollout MSE vs time step (log)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    summary_steps = [1, 50, 100, n_steps]
    for step in dict.fromkeys(s for s in summary_steps if 0 < s <= n_steps):
        print(f"step {step:3d} MSE: {mse.mean[step]:.6f}")
    print(f"finite rollouts at final step: {mse.finite_fraction[-1] * 100:.1f}%")


def plot_pos_vel_error(
    test_traj: npt.NDArray[np.floating],
    predicted: npt.NDArray[np.floating],
) -> None:
    """Plot position vs velocity error over rollout."""
    from matplotlib import pyplot as plt

    n_steps = test_traj.shape[1] - 1
    diff = predicted - test_traj
    with np.errstate(over="ignore", invalid="ignore"):
        pos_mse = (diff[..., :2] ** 2).mean(axis=(2, 3))
        vel_mse = (diff[..., 2:4] ** 2).mean(axis=(2, 3))

    mean_pos = np.nanmean(np.where(np.isfinite(pos_mse), pos_mse, np.nan), axis=0)
    mean_vel = np.nanmean(np.where(np.isfinite(vel_mse), vel_mse, np.nan), axis=0)
    steps = np.arange(n_steps + 1)

    _fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, mean_pos, label="position MSE", color="tab:blue", linewidth=1.0)
    ax.plot(steps, mean_vel, label="velocity MSE", color="tab:orange", linewidth=1.0)
    ax.set_xlabel("step")
    ax.set_ylabel("MSE")
    ax.set_title("position vs velocity error over rollout")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.show()

    print(f"final step - position MSE: {mean_pos[-1]:.6f}, velocity MSE: {mean_vel[-1]:.6f}")
    print(f"velocity/position ratio:   {mean_vel[-1] / mean_pos[-1]:.1f}x")


def plot_loss_distribution(sample_losses: npt.NDArray[np.floating]) -> None:
    """Plot histogram of per-sample single-step losses."""
    from matplotlib import pyplot as plt

    print(f"total samples: {len(sample_losses)}")
    print(f"mean loss:        {sample_losses.mean():.6f}")
    print(f"median loss:      {np.median(sample_losses):.6f}")
    print(f"max loss:         {sample_losses.max():.6f}")
    print(f"99th percentile:  {np.percentile(sample_losses, 99):.6f}")

    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(
        sample_losses,
        bins=100,
        color="tab:blue",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_xlabel("single-step MSE")
    ax.set_ylabel("count")
    ax.set_title("loss distribution (all samples)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(
        sample_losses,
        bins=100,
        color="tab:blue",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_xlabel("single-step MSE")
    ax.set_ylabel("count")
    ax.set_title("loss distribution (log scale)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_loss_vs_distance(
    min_distances: npt.NDArray[np.floating],
    sample_losses: npt.NDArray[np.floating],
) -> None:
    """Plot single-step loss vs minimum pairwise distance."""
    from matplotlib import pyplot as plt

    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(min_distances, sample_losses, alpha=0.15, s=2, color="tab:blue")
    ax.set_xlabel("min pairwise distance")
    ax.set_ylabel("single-step MSE")
    ax.set_title("loss vs closest particle pair")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(min_distances, sample_losses, alpha=0.15, s=2, color="tab:blue")
    ax.set_xlabel("min pairwise distance")
    ax.set_ylabel("single-step MSE")
    ax.set_title("loss vs closest particle pair (log-log)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
