"""Training diagnostics for monitoring health and debugging failures.

References:
    - Outlier batch analysis motivated by train loss spikes in chaotic
      3-body data (ejection trajectories with positions ~20-40).
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger("training.diagnostics")


class TrainingDiagnostics:
    """Batch-level diagnostics for the training loop."""

    def __init__(
        self,
        pos_std: float,
        vel_std: float,
        outlier_threshold: float = 100.0,
        log_dir: Path | None = None,
        dataset: Dataset | None = None,
    ) -> None:
        """Initialize diagnostics with normalization stats."""
        self.pos_std = pos_std
        self.vel_std = vel_std
        self.outlier_threshold = outlier_threshold
        self.dataset = dataset
        self._setup_logger(log_dir)

    def _setup_logger(self, log_dir: Path | None) -> None:
        """Configure file-based logging for diagnostics output."""
        logger.setLevel(logging.WARNING)
        logger.propagate = False
        logger.handlers.clear()

        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_dir / "diagnostics.log")
        else:
            handler = logging.StreamHandler()

        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s %(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)

    def check_batch(
        self,
        inputs: Tensor,
        targets: Tensor,
        preds: Tensor,
        batch_loss: float,
        batch_idx: int,
        n_batches: int,
    ) -> None:
        """Run all batch-level diagnostics."""
        if batch_loss > self.outlier_threshold:
            self._log_outlier(inputs, targets, preds, batch_loss, batch_idx, n_batches)

    def _log_outlier(
        self,
        inputs: Tensor,
        targets: Tensor,
        preds: Tensor,
        batch_loss: float,
        batch_idx: int,
        n_batches: int,
    ) -> None:
        """Log details for the worst sample in an outlier batch."""
        with torch.no_grad():
            diff = (preds - targets)[..., :4]
            per_sample = (diff**2).mean(dim=(1, 2))
            worst_idx = per_sample.argmax().item()
            worst_mse = per_sample[worst_idx].item()

            inp = inputs[worst_idx].cpu().numpy()  # (N, 5)
            tgt = targets[worst_idx].cpu().numpy()
            pred = preds[worst_idx].cpu().numpy()

            min_dist = self._min_pairwise_distance(inp[:, :2])
            comp_mse = ((pred[:, :4] - tgt[:, :4]) ** 2).mean(axis=0)
            pos_norm = inp[:, :2] / self.pos_std
            vel_norm = inp[:, 2:4] / self.vel_std

            location = self._locate_sample(targets[worst_idx])

        logger.warning(
            "[OUTLIER] batch %d/%d | batch_loss: %.3e | worst_sample_mse: %.3e\n"
            "  location:       %s\n"
            "  input pos:      %s\n"
            "  input vel:      %s\n"
            "  min dist:       %.4f\n"
            "  target pos:     %s\n"
            "  target vel:     %s\n"
            "  pred pos:       %s\n"
            "  pred vel:       %s\n"
            "  component MSE:  pos_x=%.3e  pos_y=%.3e  vel_x=%.3e  vel_y=%.3e\n"
            "  normalized pos: %s\n"
            "  normalized vel: %s",
            batch_idx,
            n_batches,
            batch_loss,
            worst_mse,
            location,
            np.array2string(inp[:, :2], precision=3, suppress_small=True),
            np.array2string(inp[:, 2:4], precision=3, suppress_small=True),
            min_dist,
            np.array2string(tgt[:, :2], precision=3, suppress_small=True),
            np.array2string(tgt[:, 2:4], precision=3, suppress_small=True),
            np.array2string(pred[:, :2], precision=3, floatmode="maxprec_equal"),
            np.array2string(pred[:, 2:4], precision=3, floatmode="maxprec_equal"),
            comp_mse[0],
            comp_mse[1],
            comp_mse[2],
            comp_mse[3],
            np.array2string(pos_norm, precision=3, suppress_small=True),
            np.array2string(vel_norm, precision=3, suppress_small=True),
        )

    def _locate_sample(self, target: Tensor) -> str:
        """Find the trajectory and step index for a target state."""
        if self.dataset is None:
            return "unknown (no dataset reference)"

        dataset_targets = self.dataset.targets
        target_cpu = target.cpu()
        if dataset_targets.ndim == 4:
            dataset_targets = dataset_targets[:, 0]

        matches = (dataset_targets == target_cpu).all(dim=(1, 2))
        idxs = matches.nonzero(as_tuple=True)[0]

        if len(idxs) == 0:
            return "unknown (no match)"

        flat_idx = idxs[0].item()
        if hasattr(self.dataset, "steps_per_traj"):
            samples_per_traj = self.dataset.steps_per_traj
        else:
            samples_per_traj = self.dataset.windows_per_traj
        traj_idx = flat_idx // samples_per_traj
        step_idx = flat_idx % samples_per_traj

        return f"trajectory {traj_idx}, step {step_idx}"

    @staticmethod
    def _min_pairwise_distance(positions: np.ndarray) -> float:
        """Compute the minimum distance between any pair of particles."""
        n = len(positions)
        min_dist = float("inf")
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.sqrt(((positions[i] - positions[j]) ** 2).sum()))
                min_dist = min(min_dist, d)
        return min_dist
