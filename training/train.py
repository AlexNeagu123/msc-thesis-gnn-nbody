"""Shared training pipeline for EGNN and HGNN.

References:
    - Training hyperparams comparison: edu/research/training-hyperparams-comparison.md
    - Architecture specs: edu/architecture-specs.md
    - HGNN (Bishnoi et al., 2023): https://arxiv.org/abs/2307.05299
    - EGNN (Satorras et al., 2021): https://arxiv.org/abs/2102.09844
"""

import argparse
import random
import subprocess
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from tqdm import tqdm

from data.dataset import NBodyDataset, TrajectoryWindowDataset
from models.egnn import EGNN
from models.hgnn import HGNN
from training._io import (
    append_metrics,
    init_metrics_csv,
    load_checkpoint,
    load_config,
    save_checkpoint,
)
from training._types import Checkpoint, EpochMetrics, TrainConfig, TrainResult
from training.diagnostics import TrainingDiagnostics
from utils import get_logger

logger = get_logger(__name__)


# re-exported so existing callers `from training.train import load_config` keep working
__all__ = ["Trainer", "build_model", "load_config", "train"]


def build_model(
    cfg: TrainConfig,
    pos_std: float = 1.0,
    vel_std: float = 1.0,
) -> nn.Module:
    """Instantiate a model based on config."""
    name = cfg.model.name

    if name == "egnn":
        return EGNN(
            hidden_dim=cfg.model.hidden_dim,
            n_layers=cfg.model.n_layers,
            pos_std=pos_std,
            vel_std=vel_std,
        )
    if name == "hgnn":
        return HGNN(
            hidden_dim=cfg.model.hidden_dim,
            n_layers=cfg.model.n_layers,
            dt=cfg.data.dt,
            pos_std=pos_std,
            vel_std=vel_std,
        )

    msg = f"Unknown model: {name}"
    raise ValueError(msg)


class Trainer:
    """Orchestrate model training, validation, checkpointing, and logging."""

    def __init__(
        self,
        cfg: TrainConfig,
        model: nn.Module | None = None,
        init_checkpoint: str | Path | None = None,
    ) -> None:
        """Set up all training components from config.

        Args:
            cfg: training configuration.
            model: optional pre-built model (dependency-injection for tests).
            init_checkpoint: optional path to a checkpoint whose model weights
                should be loaded as the starting point for a fresh run. The
                optimizer, scheduler, run_id, logs, and checkpoint dir are NOT
                inherited; they all start fresh from the current config.
        """
        horizon = cfg.training.multi_step_horizon
        if cfg.model.name == "hgnn" and horizon > 1:
            msg = (
                "multi_step_horizon > 1 is not supported for HGNN: "
                "rollout training compounds with HGNN's internal autograd "
                "and is too expensive to run by accident."
            )
            raise ValueError(msg)

        self.cfg = cfg
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._seed_everything(cfg.training.seed)
        self.device = self._resolve_device(cfg.training.device)
        logger.info(
            "run: %s | device: %s | seed: %s",
            self.run_id,
            self.device,
            cfg.training.seed,
        )

        self.train_loader, self.val_loader = self._setup_data()
        self.model = model.to(self.device) if model is not None else self._setup_model()

        if init_checkpoint is not None:
            self._load_init_checkpoint(init_checkpoint)

        self.loss_fn = self._build_loss_fn(cfg.training.loss)
        self.optimizer, self.scheduler = self._setup_optimizer()
        self.ckpt_dir = self._setup_checkpointing()
        self.csv_path = self._setup_logging()

        diag_dir = self.csv_path.parent if self.csv_path is not None else None
        self.diagnostics = TrainingDiagnostics(
            pos_std=self.pos_std,
            vel_std=self.vel_std,
            log_dir=diag_dir,
            dataset=self.train_loader.dataset,
        )

    def run(self) -> TrainResult:
        """Execute the full training loop."""
        cfg = self.cfg
        verbose = cfg.logging.enabled
        best_val_loss = float("inf")
        best_epoch = 0
        train_history: list[float] = []
        val_history: list[float] = []

        for epoch in range(1, cfg.training.epochs + 1):
            train_loss = self._run_epoch(training=True, verbose=verbose)
            val_loss = self._run_epoch(training=False, verbose=verbose)

            current_lr = self.optimizer.param_groups[0]["lr"]
            train_history.append(train_loss)
            val_history.append(val_loss)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch

            self._checkpoint(epoch, val_loss, is_best)
            self._log_epoch(epoch, train_loss, val_loss, current_lr)

        logger.info("best val loss: %.6f at epoch %d", best_val_loss, best_epoch)

        return TrainResult(
            best_val_loss=best_val_loss,
            final_train_loss=train_history[-1] if train_history else float("inf"),
            best_epoch=best_epoch,
            train_history=train_history,
            val_history=val_history,
        )

    # --- setup helpers ---

    def _setup_data(self) -> tuple[DataLoader, DataLoader]:
        """Create data loaders and training-set normalization stats."""
        cfg = self.cfg
        horizon = cfg.training.multi_step_horizon
        train_set = self._build_dataset(cfg.data.train_path, horizon, cfg.data.n_train_trajectories)
        val_set = self._build_dataset(cfg.data.val_path, horizon)

        self.pos_std = float(train_set.inputs[..., :2].std())
        self.vel_std = float(train_set.inputs[..., 2:4].std())
        logger.info("data stds: pos=%.4f, vel=%.4f", self.pos_std, self.vel_std)

        use_cuda = self.device.type == "cuda"
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            pin_memory=use_cuda,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            pin_memory=use_cuda,
        )

        logger.info(
            "data: %d train / %d val samples, batch_size=%d",
            len(train_set),
            len(val_set),
            cfg.training.batch_size,
        )

        return train_loader, val_loader

    def _setup_model(self) -> nn.Module:
        """Build the model and optionally print a summary."""
        model = build_model(self.cfg, pos_std=self.pos_std, vel_std=self.vel_std).to(self.device)

        if self.cfg.model.summary:
            n_particles = 3
            state_dim = 5
            summary(
                model,
                input_size=(self.cfg.training.batch_size, n_particles, state_dim),
                col_names=["input_size", "output_size", "num_params", "mult_adds"],
                depth=4,
                device=self.device,
            )

        return model

    def _load_init_checkpoint(self, path: str | Path) -> None:
        """Load model weights only from a previous checkpoint; everything else stays fresh.

        The optimizer, scheduler, epoch counter, run_id, logging dir, and
        checkpoint dir are all built from the current config. Only the
        model state_dict is taken from the checkpoint.

        A checkpoint with a non-None `model_name` must match `cfg.model.name`.
        Legacy checkpoints with `model_name=None` are accepted as-is.
        """
        path = Path(path)
        checkpoint = load_checkpoint(path, self.device)

        if checkpoint.model_name is not None and checkpoint.model_name != self.cfg.model.name:
            msg = (
                f"init checkpoint at {path} was trained for model "
                f"{checkpoint.model_name!r}, but current config is "
                f"{self.cfg.model.name!r}"
            )
            raise ValueError(msg)

        self.model.load_state_dict(checkpoint.model)
        logger.info(
            "init checkpoint: loaded weights from %s (epoch %d, val_loss %.6f)",
            path,
            checkpoint.epoch,
            checkpoint.val_loss,
        )

    def _setup_optimizer(
        self,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
        """Create optimizer and optional LR scheduler."""
        cfg = self.cfg
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )

        scheduler = None
        if cfg.scheduler.enabled:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=cfg.scheduler.patience,
                factor=cfg.scheduler.factor,
                min_lr=cfg.scheduler.min_lr,
            )

        return optimizer, scheduler

    def _setup_checkpointing(self) -> Path | None:
        """Create checkpoint directory if enabled."""
        if not self.cfg.checkpointing.enabled:
            return None

        ckpt_dir = Path(self.cfg.checkpointing.dir) / self.run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir

    def _setup_logging(self) -> Path | None:
        """Create log directory and CSV header if enabled."""
        if not self.cfg.logging.enabled:
            return None

        log_dir = Path(self.cfg.logging.dir) / self.run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = log_dir / "metrics.csv"
        init_metrics_csv(csv_path)
        return csv_path

    def apply_noise(self, inputs: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to positions and velocities."""
        noise = torch.zeros_like(inputs)
        noise[..., :2] = (
            torch.randn_like(inputs[..., :2]) * self.cfg.training.noise_factor * self.pos_std
        )
        noise[..., 2:4] = (
            torch.randn_like(inputs[..., 2:4]) * self.cfg.training.noise_factor * self.vel_std
        )
        return inputs + noise

    # --- epoch helpers ---

    def _run_epoch(self, *, training: bool, verbose: bool) -> float:
        """Run one train or validation epoch."""
        if training:
            self.model.train()
            loader = self.train_loader
        else:
            self.model.eval()
            loader = self.val_loader

        total_loss = 0.0
        n_batches = 0

        phase = "train" if training else "val"
        batches = tqdm(loader, desc=phase, leave=False) if verbose else loader

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for inputs, targets in batches:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                if training and self.cfg.training.noise_factor > 0:
                    inputs = self.apply_noise(inputs)

                preds, loss, diag_targets = self._compute_loss(inputs, targets)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.optimizer.step()

                batch_loss = loss.item()
                total_loss += batch_loss
                n_batches += 1

                if training:
                    self.diagnostics.check_batch(
                        inputs,
                        diag_targets,
                        preds.detach(),
                        batch_loss,
                        n_batches,
                        len(loader),
                    )

                if verbose:
                    batches.set_postfix(loss=f"{total_loss / n_batches:.6f}")

        return total_loss / n_batches

    def _compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dispatch to the configured loss path, returning (preds, loss, diag_targets).

        `diag_targets` is the single-step target tensor (B, N, 5) suitable for
        feeding to TrainingDiagnostics, regardless of which path runs.
        """
        if self.cfg.training.multi_step_horizon == 1:
            preds, loss = self._one_step_loss(inputs, targets)
            return preds, loss, targets
        preds, loss = self._multi_step_rollout_loss(inputs, targets)
        return preds, loss, targets[:, 0]

    def _one_step_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One forward pass; full-state MSE matching the legacy training path."""
        preds = self.model(inputs)
        loss = self.loss_fn(preds, targets)
        return preds, loss

    def _multi_step_rollout_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unroll the model `horizon` steps; weighted-mean MSE on (x, y, vx, vy).

        Loss is `sum_k gamma^k * MSE_k / sum_k gamma^k`, keeping the value on
        the same scale as one-step MSE so val_loss, scheduler thresholds, and
        gradient magnitudes stay comparable across horizon settings.

        Gradients flow through every rollout step; intermediate states are not
        detached. Only the initial input is noised (by `_run_epoch`); every
        subsequent state is the model's own prediction. Targets stay clean.

        Mass is excluded from the loss because the model passes it through
        unchanged at every step and the trajectory keeps it constant.
        """
        horizon = self.cfg.training.multi_step_horizon
        gamma = self.cfg.training.multi_step_gamma

        state = self.model(inputs)
        first_preds = state
        weighted_loss = self.loss_fn(state[..., :4], targets[:, 0, ..., :4])
        weight_sum = 1.0

        for k in range(1, horizon):
            state = self.model(state)
            step_loss = self.loss_fn(state[..., :4], targets[:, k, ..., :4])
            weight = gamma**k
            weighted_loss = weighted_loss + weight * step_loss
            weight_sum += weight

        return first_preds, weighted_loss / weight_sum

    def _checkpoint(self, epoch: int, val_loss: float, is_best: bool) -> None:
        """Save model checkpoint if enabled."""
        if self.ckpt_dir is None:
            return

        ckpt = Checkpoint(
            epoch=epoch,
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            val_loss=val_loss,
            config=asdict(self.cfg),
            model_name=self.cfg.model.name,
            run_id=self.run_id,
            pos_std=self.pos_std,
            vel_std=self.vel_std,
            git_commit=self._git_commit(),
        )
        save_checkpoint(self.ckpt_dir / "latest.pt", ckpt)
        if is_best:
            save_checkpoint(self.ckpt_dir / "best.pt", ckpt)

    def _log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        lr: float,
    ) -> None:
        """Write epoch metrics to CSV and console."""
        if self.csv_path is not None:
            append_metrics(
                self.csv_path,
                EpochMetrics(epoch=epoch, train_loss=train_loss, val_loss=val_loss, lr=lr),
            )

        logger.info(
            "epoch %3d/%d | train %.6f | val %.6f | lr %.2e",
            epoch,
            self.cfg.training.epochs,
            train_loss,
            val_loss,
            lr,
        )

    # --- static helpers ---

    @staticmethod
    def _build_dataset(
        path: str,
        horizon: int,
        n_trajectories: int | None = None,
    ) -> Dataset:
        """Pick the right dataset for the configured rollout horizon."""
        if horizon == 1:
            return NBodyDataset(path, n_trajectories)
        return TrajectoryWindowDataset(path, horizon=horizon, n_trajectories=n_trajectories)

    @staticmethod
    def _resolve_device(device_cfg: str) -> torch.device:
        """Resolve device string to a torch.device."""
        if device_cfg != "auto":
            return torch.device(device_cfg)

        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _seed_everything(seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _build_loss_fn(name: str) -> nn.Module:
        """Build a loss function by name."""
        losses = {
            "mse": nn.MSELoss(),
            "mae": nn.L1Loss(),
        }
        if name not in losses:
            msg = f"Unknown loss: {name}. Choose from {list(losses.keys())}"
            raise ValueError(msg)
        return losses[name]

    @staticmethod
    def _git_commit() -> str | None:
        """Return the current git commit hash when available."""
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()


def train(
    cfg: TrainConfig,
    model: nn.Module | None = None,
    init_checkpoint: str | Path | None = None,
) -> TrainResult:
    """Run Trainer with a compact function call."""
    return Trainer(cfg, model=model, init_checkpoint=init_checkpoint).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model YAML config (e.g. configs/egnn.yaml).",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=None,
        help="Override n_train_trajectories from the config (data-scaling runs).",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint whose model weights initialise this run. "
            "Optimizer, scheduler, run_id, logs, and checkpoints all start fresh."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.n_train is not None:
        config = replace(config, data=replace(config.data, n_train_trajectories=args.n_train))
    results = train(config, init_checkpoint=args.init_checkpoint)
    logger.info("results: %s", results)
