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
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from data.dataset import NBodyDataset
from models.egnn import EGNN
from models.hgnn import HGNN
from training._types import Checkpoint, TrainConfig, TrainResult
from training.diagnostics import TrainingDiagnostics
from utils import get_logger

logger = get_logger(__name__)


def load_config(path: str) -> TrainConfig:
    """Load a YAML config file into a typed TrainConfig."""
    with Path(path).open() as f:
        raw = yaml.safe_load(f)
    return TrainConfig.from_dict(raw)


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

    def __init__(self, cfg: TrainConfig, model: nn.Module | None = None) -> None:
        """Set up all training components from config."""
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
        train_set = NBodyDataset(cfg.data.train_path, cfg.data.n_train_trajectories)
        val_set = NBodyDataset(cfg.data.val_path)

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
        csv_path.write_text("epoch,train_loss,val_loss,lr\n")
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

                preds = self.model(inputs)
                loss = self.loss_fn(preds, targets)

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
                        targets,
                        preds.detach(),
                        batch_loss,
                        n_batches,
                        len(loader),
                    )

                if verbose:
                    batches.set_postfix(loss=f"{total_loss / n_batches:.6f}")

        return total_loss / n_batches

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
        torch.save(ckpt, self.ckpt_dir / "latest.pt")
        if is_best:
            torch.save(ckpt, self.ckpt_dir / "best.pt")

    def _log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        lr: float,
    ) -> None:
        """Write epoch metrics to CSV and console."""
        if self.csv_path is not None:
            with self.csv_path.open("a") as f:
                f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{lr:.2e}\n")

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


def train(cfg: TrainConfig, model: nn.Module | None = None) -> TrainResult:
    """Run Trainer with a compact function call."""
    return Trainer(cfg, model=model).run()


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
    args = parser.parse_args()

    config = load_config(args.config)
    if args.n_train is not None:
        config = replace(config, data=replace(config.data, n_train_trajectories=args.n_train))
    results = train(config)
    logger.info("results: %s", results)
