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
from training._types import (
    Checkpoint,
    EpochMetrics,
    EpochRunSummary,
    RolloutScore,
    TrainConfig,
    TrainResult,
)
from training.diagnostics import TrainingDiagnostics
from training.rollout_score import RolloutScoreEvaluator
from utils import get_logger

logger = get_logger(__name__)


# re-exported so existing callers `from training.train import load_config` keep working
__all__ = ["Trainer", "apply_artifact_dir", "build_model", "load_config", "train"]


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
        if cfg.training.checkpoint_metric not in ("val_loss", "rollout_score"):
            msg = (
                f"checkpoint_metric must be 'val_loss' or 'rollout_score', "
                f"got {cfg.training.checkpoint_metric!r}"
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
        self.current_clip_norm = cfg.training.gradient_clip_norm

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

        self.rollout_evaluator: RolloutScoreEvaluator | None = None
        if cfg.training.checkpoint_metric == "rollout_score":
            self.rollout_evaluator = RolloutScoreEvaluator(
                val_path=cfg.data.val_path,
                train_path=cfg.data.train_path,
                dt=cfg.data.dt,
                device=self.device,
            )
            logger.info("checkpoint metric: rollout_score (evaluator constructed lazily)")
        else:
            logger.info("checkpoint metric: val_loss")

    def run(self) -> TrainResult:
        """Execute the full training loop, stage by stage when curriculum is set."""
        cfg = self.cfg
        verbose = cfg.logging.enabled
        best_val_loss = float("inf")
        best_selected_score = float("inf")
        best_epoch = 0
        train_history: list[float] = []
        val_history: list[float] = []
        last_rollout: RolloutScore | None = None
        epoch_index = 0

        stages = self._stages()
        total_epochs = sum(n for _, n in stages)
        for stage_idx, (horizon, n_stage_epochs) in enumerate(stages):
            if horizon != self.current_horizon:
                self.train_loader, self.val_loader = self._setup_loaders_for_horizon(horizon)
                self.current_horizon = horizon
                self.diagnostics.dataset = self.train_loader.dataset
                # reset is gated on horizon change so it fires alongside the loader rebuild;
                # consecutive same-horizon stages keep optimizer state intact by design.
                if cfg.training.reset_optimizer_on_stage:
                    self.optimizer, self.scheduler = self._setup_optimizer()
                    logger.info("optimizer reset at stage transition | horizon=%d", horizon)

            self.current_clip_norm = self._stage_clip_norm(stage_idx)

            stage_lr = self.optimizer.param_groups[0]["lr"]
            score_before = f"{last_rollout.score:+.4f}" if last_rollout is not None else "n/a"
            logger.info(
                "stage %d/%d start | horizon=%d | epochs %d..%d | lr=%.2e "
                "| clip=%.2f | rscore_before=%s",
                stage_idx + 1,
                len(stages),
                horizon,
                epoch_index + 1,
                epoch_index + n_stage_epochs,
                stage_lr,
                self.current_clip_norm,
                score_before,
            )

            for _ in range(n_stage_epochs):
                epoch_index += 1
                train_summary = self._run_epoch(training=True, verbose=verbose)
                val_summary = self._run_epoch(training=False, verbose=verbose)
                train_loss = train_summary.loss
                val_loss = val_summary.loss

                train_history.append(train_loss)
                val_history.append(val_loss)

                rollout = (
                    self.rollout_evaluator.score(self.model) if self.rollout_evaluator else None
                )
                selected_metric = cfg.training.checkpoint_metric
                selected_score = rollout.score if rollout is not None else val_loss

                # scheduler tracks the same metric used for best-checkpoint selection so
                # LR adaptation and best.pt agree on what improvement means.
                if self.scheduler is not None:
                    self.scheduler.step(selected_score)

                # capture LR after the scheduler so a reduction this epoch is reflected
                current_lr = self.optimizer.param_groups[0]["lr"]

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                is_best = selected_score < best_selected_score
                if is_best:
                    best_selected_score = selected_score
                    best_epoch = epoch_index

                self._checkpoint(
                    epoch_index,
                    val_loss=val_loss,
                    rollout=rollout,
                    selected_metric=selected_metric,
                    selected_score=selected_score,
                    is_best=is_best,
                )
                self._log_epoch(
                    epoch_index, total_epochs, train_summary, val_loss, current_lr, rollout
                )
                last_rollout = rollout

            score_after = f"{last_rollout.score:+.4f}" if last_rollout is not None else "n/a"
            logger.info(
                "stage %d/%d done  | horizon=%d | rscore_after=%s",
                stage_idx + 1,
                len(stages),
                horizon,
                score_after,
            )

        val_loss_at_best = val_history[best_epoch - 1] if best_epoch > 0 else float("inf")
        logger.info(
            "best %s: %.6f at epoch %d (val_loss=%.6f at that epoch); "
            "min val_loss across run: %.6f",
            cfg.training.checkpoint_metric,
            best_selected_score,
            best_epoch,
            val_loss_at_best,
            best_val_loss,
        )

        return TrainResult(
            best_val_loss=best_val_loss,
            final_train_loss=train_history[-1] if train_history else float("inf"),
            best_epoch=best_epoch,
            train_history=train_history,
            val_history=val_history,
        )

    # --- setup helpers ---

    def _setup_data(self) -> tuple[DataLoader, DataLoader]:
        """Compute one-time normalization stats and build initial-stage loaders.

        Stats are computed exactly once from whichever dataset corresponds to
        the initial stage horizon. They are NOT recomputed when loaders are
        rebuilt for later curriculum stages, so the model's pos_std/vel_std
        buffers remain stable across the whole run. Practical effect: with
        the planned schedules that always start at horizon=1, stats come
        from the full one-step dataset and match prior behavior.
        """
        cfg = self.cfg
        initial_horizon = self._initial_stage_horizon()
        train_set = self._build_dataset(
            cfg.data.train_path, initial_horizon, cfg.data.n_train_trajectories
        )
        val_set = self._build_dataset(cfg.data.val_path, initial_horizon)

        self.pos_std = float(train_set.inputs[..., :2].std())
        self.vel_std = float(train_set.inputs[..., 2:4].std())
        self.current_horizon = initial_horizon
        logger.info("data stds: pos=%.4f, vel=%.4f", self.pos_std, self.vel_std)

        train_loader, val_loader = self._loaders_from_sets(train_set, val_set)
        logger.info(
            "data: %d train / %d val samples, horizon=%d, batch_size=%d",
            len(train_set),
            len(val_set),
            initial_horizon,
            cfg.training.batch_size,
        )
        return train_loader, val_loader

    def _setup_loaders_for_horizon(self, horizon: int) -> tuple[DataLoader, DataLoader]:
        """Build fresh train/val loaders at the given horizon (no stat recomputation)."""
        cfg = self.cfg
        train_set = self._build_dataset(cfg.data.train_path, horizon, cfg.data.n_train_trajectories)
        val_set = self._build_dataset(cfg.data.val_path, horizon)
        train_loader, val_loader = self._loaders_from_sets(train_set, val_set)
        logger.info(
            "loaders rebuilt: %d train / %d val samples, horizon=%d, batch_size=%d",
            len(train_set),
            len(val_set),
            horizon,
            cfg.training.batch_size,
        )
        return train_loader, val_loader

    def _loaders_from_sets(
        self, train_set: Dataset, val_set: Dataset
    ) -> tuple[DataLoader, DataLoader]:
        """Wrap train/val datasets into DataLoaders with the trainer's settings."""
        use_cuda = self.device.type == "cuda"
        train_loader = DataLoader(
            train_set,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            pin_memory=use_cuda,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            pin_memory=use_cuda,
        )
        return train_loader, val_loader

    def _initial_stage_horizon(self) -> int:
        """Horizon used for the first stage of the run.

        In curriculum mode this is the first scheduled horizon; otherwise it
        is the configured single-horizon `multi_step_horizon`.
        """
        cfg = self.cfg
        if cfg.training.curriculum_horizons is not None:
            return cfg.training.curriculum_horizons[0]
        return cfg.training.multi_step_horizon

    def _stages(self) -> list[tuple[int, int]]:
        """Return the schedule as a list of (horizon, epochs_in_stage) tuples."""
        cfg = self.cfg
        if cfg.training.curriculum_horizons is not None:
            return list(
                zip(
                    cfg.training.curriculum_horizons,
                    cfg.training.curriculum_epochs,
                    strict=True,
                )
            )
        return [(cfg.training.multi_step_horizon, cfg.training.epochs)]

    def _stage_clip_norm(self, stage_idx: int) -> float:
        """Return the gradient-clip max-norm for the given curriculum stage.

        In single-horizon mode there is one stage, so the configured
        `gradient_clip_norm` is always used. In curriculum mode, an explicit
        `curriculum_gradient_clip_norms` overrides it per stage; if the list
        is None, every stage falls back to `gradient_clip_norm`.
        """
        cfg = self.cfg
        if cfg.training.curriculum_gradient_clip_norms is not None:
            return cfg.training.curriculum_gradient_clip_norms[stage_idx]
        return cfg.training.gradient_clip_norm

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

    def _run_epoch(self, *, training: bool, verbose: bool) -> EpochRunSummary:
        """Run one train or validation epoch and return a summary.

        Skip semantics differ for the two failure modes:
            - non-finite loss respects `skip_nonfinite_batches`. With the
              flag on, the batch is dropped before backward; with it off,
              backward and `clip_grad_norm_` still run so the explosion is
              visible (e.g. via the gradient norm in the next check).
            - non-finite gradient norm is ALWAYS skipped. Stepping the
              optimizer with NaN/Inf gradients is never useful and
              corrupts every parameter, so this protection is unconditional.

        `batch_idx` reflects the loader's batch position (1..len(loader))
        so diagnostics keep their loader-relative meaning even when some
        batches are skipped. `n_taken` only counts batches whose loss
        contributed to the running mean.
        """
        if training:
            self.model.train()
            loader = self.train_loader
        else:
            self.model.eval()
            loader = self.val_loader

        total_loss = 0.0
        n_taken = 0
        grad_norms: list[float] = []
        n_clipped = 0
        skipped_batches = 0

        phase = "train" if training else "val"
        batches = tqdm(loader, desc=phase, leave=False) if verbose else loader

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch_idx, (inputs, targets) in enumerate(batches, start=1):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                if training and self.cfg.training.noise_factor > 0:
                    inputs = self.apply_noise(inputs)

                preds, loss, diag_targets = self._compute_loss(inputs, targets)

                if training:
                    if self.cfg.training.skip_nonfinite_batches and not torch.isfinite(loss).item():
                        logger.warning("skipped batch: non-finite loss (%s)", loss.item())
                        skipped_batches += 1
                        self.optimizer.zero_grad(set_to_none=True)
                        continue

                    self.optimizer.zero_grad()
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.current_clip_norm
                    )

                    # always skip on non-finite gradients; stepping corrupts every param
                    if not torch.isfinite(grad_norm).item():
                        logger.warning(
                            "skipped batch: non-finite grad norm (%s)",
                            grad_norm.item(),
                        )
                        skipped_batches += 1
                        self.optimizer.zero_grad(set_to_none=True)
                        continue

                    grad_norm_value = float(grad_norm)
                    grad_norms.append(grad_norm_value)
                    if grad_norm_value > self.current_clip_norm:
                        n_clipped += 1
                    self.optimizer.step()

                batch_loss = loss.item()
                total_loss += batch_loss
                n_taken += 1

                if training:
                    self.diagnostics.check_batch(
                        inputs,
                        diag_targets,
                        preds.detach(),
                        batch_loss,
                        batch_idx,
                        len(loader),
                    )

                if verbose:
                    batches.set_postfix(loss=f"{total_loss / n_taken:.6f}")

        avg_loss = total_loss / n_taken if n_taken > 0 else float("nan")

        if not training:
            return EpochRunSummary(loss=avg_loss)

        if grad_norms:
            return EpochRunSummary(
                loss=avg_loss,
                grad_norm_mean=float(np.mean(grad_norms)),
                grad_norm_max=float(np.max(grad_norms)),
                grad_clip_fraction=n_clipped / len(grad_norms),
                skipped_batches=skipped_batches,
            )

        if skipped_batches > 0:
            logger.warning(
                "epoch ended with all %d batches skipped; train loss is nan",
                skipped_batches,
            )
        return EpochRunSummary(loss=avg_loss, skipped_batches=skipped_batches)

    def _compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dispatch to the configured loss path, returning (preds, loss, diag_targets).

        `diag_targets` is the single-step target tensor (B, N, 5) suitable for
        feeding to TrainingDiagnostics, regardless of which path runs.
        """
        if self.current_horizon == 1:
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
        horizon = self.current_horizon
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

    def _checkpoint(
        self,
        epoch: int,
        *,
        val_loss: float,
        rollout: RolloutScore | None,
        selected_metric: str,
        selected_score: float,
        is_best: bool,
    ) -> None:
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
            selected_metric=selected_metric,
            selected_score=selected_score,
            rollout_score=rollout.score if rollout is not None else None,
        )
        save_checkpoint(self.ckpt_dir / "latest.pt", ckpt)
        if is_best:
            save_checkpoint(self.ckpt_dir / "best.pt", ckpt)

    def _log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_summary: EpochRunSummary,
        val_loss: float,
        lr: float,
        rollout: RolloutScore | None,
    ) -> None:
        """Write epoch metrics to CSV and console.

        `total_epochs` is the run-wide denominator. In single-horizon mode it
        equals `cfg.training.epochs`; in curriculum mode it is the sum of
        `curriculum_epochs`. Computing it in `run` avoids the `epoch/0`
        rendering artefact that would otherwise show up under curriculum.

        Gradient diagnostics for the epoch come from `train_summary` and are
        only populated for training epochs that took at least one step.
        """
        train_loss = train_summary.loss

        if self.csv_path is not None:
            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                lr=lr,
                rollout_score=rollout.score if rollout is not None else None,
                dominance_horizon=rollout.dominance_horizon if rollout is not None else None,
                fraction_beating_baseline=(
                    rollout.fraction_beating_baseline if rollout is not None else None
                ),
                final_ratio=rollout.final_ratio if rollout is not None else None,
                grad_norm_mean=train_summary.grad_norm_mean,
                grad_norm_max=train_summary.grad_norm_max,
                grad_clip_fraction=train_summary.grad_clip_fraction,
                skipped_batches=train_summary.skipped_batches,
            )
            append_metrics(self.csv_path, metrics)

        if rollout is None:
            logger.info(
                "epoch %3d/%d | train %.6f | val %.6f | lr %.2e",
                epoch,
                total_epochs,
                train_loss,
                val_loss,
                lr,
            )
        else:
            logger.info(
                "epoch %3d/%d | train %.6f | val %.6f | rscore %+.4f "
                "| dom %d | beat %.2f | lr %.2e",
                epoch,
                total_epochs,
                train_loss,
                val_loss,
                rollout.score,
                rollout.dominance_horizon,
                rollout.fraction_beating_baseline,
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


def apply_artifact_dir(cfg: TrainConfig, artifact_dir: str | Path) -> TrainConfig:
    """Override checkpointing.dir and logging.dir to a single artifact root.

    Force-enables both checkpointing and logging because the override only
    makes sense when artifacts are actually written. Use this at the CLI
    entry point or from orchestration scripts (scaling, sweep) to put every
    run under the canonical `runs/<mode>/...` layout without editing YAMLs.
    """
    artifact_dir = str(artifact_dir)
    return replace(
        cfg,
        checkpointing=replace(cfg.checkpointing, enabled=True, dir=artifact_dir),
        logging=replace(cfg.logging, enabled=True, dir=artifact_dir),
    )


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
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help=(
            "Single artifact root under which checkpoints and logs are both "
            "written for this run, force-enabling both. Lets local runs match "
            "the canonical runs/<mode>/<model>/n<N>/ layout without editing the "
            "YAML. The trainer still appends <run_id> as a per-run subdirectory."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.n_train is not None:
        config = replace(config, data=replace(config.data, n_train_trajectories=args.n_train))
    if args.artifact_dir is not None:
        config = apply_artifact_dir(config, args.artifact_dir)
    results = train(config, init_checkpoint=args.init_checkpoint)
    logger.info("results: %s", results)
