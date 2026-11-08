"""Shared training pipeline for EGNN and HGNN.

References:
    - Training hyperparams comparison: edu/research/training-hyperparams-comparison.md
    - Architecture specs: edu/architecture-specs.md
    - HGNN (Bishnoi et al., 2023): https://arxiv.org/abs/2307.05299
    - EGNN (Satorras et al., 2021): https://arxiv.org/abs/2102.09844
"""

import argparse
import copy
import json
import math
import random
import subprocess
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchinfo import summary
from tqdm import tqdm

from data.dataset import NBodyDataset, TrajectoryWindowDataset
from data.lazy_dataset import LazyTrajectoryWindowDataset
from evaluation._io import _json_safe
from models.egnn import EGNN
from models.hgnn import HGNN
from models.hgnn_canonical import MODEL_ID as CANONICAL_MODEL_ID
from models.hgnn_canonical import CanonicalHGNN
from training._io import (
    append_metrics,
    init_metrics_csv,
    load_checkpoint,
    load_config,
    save_checkpoint,
)
from training._types import (
    BucketRolloutScore,
    Checkpoint,
    EpochMetrics,
    EpochRunSummary,
    RolloutScore,
    TrainConfig,
    TrainResult,
)
from training.diagnostics import TrainingDiagnostics
from training.fixed_validation import METRIC as FIXED_POSITION_MAE
from training.fixed_validation import FixedRolloutEvaluator, file_sha256
from training.rollout_score import BucketRolloutScoreEvaluator, RolloutScoreEvaluator
from utils import get_logger

logger = get_logger(__name__)


# re-exported so existing callers `from training.train import load_config` keep working
__all__ = ["Trainer", "apply_artifact_dir", "build_model", "load_config", "train"]


def _format_rollout_or_bucket(
    rollout: RolloutScore | None,
    bucket: BucketRolloutScore | None,
) -> str:
    """Render the rollout (or bucket-macro) score for log lines, "n/a" in val_loss mode."""
    if rollout is not None:
        return f"{rollout.score:+.4f}"
    if bucket is not None:
        return f"{bucket.macro:+.4f}"
    return "n/a"


def build_model(
    cfg: TrainConfig,
    pos_std: float = 1.0,
    vel_std: float = 1.0,
    momentum_std: float | None = None,
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

    if name == CANONICAL_MODEL_ID:
        if momentum_std is None:
            raise ValueError("canonical HGNN requires an explicit training-fitted momentum_std")
        return CanonicalHGNN(
            hidden_dim=cfg.model.hidden_dim,
            n_layers=cfg.model.n_layers,
            dt=cfg.data.dt,
            pos_std=pos_std,
            momentum_std=momentum_std,
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
        resume_checkpoint: str | Path | None = None,
        allow_runtime_change: bool = False,
    ) -> None:
        """Set up all training components from config.

        `model` is a DI hook for tests. `init_checkpoint` seeds model weights only;
        optimizer, scheduler, run_id, logs, and checkpoint dir all start fresh.
        """
        valid_metrics = (
            "val_loss",
            "rollout_score",
            "bucket_macro_rollout_score",
            FIXED_POSITION_MAE,
        )
        if cfg.training.checkpoint_metric not in valid_metrics:
            msg = (
                f"checkpoint_metric must be one of {valid_metrics}, "
                f"got {cfg.training.checkpoint_metric!r}"
            )
            raise ValueError(msg)

        if (
            cfg.model.name == CANONICAL_MODEL_ID
            and init_checkpoint is not None
            and resume_checkpoint is None
        ):
            raise ValueError(
                "canonical HGNN benchmark starts from scratch; init_checkpoint is unsupported"
            )
        if cfg.training.dynamic_loss_scaling != "none" and cfg.model.name != CANONICAL_MODEL_ID:
            raise ValueError("elapsed-time loss scaling requires canonical HGNN")
        from training import resume

        if resume_checkpoint is not None and (init_checkpoint is not None or model is not None):
            raise ValueError("resume cannot be combined with init_checkpoint or model injection")
        self._resume_checkpoint = (
            load_checkpoint(Path(resume_checkpoint), torch.device("cpu"))
            if resume_checkpoint is not None
            else None
        )
        if self._resume_checkpoint is not None:
            resume.validate(self._resume_checkpoint, cfg)
        self._progress: dict = {}
        self._best_checkpoint: Checkpoint | None = None
        self._resume_lineage: list = []
        self.cfg = cfg
        run_format = "%Y%m%d_%H%M%S_%f" if cfg.model.name == CANONICAL_MODEL_ID else "%Y%m%d_%H%M%S"
        self.run_id = datetime.now().strftime(run_format)
        if self._resume_checkpoint is not None:
            self.run_id = self._resume_checkpoint.run_id
        self.artifact_id = (
            self.run_id
            if self._resume_checkpoint is None
            else f"{self.run_id}-resume-{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        )
        self.optimizer_steps = 0
        self.attempted_train_batches = 0
        self.validation_history: list[dict] = []
        self._train_generator = torch.Generator().manual_seed(cfg.training.seed)

        self._seed_everything(cfg.training.seed)
        self.device = self._resolve_device(cfg.training.device)
        logger.info(
            "run: %s | device: %s | seed: %s",
            self.run_id,
            self.device,
            cfg.training.seed,
        )

        self.data_hashes = {}
        self.source_hashes = {}
        if cfg.model.name == CANONICAL_MODEL_ID:
            self.data_hashes = {
                name: file_sha256(Path(path))
                for name, path in (("train", cfg.data.train_path), ("val", cfg.data.val_path))
            }
            root = Path(__file__).resolve().parents[1]
            sources = (
                "models/hgnn.py",
                "models/hgnn_canonical.py",
                "models/baselines.py",
                "data/lazy_dataset.py",
                "training/train.py",
                "training/_io.py",
                "training/resume.py",
                "training/_types.py",
                "training/fixed_validation.py",
                "training/rollout_score.py",
                "evaluation/rk4_metrics.py",
            )
            self.source_hashes = {name: file_sha256(root / name) for name in sources}
        self.train_loader, self.val_loader = self._setup_data()
        for name, path in (("train", cfg.data.train_path), ("val", cfg.data.val_path)):
            if name in self.data_hashes and file_sha256(Path(path)) != self.data_hashes[name]:
                raise RuntimeError(f"{name} source changed during training setup")
        self.model = model.to(self.device) if model is not None else self._setup_model()
        self.current_clip_norm = cfg.training.gradient_clip_norm

        if init_checkpoint is not None:
            self._load_init_checkpoint(init_checkpoint)

        self.loss_fn = self._build_loss_fn(cfg.training.loss)
        self.optimizer, self.scheduler = self._setup_optimizer()
        self.ckpt_dir = self._setup_checkpointing()

        # built before logging so the CSV header can widen with bin_order in bucket mode
        self.rollout_evaluator: RolloutScoreEvaluator | None = None
        self.bucket_evaluator: BucketRolloutScoreEvaluator | None = None
        self.fixed_evaluator: FixedRolloutEvaluator | None = None
        self.bucket_bin_order: tuple[str, ...] = ()
        if cfg.training.checkpoint_metric == FIXED_POSITION_MAE:
            self.fixed_evaluator = FixedRolloutEvaluator(
                val_path=cfg.data.val_path,
                train_path=cfg.data.train_path,
                dt=cfg.data.dt,
                device=self.device,
                steps=cfg.training.validation_rollout_steps,
                batch_size=cfg.training.validation_batch_size,
                n_train_trajectories=cfg.data.n_train_trajectories,
                expected_train_sha256=self.data_hashes.get("train"),
                expected_val_sha256=self.data_hashes.get("val"),
            )
            logger.info("checkpoint metric: fixed position MAE (independent of training horizon)")
        elif cfg.training.checkpoint_metric == "rollout_score":
            self.rollout_evaluator = RolloutScoreEvaluator(
                val_path=cfg.data.val_path,
                train_path=cfg.data.train_path,
                dt=cfg.data.dt,
                device=self.device,
            )
            logger.info("checkpoint metric: rollout_score (evaluator constructed lazily)")
        elif cfg.training.checkpoint_metric == "bucket_macro_rollout_score":
            self.bucket_evaluator = BucketRolloutScoreEvaluator(
                val_path=cfg.data.val_path,
                train_path=cfg.data.train_path,
                dt=cfg.data.dt,
                device=self.device,
            )
            self.bucket_bin_order = self.bucket_evaluator.bin_order
            logger.info(
                "checkpoint metric: bucket_macro_rollout_score | bins=%s",
                self.bucket_bin_order,
            )
        else:
            logger.info("checkpoint metric: val_loss")

        self.csv_path = self._setup_logging()

        diag_dir = self.csv_path.parent if self.csv_path is not None else None
        self.diagnostics = TrainingDiagnostics(
            pos_std=self.pos_std,
            vel_std=self.vel_std,
            log_dir=diag_dir,
            dataset=self.train_loader.dataset,
        )

        if self._resume_checkpoint is not None:
            resume.restore(self, self._resume_checkpoint, allow_runtime_change=allow_runtime_change)

    def run(self, *, max_rounds: int | None = None) -> TrainResult:
        """Execute the full training loop, stage by stage when curriculum is set."""
        if max_rounds is not None and (
            not isinstance(max_rounds, int) or isinstance(max_rounds, bool) or max_rounds < 1
        ):
            raise ValueError("max_rounds must be positive")
        cfg = self.cfg
        verbose = cfg.logging.enabled
        best_val_loss = self._progress.get("best_val_loss", float("inf"))
        best_selected_score = self._progress.get("best_selected_score", float("inf"))
        best_epoch = self._progress.get("best_epoch", 0)
        train_history: list[float] = list(self._progress.get("train_history", []))
        val_history: list[float] = list(self._progress.get("val_history", []))
        last_rollout: RolloutScore | None = None
        epoch_index = len(train_history)
        completed = epoch_index
        rounds_this_call = 0
        stage_start = 0

        last_bucket: BucketRolloutScore | None = None

        stages = self._stages()
        total_epochs = sum(n for _, n in stages)
        for stage_idx, (horizon, n_stage_epochs) in enumerate(stages):
            first_round = max(0, completed - stage_start)
            stage_start += n_stage_epochs
            if first_round >= n_stage_epochs:
                continue
            target_stage_lr = self._stage_lr(stage_idx)
            if horizon != self.current_horizon:
                self.train_loader, self.val_loader = self._setup_loaders_for_horizon(horizon)
                self.current_horizon = horizon
                self.diagnostics.dataset = self.train_loader.dataset
                # gated on horizon change: consecutive same-horizon stages keep optimizer state
                if cfg.training.reset_optimizer_on_stage:
                    self.optimizer, self.scheduler = self._setup_optimizer(lr=target_stage_lr)
                    logger.info("optimizer reset at stage transition | horizon=%d", horizon)

            # apply per-stage LR before the stage-start log line; idempotent after a reset
            if cfg.training.curriculum_lrs is not None and first_round == 0:
                for pg in self.optimizer.param_groups:
                    pg["lr"] = target_stage_lr

            self.current_clip_norm = self._stage_clip_norm(stage_idx)

            stage_lr = self.optimizer.param_groups[0]["lr"]
            score_before = _format_rollout_or_bucket(last_rollout, last_bucket)
            logger.info(
                "stage %d/%d start | horizon=%d | epochs %d..%d | lr=%.2e "
                "| clip=%.2f | rscore_before=%s",
                stage_idx + 1,
                len(stages),
                horizon,
                epoch_index + 1,
                epoch_index + n_stage_epochs - first_round,
                stage_lr,
                self.current_clip_norm,
                score_before,
            )

            for _ in range(first_round, n_stage_epochs):
                epoch_index += 1
                train_summary = self._run_epoch(training=True, verbose=verbose)
                fixed_report = None
                train_loss = train_summary.loss
                if self.fixed_evaluator is not None:
                    fixed_report = self.fixed_evaluator.score(self.model)
                    val_loss = fixed_report["model"]["state4_mse"]
                else:
                    val_summary = self._run_epoch(training=False, verbose=verbose)
                    val_loss = val_summary.loss

                train_history.append(train_loss)
                val_history.append(val_loss)

                rollout: RolloutScore | None = None
                bucket: BucketRolloutScore | None = None
                if self.rollout_evaluator is not None:
                    rollout = self.rollout_evaluator.score(self.model)
                elif self.bucket_evaluator is not None:
                    bucket = self.bucket_evaluator.score(self.model)

                selected_metric = cfg.training.checkpoint_metric
                if fixed_report is not None:
                    selected_score = fixed_report["model"]["selection_score"]
                    self._record_fixed_validation(epoch_index, fixed_report)
                elif rollout is not None:
                    selected_score = rollout.score
                elif bucket is not None:
                    selected_score = bucket.macro
                else:
                    selected_score = val_loss

                # scheduler tracks the same metric as best-checkpoint selection
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

                if self.cfg.model.name == CANONICAL_MODEL_ID:
                    self._progress = {
                        "best_val_loss": best_val_loss,
                        "best_selected_score": best_selected_score,
                        "best_epoch": best_epoch,
                        "train_history": train_history,
                        "val_history": val_history,
                    }
                self._checkpoint(
                    epoch_index,
                    val_loss=val_loss,
                    rollout=rollout,
                    bucket=bucket,
                    selected_metric=selected_metric,
                    selected_score=selected_score,
                    is_best=is_best,
                )
                self._log_epoch(
                    epoch_index,
                    total_epochs,
                    train_summary,
                    val_loss,
                    current_lr,
                    rollout,
                    bucket,
                )
                last_rollout = rollout
                last_bucket = bucket
                rounds_this_call += 1
                if max_rounds is not None and rounds_this_call >= max_rounds:
                    return TrainResult(
                        best_val_loss, train_history[-1], best_epoch, train_history, val_history
                    )

            score_after = _format_rollout_or_bucket(last_rollout, last_bucket)
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

    # setup helpers
    def _setup_data(self) -> tuple[DataLoader, DataLoader]:
        """Compute one-time normalization stats and build initial-stage loaders.

        Stats are computed once from the initial-stage dataset and never recomputed,
        so pos_std/vel_std stay fixed across curriculum stages.
        """
        cfg = self.cfg
        initial_horizon = self._initial_stage_horizon()
        train_set = self._build_dataset(
            cfg.data.train_path,
            initial_horizon,
            cfg.data.n_train_trajectories,
            lazy=cfg.model.name == CANONICAL_MODEL_ID,
        )
        val_set = self._build_dataset(
            cfg.data.val_path,
            initial_horizon,
            lazy=cfg.model.name == CANONICAL_MODEL_ID,
        )

        if self._resume_checkpoint is not None:
            saved = self._resume_checkpoint.resume_state
            self.pos_std, self.vel_std = saved["pos_std"], saved["vel_std"]
            self.momentum_std = saved["momentum_std"]
            if not all(
                math.isfinite(value) and value > 0
                for value in (self.pos_std, self.vel_std, self.momentum_std)
            ):
                raise ValueError("resume normalization scales must be finite and positive")
        else:
            self.pos_std = float(train_set.inputs[..., :2].std())
            self.vel_std = float(train_set.inputs[..., 2:4].std())
            if cfg.model.name == CANONICAL_MODEL_ID:
                momentum = train_set.inputs[..., 2:4] * train_set.inputs[..., 4:]
                self.momentum_std = float(momentum.std())
                logger.info("canonical momentum std (training only): %.6f", self.momentum_std)
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
        if cfg.model.name == CANONICAL_MODEL_ID:
            train_set = self.train_loader.dataset.with_horizon(horizon)
            val_set = self.val_loader.dataset.with_horizon(horizon)
        else:
            train_set = self._build_dataset(
                cfg.data.train_path, horizon, cfg.data.n_train_trajectories
            )
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
        cap = self.cfg.training.max_train_batches_per_epoch
        if cap is None:
            train_loader = DataLoader(
                train_set,
                batch_size=self.cfg.training.batch_size,
                shuffle=True,
                pin_memory=use_cuda,
            )
        else:
            # Bounded attempts, not a prefix of simulations/windows or an unbounded retry loop.
            sampler = RandomSampler(
                train_set,
                replacement=True,
                num_samples=cap * self.cfg.training.batch_size,
                generator=self._train_generator,
            )
            train_loader = DataLoader(
                train_set,
                batch_size=self.cfg.training.batch_size,
                sampler=sampler,
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
        """First-stage horizon: the first curriculum entry, else multi_step_horizon."""
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
        """Gradient-clip max-norm for the stage: curriculum_gradient_clip_norms, else the base."""
        cfg = self.cfg
        if cfg.training.curriculum_gradient_clip_norms is not None:
            return cfg.training.curriculum_gradient_clip_norms[stage_idx]
        return cfg.training.gradient_clip_norm

    def _stage_lr(self, stage_idx: int) -> float:
        """Optimizer LR for the stage: curriculum_lrs, else the base lr."""
        cfg = self.cfg
        if cfg.training.curriculum_lrs is not None:
            return cfg.training.curriculum_lrs[stage_idx]
        return cfg.training.lr

    def _setup_model(self) -> nn.Module:
        """Build the model and optionally print a summary."""
        if self.cfg.model.name == CANONICAL_MODEL_ID:
            model = build_model(
                self.cfg,
                pos_std=self.pos_std,
                vel_std=self.vel_std,
                momentum_std=self.momentum_std,
            ).to(self.device)
        else:
            model = build_model(self.cfg, pos_std=self.pos_std, vel_std=self.vel_std).to(
                self.device
            )

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
        """Load model weights only from a checkpoint; everything else stays fresh.

        A non-None checkpoint `model_name` must match `cfg.model.name`.
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
        lr: float | None = None,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
        """Create optimizer and optional LR scheduler; `lr` overrides cfg.training.lr."""
        cfg = self.cfg
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr if lr is not None else cfg.training.lr,
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

        ckpt_dir = Path(self.cfg.checkpointing.dir) / self.artifact_id
        ckpt_dir.mkdir(parents=True, exist_ok=self.cfg.model.name != CANONICAL_MODEL_ID)
        return ckpt_dir

    def _setup_logging(self) -> Path | None:
        """Create log directory and CSV header if enabled; width follows bucket_bin_order."""
        if not self.cfg.logging.enabled:
            return None

        log_dir = Path(self.cfg.logging.dir) / self.artifact_id
        # Checkpoints and logs may intentionally share this run directory.
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = log_dir / "metrics.csv"
        init_metrics_csv(csv_path, bin_names=self.bucket_bin_order)
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

    # epoch helpers
    def _run_epoch(self, *, training: bool, verbose: bool) -> EpochRunSummary:
        """Run one train or validation epoch and return a summary.

        Non-finite loss is skipped only when `skip_nonfinite_batches` is set;
        a non-finite gradient norm is always skipped.
        """
        if training:
            self.model.train()
            loader = self.train_loader
        else:
            self.model.eval()
            loader = self.val_loader

        total_loss = 0.0
        total_weight = 0
        n_taken = 0
        grad_norms: list[float] = []
        n_clipped = 0
        skipped_batches = 0

        phase = "train" if training else "val"
        batches = tqdm(loader, desc=phase, leave=False) if verbose else loader

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch_idx, (inputs, targets) in enumerate(batches, start=1):
                if training:
                    self.attempted_train_batches += 1
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
                    self.optimizer_steps += 1

                batch_loss = loss.item()
                weight = len(inputs) if self.cfg.model.name == CANONICAL_MODEL_ID else 1
                total_loss += batch_loss * weight
                total_weight += weight
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
                    batches.set_postfix(loss=f"{total_loss / total_weight:.6f}")

        avg_loss = total_loss / total_weight if n_taken > 0 else float("nan")

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
        """Dispatch to one-step or rollout loss; returns (preds, loss, single-step diag_targets)."""
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
        """One forward pass; full-state MSE against the next-step target."""
        preds = self.model(inputs)
        if self.cfg.model.name == CANONICAL_MODEL_ID:
            loss = self._dynamic_step_loss(preds, targets, step=1)
        else:
            loss = self.loss_fn(preds, targets)
        return preds, loss

    def _dynamic_step_loss(
        self, predicted: torch.Tensor, target: torch.Tensor, *, step: int
    ) -> torch.Tensor:
        """Optional velocity-unit residual scaling; original unweighted loss is unchanged."""
        if self.cfg.training.dynamic_loss_scaling == "none":
            return self.loss_fn(predicted[..., :4], target[..., :4])
        if not math.isfinite(self.vel_std) or self.vel_std <= 0:
            raise ValueError("elapsed-time loss requires a positive training velocity scale")
        elapsed = step * self.cfg.data.dt
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("elapsed-time loss requires positive elapsed time")
        error = predicted[..., :4] - target[..., :4]
        return 0.5 * (
            (error[..., :2] / (elapsed * self.vel_std)).square().mean()
            + (error[..., 2:4] / self.vel_std).square().mean()
        )

    def _multi_step_rollout_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unroll `horizon` steps; gamma-weighted-mean MSE on (x, y, vx, vy).

        Loss is sum_k gamma^k * MSE_k / sum_k gamma^k, on the same scale as one-step MSE.
        Gradients flow through every step; mass is excluded (passed through unchanged).
        """
        horizon = self.current_horizon
        gamma = self.cfg.training.multi_step_gamma

        state = self.model(inputs)
        first_preds = state
        weighted_loss = self._dynamic_step_loss(state, targets[:, 0], step=1)
        weight_sum = 1.0

        for k in range(1, horizon):
            state = self.model(state)
            step_loss = self._dynamic_step_loss(state, targets[:, k], step=k + 1)
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
        bucket: BucketRolloutScore | None,
        selected_metric: str,
        selected_score: float,
        is_best: bool,
    ) -> None:
        """Save latest.pt (and best.pt when is_best) if checkpointing is enabled."""
        if self.ckpt_dir is None:
            return

        if rollout is not None:
            ckpt_rollout_score: float | None = rollout.score
        elif bucket is not None:
            ckpt_rollout_score = bucket.macro
        else:
            ckpt_rollout_score = None

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
            rollout_score=ckpt_rollout_score,
        )
        if self.cfg.model.name == CANONICAL_MODEL_ID and self._progress:
            from training.resume import capture

            if is_best:
                self._best_checkpoint = copy.deepcopy(ckpt)
            ckpt.resume_state = capture(self)
            save_checkpoint(self.ckpt_dir / "latest.pt", ckpt)
            if self._best_checkpoint is not None:
                save_checkpoint(self.ckpt_dir / "best.pt", self._best_checkpoint)
        else:
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
        bucket: BucketRolloutScore | None,
    ) -> None:
        """Write epoch metrics to CSV and console."""
        train_loss = train_summary.loss

        if rollout is not None:
            scalar_score: float | None = rollout.score
        elif bucket is not None:
            scalar_score = bucket.macro
        else:
            scalar_score = None

        if self.csv_path is not None:
            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                lr=lr,
                rollout_score=scalar_score,
                dominance_horizon=rollout.dominance_horizon if rollout is not None else None,
                fraction_beating_baseline=(
                    rollout.fraction_beating_baseline if rollout is not None else None
                ),
                final_ratio=rollout.final_ratio if rollout is not None else None,
                grad_norm_mean=train_summary.grad_norm_mean,
                grad_norm_max=train_summary.grad_norm_max,
                grad_clip_fraction=train_summary.grad_clip_fraction,
                skipped_batches=train_summary.skipped_batches,
                bucket_per_bin=bucket.per_bin if bucket is not None else None,
            )
            append_metrics(self.csv_path, metrics, bin_names=self.bucket_bin_order)

        if rollout is not None:
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
        elif bucket is not None:
            logger.info(
                "epoch %3d/%d | train %.6f | val %.6f | macro %+.4f | bins %d | lr %.2e",
                epoch,
                total_epochs,
                train_loss,
                val_loss,
                bucket.macro,
                len(bucket.per_bin),
                lr,
            )
        else:
            logger.info(
                "epoch %3d/%d | train %.6f | val %.6f | lr %.2e",
                epoch,
                total_epochs,
                train_loss,
                val_loss,
                lr,
            )

    def _record_fixed_validation(self, epoch: int, report: dict) -> None:
        """Log explicitly named primary metrics and update counts, separate from legacy CSV."""
        record = {
            "epoch": epoch,
            "training_horizon": self.current_horizon,
            "optimizer_steps": self.optimizer_steps,
            "attempted_train_batches": self.attempted_train_batches,
            "selection_metric": FIXED_POSITION_MAE,
            "config": asdict(self.cfg),
            "source_sha256": self.source_hashes,
            **report,
        }
        self.validation_history.append(record)
        artifact_dir = self.csv_path.parent if self.csv_path is not None else self.ckpt_dir
        if artifact_dir is not None:
            with (artifact_dir / "fixed_validation.jsonl").open("a") as handle:
                handle.write(json.dumps(_json_safe(record), allow_nan=False) + "\n")
        logger.info(
            "fixed val position MAE=%s | failed runs=%d | optimizer steps=%d | attempts=%d",
            report["model"]["selection_score"],
            report["model"]["failed_runs"],
            self.optimizer_steps,
            self.attempted_train_batches,
        )

    # static helpers
    @staticmethod
    def _build_dataset(
        path: str,
        horizon: int,
        n_trajectories: int | None = None,
        *,
        lazy: bool = False,
    ) -> Dataset:
        """Keep legacy materialization unless the canonical path explicitly opts in."""
        if lazy:
            return LazyTrajectoryWindowDataset(path, horizon, n_trajectories)
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
    *,
    resume_checkpoint: str | Path | None = None,
    allow_runtime_change: bool = False,
    max_rounds: int | None = None,
) -> TrainResult:
    """Run training, optionally continuing a committed canonical round."""
    return Trainer(
        cfg,
        model=model,
        init_checkpoint=init_checkpoint,
        resume_checkpoint=resume_checkpoint,
        allow_runtime_change=allow_runtime_change,
    ).run(max_rounds=max_rounds)


def apply_artifact_dir(cfg: TrainConfig, artifact_dir: str | Path) -> TrainConfig:
    """Point checkpointing.dir and logging.dir at one artifact root, force-enabling both."""
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
        help="Override n_train_trajectories from the config (training subset runs).",
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
    parser.add_argument(
        "--resume",
        type=Path,
        help="Continue canonical latest.pt at its committed round boundary, not a warm start",
    )
    parser.add_argument(
        "--allow-runtime-change",
        action="store_true",
        help="Explicitly allow non-bitwise continuation on changed runtime/hardware",
    )
    parser.add_argument(
        "--max-rounds", type=int, help="Stop cleanly after this many additional committed rounds"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.n_train is not None:
        config = replace(config, data=replace(config.data, n_train_trajectories=args.n_train))
    if args.artifact_dir is not None:
        config = apply_artifact_dir(config, args.artifact_dir)
    results = train(
        config,
        init_checkpoint=args.init_checkpoint,
        resume_checkpoint=args.resume,
        allow_runtime_change=args.allow_runtime_change,
        max_rounds=args.max_rounds,
    )
    logger.info("results: %s", results)
