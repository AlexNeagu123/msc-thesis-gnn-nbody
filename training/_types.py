"""Typed contracts for the training pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class ModelConfig:
    """Model architecture parameters."""

    name: str
    hidden_dim: int
    n_layers: int
    summary: bool = False


@dataclass
class DataConfig:
    """Dataset paths and time step."""

    train_path: str
    val_path: str
    dt: float
    n_train_trajectories: int | None = None


@dataclass
class TrainingParams:
    """Core training hyperparameters.

    Single-horizon mode uses `epochs` and `multi_step_horizon`; curriculum mode uses
    `curriculum_horizons` and `curriculum_epochs` (which then take precedence).
    """

    batch_size: int
    lr: float
    weight_decay: float
    epochs: int = 0
    loss: str = "mse"
    noise_factor: float = 0.0
    seed: int = 42
    device: str = "auto"
    multi_step_horizon: int = 1
    multi_step_gamma: float = 1.0
    checkpoint_metric: str = "val_loss"
    curriculum_horizons: list[int] | None = None
    curriculum_epochs: list[int] | None = None
    gradient_clip_norm: float = 10.0
    curriculum_gradient_clip_norms: list[float] | None = None
    curriculum_lrs: list[float] | None = None
    skip_nonfinite_batches: bool = True
    reset_optimizer_on_stage: bool = False

    def __post_init__(self) -> None:
        """Validate single-horizon vs curriculum field combinations."""
        if self.gradient_clip_norm <= 0:
            msg = f"gradient_clip_norm must be > 0, got {self.gradient_clip_norm}"
            raise ValueError(msg)

        horizons = self.curriculum_horizons
        epochs_list = self.curriculum_epochs
        clip_norms = self.curriculum_gradient_clip_norms
        lrs = self.curriculum_lrs

        if (horizons is None) != (epochs_list is None):
            msg = "curriculum_horizons and curriculum_epochs must both be set or both be None"
            raise ValueError(msg)

        if horizons is None:
            if self.epochs <= 0:
                msg = "epochs must be > 0 in single-horizon mode (no curriculum configured)"
                raise ValueError(msg)
            if clip_norms is not None:
                msg = (
                    "curriculum_gradient_clip_norms is only valid with a curriculum schedule; "
                    "use gradient_clip_norm in single-horizon mode"
                )
                raise ValueError(msg)
            if lrs is not None:
                msg = (
                    "curriculum_lrs is only valid with a curriculum schedule; "
                    "use lr in single-horizon mode"
                )
                raise ValueError(msg)
            return

        # curriculum mode: validate the two lists describe a consistent schedule
        if len(horizons) != len(epochs_list):  # type: ignore[arg-type]
            msg = (
                f"curriculum_horizons ({len(horizons)}) and curriculum_epochs "
                f"({len(epochs_list)}) must have the same length"  # type: ignore[arg-type]
            )
            raise ValueError(msg)
        if len(horizons) == 0:
            msg = "curriculum schedule must contain at least one stage"
            raise ValueError(msg)
        if any(h < 1 for h in horizons):
            msg = f"every curriculum_horizons entry must be >= 1, got {horizons}"
            raise ValueError(msg)
        if any(e < 1 for e in epochs_list):  # type: ignore[union-attr]
            msg = f"every curriculum_epochs entry must be >= 1, got {epochs_list}"
            raise ValueError(msg)

        if clip_norms is not None:
            if len(clip_norms) != len(horizons):
                msg = (
                    f"curriculum_gradient_clip_norms ({len(clip_norms)}) and "
                    f"curriculum_horizons ({len(horizons)}) must have the same length"
                )
                raise ValueError(msg)
            if any(not math.isfinite(n) or n <= 0 for n in clip_norms):
                msg = (
                    "every curriculum_gradient_clip_norms entry must be a finite float > 0, "
                    f"got {clip_norms}"
                )
                raise ValueError(msg)

        if lrs is not None:
            if len(lrs) != len(horizons):
                msg = (
                    f"curriculum_lrs ({len(lrs)}) and "
                    f"curriculum_horizons ({len(horizons)}) must have the same length"
                )
                raise ValueError(msg)
            if any(not math.isfinite(v) or v <= 0 for v in lrs):
                msg = f"every curriculum_lrs entry must be a finite float > 0, got {lrs}"
                raise ValueError(msg)


@dataclass
class SchedulerConfig:
    """ReduceLROnPlateau scheduler settings."""

    enabled: bool = False
    patience: int = 50
    factor: float = 0.5
    min_lr: float = 1e-6


@dataclass
class CheckpointConfig:
    """Checkpointing settings."""

    enabled: bool = False
    dir: str = "checkpoints/"


@dataclass
class LoggingConfig:
    """CSV logging and tqdm settings."""

    enabled: bool = False
    dir: str = "logs/"


@dataclass
class TrainConfig:
    """Top-level training configuration."""

    model: ModelConfig
    data: DataConfig
    training: TrainingParams
    scheduler: SchedulerConfig
    checkpointing: CheckpointConfig
    logging: LoggingConfig

    @staticmethod
    def from_dict(d: dict) -> TrainConfig:
        """Build a TrainConfig from a parsed YAML dict."""
        return TrainConfig(
            model=ModelConfig(**d["model"]),
            data=DataConfig(**d["data"]),
            training=TrainingParams(**d["training"]),
            scheduler=SchedulerConfig(**d.get("scheduler", {})),
            checkpointing=CheckpointConfig(**d.get("checkpointing", {})),
            logging=LoggingConfig(**d.get("logging", {})),
        )


@dataclass
class Checkpoint:
    """State saved to disk at each epoch.

    `val_loss` is always the one-step validation MSE; `selected_metric`/`selected_score`
    record which metric drove `is_best`.
    """

    epoch: int
    model: dict
    optimizer: dict
    val_loss: float
    config: dict | None = None
    model_name: str | None = None
    run_id: str | None = None
    pos_std: float | None = None
    vel_std: float | None = None
    git_commit: str | None = None
    selected_metric: str | None = None
    selected_score: float | None = None
    rollout_score: float | None = None


@dataclass
class TrainResult:
    """Results returned by a training run."""

    best_val_loss: float
    final_train_loss: float
    best_epoch: int
    train_history: list[float] = field(default_factory=list)
    val_history: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class EpochRunSummary:
    """One train or val epoch summary; `loss` is the mean over non-skipped batches.

    Gradient diagnostics are populated only for training epochs.
    """

    loss: float
    grad_norm_mean: float | None = None
    grad_norm_max: float | None = None
    grad_clip_fraction: float | None = None
    skipped_batches: int | None = None


@dataclass(frozen=True)
class RolloutScore:
    """Baseline-normalized rollout score on the validation set.

    Lower is better. score < 0 means the model beats the baseline envelope
    on average across the rollout horizon (geometric mean of ratios < 1).
    """

    score: float
    ratios: npt.NDArray[np.floating]
    dominance_horizon: int
    fraction_beating_baseline: float
    final_ratio: float
    ratios_at_step: dict[int, float]


@dataclass(frozen=True)
class BucketRolloutScore:
    """Bucket-aware rollout score for checkpoint selection on stratified val.

    `macro` is the unweighted mean of per-bin scores over non-empty bins, so common
    regimes cannot mask close-encounter failures. `bin_order` fixes per-bin column order.
    """

    macro: float
    per_bin: dict[str, RolloutScore]
    bin_order: tuple[str, ...]


@dataclass
class EpochMetrics:
    """One row of the training metrics CSV; None fields render as empty strings.

    In bucket mode `rollout_score` is the macro score and `bucket_per_bin` feeds the
    dynamic per-bin columns, while the single-curve diagnostic columns stay blank.
    """

    epoch: int
    train_loss: float
    val_loss: float
    lr: float
    rollout_score: float | None = None
    dominance_horizon: int | None = None
    fraction_beating_baseline: float | None = None
    final_ratio: float | None = None
    grad_norm_mean: float | None = None
    grad_norm_max: float | None = None
    grad_clip_fraction: float | None = None
    skipped_batches: int | None = None
    bucket_per_bin: dict[str, RolloutScore] | None = None

    @classmethod
    def csv_header(cls, bin_names: tuple[str, ...] = ()) -> str:
        """CSV header line; each bin in `bin_names` appends four per-bin columns."""
        base = (
            "epoch,train_loss,val_loss,lr,"
            "rollout_score,dominance_horizon,fraction_beating_baseline,final_ratio,"
            "grad_norm_mean,grad_norm_max,grad_clip_fraction,skipped_batches"
        )
        if not bin_names:
            return base
        bucket_cols = ",".join(
            f"rollout_score_{name},dominance_horizon_{name},"
            f"fraction_beating_baseline_{name},final_ratio_{name}"
            for name in bin_names
        )
        return f"{base},{bucket_cols}"

    def to_csv_row(self, bin_names: tuple[str, ...] = ()) -> str:
        """One CSV row matching csv_header order; bins absent from per_bin render blank."""
        rs = "" if self.rollout_score is None else f"{self.rollout_score:.6f}"
        dh = "" if self.dominance_horizon is None else str(self.dominance_horizon)
        fb = (
            ""
            if self.fraction_beating_baseline is None
            else f"{self.fraction_beating_baseline:.6f}"
        )
        fr = "" if self.final_ratio is None else f"{self.final_ratio:.6f}"
        gnm = "" if self.grad_norm_mean is None else f"{self.grad_norm_mean:.6f}"
        gnx = "" if self.grad_norm_max is None else f"{self.grad_norm_max:.6f}"
        gcf = "" if self.grad_clip_fraction is None else f"{self.grad_clip_fraction:.6f}"
        skip = "" if self.skipped_batches is None else str(self.skipped_batches)
        base = (
            f"{self.epoch},{self.train_loss:.6f},{self.val_loss:.6f},{self.lr:.2e},"
            f"{rs},{dh},{fb},{fr},{gnm},{gnx},{gcf},{skip}"
        )
        if not bin_names:
            return base

        per_bin = self.bucket_per_bin or {}
        bucket_parts: list[str] = []
        for name in bin_names:
            score = per_bin.get(name)
            if score is None:
                bucket_parts.extend(["", "", "", ""])
            else:
                bucket_parts.append(f"{score.score:.6f}")
                bucket_parts.append(str(score.dominance_horizon))
                bucket_parts.append(f"{score.fraction_beating_baseline:.6f}")
                bucket_parts.append(f"{score.final_ratio:.6f}")
        return f"{base},{','.join(bucket_parts)}"
