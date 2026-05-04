"""Typed contracts for the training pipeline."""

from __future__ import annotations

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
    """Core training hyperparameters."""

    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    loss: str = "mse"
    noise_factor: float = 0.0
    seed: int = 42
    device: str = "auto"
    multi_step_horizon: int = 1
    multi_step_gamma: float = 1.0


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
    """State saved to disk at each epoch."""

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


@dataclass
class TrainResult:
    """Results returned by a training run."""

    best_val_loss: float
    final_train_loss: float
    best_epoch: int
    train_history: list[float] = field(default_factory=list)
    val_history: list[float] = field(default_factory=list)


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


@dataclass
class EpochMetrics:
    """One row of the training metrics CSV."""

    epoch: int
    train_loss: float
    val_loss: float
    lr: float

    @classmethod
    def csv_header(cls) -> str:
        """Return the CSV header line (no trailing newline)."""
        return "epoch,train_loss,val_loss,lr"

    def to_csv_row(self) -> str:
        """Return one CSV row matching csv_header column order."""
        return f"{self.epoch},{self.train_loss:.6f},{self.val_loss:.6f},{self.lr:.2e}"
