"""Tests for training/train.py."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from training._types import (
    Checkpoint,
    CheckpointConfig,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    SchedulerConfig,
    TrainConfig,
    TrainingParams,
    TrainResult,
)
from training.train import Trainer, load_config, train


class DummyModel(nn.Module):
    """Minimal model that maps (batch, 3, 5) -> (batch, 3, 5)."""

    def __init__(self) -> None:
        """Initialize with a single linear layer."""
        super().__init__()
        self.net = nn.Linear(5, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, applied per-particle."""
        return self.net(x)


@pytest.fixture
def sample_h5(tmp_path: Path) -> tuple[str, str]:
    """Create small train and val HDF5 files."""
    rng = np.random.default_rng(42)

    for name in ("train.h5", "val.h5"):
        trajectories = rng.normal(size=(5, 10, 3, 5))
        energies = rng.normal(size=(5, 10))
        path = tmp_path / name
        with h5py.File(path, "w") as f:
            f.create_dataset("trajectories", data=trajectories)
            f.create_dataset("energies", data=energies)

    return str(tmp_path / "train.h5"), str(tmp_path / "val.h5")


@pytest.fixture
def make_cfg(sample_h5: tuple[str, str], tmp_path: Path) -> TrainConfig:
    """Create a minimal TrainConfig pointing to test data."""
    train_path, val_path = sample_h5
    return TrainConfig(
        model=ModelConfig(name="dummy", hidden_dim=4, n_layers=1),
        data=DataConfig(train_path=train_path, val_path=val_path, dt=0.05),
        training=TrainingParams(
            epochs=3,
            batch_size=8,
            lr=1e-3,
            weight_decay=0.0,
            loss="mse",
            seed=42,
            device="cpu",
        ),
        scheduler=SchedulerConfig(enabled=False),
        checkpointing=CheckpointConfig(enabled=True, dir=str(tmp_path / "ckpt")),
        logging=LoggingConfig(enabled=True, dir=str(tmp_path / "logs")),
    )


def test_train_returns_result(make_cfg: TrainConfig) -> None:
    """Train function returns a TrainResult with expected fields."""
    result = train(make_cfg, model=DummyModel())

    assert isinstance(result, TrainResult)
    assert result.best_epoch >= 1
    assert result.best_val_loss < float("inf")
    assert result.final_train_loss < float("inf")


def test_train_histories_length(make_cfg: TrainConfig) -> None:
    """History lists have one entry per epoch."""
    result = train(make_cfg, model=DummyModel())

    assert len(result.train_history) == make_cfg.training.epochs
    assert len(result.val_history) == make_cfg.training.epochs


def _find_run_dir(base: str) -> Path:
    """Find the single run subdirectory inside a base directory."""
    subdirs = sorted(Path(base).iterdir())
    assert len(subdirs) == 1
    return subdirs[0]


def test_checkpoint_saved(make_cfg: TrainConfig) -> None:
    """Best and latest checkpoints are written to disk."""
    train(make_cfg, model=DummyModel())

    run_dir = _find_run_dir(make_cfg.checkpointing.dir)
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "latest.pt").exists()


def test_checkpoint_contents(make_cfg: TrainConfig) -> None:
    """Checkpoint contains expected fields."""
    train(make_cfg, model=DummyModel())

    run_dir = _find_run_dir(make_cfg.checkpointing.dir)
    ckpt = torch.load(run_dir / "best.pt", weights_only=False)

    assert isinstance(ckpt, Checkpoint)
    assert isinstance(ckpt.epoch, int)
    assert isinstance(ckpt.val_loss, float)
    assert isinstance(ckpt.model, dict)
    assert isinstance(ckpt.optimizer, dict)
    assert ckpt.config is not None
    assert ckpt.model_name == make_cfg.model.name
    assert ckpt.run_id == run_dir.name
    assert isinstance(ckpt.pos_std, float)
    assert isinstance(ckpt.vel_std, float)


def test_csv_log_written(make_cfg: TrainConfig) -> None:
    """CSV metrics file is created with header and one row per epoch."""
    train(make_cfg, model=DummyModel())

    run_dir = _find_run_dir(make_cfg.logging.dir)
    csv_path = run_dir / "metrics.csv"
    assert csv_path.exists()

    lines = csv_path.read_text().strip().split("\n")
    assert lines[0] == "epoch,train_loss,val_loss,lr"
    assert len(lines) == make_cfg.training.epochs + 1


def test_loss_decreases(make_cfg: TrainConfig) -> None:
    """Training loss should generally decrease over a few epochs."""
    result = train(make_cfg, model=DummyModel())

    # not strictly monotonic, but first should be larger than last
    assert result.train_history[0] > result.train_history[-1]


def test_mae_loss(make_cfg: TrainConfig) -> None:
    """Training works with MAE loss."""
    cfg = TrainConfig(
        model=make_cfg.model,
        data=make_cfg.data,
        training=TrainingParams(
            epochs=2,
            batch_size=8,
            lr=1e-3,
            weight_decay=0.0,
            loss="mae",
            seed=42,
            device="cpu",
        ),
        scheduler=make_cfg.scheduler,
        checkpointing=CheckpointConfig(enabled=False),
        logging=LoggingConfig(enabled=False),
    )
    result = train(cfg, model=DummyModel())

    assert isinstance(result, TrainResult)
    assert result.final_train_loss < float("inf")


def test_no_checkpointing(make_cfg: TrainConfig, tmp_path: Path) -> None:
    """Training runs fine with checkpointing disabled."""
    cfg = TrainConfig(
        model=make_cfg.model,
        data=make_cfg.data,
        training=make_cfg.training,
        scheduler=make_cfg.scheduler,
        checkpointing=CheckpointConfig(enabled=False),
        logging=LoggingConfig(enabled=False),
    )
    result = train(cfg, model=DummyModel())

    assert isinstance(result, TrainResult)


def test_scheduler_enabled(make_cfg: TrainConfig) -> None:
    """Training runs fine with scheduler enabled."""
    cfg = TrainConfig(
        model=make_cfg.model,
        data=make_cfg.data,
        training=make_cfg.training,
        scheduler=SchedulerConfig(enabled=True, patience=1, factor=0.5, min_lr=1e-6),
        checkpointing=CheckpointConfig(enabled=False),
        logging=LoggingConfig(enabled=False),
    )
    result = train(cfg, model=DummyModel())

    assert isinstance(result, TrainResult)


def test_noise_injection_modifies_pos_vel_only(make_cfg: TrainConfig) -> None:
    """Noise is applied to positions and velocities but not mass."""
    cfg = TrainConfig(
        model=make_cfg.model,
        data=make_cfg.data,
        training=TrainingParams(
            epochs=1,
            batch_size=8,
            lr=1e-3,
            weight_decay=0.0,
            loss="mse",
            noise_factor=0.05,
            seed=42,
            device="cpu",
        ),
        scheduler=SchedulerConfig(enabled=False),
        checkpointing=CheckpointConfig(enabled=False),
        logging=LoggingConfig(enabled=False),
    )
    trainer = Trainer(cfg, model=DummyModel())

    inputs, _targets = next(iter(trainer.train_loader))
    original_mass = inputs[..., 4:].clone()
    original_pos = inputs[..., :2].clone()

    noisy = trainer.apply_noise(inputs)

    # mass column unchanged
    assert torch.equal(noisy[..., 4:], original_mass)
    # position and velocity columns changed
    assert not torch.equal(noisy[..., :2], original_pos)
    assert not torch.equal(noisy[..., 2:4], inputs[..., 2:4])


def test_noise_injection_runs(make_cfg: TrainConfig) -> None:
    """Training completes with noise injection enabled."""
    cfg = TrainConfig(
        model=make_cfg.model,
        data=make_cfg.data,
        training=TrainingParams(
            epochs=2,
            batch_size=8,
            lr=1e-3,
            weight_decay=0.0,
            loss="mse",
            noise_factor=0.05,
            seed=42,
            device="cpu",
        ),
        scheduler=SchedulerConfig(enabled=False),
        checkpointing=CheckpointConfig(enabled=False),
        logging=LoggingConfig(enabled=False),
    )
    result = train(cfg, model=DummyModel())

    assert isinstance(result, TrainResult)
    assert result.final_train_loss < float("inf")


def test_load_config(tmp_path: Path) -> None:
    """Load_config parses a YAML file into a TrainConfig."""
    yaml_content = """
model:
  name: egnn
  hidden_dim: 64
  n_layers: 4
data:
  train_path: train.h5
  val_path: val.h5
  dt: 0.05
training:
  epochs: 10
  batch_size: 32
  lr: 0.001
  weight_decay: 0.00001
"""
    config_path = tmp_path / "test.yaml"
    config_path.write_text(yaml_content)

    cfg = load_config(str(config_path))

    assert isinstance(cfg, TrainConfig)
    assert cfg.model.name == "egnn"
    assert cfg.training.epochs == 10
    assert cfg.scheduler.enabled is False  # default
    assert cfg.checkpointing.enabled is False  # default
