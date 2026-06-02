"""Tests for evaluation/_loader.py."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from evaluation._loader import _resolve_normalization_stats
from training._types import (
    Checkpoint,
    CheckpointConfig,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    SchedulerConfig,
    TrainConfig,
    TrainingParams,
)


def _write_h5(path: Path, n_traj: int = 2, n_steps: int = 4) -> None:
    """Write a tiny deterministic trajectory file."""
    rng = np.random.default_rng(42)
    trajectories = rng.normal(size=(n_traj, n_steps, 3, 5)).astype(np.float32)
    trajectories[..., 4] = 1.0
    with h5py.File(path, "w") as f:
        f.create_dataset("trajectories", data=trajectories)
        f.create_dataset("energies", data=np.zeros((n_traj, n_steps), dtype=np.float32))


def _cfg(train_path: Path) -> TrainConfig:
    """Minimal config whose only relevant field is data.train_path."""
    return TrainConfig(
        model=ModelConfig(name="egnn", hidden_dim=8, n_layers=1),
        data=DataConfig(train_path=str(train_path), val_path=str(train_path), dt=0.05),
        training=TrainingParams(epochs=1, batch_size=2, lr=1e-3, weight_decay=0.0, device="cpu"),
        scheduler=SchedulerConfig(enabled=False),
        checkpointing=CheckpointConfig(enabled=False),
        logging=LoggingConfig(enabled=False),
    )


def _checkpoint(*, pos_std: float | None, vel_std: float | None) -> Checkpoint:
    """Checkpoint carrying only the fields the resolver reads."""
    return Checkpoint(
        epoch=1, model={}, optimizer={}, val_loss=0.1, pos_std=pos_std, vel_std=vel_std
    )


def test_prefers_checkpoint_stats(tmp_path: Path) -> None:
    """Checkpoint stats win when present, without touching train data."""
    cfg = _cfg(tmp_path / "missing.h5")
    checkpoint = _checkpoint(pos_std=12.5, vel_std=3.25)
    assert _resolve_normalization_stats(cfg, checkpoint) == (12.5, 3.25)


def test_falls_back_to_train_data(tmp_path: Path) -> None:
    """Missing checkpoint stats are refit from the train file."""
    train_path = tmp_path / "train.h5"
    _write_h5(train_path)
    pos_std, vel_std = _resolve_normalization_stats(
        _cfg(train_path), _checkpoint(pos_std=None, vel_std=None)
    )
    assert pos_std > 0
    assert vel_std > 0


def test_raises_when_stats_and_data_both_missing(tmp_path: Path) -> None:
    """No checkpoint stats and no train file is a hard error."""
    with pytest.raises(FileNotFoundError):
        _resolve_normalization_stats(
            _cfg(tmp_path / "missing.h5"), _checkpoint(pos_std=None, vel_std=None)
        )
