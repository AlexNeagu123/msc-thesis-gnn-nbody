"""Tests for training/_io.py."""

from pathlib import Path

import pytest
import torch
import yaml

from training._io import (
    append_metrics,
    init_metrics_csv,
    load_checkpoint,
    load_config,
    save_checkpoint,
)
from training._types import Checkpoint, EpochMetrics


def test_load_config_parses_yaml(tmp_path: Path) -> None:
    """load_config returns a typed TrainConfig from YAML."""
    yaml_text = """
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
  weight_decay: 1e-5
"""
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml_text)

    cfg = load_config(path)
    assert cfg.model.name == "egnn"
    assert cfg.training.epochs == 10
    assert cfg.scheduler.enabled is False  # default


def test_save_and_load_checkpoint_round_trip(tmp_path: Path) -> None:
    """A typed Checkpoint survives save -> load through torch.save."""
    ckpt = Checkpoint(
        epoch=5,
        model={"weight": torch.tensor([1.0, 2.0])},
        optimizer={"state": {}},
        val_loss=0.123,
        run_id="abc",
        pos_std=0.97,
        vel_std=0.85,
    )

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, ckpt)
    loaded = load_checkpoint(path, torch.device("cpu"))

    assert isinstance(loaded, Checkpoint)
    assert loaded.epoch == 5
    assert loaded.run_id == "abc"
    assert loaded.pos_std == 0.97
    assert torch.equal(loaded.model["weight"], ckpt.model["weight"])


def test_load_checkpoint_normalizes_legacy_dict(tmp_path: Path) -> None:
    """A legacy dict-shaped checkpoint is converted to a typed Checkpoint."""
    legacy = {
        "epoch": 7,
        "model": {"weight": torch.tensor([0.5])},
        "optimizer": {},
        "val_loss": 0.42,
        "run_id": "legacy",
    }
    path = tmp_path / "legacy.pt"
    torch.save(legacy, path)

    loaded = load_checkpoint(path, torch.device("cpu"))

    assert isinstance(loaded, Checkpoint)
    assert loaded.epoch == 7
    assert loaded.val_loss == 0.42
    assert loaded.run_id == "legacy"
    # missing fields are filled with dataclass defaults
    assert loaded.pos_std is None
    assert loaded.git_commit is None


def test_load_checkpoint_rejects_bad_type(tmp_path: Path) -> None:
    """Loading a non-Checkpoint, non-dict object raises TypeError."""
    path = tmp_path / "junk.pt"
    torch.save("not a checkpoint", path)

    with pytest.raises(TypeError, match="Unsupported checkpoint type"):
        load_checkpoint(path, torch.device("cpu"))


def test_load_checkpoint_rejects_dict_missing_model(tmp_path: Path) -> None:
    """Legacy dict missing a dict-shaped 'model' state fails at I/O boundary."""
    path = tmp_path / "missing_model.pt"
    torch.save({"epoch": 1, "optimizer": {}, "val_loss": 0.1}, path)

    with pytest.raises(ValueError, match="missing a dict-shaped 'model'"):
        load_checkpoint(path, torch.device("cpu"))


def test_load_checkpoint_rejects_dict_missing_optimizer(tmp_path: Path) -> None:
    """Legacy dict missing a dict-shaped 'optimizer' state fails at I/O boundary."""
    path = tmp_path / "missing_opt.pt"
    torch.save({"epoch": 1, "model": {"w": torch.tensor([1.0])}, "val_loss": 0.1}, path)

    with pytest.raises(ValueError, match="missing a dict-shaped 'optimizer'"):
        load_checkpoint(path, torch.device("cpu"))


def test_init_metrics_csv_writes_header(tmp_path: Path) -> None:
    """init_metrics_csv creates the file with the EpochMetrics header."""
    path = tmp_path / "metrics.csv"
    init_metrics_csv(path)

    assert path.read_text() == "epoch,train_loss,val_loss,lr\n"


def test_append_metrics_writes_row(tmp_path: Path) -> None:
    """append_metrics adds one row in the EpochMetrics format."""
    path = tmp_path / "metrics.csv"
    init_metrics_csv(path)
    append_metrics(path, EpochMetrics(epoch=1, train_loss=0.5, val_loss=0.6, lr=1e-3))
    append_metrics(path, EpochMetrics(epoch=2, train_loss=0.4, val_loss=0.55, lr=5e-4))

    lines = path.read_text().splitlines()
    assert lines[0] == "epoch,train_loss,val_loss,lr"
    assert lines[1] == "1,0.500000,0.600000,1.00e-03"
    assert lines[2] == "2,0.400000,0.550000,5.00e-04"


def test_load_config_handles_yaml_default_keys(tmp_path: Path) -> None:
    """Missing optional sections produce sensible defaults via yaml.safe_load."""
    path = tmp_path / "cfg.yaml"
    minimal = {
        "model": {"name": "x", "hidden_dim": 8, "n_layers": 1},
        "data": {"train_path": "t", "val_path": "v", "dt": 0.05},
        "training": {"epochs": 1, "batch_size": 1, "lr": 1e-3, "weight_decay": 0.0},
    }
    path.write_text(yaml.safe_dump(minimal))

    cfg = load_config(path)
    assert cfg.checkpointing.enabled is False
    assert cfg.logging.enabled is False
