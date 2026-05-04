"""Tests for training/scaling.py."""

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from training._types import (
    CheckpointConfig,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    SchedulerConfig,
    TrainConfig,
    TrainingParams,
)
from training.scaling import DEFAULT_ARTIFACT_ROOT, run_scaling, scaling_artifact_dir
from training.train import Trainer


class _DummyModel(nn.Module):
    """Tiny per-particle linear used in place of a real EGNN/HGNN for tests."""

    def __init__(self) -> None:
        """Initialize with one linear layer."""
        super().__init__()
        self.net = nn.Linear(5, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear per-particle."""
        return self.net(x)


@pytest.fixture
def sample_h5(tmp_path: Path) -> tuple[str, str]:
    """Create small synthetic train and val HDF5 files."""
    rng = np.random.default_rng(42)
    for name in ("train.h5", "val.h5"):
        trajectories = rng.normal(size=(5, 10, 3, 5)).astype(np.float32)
        trajectories[..., 4] = 1.0
        with h5py.File(tmp_path / name, "w") as f:
            f.create_dataset("trajectories", data=trajectories)
            f.create_dataset("energies", data=np.zeros((5, 10), dtype=np.float32))
    return str(tmp_path / "train.h5"), str(tmp_path / "val.h5")


@pytest.fixture
def base_cfg(sample_h5: tuple[str, str]) -> TrainConfig:
    """Minimal trainable config; logging/checkpointing replaced by run_scaling override."""
    train_path, val_path = sample_h5
    return TrainConfig(
        model=ModelConfig(name="egnn", hidden_dim=4, n_layers=1),
        data=DataConfig(train_path=train_path, val_path=val_path, dt=0.05),
        training=TrainingParams(epochs=1, batch_size=8, lr=1e-3, weight_decay=0.0, device="cpu"),
        scheduler=SchedulerConfig(enabled=False),
        checkpointing=CheckpointConfig(enabled=False),
        logging=LoggingConfig(enabled=False),
    )


def test_scaling_artifact_dir_layout() -> None:
    """Path is `<root>/<model>/n<N_TRAIN>` regardless of root style."""
    assert scaling_artifact_dir("egnn", 5000, "runs/scaling") == Path("runs/scaling/egnn/n5000")
    assert scaling_artifact_dir("hgnn", 100, Path("/tmp/foo")) == Path("/tmp/foo/hgnn/n100")


def test_default_artifact_root_is_runs_scaling() -> None:
    """The advertised default lives under the canonical runs/ layout."""
    assert DEFAULT_ARTIFACT_ROOT == "runs/scaling"


def test_run_scaling_writes_one_dir_per_size(base_cfg: TrainConfig, tmp_path: Path) -> None:
    """Each n_train value gets its own `n<N>` subdirectory under the artifact root."""
    artifact_root = tmp_path / "runs" / "scaling"
    sizes = [2, 3]

    run_scaling(
        base_cfg,
        sizes,
        artifact_root=artifact_root,
        trainer_factory=lambda cfg: Trainer(cfg, model=_DummyModel()),
    )

    for n_train in sizes:
        size_dir = artifact_root / "egnn" / f"n{n_train}"
        run_dirs = list(size_dir.iterdir())
        assert len(run_dirs) == 1, f"expected one run dir under {size_dir}"
        run_dir = run_dirs[0]
        assert (run_dir / "best.pt").exists()
        assert (run_dir / "metrics.csv").exists()


def test_run_scaling_does_not_share_dirs_between_sizes(
    base_cfg: TrainConfig, tmp_path: Path
) -> None:
    """Two distinct sizes must not collide on the same run folder."""
    artifact_root = tmp_path / "runs" / "scaling"

    run_scaling(
        base_cfg,
        [2, 3],
        artifact_root=artifact_root,
        trainer_factory=lambda cfg: Trainer(cfg, model=_DummyModel()),
    )

    n2_dir = artifact_root / "egnn" / "n2"
    n3_dir = artifact_root / "egnn" / "n3"
    assert n2_dir.exists()
    assert n3_dir.exists()
    assert next(n2_dir.iterdir()) != next(n3_dir.iterdir())


def test_run_scaling_uses_model_name_in_path(base_cfg: TrainConfig, tmp_path: Path) -> None:
    """Different models go to sibling directories under the artifact root."""
    artifact_root = tmp_path / "runs" / "scaling"
    cfg = replace(base_cfg, model=replace(base_cfg.model, name="hgnn"))

    run_scaling(
        cfg,
        [2],
        artifact_root=artifact_root,
        trainer_factory=lambda c: Trainer(c, model=_DummyModel()),
    )

    assert (artifact_root / "hgnn" / "n2").exists()
    assert not (artifact_root / "egnn").exists()
