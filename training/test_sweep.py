"""Tests for training/sweep.py."""

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
from training.sweep import (
    DEFAULT_ARTIFACT_ROOT,
    LRS,
    NOISE_FACTORS,
    _label_lr,
    _label_nf,
    run_sweep,
    sweep_artifact_dir,
)
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
    """Minimal trainable config; logging/checkpointing replaced by run_sweep override."""
    train_path, val_path = sample_h5
    return TrainConfig(
        model=ModelConfig(name="egnn", hidden_dim=4, n_layers=1),
        data=DataConfig(train_path=train_path, val_path=val_path, dt=0.05),
        training=TrainingParams(epochs=1, batch_size=8, lr=1e-3, weight_decay=0.0, device="cpu"),
        scheduler=SchedulerConfig(enabled=False),
        checkpointing=CheckpointConfig(enabled=False),
        logging=LoggingConfig(enabled=False),
    )


def test_label_lr_is_path_safe_scientific() -> None:
    """LR labels render as compact scientific notation suitable for path components."""
    assert _label_lr(5e-4) == "5e-04"
    assert _label_lr(1e-3) == "1e-03"
    assert _label_lr(2e-3) == "2e-03"


def test_label_nf_uses_p_separator() -> None:
    """Noise-factor labels avoid '.' so they survive in file paths."""
    assert _label_nf(0.0) == "0"
    assert _label_nf(0.03) == "0p03"
    assert _label_nf(0.05) == "0p05"


def test_sweep_artifact_dir_layout() -> None:
    """Path is `<root>/<model>/lr_<lr>_nf_<nf>` so each grid cell is unique."""
    assert sweep_artifact_dir("egnn", 5e-4, 0.03, "runs/sweep") == Path(
        "runs/sweep/egnn/lr_5e-04_nf_0p03"
    )


def test_default_artifact_root_is_runs_sweep() -> None:
    """Default lives under runs/sweep, separate from Colab's runs/noise_sweep."""
    assert DEFAULT_ARTIFACT_ROOT == "runs/sweep"


def test_run_sweep_writes_one_dir_per_grid_cell(base_cfg: TrainConfig, tmp_path: Path) -> None:
    """Every (lr, nf) cell lands in its own directory; no two cells share a folder."""
    artifact_root = tmp_path / "runs" / "sweep"

    run_sweep(
        base_cfg,
        epochs=1,
        artifact_root=artifact_root,
        trainer_factory=lambda cfg: Trainer(cfg, model=_DummyModel()),
    )

    expected_cells = {(lr, nf) for lr in LRS for nf in NOISE_FACTORS}
    assert len(expected_cells) == len(LRS) * len(NOISE_FACTORS)

    cell_dirs = sorted((artifact_root / "egnn").iterdir())
    assert len(cell_dirs) == len(expected_cells)

    # each cell dir contains exactly one run with the standard artifacts
    for cell_dir in cell_dirs:
        run_dirs = list(cell_dir.iterdir())
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]
        assert (run_dir / "best.pt").exists()
        assert (run_dir / "metrics.csv").exists()
