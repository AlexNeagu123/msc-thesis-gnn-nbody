"""Tests for evaluation/evaluate.py."""

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from evaluation.evaluate import (
    _normalization_stats,
    _output_dir,
    evaluate_checkpoint,
)
from models.egnn import EGNN
from models.hgnn import HGNN
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


def _cfg(train_path: Path, val_path: Path, model_name: str) -> TrainConfig:
    """Create a minimal evaluation config."""
    return TrainConfig(
        model=ModelConfig(name=model_name, hidden_dim=8, n_layers=1),
        data=DataConfig(train_path=str(train_path), val_path=str(val_path), dt=0.05),
        training=TrainingParams(
            epochs=1,
            batch_size=2,
            lr=1e-3,
            weight_decay=0.0,
            device="cpu",
        ),
        scheduler=SchedulerConfig(enabled=False),
        checkpointing=CheckpointConfig(enabled=False),
        logging=LoggingConfig(enabled=False),
    )


def _write_config(path: Path, cfg: TrainConfig) -> None:
    """Write the minimal YAML shape expected by load_config."""
    path.write_text(
        f"""
model:
  name: {cfg.model.name}
  hidden_dim: {cfg.model.hidden_dim}
  n_layers: {cfg.model.n_layers}
data:
  train_path: {cfg.data.train_path}
  val_path: {cfg.data.val_path}
  dt: {cfg.data.dt}
training:
  epochs: {cfg.training.epochs}
  batch_size: {cfg.training.batch_size}
  lr: {cfg.training.lr}
  weight_decay: {cfg.training.weight_decay}
  device: cpu
"""
    )


def test_normalization_prefers_checkpoint_metadata(tmp_path: Path) -> None:
    """Evaluation should use the stats saved with the trained checkpoint."""
    train_path = tmp_path / "train.h5"
    val_path = tmp_path / "val.h5"
    _write_h5(train_path)
    _write_h5(val_path)

    cfg = _cfg(train_path, val_path, "egnn")
    checkpoint = Checkpoint(
        epoch=1,
        model={},
        optimizer={},
        val_loss=0.1,
        pos_std=12.5,
        vel_std=3.25,
    )

    assert _normalization_stats(cfg, checkpoint) == (12.5, 3.25)


def test_default_output_dir_for_legacy_checkpoint_falls_back_to_results() -> None:
    """Legacy `checkpoints/...` layout keeps writing reports under results/."""
    path = _output_dir(None, "egnn", Path("checkpoints/egnn/20260416_234825/best.pt"))

    assert path == Path("results/evaluation/egnn/20260416_234825")


def test_default_output_dir_for_canonical_runs_checkpoint_colocates() -> None:
    """Canonical `runs/...` checkpoint puts the report next to the checkpoint."""
    ckpt = Path("runs/curriculum/egnn/n5000/20260504_120000/best.pt")
    path = _output_dir(None, "egnn", ckpt)

    assert path == Path("runs/curriculum/egnn/n5000/20260504_120000/evaluation")


def test_default_output_dir_works_for_other_canonical_modes() -> None:
    """Detection should not be specific to one mode; any `runs/` ancestor counts."""
    for mode in ("single", "scaling", "sweep", "noise_sweep"):
        ckpt = Path(f"runs/{mode}/egnn/foo/20260101_000000/best.pt")
        assert _output_dir(None, "egnn", ckpt) == ckpt.parent / "evaluation"


def test_explicit_output_dir_wins_over_canonical_default() -> None:
    """Explicit --output-dir is honored even when the checkpoint sits under runs/."""
    ckpt = Path("runs/single/egnn/n5000/20260504_120000/best.pt")
    explicit = Path("/tmp/somewhere/else")

    assert _output_dir(explicit, "egnn", ckpt) == explicit


def test_evaluate_checkpoint_writes_json_and_csv(tmp_path: Path) -> None:
    """EGNN evaluation writes official report artifacts."""
    train_path = tmp_path / "train.h5"
    val_path = tmp_path / "val.h5"
    test_path = tmp_path / "test.h5"
    _write_h5(train_path)
    _write_h5(val_path)
    _write_h5(test_path)

    cfg = _cfg(train_path, val_path, "egnn")
    config_path = tmp_path / "egnn.yaml"
    _write_config(config_path, cfg)

    model = EGNN(hidden_dim=8, n_layers=1)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        Checkpoint(
            epoch=1,
            model=model.state_dict(),
            optimizer={},
            val_loss=0.1,
        ),
        checkpoint_path,
    )

    output_dir = tmp_path / "eval"
    report = evaluate_checkpoint(
        cfg,
        checkpoint_path,
        config_path=config_path,
        test_path=test_path,
        output_dir=output_dir,
        device="cpu",
    )

    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "summary.csv").exists()
    assert report.metadata.model_name == "egnn"
    assert report.energy.learned_hamiltonian is None

    data = json.loads((output_dir / "metrics.json").read_text())
    assert data["metadata"]["checkpoint_epoch"] == 1
    assert data["metadata"]["n_frames"] == 4
    assert data["metadata"]["n_transitions"] == 3
    assert "state_mse" in data["single_step"]
    assert "position_mse" in data["single_step"]
    assert "velocity_mse" in data["single_step"]
    assert "p95" in data["rollout"]["steps"]["1"]["position_mse"]
    assert "mean_finite" in data["rollout"]["steps"]["1"]["state_mse"]
    assert data["rollout"]["curves"]["step"] == [0, 1, 2, 3]
    assert len(data["rollout"]["curves"]["position_mse"]["median"]) == 4
    assert len(data["rollout"]["curves"]["state_mse"]["p95"]) == 4
    assert data["rollout"]["curves"]["state_mse"]["finite_fraction"][0] == 1.0
    assert "state_mse_thresholds" in data["rollout"]
    assert "position_mse_thresholds" in data["rollout"]
    assert "10" in data["rollout"]["state_mse_thresholds"]

    with (output_dir / "summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["model_name"] == "egnn"
    assert rows[0]["n_frames"] == "4"
    assert rows[0]["n_transitions"] == "3"
    assert "rollout_step_1_state_mse_mean_finite" in rows[0]
    assert "rollout_step_1_position_mse_p95" in rows[0]
    assert "rollout_final_fraction_below_state_mse_10" in rows[0]
    assert "physical_energy_final_drift_mean" in rows[0]
    assert "true_energy_final_drift_mean" not in rows[0]


def test_evaluate_hgnn_reports_learned_hamiltonian(tmp_path: Path) -> None:
    """HGNN evaluation includes learned Hamiltonian drift."""
    train_path = tmp_path / "train.h5"
    val_path = tmp_path / "val.h5"
    test_path = tmp_path / "test.h5"
    _write_h5(train_path, n_traj=1, n_steps=3)
    _write_h5(val_path, n_traj=1, n_steps=3)
    _write_h5(test_path, n_traj=1, n_steps=3)

    cfg = _cfg(train_path, val_path, "hgnn")
    config_path = tmp_path / "hgnn.yaml"
    _write_config(config_path, cfg)

    model = HGNN(hidden_dim=8, n_layers=1)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        Checkpoint(
            epoch=1,
            model=model.state_dict(),
            optimizer={},
            val_loss=0.1,
            model_name="hgnn",
        ),
        checkpoint_path,
    )

    report = evaluate_checkpoint(
        cfg,
        checkpoint_path,
        config_path=config_path,
        test_path=test_path,
        output_dir=tmp_path / "eval_hgnn",
        device="cpu",
    )

    assert report.energy.learned_hamiltonian is not None
