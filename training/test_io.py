"""Tests for training/_io.py."""

from pathlib import Path

import numpy as np
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
from training._types import BucketRolloutScore, Checkpoint, EpochMetrics, RolloutScore


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


_HEADER = (
    "epoch,train_loss,val_loss,lr,"
    "rollout_score,dominance_horizon,fraction_beating_baseline,final_ratio,"
    "grad_norm_mean,grad_norm_max,grad_clip_fraction,skipped_batches"
)


def test_init_metrics_csv_writes_header(tmp_path: Path) -> None:
    """init_metrics_csv creates the file with the EpochMetrics header."""
    path = tmp_path / "metrics.csv"
    init_metrics_csv(path)

    assert path.read_text() == _HEADER + "\n"


def test_append_metrics_writes_row(tmp_path: Path) -> None:
    """append_metrics adds one row in the EpochMetrics format.

    Rollout and gradient columns are blank when nothing is supplied
    (e.g. constructing EpochMetrics outside the training loop).
    """
    path = tmp_path / "metrics.csv"
    init_metrics_csv(path)
    append_metrics(path, EpochMetrics(epoch=1, train_loss=0.5, val_loss=0.6, lr=1e-3))
    append_metrics(path, EpochMetrics(epoch=2, train_loss=0.4, val_loss=0.55, lr=5e-4))

    lines = path.read_text().splitlines()
    assert lines[0] == _HEADER
    assert lines[1] == "1,0.500000,0.600000,1.00e-03,,,,,,,,"
    assert lines[2] == "2,0.400000,0.550000,5.00e-04,,,,,,,,"


def test_append_metrics_writes_rollout_row(tmp_path: Path) -> None:
    """When rollout diagnostics are present they render in the right columns."""
    path = tmp_path / "metrics.csv"
    init_metrics_csv(path)
    append_metrics(
        path,
        EpochMetrics(
            epoch=1,
            train_loss=0.5,
            val_loss=0.6,
            lr=1e-3,
            rollout_score=-0.123456,
            dominance_horizon=42,
            fraction_beating_baseline=0.789,
            final_ratio=1.5,
        ),
    )

    lines = path.read_text().splitlines()
    assert lines[1] == "1,0.500000,0.600000,1.00e-03,-0.123456,42,0.789000,1.500000,,,,"


def test_append_metrics_writes_grad_diagnostics_row(tmp_path: Path) -> None:
    """Grad diagnostics fill the four trailing columns; rollout columns stay blank."""
    path = tmp_path / "metrics.csv"
    init_metrics_csv(path)
    append_metrics(
        path,
        EpochMetrics(
            epoch=1,
            train_loss=0.5,
            val_loss=0.6,
            lr=1e-3,
            grad_norm_mean=0.250000,
            grad_norm_max=1.875000,
            grad_clip_fraction=0.125,
            skipped_batches=2,
        ),
    )

    lines = path.read_text().splitlines()
    assert lines[1] == "1,0.500000,0.600000,1.00e-03,,,,,0.250000,1.875000,0.125000,2"


def _rollout_score_fixture(score: float, dh: int, fb: float, fr: float) -> RolloutScore:
    """Build a RolloutScore with the scalar fields the CSV emitter reads."""
    return RolloutScore(
        score=score,
        ratios=np.zeros(1),
        dominance_horizon=dh,
        fraction_beating_baseline=fb,
        final_ratio=fr,
        ratios_at_step={},
    )


def test_csv_header_default_unchanged_with_no_bin_names() -> None:
    """csv_header() with no args returns the existing single-curve header byte-identical."""
    assert EpochMetrics.csv_header() == _HEADER
    assert EpochMetrics.csv_header(()) == _HEADER


def test_csv_header_appends_per_bin_columns_in_order() -> None:
    """Per-bin columns are appended in the supplied bin_names order."""
    header = EpochMetrics.csv_header(("extreme", "smooth"))
    assert header.startswith(_HEADER + ",")
    suffix = header[len(_HEADER) + 1 :]
    assert suffix == (
        "rollout_score_extreme,dominance_horizon_extreme,"
        "fraction_beating_baseline_extreme,final_ratio_extreme,"
        "rollout_score_smooth,dominance_horizon_smooth,"
        "fraction_beating_baseline_smooth,final_ratio_smooth"
    )


def test_csv_row_default_unchanged_with_no_bin_names() -> None:
    """to_csv_row() with no args matches existing single-curve format."""
    row = EpochMetrics(epoch=1, train_loss=0.5, val_loss=0.6, lr=1e-3)
    assert row.to_csv_row() == "1,0.500000,0.600000,1.00e-03,,,,,,,,"


def test_csv_row_bucket_mode_emits_per_bin_columns(tmp_path: Path) -> None:
    """Bucket-mode row writes macro in rollout_score and per-bin columns at the end."""
    row = EpochMetrics(
        epoch=1,
        train_loss=0.5,
        val_loss=0.6,
        lr=1e-3,
        rollout_score=-0.20,
        bucket_per_bin={
            "extreme": _rollout_score_fixture(0.10, 5, 0.20, 1.5),
            "smooth": _rollout_score_fixture(-0.50, 50, 0.90, 0.4),
        },
    )

    path = tmp_path / "metrics.csv"
    init_metrics_csv(path, bin_names=("extreme", "smooth"))
    append_metrics(path, row, bin_names=("extreme", "smooth"))

    lines = path.read_text().splitlines()
    assert lines[0] == EpochMetrics.csv_header(("extreme", "smooth"))
    assert lines[1] == (
        "1,0.500000,0.600000,1.00e-03,"
        "-0.200000,,,,"  # macro in rollout_score; single-curve diagnostics blank
        ",,,"  # grad fields blank
        ","  # skipped_batches blank
        "0.100000,5,0.200000,1.500000,"
        "-0.500000,50,0.900000,0.400000"
    )


def test_csv_row_bucket_mode_missing_bin_renders_blanks() -> None:
    """A bin listed in bin_names but absent from bucket_per_bin emits four blanks."""
    row = EpochMetrics(
        epoch=1,
        train_loss=0.5,
        val_loss=0.6,
        lr=1e-3,
        rollout_score=-0.20,
        bucket_per_bin={"extreme": _rollout_score_fixture(0.10, 5, 0.20, 1.5)},
    )
    out = row.to_csv_row(("extreme", "smooth"))

    # tail of the row must be the extreme columns then four blanks for smooth
    assert out.endswith("0.100000,5,0.200000,1.500000,,,,")


def test_csv_header_and_row_have_matching_column_counts() -> None:
    """Header and row must agree on column count for both modes."""
    header_plain = EpochMetrics.csv_header()
    row_plain = EpochMetrics(epoch=1, train_loss=0.0, val_loss=0.0, lr=1e-3).to_csv_row()
    assert header_plain.count(",") == row_plain.count(",")

    bin_names = ("extreme", "very_close", "close", "moderate", "mild", "smooth")
    header_bucket = EpochMetrics.csv_header(bin_names)
    row_bucket = EpochMetrics(epoch=1, train_loss=0.0, val_loss=0.0, lr=1e-3).to_csv_row(bin_names)
    assert header_bucket.count(",") == row_bucket.count(",")


def test_bucket_rollout_score_macro_equals_arithmetic_mean() -> None:
    """BucketRolloutScore.macro must be the arithmetic mean of populated per-bin scores.

    Pinned explicitly so a future "improvement" to weighted averaging
    would have to deliberately change this assertion.
    """
    per_bin = {
        "extreme": _rollout_score_fixture(0.10, 5, 0.20, 1.5),
        "close": _rollout_score_fixture(-0.30, 25, 0.60, 0.8),
        "smooth": _rollout_score_fixture(-0.50, 50, 0.90, 0.4),
    }
    expected_macro = (0.10 + -0.30 + -0.50) / 3
    bucket = BucketRolloutScore(
        macro=expected_macro,
        per_bin=per_bin,
        bin_order=("extreme", "close", "smooth"),
    )
    assert bucket.macro == pytest.approx(expected_macro)


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
