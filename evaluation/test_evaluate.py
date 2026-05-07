"""Tests for evaluation/evaluate.py."""

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from data._io import write_trajectories
from data._types import EncounterBin, Trajectories
from evaluation.evaluate import (
    _normalization_stats,
    _output_dir,
    evaluate_checkpoint,
)
from evaluation.metrics import compute_rollout_mse, run_all_rollouts
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


def _write_stratified_h5(path: Path, n_steps: int = 4) -> Trajectories:
    """Write a 4-trajectory stratified fixture with 2 bins, return the bundle.

    Trajectories alternate bins so per-bin slicing has 2 trajectories each,
    enough to exercise the per-bin aggregation path while keeping the
    fixture cheap to evaluate against a tiny model.
    """
    n_traj = 4
    rng = np.random.default_rng(7)
    states = rng.normal(size=(n_traj, n_steps, 3, 5)).astype(np.float32)
    states[..., 4] = 1.0
    energies = np.zeros((n_traj, n_steps), dtype=np.float32)

    bins = (
        EncounterBin(name="extreme", lo=0.0, hi=0.05),
        EncounterBin(name="smooth", lo=0.05, hi=float("inf")),
    )
    bin_id = np.array([0, 1, 0, 1], dtype=np.int64)
    bin_name = np.array(["extreme", "smooth", "extreme", "smooth"])
    d_min = np.array([0.02, 0.5, 0.03, 1.0], dtype=np.float64)

    bundle = Trajectories(
        states=states,
        energies=energies,
        encounter_bin_id=bin_id,
        encounter_bin_name=bin_name,
        min_pairwise_distance=d_min,
        encounter_bins=bins,
    )
    write_trajectories(path, bundle)
    return bundle


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


def test_legacy_test_file_yields_no_encounter_bins_block(tmp_path: Path) -> None:
    """Non-stratified h5 evaluation produces report.encounter_bins=None and omits the key."""
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
        Checkpoint(epoch=1, model=model.state_dict(), optimizer={}, val_loss=0.1),
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

    assert report.encounter_bins is None
    data = json.loads((output_dir / "metrics.json").read_text())
    assert "encounter_bins" not in data


def test_stratified_test_file_populates_encounter_bins(tmp_path: Path) -> None:
    """Stratified h5 evaluation produces a populated per-bin report block."""
    train_path = tmp_path / "train.h5"
    val_path = tmp_path / "val.h5"
    test_path = tmp_path / "test.h5"
    _write_h5(train_path)
    _write_h5(val_path)
    _write_stratified_h5(test_path)

    cfg = _cfg(train_path, val_path, "egnn")
    config_path = tmp_path / "egnn.yaml"
    _write_config(config_path, cfg)

    model = EGNN(hidden_dim=8, n_layers=1)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        Checkpoint(epoch=1, model=model.state_dict(), optimizer={}, val_loss=0.1),
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

    assert report.encounter_bins is not None
    bins = report.encounter_bins
    assert [b.name for b in bins.bins] == ["extreme", "smooth"]
    assert [b.id for b in bins.bins] == [0, 1]
    assert bins.bins[1].hi == float("inf")
    assert set(bins.by_name) == {"extreme", "smooth"}

    extreme = bins.by_name["extreme"]
    smooth = bins.by_name["smooth"]
    assert extreme.count == 2
    assert smooth.count == 2
    assert extreme.count + smooth.count == report.metadata.n_trajectories

    # d_min summary tracks the bundle d_mins, not predicted values
    assert extreme.d_min.median == 0.025  # median([0.02, 0.03])
    assert smooth.d_min.median == 0.75  # median([0.5, 1.0])

    # JSON output round-trips and includes the encounter_bins top-level block
    data = json.loads((output_dir / "metrics.json").read_text())
    assert "encounter_bins" in data
    assert data["encounter_bins"]["bins"][1]["hi"] == "inf"
    assert data["encounter_bins"]["by_name"]["extreme"]["count"] == 2


def test_per_bin_rollout_matches_manual_slicing(tmp_path: Path) -> None:
    """Per-bin rollout summaries equal a manual slice of the full rollout MSE."""
    train_path = tmp_path / "train.h5"
    val_path = tmp_path / "val.h5"
    test_path = tmp_path / "test.h5"
    _write_h5(train_path)
    _write_h5(val_path)
    bundle = _write_stratified_h5(test_path)

    cfg = _cfg(train_path, val_path, "egnn")
    config_path = tmp_path / "egnn.yaml"
    _write_config(config_path, cfg)

    model = EGNN(hidden_dim=8, n_layers=1)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        Checkpoint(epoch=1, model=model.state_dict(), optimizer={}, val_loss=0.1),
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

    # reproduce the model's rollout outside the orchestrator and slice manually
    model.eval()
    predicted = run_all_rollouts(model, bundle.states, torch.device("cpu"))
    full_mse = compute_rollout_mse(bundle.states, predicted)

    extreme_mask = bundle.encounter_bin_id == 0
    expected_extreme_median = np.nanmedian(full_mse.state.per_trajectory[extreme_mask], axis=0)

    assert report.encounter_bins is not None
    extreme = report.encounter_bins.by_name["extreme"]
    assert extreme.rollout.curves is not None
    actual_extreme_median = np.array(
        [m if m is not None else np.nan for m in extreme.rollout.curves.state_mse.median]
    )
    np.testing.assert_allclose(actual_extreme_median, expected_extreme_median, atol=1e-6)


def test_summary_csv_unchanged_for_stratified_test_file(tmp_path: Path) -> None:
    """Block 2 must not widen summary.csv columns even with stratified data."""
    # baseline: legacy test file
    legacy_test = tmp_path / "legacy_test.h5"
    train_path = tmp_path / "train.h5"
    val_path = tmp_path / "val.h5"
    _write_h5(train_path)
    _write_h5(val_path)
    _write_h5(legacy_test)

    # stratified test file
    strat_test = tmp_path / "strat_test.h5"
    _write_stratified_h5(strat_test)

    cfg = _cfg(train_path, val_path, "egnn")
    config_path = tmp_path / "egnn.yaml"
    _write_config(config_path, cfg)

    model = EGNN(hidden_dim=8, n_layers=1)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        Checkpoint(epoch=1, model=model.state_dict(), optimizer={}, val_loss=0.1),
        checkpoint_path,
    )

    legacy_dir = tmp_path / "eval_legacy"
    strat_dir = tmp_path / "eval_strat"
    evaluate_checkpoint(
        cfg,
        checkpoint_path,
        config_path=config_path,
        test_path=legacy_test,
        output_dir=legacy_dir,
        device="cpu",
    )
    evaluate_checkpoint(
        cfg,
        checkpoint_path,
        config_path=config_path,
        test_path=strat_test,
        output_dir=strat_dir,
        device="cpu",
    )

    with (legacy_dir / "summary.csv").open() as f:
        legacy_cols = list(csv.DictReader(f).fieldnames or [])
    with (strat_dir / "summary.csv").open() as f:
        strat_cols = list(csv.DictReader(f).fieldnames or [])

    assert legacy_cols == strat_cols


def test_per_bin_baseline_ratios_populated(tmp_path: Path) -> None:
    """Stratified evaluation attaches baseline_ratios to each non-empty bin."""
    train_path = tmp_path / "train.h5"
    val_path = tmp_path / "val.h5"
    test_path = tmp_path / "test.h5"
    _write_h5(train_path)
    _write_h5(val_path)
    _write_stratified_h5(test_path)

    cfg = _cfg(train_path, val_path, "egnn")
    config_path = tmp_path / "egnn.yaml"
    _write_config(config_path, cfg)

    model = EGNN(hidden_dim=8, n_layers=1)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        Checkpoint(epoch=1, model=model.state_dict(), optimizer={}, val_loss=0.1),
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

    assert report.encounter_bins is not None
    for bin_name in ("extreme", "smooth"):
        ratios = report.encounter_bins.by_name[bin_name].baseline_ratios
        assert ratios is not None, f"baseline_ratios missing for bin {bin_name!r}"
        assert isinstance(ratios.dominance_horizon, int)
        # anchor-step keys are ints in memory
        assert all(isinstance(k, int) for k in ratios.state_mse_ratios)


def test_baseline_ratios_json_shape(tmp_path: Path) -> None:
    """JSON output of baseline_ratios pins the exact agreed shape."""
    train_path = tmp_path / "train.h5"
    val_path = tmp_path / "val.h5"
    test_path = tmp_path / "test.h5"
    _write_h5(train_path)
    _write_h5(val_path)
    _write_stratified_h5(test_path)

    cfg = _cfg(train_path, val_path, "egnn")
    config_path = tmp_path / "egnn.yaml"
    _write_config(config_path, cfg)

    model = EGNN(hidden_dim=8, n_layers=1)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        Checkpoint(epoch=1, model=model.state_dict(), optimizer={}, val_loss=0.1),
        checkpoint_path,
    )

    output_dir = tmp_path / "eval"
    evaluate_checkpoint(
        cfg,
        checkpoint_path,
        config_path=config_path,
        test_path=test_path,
        output_dir=output_dir,
        device="cpu",
    )

    data = json.loads((output_dir / "metrics.json").read_text())
    ratios_json = data["encounter_bins"]["by_name"]["extreme"]["baseline_ratios"]
    assert set(ratios_json) == {
        "score",
        "state_mse",
        "dominance_horizon",
        "fraction_beating_baseline",
        "final_ratio",
    }
    # JSON anchor-step keys are strings; values are floats or null
    assert all(isinstance(k, str) for k in ratios_json["state_mse"])
    for v in ratios_json["state_mse"].values():
        assert v is None or isinstance(v, float | int)
