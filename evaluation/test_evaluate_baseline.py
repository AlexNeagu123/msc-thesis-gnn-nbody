"""Tests for evaluation/evaluate_baseline.py."""

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from data._io import write_trajectories
from data._types import EncounterBin, Trajectories
from evaluation._types import EvaluationReport
from evaluation.evaluate_baseline import _resolve_output_dir, evaluate_baseline


def _write_h5(path: Path, n_traj: int = 2, n_steps: int = 4) -> None:
    """Write a tiny deterministic trajectory file."""
    rng = np.random.default_rng(42)
    trajectories = rng.normal(size=(n_traj, n_steps, 3, 5)).astype(np.float32)
    trajectories[..., 4] = 1.0
    with h5py.File(path, "w") as f:
        f.create_dataset("trajectories", data=trajectories)
        f.create_dataset("energies", data=np.zeros((n_traj, n_steps), dtype=np.float32))


def _write_stratified_h5(path: Path, n_steps: int = 4) -> None:
    """Write a 4-trajectory stratified fixture with two bins (extreme and smooth)."""
    n_traj = 4
    rng = np.random.default_rng(7)
    states = rng.normal(size=(n_traj, n_steps, 3, 5)).astype(np.float32)
    states[..., 4] = 1.0
    energies = np.zeros((n_traj, n_steps), dtype=np.float32)

    bins = (
        EncounterBin(name="extreme", lo=0.0, hi=0.05),
        EncounterBin(name="smooth", lo=0.05, hi=float("inf")),
    )
    bundle = Trajectories(
        states=states,
        energies=energies,
        encounter_bin_id=np.array([0, 1, 0, 1], dtype=np.int64),
        encounter_bin_name=np.array(["extreme", "smooth", "extreme", "smooth"]),
        min_pairwise_distance=np.array([0.02, 0.5, 0.03, 1.0], dtype=np.float64),
        encounter_bins=bins,
    )
    write_trajectories(path, bundle)


def test_mean_state_baseline_writes_artifacts(tmp_path: Path) -> None:
    """Mean_state baseline fits from train HDF5, evaluates on test, writes JSON+CSV."""
    train_path = tmp_path / "train.h5"
    test_path = tmp_path / "test.h5"
    _write_h5(train_path, n_traj=3, n_steps=5)
    _write_h5(test_path, n_traj=2, n_steps=4)
    output_dir = tmp_path / "out"

    report = evaluate_baseline(
        baseline="mean_state",
        test_path=test_path,
        train_path=train_path,
        output_dir=output_dir,
        device="cpu",
    )

    assert isinstance(report, EvaluationReport)
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "summary.csv").exists()

    metadata = report.metadata
    assert metadata.model_name == "baseline_mean_state"
    assert metadata.run_id == "baseline_mean_state"
    assert metadata.checkpoint_path is None
    assert metadata.config_path is None
    assert metadata.checkpoint_epoch is None
    assert metadata.checkpoint_val_loss is None
    assert metadata.pos_std > 0
    assert metadata.vel_std > 0
    assert metadata.n_trajectories == 2
    assert metadata.n_frames == 4
    assert metadata.n_transitions == 3
    assert metadata.n_particles == 3

    assert report.energy.learned_hamiltonian is None

    data = json.loads((output_dir / "metrics.json").read_text())
    assert data["metadata"]["model_name"] == "baseline_mean_state"
    assert data["metadata"]["checkpoint_path"] is None
    assert "single_step" in data
    assert "rollout" in data
    assert "energy" in data
    assert "physical" in data["energy"]

    with (output_dir / "summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["model_name"] == "baseline_mean_state"
    assert rows[0]["run_id"] == "baseline_mean_state"
    assert rows[0]["checkpoint_epoch"] == ""
    assert rows[0]["checkpoint_val_loss"] == ""


def test_persistence_baseline_runs_without_train_path(tmp_path: Path) -> None:
    """Persistence baseline does not require --train-path."""
    test_path = tmp_path / "test.h5"
    _write_h5(test_path)
    output_dir = tmp_path / "out"

    report = evaluate_baseline(
        baseline="persistence",
        test_path=test_path,
        output_dir=output_dir,
    )

    assert (output_dir / "metrics.json").exists()
    assert report.metadata.model_name == "baseline_persistence"


def test_default_baseline_output_dir_uses_runs_archive() -> None:
    """Baseline reports default into the canonical runs/ artifact archive."""
    assert _resolve_output_dir(None, "persistence") == Path("runs/baselines/persistence/evaluation")


def test_baseline_rejects_fitted_without_train_path(tmp_path: Path) -> None:
    """Mean_velocity and mean_state require --train-path."""
    test_path = tmp_path / "test.h5"
    _write_h5(test_path)

    with pytest.raises(ValueError, match="requires --train-path"):
        evaluate_baseline(
            baseline="mean_state",
            test_path=test_path,
            output_dir=tmp_path / "out",
        )


def test_baseline_rejects_unknown_kind(tmp_path: Path) -> None:
    """Unknown baseline name raises."""
    test_path = tmp_path / "test.h5"
    _write_h5(test_path)

    with pytest.raises(ValueError, match="unknown baseline"):
        evaluate_baseline(
            baseline="foo_bar",
            test_path=test_path,
            output_dir=tmp_path / "out",
        )


def test_baseline_on_stratified_test_emits_encounter_bins(tmp_path: Path) -> None:
    """Baseline evaluation on a stratified test file produces the per-bin block."""
    train_path = tmp_path / "train.h5"
    test_path = tmp_path / "test.h5"
    _write_h5(train_path)
    _write_stratified_h5(test_path)
    output_dir = tmp_path / "out"

    report = evaluate_baseline(
        baseline="constant_velocity",
        test_path=test_path,
        train_path=train_path,
        output_dir=output_dir,
        device="cpu",
    )

    assert report.encounter_bins is not None
    bins = report.encounter_bins
    assert {b.name for b in bins.bins} == {"extreme", "smooth"}
    assert bins.bins[1].hi == float("inf")
    assert bins.by_name["extreme"].count == 2
    assert bins.by_name["smooth"].count == 2

    data = json.loads((output_dir / "metrics.json").read_text())
    assert "encounter_bins" in data
    assert data["encounter_bins"]["bins"][1]["hi"] == "inf"
    assert data["encounter_bins"]["by_name"]["smooth"]["count"] == 2


def test_baseline_on_stratified_without_train_path_raises(tmp_path: Path) -> None:
    """Stratified test file requires train_path even for non-fitted baselines.

    train_path is needed to build the baseline envelope; dropping per-bin ratios silently
    would produce inconsistent reports.
    """
    test_path = tmp_path / "test.h5"
    _write_stratified_h5(test_path)

    with pytest.raises(ValueError, match="stratified baseline ratios require --train-path"):
        evaluate_baseline(
            baseline="persistence",
            test_path=test_path,
            output_dir=tmp_path / "out",
            device="cpu",
        )


def test_baseline_on_stratified_emits_baseline_ratios(tmp_path: Path) -> None:
    """Per-bin baseline_ratios block is populated for non-empty bins."""
    train_path = tmp_path / "train.h5"
    test_path = tmp_path / "test.h5"
    _write_h5(train_path)
    _write_stratified_h5(test_path)
    output_dir = tmp_path / "out"

    report = evaluate_baseline(
        baseline="persistence",
        test_path=test_path,
        train_path=train_path,
        output_dir=output_dir,
        device="cpu",
    )

    assert report.encounter_bins is not None
    extreme = report.encounter_bins.by_name["extreme"]
    assert extreme.baseline_ratios is not None
    # anchor-step keys are ints in memory, stringified in JSON
    assert all(isinstance(k, int) for k in extreme.baseline_ratios.state_mse_ratios)

    data = json.loads((output_dir / "metrics.json").read_text())
    ratios_json = data["encounter_bins"]["by_name"]["extreme"]["baseline_ratios"]
    assert "score" in ratios_json
    assert "dominance_horizon" in ratios_json
    assert "fraction_beating_baseline" in ratios_json
    assert "final_ratio" in ratios_json
    assert all(isinstance(k, str) for k in ratios_json["state_mse"])
