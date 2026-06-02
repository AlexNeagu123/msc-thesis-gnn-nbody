"""Tests for evaluation/report_generalization.py."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from evaluation._types import (
    DistanceSummary,
    DriftSummary,
    EnergyDriftCurves,
    EnergyDriftReport,
    EnergyReport,
    EvaluationMetadata,
    EvaluationReport,
    MseSummary,
    RolloutCurves,
    RolloutMetricCurves,
    RolloutMetricSummary,
    RolloutReport,
    RolloutStepMetrics,
    SingleStepReport,
)
from evaluation.report_generalization import (
    SUMMARY_CSV_NAME,
    GeneralizationReporter,
)

_N_FRAMES = 200


def _metric_curves(n: int) -> RolloutMetricCurves:
    """Build a simple increasing metric curve of length n."""
    vals = [float(i) for i in range(n)]
    return RolloutMetricCurves(mean_finite=vals, median=vals, p95=vals, finite_fraction=[1.0] * n)


def _rollout_report(n: int) -> RolloutReport:
    """Build a rollout report with full per-step curves and a few anchor steps."""
    summary = RolloutMetricSummary(mean_finite=1.0, median=1.0, p95=1.0, finite_fraction=1.0)
    step_metrics = RolloutStepMetrics(state_mse=summary, position_mse=summary, velocity_mse=summary)
    return RolloutReport(
        steps={str(k): step_metrics for k in (1, 10, 50, 100, 199)},
        first_nonfinite_step=[None],
        state_mse_thresholds={},
        position_mse_thresholds={},
        state_final_finite_fraction=1.0,
        curves=RolloutCurves(
            step=list(range(n)),
            state_mse=_metric_curves(n),
            position_mse=_metric_curves(n),
            velocity_mse=_metric_curves(n),
        ),
    )


def _energy_report(n: int) -> EnergyReport:
    """Build an energy report with full per-step drift curves."""
    drift = DriftSummary(mean=1.0, median=1.0, max=1.0, p95=1.0)
    vals = [float(i) for i in range(n)]
    return EnergyReport(
        physical=EnergyDriftReport(
            final_relative_drift=drift,
            max_relative_drift=drift,
            per_trajectory_final=[1.0],
            per_trajectory_max=[1.0],
            steps={},
            curves=EnergyDriftCurves(
                step=list(range(n)),
                mean_finite=vals,
                median=vals,
                p95=vals,
                finite_fraction=[1.0] * n,
            ),
            per_trajectory_at_steps={},
        )
    )


def _single_step() -> SingleStepReport:
    """Build a trivial single-step block."""
    mse = MseSummary(mean=1.0, median=1.0, max=1.0, p95=1.0, p99=1.0)
    distance = DistanceSummary(mean=1.0, median=1.0, max=1.0, p5=1.0, p50=1.0)
    return SingleStepReport(
        state_mse=mse,
        position_mse=mse,
        velocity_mse=mse,
        min_pairwise_distance=distance,
    )


def _make_report(model_name: str, n_particles: int, n_frames: int = _N_FRAMES) -> EvaluationReport:
    """Build a minimal but schema-complete EvaluationReport for one model."""
    metadata = EvaluationMetadata(
        model_name=model_name,
        checkpoint_path="ckpt.pt",
        config_path="cfg.yaml",
        test_path="test.h5",
        device="cpu",
        checkpoint_epoch=1,
        checkpoint_val_loss=0.1,
        run_id="run",
        git_commit=None,
        pos_std=1.0,
        vel_std=1.0,
        n_trajectories=200,
        n_frames=n_frames,
        n_transitions=n_frames - 1,
        n_particles=n_particles,
    )
    return EvaluationReport(
        metadata=metadata,
        single_step=_single_step(),
        rollout=_rollout_report(n_frames),
        energy=_energy_report(n_frames),
        encounter_bins=None,
    )


def _dump(report: EvaluationReport, path: Path) -> None:
    """Serialize a report to a metrics.json file."""
    path.write_text(json.dumps(report.to_dict()))


def _write_triple(
    tmp_path: Path,
    *,
    egnn_n: int = 4,
    hgnn_n: int = 4,
    baseline_name: str = "baseline_constant_velocity",
) -> tuple[Path, Path, Path]:
    """Write EGNN/HGNN/baseline metrics.json files and return their paths."""
    egnn_path = tmp_path / "egnn.json"
    hgnn_path = tmp_path / "hgnn.json"
    baseline_path = tmp_path / "baseline.json"
    _dump(_make_report("egnn", egnn_n), egnn_path)
    _dump(_make_report("hgnn", hgnn_n), hgnn_path)
    _dump(_make_report(baseline_name, egnn_n), baseline_path)
    return egnn_path, hgnn_path, baseline_path


def test_run_writes_all_artifacts(tmp_path: Path) -> None:
    """A valid triple produces figures, the summary CSV, and report.md."""
    egnn_path, hgnn_path, baseline_path = _write_triple(tmp_path)
    out = tmp_path / "out"

    GeneralizationReporter(egnn_path, hgnn_path, baseline_path, out).run()

    assert (out / "report.md").is_file()
    assert (out / "tables" / SUMMARY_CSV_NAME).is_file()
    for figure in (
        "01_rollout_position_mse.png",
        "02_energy_drift.png",
        "03_horizon_snapshots_early.png",
        "04_horizon_snapshots_late.png",
    ):
        assert (out / "figures" / figure).is_file()

    rows = list(csv.DictReader((out / "tables" / SUMMARY_CSV_NAME).open()))
    assert [r["model_name"] for r in rows] == ["egnn", "hgnn", "baseline_constant_velocity"]
    assert "N=4" in (out / "report.md").read_text()


def test_exclude_egnn_still_writes_figures_and_full_csv(tmp_path: Path) -> None:
    """include_egnn=False renders all figures (HGNN and baseline only) but keeps EGNN in the CSV."""
    egnn_path, hgnn_path, baseline_path = _write_triple(tmp_path)
    out = tmp_path / "out"

    GeneralizationReporter(egnn_path, hgnn_path, baseline_path, out, include_egnn=False).run()

    for figure in (
        "01_rollout_position_mse.png",
        "02_energy_drift.png",
        "03_horizon_snapshots_early.png",
        "04_horizon_snapshots_late.png",
    ):
        assert (out / "figures" / figure).is_file()

    rows = list(csv.DictReader((out / "tables" / SUMMARY_CSV_NAME).open()))
    assert [r["model_name"] for r in rows] == ["egnn", "hgnn", "baseline_constant_velocity"]


def test_rejects_non_constant_velocity_baseline(tmp_path: Path) -> None:
    """A baseline that is not constant-velocity is rejected."""
    egnn_path, hgnn_path, baseline_path = _write_triple(tmp_path, baseline_name="baseline_mean")
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="constant_velocity"):
        GeneralizationReporter(egnn_path, hgnn_path, baseline_path, out).run()


def test_rejects_n_particles_mismatch(tmp_path: Path) -> None:
    """Reports evaluated on different body counts cannot be combined."""
    egnn_path, hgnn_path, baseline_path = _write_triple(tmp_path, egnn_n=4, hgnn_n=5)
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="n_particles"):
        GeneralizationReporter(egnn_path, hgnn_path, baseline_path, out).run()
