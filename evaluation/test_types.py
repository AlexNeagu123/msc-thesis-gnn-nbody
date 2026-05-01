"""Round-trip tests for evaluation/_types.py.

Uses hand-written fixture dicts (not smoke-run outputs) to keep tests fast
and decoupled from the model code. The on-disk metrics.json check is a
soft guard: it loads if present, skips silently otherwise.
"""

import json
from pathlib import Path

from evaluation._types import EvaluationReport, SummaryRow


def _egnn_report_dict() -> dict:
    """Hand-written EGNN-shape fixture (no learned_hamiltonian, with curves)."""
    return {
        "metadata": {
            "model_name": "egnn",
            "checkpoint_path": "checkpoints/egnn/x/best.pt",
            "config_path": "configs/egnn.yaml",
            "test_path": "data/output/test.h5",
            "device": "cpu",
            "checkpoint_epoch": 366,
            "checkpoint_val_loss": 0.016,
            "run_id": "x",
            "git_commit": None,
            "pos_std": 0.97,
            "vel_std": 0.85,
            "n_trajectories": 100,
            "n_frames": 200,
            "n_transitions": 199,
            "n_particles": 3,
        },
        "single_step": {
            "mse": {
                "mean": 0.06,
                "median": 0.0002,
                "max": 77.8,
                "p95": 0.055,
                "p99": 0.69,
            },
            "min_pairwise_distance": {
                "mean": 0.75,
                "median": 0.6,
                "max": 5.43,
                "p5": 0.12,
                "p50": 0.6,
            },
        },
        "rollout": {
            "steps": {
                "1": {
                    "mean_finite_mse": 0.22,
                    "median_mse": 0.0001,
                    "p95_mse": 0.0013,
                    "finite_fraction": 1.0,
                },
                "10": {
                    "mean_finite_mse": 1.09,
                    "median_mse": 0.06,
                    "p95_mse": 6.3,
                    "finite_fraction": 1.0,
                },
            },
            "curves": {
                "step": [0, 1, 2, 3],
                "mean_finite_mse": [0.0, 0.22, 0.5, None],
                "median_mse": [0.0, 0.0001, 0.06, 23.7],
                "p95_mse": [0.0, 0.0013, 6.3, None],
                "finite_fraction": [1.0, 1.0, 1.0, 0.85],
            },
            "first_nonfinite_step": [None, None, 145, None],
            "thresholds": {
                "1": {"first_step": [16, 7, None, 9], "final_fraction_below": 0.0},
                "10": {"first_step": [21, 13, 33, 48], "final_fraction_below": 0.0},
            },
            "finite_final_fraction": 0.67,
        },
        "energy": {
            "physical": {
                "final_relative_drift": {
                    "mean": 2.21,
                    "median": 1.39,
                    "max": 10.5,
                    "p95": 6.68,
                },
                "max_relative_drift": {
                    "mean": 2.08e35,
                    "median": 9.05,
                    "max": 1.8e37,
                    "p95": 5.11e30,
                },
                "per_trajectory_final": [4.89, 2.83, None, 1.6],
                "per_trajectory_max": [15.49, 20.46, 1.35e28, 3.87],
            },
        },
    }


def _hgnn_report_dict() -> dict:
    """Hand-written HGNN-shape fixture (with learned_hamiltonian)."""
    base = _egnn_report_dict()
    base["metadata"]["model_name"] = "hgnn"
    base["energy"]["learned_hamiltonian"] = {
        "final_relative_drift": {"mean": 0.05, "median": 0.05, "max": 0.1, "p95": 0.08},
        "max_relative_drift": {"mean": 0.1, "median": 0.1, "max": 0.2, "p95": 0.15},
        "per_trajectory_final": [0.05, 0.04, 0.06, None],
        "per_trajectory_max": [0.1, 0.08, 0.12, 0.15],
    }
    return base


def _legacy_report_dict() -> dict:
    """Old-shape fixture without rollout.curves (matches early metrics.json files)."""
    base = _egnn_report_dict()
    del base["rollout"]["curves"]
    return base


def test_evaluation_report_round_trip_egnn() -> None:
    """EGNN-shape report round-trips through from_dict/to_dict."""
    d = _egnn_report_dict()
    assert EvaluationReport.from_dict(d).to_dict() == d


def test_evaluation_report_round_trip_hgnn() -> None:
    """HGNN-shape report (with learned_hamiltonian) round-trips."""
    d = _hgnn_report_dict()
    assert EvaluationReport.from_dict(d).to_dict() == d


def test_evaluation_report_legacy_no_curves_round_trip() -> None:
    """Old metrics.json without rollout.curves round-trips with curves still absent."""
    d = _legacy_report_dict()
    out = EvaluationReport.from_dict(d).to_dict()
    assert "curves" not in out["rollout"]
    assert out == d


def test_evaluation_report_typed_access() -> None:
    """Loaded report exposes attribute access on nested fields."""
    report = EvaluationReport.from_dict(_egnn_report_dict())
    assert report.metadata.model_name == "egnn"
    assert report.metadata.checkpoint_val_loss == 0.016
    assert report.single_step.mse.median == 0.0002
    assert report.rollout.curves is not None
    assert report.rollout.curves.median_mse[2] == 0.06
    assert report.energy.physical.final_relative_drift.median == 1.39
    assert report.energy.learned_hamiltonian is None


def test_hgnn_learned_hamiltonian_present() -> None:
    """HGNN report exposes typed learned_hamiltonian access."""
    report = EvaluationReport.from_dict(_hgnn_report_dict())
    assert report.energy.learned_hamiltonian is not None
    assert report.energy.learned_hamiltonian.final_relative_drift.median == 0.05


def test_egnn_learned_hamiltonian_omitted_in_to_dict() -> None:
    """EGNN report serializes without a learned_hamiltonian key."""
    report = EvaluationReport.from_dict(_egnn_report_dict())
    out = report.to_dict()
    assert "learned_hamiltonian" not in out["energy"]


def test_summary_row_includes_dynamic_step_keys() -> None:
    """CSV row has rollout_step_<n>_* columns for each step in the report."""
    report = EvaluationReport.from_dict(_egnn_report_dict())
    row = SummaryRow.from_report(report).to_csv_row()
    assert "rollout_step_1_mean_finite_mse" in row
    assert "rollout_step_10_p95_mse" in row
    assert row["rollout_step_1_finite_fraction"] == 1.0


def test_summary_row_includes_dynamic_threshold_keys() -> None:
    """CSV row has rollout_final_fraction_below_mse_<t> for each threshold."""
    report = EvaluationReport.from_dict(_egnn_report_dict())
    row = SummaryRow.from_report(report).to_csv_row()
    assert "rollout_final_fraction_below_mse_1" in row
    assert row["rollout_final_fraction_below_mse_10"] == 0.0


def test_summary_row_excludes_learned_h_for_egnn() -> None:
    """EGNN row contains no learned_h_* columns."""
    report = EvaluationReport.from_dict(_egnn_report_dict())
    row = SummaryRow.from_report(report).to_csv_row()
    assert "learned_h_final_drift_mean" not in row


def test_summary_row_includes_learned_h_for_hgnn() -> None:
    """HGNN row includes learned_h_* columns."""
    report = EvaluationReport.from_dict(_hgnn_report_dict())
    row = SummaryRow.from_report(report).to_csv_row()
    assert row["learned_h_final_drift_median"] == 0.05
    assert "learned_h_max_drift_max" in row


def test_existing_metrics_json_round_trip() -> None:
    """Every on-disk metrics.json must survive from_dict -> to_dict unchanged."""
    files = sorted(Path("results/evaluation").glob("**/metrics.json"))
    if not files:
        return
    for path in files:
        with path.open() as f:
            original = json.load(f)
        out = EvaluationReport.from_dict(original).to_dict()
        assert out == original, f"round-trip mismatch for {path}"


_EXPECTED_STATIC_COLUMNS = (
    "model_name",
    "run_id",
    "checkpoint_epoch",
    "checkpoint_val_loss",
    "n_trajectories",
    "n_frames",
    "n_transitions",
    "n_particles",
    "single_step_mse_mean",
    "single_step_mse_median",
    "single_step_mse_p95",
    "single_step_mse_p99",
    "single_step_mse_max",
    "physical_energy_final_drift_mean",
    "physical_energy_final_drift_median",
    "physical_energy_final_drift_p95",
    "physical_energy_final_drift_max",
    "physical_energy_max_drift_mean",
    "physical_energy_max_drift_median",
    "physical_energy_max_drift_p95",
    "physical_energy_max_drift_max",
)

_EXPECTED_LEARNED_H_COLUMNS = (
    "learned_h_final_drift_mean",
    "learned_h_final_drift_median",
    "learned_h_final_drift_p95",
    "learned_h_final_drift_max",
    "learned_h_max_drift_mean",
    "learned_h_max_drift_median",
    "learned_h_max_drift_p95",
    "learned_h_max_drift_max",
)


def test_summary_row_egnn_column_order_pinned() -> None:
    """EGNN CSV header order is pinned: static + per-step + final + per-threshold."""
    report = EvaluationReport.from_dict(_egnn_report_dict())
    cols = tuple(SummaryRow.from_report(report).to_csv_row().keys())

    expected = (
        *_EXPECTED_STATIC_COLUMNS,
        # per-step in report.rollout.steps insertion order ("1", "10")
        "rollout_step_1_mean_finite_mse",
        "rollout_step_1_median_mse",
        "rollout_step_1_p95_mse",
        "rollout_step_1_finite_fraction",
        "rollout_step_10_mean_finite_mse",
        "rollout_step_10_median_mse",
        "rollout_step_10_p95_mse",
        "rollout_step_10_finite_fraction",
        "rollout_final_finite_fraction",
        # per-threshold in report.rollout.thresholds insertion order ("1", "10")
        "rollout_final_fraction_below_mse_1",
        "rollout_final_fraction_below_mse_10",
    )
    assert cols == expected


def test_summary_row_hgnn_appends_learned_h_columns() -> None:
    """HGNN CSV header is the EGNN header followed by learned_h_* columns."""
    egnn_report = EvaluationReport.from_dict(_egnn_report_dict())
    hgnn_report = EvaluationReport.from_dict(_hgnn_report_dict())

    egnn_cols = tuple(SummaryRow.from_report(egnn_report).to_csv_row().keys())
    hgnn_cols = tuple(SummaryRow.from_report(hgnn_report).to_csv_row().keys())

    assert hgnn_cols == egnn_cols + _EXPECTED_LEARNED_H_COLUMNS
