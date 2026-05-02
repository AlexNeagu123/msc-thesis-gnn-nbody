"""Tests for evaluation/_io.py.

File-system round-trip: a typed report written to disk and read back must
be equal to the original. Also pins the CSV header produced by the writer.
"""

import csv
from pathlib import Path

import numpy as np

from evaluation._io import (
    _json_safe,
    read_evaluation_report,
    write_evaluation_report,
    write_summary_csv,
)
from evaluation._types import EvaluationReport


def _egnn_fixture() -> dict:
    """Minimal EGNN-shape report fixture (no learned_hamiltonian)."""
    return {
        "metadata": {
            "model_name": "egnn",
            "checkpoint_path": "x/best.pt",
            "config_path": "x.yaml",
            "test_path": "test.h5",
            "device": "cpu",
            "checkpoint_epoch": 1,
            "checkpoint_val_loss": 0.1,
            "run_id": "x",
            "git_commit": None,
            "pos_std": 1.0,
            "vel_std": 1.0,
            "n_trajectories": 1,
            "n_frames": 2,
            "n_transitions": 1,
            "n_particles": 3,
        },
        "single_step": {
            "state_mse": {"mean": 0.1, "median": 0.1, "max": 0.1, "p95": 0.1, "p99": 0.1},
            "position_mse": {
                "mean": 0.05,
                "median": 0.05,
                "max": 0.05,
                "p95": 0.05,
                "p99": 0.05,
            },
            "velocity_mse": {
                "mean": 0.15,
                "median": 0.15,
                "max": 0.15,
                "p95": 0.15,
                "p99": 0.15,
            },
            "min_pairwise_distance": {
                "mean": 0.5,
                "median": 0.5,
                "max": 0.5,
                "p5": 0.5,
                "p50": 0.5,
            },
        },
        "rollout": {
            "steps": {
                "1": {
                    "state_mse": {
                        "mean_finite": 0.1,
                        "median": 0.1,
                        "p95": 0.1,
                        "finite_fraction": 1.0,
                    },
                    "position_mse": {
                        "mean_finite": 0.05,
                        "median": 0.05,
                        "p95": 0.05,
                        "finite_fraction": 1.0,
                    },
                    "velocity_mse": {
                        "mean_finite": 0.15,
                        "median": 0.15,
                        "p95": 0.15,
                        "finite_fraction": 1.0,
                    },
                },
            },
            "curves": {
                "step": [0, 1],
                "state_mse": {
                    "mean_finite": [0.0, 0.1],
                    "median": [0.0, 0.1],
                    "p95": [0.0, 0.1],
                    "finite_fraction": [1.0, 1.0],
                },
                "position_mse": {
                    "mean_finite": [0.0, 0.05],
                    "median": [0.0, 0.05],
                    "p95": [0.0, 0.05],
                    "finite_fraction": [1.0, 1.0],
                },
                "velocity_mse": {
                    "mean_finite": [0.0, 0.15],
                    "median": [0.0, 0.15],
                    "p95": [0.0, 0.15],
                    "finite_fraction": [1.0, 1.0],
                },
            },
            "first_nonfinite_step": [None],
            "state_mse_thresholds": {
                "1": {"first_step": [None], "final_fraction_below": 1.0},
            },
            "position_mse_thresholds": {
                "1": {"first_step": [None], "final_fraction_below": 1.0},
            },
            "state_final_finite_fraction": 1.0,
        },
        "energy": {
            "physical": {
                "final_relative_drift": {"mean": 0.0, "median": 0.0, "max": 0.0, "p95": 0.0},
                "max_relative_drift": {"mean": 0.0, "median": 0.0, "max": 0.0, "p95": 0.0},
                "per_trajectory_final": [0.0],
                "per_trajectory_max": [0.0],
            },
        },
    }


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    """A report written to disk and read back equals the original."""
    fixture = _egnn_fixture()
    report = EvaluationReport.from_dict(fixture)

    path = tmp_path / "metrics.json"
    write_evaluation_report(path, report)
    loaded = read_evaluation_report(path)

    assert loaded.to_dict() == fixture


def test_write_summary_csv_header_matches_summary_row(tmp_path: Path) -> None:
    """CSV writer header order is identical to SummaryRow.to_csv_row keys."""
    report = EvaluationReport.from_dict(_egnn_fixture())

    path = tmp_path / "summary.csv"
    write_summary_csv(path, report)

    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["model_name"] == "egnn"
    assert "rollout_step_1_state_mse_mean_finite" in rows[0]
    assert "rollout_final_fraction_below_state_mse_1" in rows[0]
    # learned_h_* columns absent for EGNN
    assert all(not k.startswith("learned_h_") for k in rows[0])


def test_json_safe_drops_non_finite_floats() -> None:
    """NaN and inf become None; numpy scalars become Python primitives."""
    assert _json_safe(float("nan")) is None
    assert _json_safe(float("inf")) is None
    assert _json_safe(np.float64(1.5)) == 1.5
    assert _json_safe(np.int64(7)) == 7
    assert _json_safe({"x": [np.float32(1.0), float("nan")]}) == {"x": [1.0, None]}
