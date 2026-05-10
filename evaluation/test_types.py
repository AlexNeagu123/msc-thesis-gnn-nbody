"""Round-trip tests for evaluation/_types.py.

Uses hand-written fixture dicts (not smoke-run outputs) to keep tests fast
and decoupled from the model code.
"""

from evaluation._types import (
    EncounterBinDefinition,
    EncounterBinReport,
    EncounterBinsReport,
    EvaluationReport,
    PerBinBaselineRatios,
    SummaryRow,
)


def _egnn_report_dict() -> dict:
    """Hand-written EGNN-shape fixture (no learned_hamiltonian, with curves)."""
    return {
        "metadata": {
            "model_name": "egnn",
            "checkpoint_path": "runs/egnn/x/best.pt",
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
            "state_mse": {
                "mean": 0.06,
                "median": 0.0002,
                "max": 77.8,
                "p95": 0.055,
                "p99": 0.69,
            },
            "position_mse": {
                "mean": 0.03,
                "median": 0.0001,
                "max": 38.9,
                "p95": 0.028,
                "p99": 0.34,
            },
            "velocity_mse": {
                "mean": 0.09,
                "median": 0.0003,
                "max": 116.7,
                "p95": 0.083,
                "p99": 1.04,
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
                    "state_mse": {
                        "mean_finite": 0.22,
                        "median": 0.0001,
                        "p95": 0.0013,
                        "finite_fraction": 1.0,
                    },
                    "position_mse": {
                        "mean_finite": 0.11,
                        "median": 0.00005,
                        "p95": 0.0007,
                        "finite_fraction": 1.0,
                    },
                    "velocity_mse": {
                        "mean_finite": 0.33,
                        "median": 0.00015,
                        "p95": 0.002,
                        "finite_fraction": 1.0,
                    },
                },
                "10": {
                    "state_mse": {
                        "mean_finite": 1.09,
                        "median": 0.06,
                        "p95": 6.3,
                        "finite_fraction": 1.0,
                    },
                    "position_mse": {
                        "mean_finite": 0.55,
                        "median": 0.03,
                        "p95": 3.2,
                        "finite_fraction": 1.0,
                    },
                    "velocity_mse": {
                        "mean_finite": 1.63,
                        "median": 0.09,
                        "p95": 9.4,
                        "finite_fraction": 1.0,
                    },
                },
            },
            "curves": {
                "step": [0, 1, 2, 3],
                "state_mse": {
                    "mean_finite": [0.0, 0.22, 0.5, None],
                    "median": [0.0, 0.0001, 0.06, 23.7],
                    "p95": [0.0, 0.0013, 6.3, None],
                    "finite_fraction": [1.0, 1.0, 1.0, 0.85],
                },
                "position_mse": {
                    "mean_finite": [0.0, 0.11, 0.25, None],
                    "median": [0.0, 0.00005, 0.03, 12.0],
                    "p95": [0.0, 0.0007, 3.2, None],
                    "finite_fraction": [1.0, 1.0, 1.0, 0.9],
                },
                "velocity_mse": {
                    "mean_finite": [0.0, 0.33, 0.75, None],
                    "median": [0.0, 0.00015, 0.09, 35.0],
                    "p95": [0.0, 0.002, 9.4, None],
                    "finite_fraction": [1.0, 1.0, 1.0, 0.85],
                },
            },
            "first_nonfinite_step": [None, None, 145, None],
            "state_mse_thresholds": {
                "1": {"first_step": [16, 7, None, 9], "final_fraction_below": 0.0},
                "10": {"first_step": [21, 13, 33, 48], "final_fraction_below": 0.0},
            },
            "position_mse_thresholds": {
                "1": {"first_step": [18, 8, None, 11], "final_fraction_below": 0.25},
                "10": {"first_step": [23, 14, 35, 49], "final_fraction_below": 0.0},
            },
            "state_final_finite_fraction": 0.67,
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


def test_evaluation_report_round_trip_egnn() -> None:
    """EGNN-shape report round-trips through from_dict/to_dict."""
    d = _egnn_report_dict()
    assert EvaluationReport.from_dict(d).to_dict() == d


def test_evaluation_report_round_trip_hgnn() -> None:
    """HGNN-shape report (with learned_hamiltonian) round-trips."""
    d = _hgnn_report_dict()
    assert EvaluationReport.from_dict(d).to_dict() == d


def test_evaluation_report_typed_access() -> None:
    """Loaded report exposes attribute access on nested fields."""
    report = EvaluationReport.from_dict(_egnn_report_dict())
    assert report.metadata.model_name == "egnn"
    assert report.metadata.checkpoint_val_loss == 0.016
    assert report.single_step.state_mse.median == 0.0002
    assert report.single_step.position_mse.median == 0.0001
    assert report.rollout.curves is not None
    assert report.rollout.curves.state_mse.median[2] == 0.06
    assert report.rollout.curves.position_mse.median[2] == 0.03
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
    assert "rollout_step_1_state_mse_mean_finite" in row
    assert "rollout_step_10_position_mse_p95" in row
    assert row["rollout_step_1_state_mse_finite_fraction"] == 1.0


def test_summary_row_includes_dynamic_threshold_keys() -> None:
    """CSV row has state and position threshold columns."""
    report = EvaluationReport.from_dict(_egnn_report_dict())
    row = SummaryRow.from_report(report).to_csv_row()
    assert "rollout_final_fraction_below_state_mse_1" in row
    assert "rollout_final_fraction_below_position_mse_1" in row
    assert row["rollout_final_fraction_below_position_mse_1"] == 0.25


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


_EXPECTED_STATIC_COLUMNS = (
    "model_name",
    "run_id",
    "checkpoint_epoch",
    "checkpoint_val_loss",
    "n_trajectories",
    "n_frames",
    "n_transitions",
    "n_particles",
    "single_step_state_mse_mean",
    "single_step_state_mse_median",
    "single_step_state_mse_p95",
    "single_step_state_mse_p99",
    "single_step_state_mse_max",
    "single_step_position_mse_mean",
    "single_step_position_mse_median",
    "single_step_position_mse_p95",
    "single_step_position_mse_p99",
    "single_step_position_mse_max",
    "single_step_velocity_mse_mean",
    "single_step_velocity_mse_median",
    "single_step_velocity_mse_p95",
    "single_step_velocity_mse_p99",
    "single_step_velocity_mse_max",
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
        "rollout_step_1_state_mse_mean_finite",
        "rollout_step_1_state_mse_median",
        "rollout_step_1_state_mse_p95",
        "rollout_step_1_state_mse_finite_fraction",
        "rollout_step_1_position_mse_mean_finite",
        "rollout_step_1_position_mse_median",
        "rollout_step_1_position_mse_p95",
        "rollout_step_1_position_mse_finite_fraction",
        "rollout_step_1_velocity_mse_mean_finite",
        "rollout_step_1_velocity_mse_median",
        "rollout_step_1_velocity_mse_p95",
        "rollout_step_1_velocity_mse_finite_fraction",
        "rollout_step_10_state_mse_mean_finite",
        "rollout_step_10_state_mse_median",
        "rollout_step_10_state_mse_p95",
        "rollout_step_10_state_mse_finite_fraction",
        "rollout_step_10_position_mse_mean_finite",
        "rollout_step_10_position_mse_median",
        "rollout_step_10_position_mse_p95",
        "rollout_step_10_position_mse_finite_fraction",
        "rollout_step_10_velocity_mse_mean_finite",
        "rollout_step_10_velocity_mse_median",
        "rollout_step_10_velocity_mse_p95",
        "rollout_step_10_velocity_mse_finite_fraction",
        "rollout_state_final_finite_fraction",
        # per-threshold in insertion order ("1", "10"), state then position
        "rollout_final_fraction_below_state_mse_1",
        "rollout_final_fraction_below_state_mse_10",
        "rollout_final_fraction_below_position_mse_1",
        "rollout_final_fraction_below_position_mse_10",
    )
    assert cols == expected


def test_summary_row_hgnn_appends_learned_h_columns() -> None:
    """HGNN CSV header is the EGNN header followed by learned_h_* columns."""
    egnn_report = EvaluationReport.from_dict(_egnn_report_dict())
    hgnn_report = EvaluationReport.from_dict(_hgnn_report_dict())

    egnn_cols = tuple(SummaryRow.from_report(egnn_report).to_csv_row().keys())
    hgnn_cols = tuple(SummaryRow.from_report(hgnn_report).to_csv_row().keys())

    assert hgnn_cols == egnn_cols + _EXPECTED_LEARNED_H_COLUMNS


def _bin_block_fixture() -> dict:
    """Hand-written encounter_bins block reusing EGNN-shape sub-dicts.

    Three bins, with the last one using the +inf top-of-range so the
    sentinel encoding is exercised. Each per-bin block reuses the global
    single_step / rollout / energy sub-dicts wholesale; this tests the
    schema, not the per-bin numerics (those come in Block 2).
    """
    base = _egnn_report_dict()
    return {
        "bins": [
            {"id": 0, "name": "extreme", "lo": 0.0, "hi": 0.01},
            {"id": 1, "name": "close", "lo": 0.01, "hi": 0.05},
            {"id": 2, "name": "smooth", "lo": 0.05, "hi": "inf"},
        ],
        "by_name": {
            "extreme": {
                "count": 12,
                "d_min": {
                    "mean": 0.005,
                    "median": 0.005,
                    "max": 0.0099,
                    "p5": 0.001,
                    "p50": 0.005,
                },
                "single_step": base["single_step"],
                "rollout": base["rollout"],
                "energy": base["energy"],
            },
            "close": {
                "count": 30,
                "d_min": {
                    "mean": 0.025,
                    "median": 0.025,
                    "max": 0.0499,
                    "p5": 0.012,
                    "p50": 0.025,
                },
                "single_step": base["single_step"],
                "rollout": base["rollout"],
                "energy": base["energy"],
            },
            "smooth": {
                "count": 58,
                "d_min": {
                    "mean": 0.5,
                    "median": 0.4,
                    "max": 5.0,
                    "p5": 0.06,
                    "p50": 0.4,
                },
                "single_step": base["single_step"],
                "rollout": base["rollout"],
                "energy": base["energy"],
            },
        },
    }


def _stratified_report_dict() -> dict:
    """EGNN-shape report with an attached encounter_bins block."""
    base = _egnn_report_dict()
    base["encounter_bins"] = _bin_block_fixture()
    return base


def test_report_without_encounter_bins_field_parses() -> None:
    """A non-stratified report (no encounter_bins key) parses with field=None."""
    report = EvaluationReport.from_dict(_egnn_report_dict())
    assert report.encounter_bins is None


def test_to_dict_omits_encounter_bins_when_none() -> None:
    """encounter_bins=None must not emit the key in JSON output."""
    report = EvaluationReport.from_dict(_egnn_report_dict())
    out = report.to_dict()
    assert "encounter_bins" not in out


def test_stratified_report_round_trip() -> None:
    """Report with encounter_bins block round-trips through from_dict/to_dict."""
    d = _stratified_report_dict()
    assert EvaluationReport.from_dict(d).to_dict() == d


def test_encounter_bin_definition_inf_round_trip() -> None:
    """Top-of-range hi=+inf encodes as "inf" sentinel and decodes back to +inf."""
    definition = EncounterBinDefinition(id=5, name="smooth", lo=0.2, hi=float("inf"))
    encoded = definition.to_dict()

    assert encoded["hi"] == "inf"

    decoded = EncounterBinDefinition.from_dict(encoded)
    assert decoded.hi == float("inf")
    assert decoded == definition


def test_encounter_bin_definition_finite_hi_passes_through() -> None:
    """Finite hi values are not touched by the sentinel encoding."""
    definition = EncounterBinDefinition(id=0, name="extreme", lo=0.0, hi=0.01)
    encoded = definition.to_dict()

    assert encoded["hi"] == 0.01
    assert EncounterBinDefinition.from_dict(encoded) == definition


def test_encounter_bin_definition_negative_inf_not_aliased_to_sentinel() -> None:
    """hi=-inf must not encode as the "inf" sentinel.

    The "inf" sentinel is reserved for the top-of-range +inf only. Negative
    infinity is invalid for a bin's upper bound, but the schema layer should
    not silently relabel it as positive: it passes the raw float through and
    lets the JSON write step (allow_nan=False) reject it.
    """
    definition = EncounterBinDefinition(id=0, name="bad", lo=0.0, hi=float("-inf"))
    encoded = definition.to_dict()

    assert encoded["hi"] != "inf"
    assert encoded["hi"] == float("-inf")


def test_encounter_bins_report_preserves_order() -> None:
    """Canonical bin order in `bins` survives the to_dict/from_dict round-trip."""
    block = _bin_block_fixture()
    report = EncounterBinsReport.from_dict(block)

    assert [b.name for b in report.bins] == ["extreme", "close", "smooth"]
    assert [b.id for b in report.bins] == [0, 1, 2]

    out = report.to_dict()
    assert [b["name"] for b in out["bins"]] == ["extreme", "close", "smooth"]


def test_encounter_bins_report_typed_access() -> None:
    """EncounterBinsReport exposes attribute access on per-bin fields."""
    report = EvaluationReport.from_dict(_stratified_report_dict())
    assert report.encounter_bins is not None
    assert report.encounter_bins.by_name["extreme"].count == 12
    assert report.encounter_bins.by_name["smooth"].count == 58
    assert report.encounter_bins.by_name["smooth"].d_min.median == 0.4
    assert report.encounter_bins.by_name["close"].baseline_ratios is None


def test_summary_row_columns_unchanged_for_stratified_report() -> None:
    """Block 1 must NOT widen summary.csv: stratified report yields same columns."""
    plain = EvaluationReport.from_dict(_egnn_report_dict())
    stratified = EvaluationReport.from_dict(_stratified_report_dict())

    plain_cols = tuple(SummaryRow.from_report(plain).to_csv_row().keys())
    stratified_cols = tuple(SummaryRow.from_report(stratified).to_csv_row().keys())

    assert stratified_cols == plain_cols


def _baseline_ratios_fixture() -> dict:
    """Hand-written baseline_ratios block matching the JSON shape."""
    return {
        "score": -0.12,
        "state_mse": {
            "10": 0.82,
            "20": 1.04,
            "50": 2.10,
            "100": None,
            "199": None,
        },
        "dominance_horizon": 12,
        "fraction_beating_baseline": 0.18,
        "final_ratio": 4.2,
    }


def test_per_bin_baseline_ratios_round_trip_preserves_int_keys() -> None:
    """JSON anchor-step keys are strings on disk and ints in memory."""
    d = _baseline_ratios_fixture()
    parsed = PerBinBaselineRatios.from_dict(d)

    # in-memory keys must be ints, not strings
    assert set(parsed.state_mse_ratios.keys()) == {10, 20, 50, 100, 199}
    assert all(isinstance(k, int) for k in parsed.state_mse_ratios)
    assert parsed.score == -0.12
    assert parsed.dominance_horizon == 12

    # round-trip back to JSON: keys stringified again
    out = parsed.to_dict()
    assert out == d
    assert all(isinstance(k, str) for k in out["state_mse"])


def test_per_bin_baseline_ratios_in_encounter_bin_report_round_trip() -> None:
    """Populated baseline_ratios survives EncounterBinReport.from_dict/to_dict."""
    base = _egnn_report_dict()
    bin_dict = {
        "count": 12,
        "d_min": {"mean": 0.005, "median": 0.005, "max": 0.0099, "p5": 0.001, "p50": 0.005},
        "single_step": base["single_step"],
        "rollout": base["rollout"],
        "energy": base["energy"],
        "baseline_ratios": _baseline_ratios_fixture(),
    }
    report = EncounterBinReport.from_dict(bin_dict)

    assert report.baseline_ratios is not None
    assert report.baseline_ratios.state_mse_ratios[10] == 0.82
    assert report.baseline_ratios.state_mse_ratios[100] is None

    assert report.to_dict() == bin_dict


def test_encounter_bin_report_omits_baseline_ratios_when_none() -> None:
    """Empty-bin path: baseline_ratios=None must not emit the key."""
    base = _egnn_report_dict()
    bin_dict = {
        "count": 0,
        "d_min": {"mean": None, "median": None, "max": None, "p5": None, "p50": None},
        "single_step": base["single_step"],
        "rollout": base["rollout"],
        "energy": base["energy"],
    }
    report = EncounterBinReport.from_dict(bin_dict)

    assert report.baseline_ratios is None
    assert "baseline_ratios" not in report.to_dict()
