"""Tests for evaluation/report_tables.py."""

import copy
import csv
from pathlib import Path

import pytest

from evaluation._types import EvaluationReport
from evaluation.report_tables import (
    KEY_TIMESTEP_SUMMARY_COLUMNS,
    PER_BIN_SUMMARY_COLUMNS,
    write_key_timestep_summary_csv,
    write_per_bin_summary_csv,
    write_report_markdown,
)
from evaluation.test_types import _hgnn_report_dict, _stratified_report_dict

HORIZON_FIXTURE_STEPS = [0, 1, 3, 5, 10, 25, 50, 100, 150, 199]
"""Step axis for fixtures that exercise the horizon plotters; covers HORIZON_ANCHORS + 0."""


def horizonify(report_dict: dict) -> dict:
    """Rewrite rollout/energy curves so the step axis covers every HORIZON_ANCHORS value."""
    out = copy.deepcopy(report_dict)
    _rewrite_rollout_curves(out["rollout"]["curves"])
    if "encounter_bins" in out:
        for bin_block in out["encounter_bins"]["by_name"].values():
            _rewrite_rollout_curves(bin_block["rollout"]["curves"])
            _rewrite_energy_curves(bin_block["energy"]["physical"]["curves"])
    return out


def _rewrite_rollout_curves(curves: dict) -> None:
    """Replace step and per-metric arrays in a rollout curves block, in place."""
    n = len(HORIZON_FIXTURE_STEPS)
    curves["step"] = list(HORIZON_FIXTURE_STEPS)
    for metric in ("state_mse", "position_mse", "velocity_mse"):
        curves[metric]["mean_finite"] = [0.01 * i for i in range(n)]
        curves[metric]["median"] = [0.005 * i for i in range(n)]
        curves[metric]["p95"] = [0.02 * i for i in range(n)]
        curves[metric]["finite_fraction"] = [1.0] * n


def _rewrite_energy_curves(curves: dict) -> None:
    """Replace step and per-metric arrays in an energy curves block, in place."""
    n = len(HORIZON_FIXTURE_STEPS)
    curves["step"] = list(HORIZON_FIXTURE_STEPS)
    curves["mean_finite"] = [0.001 * i for i in range(n)]
    curves["median"] = [0.0005 * i for i in range(n)]
    curves["p95"] = [0.002 * i for i in range(n)]
    curves["finite_fraction"] = [1.0] * n


def _shift_final_mse(report: dict, factor: float) -> dict:
    """Multiply the final-step median state and position MSE of every bin by `factor`.

    Gives the three fixtures distinct signatures so tests can confirm CSV rows pull from
    the right report.
    """
    out = copy.deepcopy(report)
    for bin_entry in out["encounter_bins"]["by_name"].values():
        for metric in ("state_mse", "position_mse"):
            median_curve = bin_entry["rollout"]["curves"][metric]["median"]
            if median_curve and median_curve[-1] is not None:
                median_curve[-1] = median_curve[-1] * factor
    return out


def _build_reports() -> tuple[EvaluationReport, EvaluationReport, EvaluationReport]:
    """Build a horizon-aware matched EGNN/HGNN/baseline trio with distinct final MSEs."""
    egnn_dict = _shift_final_mse(horizonify(_stratified_report_dict()), factor=1.0)

    hgnn_dict = horizonify(_hgnn_report_dict())
    hgnn_dict["encounter_bins"] = copy.deepcopy(egnn_dict["encounter_bins"])
    hgnn_dict["metadata"]["model_name"] = "hgnn"
    hgnn_dict["metadata"]["run_id"] = "hgnn_run_xyz"
    hgnn_dict = _shift_final_mse(hgnn_dict, factor=0.5)

    baseline_dict = copy.deepcopy(egnn_dict)
    baseline_dict["metadata"]["model_name"] = "baseline_constant_velocity"
    baseline_dict["metadata"]["run_id"] = "baseline_constant_velocity"
    baseline_dict["metadata"]["checkpoint_path"] = None
    baseline_dict = _shift_final_mse(baseline_dict, factor=2.0)

    return (
        EvaluationReport.from_dict(egnn_dict),
        EvaluationReport.from_dict(hgnn_dict),
        EvaluationReport.from_dict(baseline_dict),
    )


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV back into (column_order, list of row dicts)."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def test_per_bin_summary_columns_are_pinned(tmp_path: Path) -> None:
    """Column order is part of the contract; downstream notebooks rely on it."""
    egnn, hgnn, baseline = _build_reports()
    path = tmp_path / "per_bin_summary.csv"

    write_per_bin_summary_csv(egnn, hgnn, baseline, path)

    columns, _ = _read_csv(path)
    assert tuple(columns) == PER_BIN_SUMMARY_COLUMNS


def test_per_bin_summary_drops_baseline_score_columns(tmp_path: Path) -> None:
    """Baseline-normalised score columns must be absent from the public CSV."""
    egnn, hgnn, baseline = _build_reports()
    path = tmp_path / "per_bin_summary.csv"

    write_per_bin_summary_csv(egnn, hgnn, baseline, path)

    columns, _ = _read_csv(path)
    assert "egnn_baseline_score" not in columns
    assert "hgnn_baseline_score" not in columns
    assert "egnn_fraction_beating_baseline" not in columns
    assert "egnn_dominance_horizon" not in columns


def test_per_bin_summary_includes_baseline_physical_columns(tmp_path: Path) -> None:
    """Baseline final MSE (position and state) and energy drift columns must be present."""
    egnn, hgnn, baseline = _build_reports()
    path = tmp_path / "per_bin_summary.csv"

    write_per_bin_summary_csv(egnn, hgnn, baseline, path)

    columns, _ = _read_csv(path)
    assert "baseline_final_median_position_mse" in columns
    assert "baseline_final_median_state_mse" in columns
    assert "baseline_energy_drift_median" in columns


def test_per_bin_summary_position_mse_columns_lead_state_mse(tmp_path: Path) -> None:
    """Position MSE is the audience-facing headline; its columns sit before state MSE."""
    egnn, hgnn, baseline = _build_reports()
    path = tmp_path / "per_bin_summary.csv"

    write_per_bin_summary_csv(egnn, hgnn, baseline, path)

    columns, _ = _read_csv(path)
    pos_idx = columns.index("egnn_final_median_position_mse")
    state_idx = columns.index("egnn_final_median_state_mse")
    assert pos_idx < state_idx


def test_per_bin_summary_row_order_matches_canonical_bin_order(tmp_path: Path) -> None:
    """Rows follow `egnn.encounter_bins.bins`, not python dict iteration order."""
    egnn, hgnn, baseline = _build_reports()
    path = tmp_path / "per_bin_summary.csv"

    write_per_bin_summary_csv(egnn, hgnn, baseline, path)

    _, rows = _read_csv(path)
    assert [r["bin"] for r in rows] == ["extreme", "close", "smooth"]
    assert [r["bin_id"] for r in rows] == ["0", "1", "2"]


def test_per_bin_summary_values_pull_from_correct_model(tmp_path: Path) -> None:
    """EGNN, HGNN, baseline columns must come from their respective reports."""
    egnn, hgnn, baseline = _build_reports()
    path = tmp_path / "per_bin_summary.csv"

    write_per_bin_summary_csv(egnn, hgnn, baseline, path)

    _, rows = _read_csv(path)
    close_row = next(r for r in rows if r["bin"] == "close")
    egnn_close = egnn.encounter_bins.by_name["close"]
    hgnn_close = hgnn.encounter_bins.by_name["close"]
    baseline_close = baseline.encounter_bins.by_name["close"]
    assert float(close_row["egnn_final_median_position_mse"]) == pytest.approx(
        egnn_close.rollout.curves.position_mse.median[-1]
    )
    assert float(close_row["hgnn_final_median_position_mse"]) == pytest.approx(
        hgnn_close.rollout.curves.position_mse.median[-1]
    )
    assert float(close_row["baseline_final_median_position_mse"]) == pytest.approx(
        baseline_close.rollout.curves.position_mse.median[-1]
    )
    assert float(close_row["egnn_final_median_state_mse"]) == pytest.approx(
        egnn_close.rollout.curves.state_mse.median[-1]
    )
    assert float(close_row["hgnn_final_median_state_mse"]) == pytest.approx(
        hgnn_close.rollout.curves.state_mse.median[-1]
    )
    assert float(close_row["baseline_final_median_state_mse"]) == pytest.approx(
        baseline_close.rollout.curves.state_mse.median[-1]
    )


def test_key_timestep_summary_columns_are_pinned(tmp_path: Path) -> None:
    """Column order is part of the contract."""
    egnn, hgnn, baseline = _build_reports()
    path = tmp_path / "key_timestep_summary.csv"

    write_key_timestep_summary_csv(egnn, hgnn, baseline, path)

    columns, _ = _read_csv(path)
    assert tuple(columns) == KEY_TIMESTEP_SUMMARY_COLUMNS


def test_key_timestep_summary_includes_baseline_columns(tmp_path: Path) -> None:
    """The per-step CSV must include baseline position and state MSE columns."""
    egnn, hgnn, baseline = _build_reports()
    path = tmp_path / "key_timestep_summary.csv"

    write_key_timestep_summary_csv(egnn, hgnn, baseline, path)

    columns, _ = _read_csv(path)
    assert "baseline_position_mse_median" in columns
    assert "baseline_position_mse_p95" in columns
    assert "baseline_state_mse_median" in columns
    assert "baseline_state_mse_p95" in columns
    assert "baseline_finite_fraction" in columns


def test_key_timestep_summary_position_columns_lead_state_columns(tmp_path: Path) -> None:
    """Position MSE columns appear before state MSE columns in the CSV header."""
    egnn, hgnn, baseline = _build_reports()
    path = tmp_path / "key_timestep_summary.csv"

    write_key_timestep_summary_csv(egnn, hgnn, baseline, path)

    columns, _ = _read_csv(path)
    pos_idx = columns.index("egnn_position_mse_median")
    state_idx = columns.index("egnn_state_mse_median")
    assert pos_idx < state_idx


def test_key_timestep_summary_iterates_bins_outer_steps_inner(tmp_path: Path) -> None:
    """Outer loop = bin order; inner loop = numeric step order."""
    egnn, hgnn, baseline = _build_reports()
    path = tmp_path / "key_timestep_summary.csv"

    write_key_timestep_summary_csv(egnn, hgnn, baseline, path)

    _, rows = _read_csv(path)
    # fixture has steps "1" and "10" -> we expect 3 bins x 2 steps = 6 rows
    assert len(rows) == 6
    assert [r["bin"] for r in rows] == ["extreme", "extreme", "close", "close", "smooth", "smooth"]
    assert [int(r["step"]) for r in rows] == [1, 10, 1, 10, 1, 10]


def test_key_timestep_summary_intersects_step_keys(tmp_path: Path) -> None:
    """If any report drops an anchor step, the union is not emitted."""
    egnn_dict = _stratified_report_dict()
    hgnn_dict = _hgnn_report_dict()
    hgnn_dict["encounter_bins"] = copy.deepcopy(egnn_dict["encounter_bins"])
    hgnn_dict["metadata"]["model_name"] = "hgnn"
    baseline_dict = copy.deepcopy(egnn_dict)
    baseline_dict["metadata"]["model_name"] = "baseline_constant_velocity"
    # drop step "10" from the baseline report so intersection is just {1}
    baseline_dict["rollout"]["steps"].pop("10")

    egnn = EvaluationReport.from_dict(egnn_dict)
    hgnn = EvaluationReport.from_dict(hgnn_dict)
    baseline = EvaluationReport.from_dict(baseline_dict)
    path = tmp_path / "key_timestep_summary.csv"

    write_key_timestep_summary_csv(egnn, hgnn, baseline, path)

    _, rows = _read_csv(path)
    # 3 bins x 1 shared step = 3 rows
    assert len(rows) == 3
    assert {int(r["step"]) for r in rows} == {1}


def test_report_markdown_includes_provenance_and_artifacts(tmp_path: Path) -> None:
    """report.md mentions all three run_ids, all bin names, and the artifact filenames."""
    egnn, hgnn, baseline = _build_reports()
    output_dir = tmp_path
    tables = ["per_bin_summary.csv", "key_timestep_summary.csv"]
    figures = ["01_rollout_position_mse_by_bin.png", "02_energy_drift_by_bin.png"]

    write_report_markdown(
        egnn,
        hgnn,
        baseline,
        output_dir,
        figures=figures,
        tables=tables,
    )

    md = (output_dir / "report.md").read_text()
    assert "Constant Velocity" in md
    assert egnn.metadata.run_id in md
    assert hgnn.metadata.run_id in md
    assert baseline.metadata.run_id in md
    assert "extreme" in md
    assert "close" in md
    assert "smooth" in md
    assert "tables/per_bin_summary.csv" in md
    assert "tables/key_timestep_summary.csv" in md
    assert "figures/01_rollout_position_mse_by_bin.png" in md
    assert "figures/02_energy_drift_by_bin.png" in md


def test_report_markdown_headline_uses_physical_metrics(tmp_path: Path) -> None:
    """The headline table lives on raw position MSE and energy drift across all three models."""
    egnn, hgnn, baseline = _build_reports()

    write_report_markdown(
        egnn,
        hgnn,
        baseline,
        tmp_path,
        figures=[],
        tables=[],
    )

    md = (tmp_path / "report.md").read_text()
    assert "Headline: Metrics by Trajectory Class" in md
    assert "EGNN final position MSE" in md
    assert "HGNN final position MSE" in md
    assert "Baseline final position MSE" in md
    assert "EGNN energy drift" in md
    assert "Baseline energy drift" in md
    # state MSE no longer surfaces in the public headline; it remains in CSVs only.
    assert "EGNN final MSE" not in md
    assert "median state MSE" not in md


def test_report_markdown_drops_baseline_score_section(tmp_path: Path) -> None:
    """The supplementary baseline-normalised score section must be gone."""
    egnn, hgnn, baseline = _build_reports()

    write_report_markdown(
        egnn,
        hgnn,
        baseline,
        tmp_path,
        figures=[],
        tables=[],
    )

    md = (tmp_path / "report.md").read_text()
    assert "Baseline-normalised Score" not in md
    assert "bucket_macro_rollout_score" not in md
    assert "score heatmap" not in md.lower()


def test_per_bin_summary_raises_when_egnn_lacks_encounter_bins(tmp_path: Path) -> None:
    """Defensive narrowing: an unstratified report short-circuits with a clear error."""
    from evaluation.test_types import _egnn_report_dict

    egnn = EvaluationReport.from_dict(_egnn_report_dict())  # no encounter_bins
    hgnn_dict = _hgnn_report_dict()
    hgnn_dict["encounter_bins"] = copy.deepcopy(_stratified_report_dict()["encounter_bins"])
    hgnn = EvaluationReport.from_dict(hgnn_dict)
    baseline_dict = _stratified_report_dict()
    baseline_dict["metadata"]["model_name"] = "baseline_constant_velocity"
    baseline = EvaluationReport.from_dict(baseline_dict)

    with pytest.raises(ValueError, match="no encounter_bins block"):
        write_per_bin_summary_csv(egnn, hgnn, baseline, tmp_path / "per_bin_summary.csv")
