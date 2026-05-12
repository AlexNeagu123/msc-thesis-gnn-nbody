"""CSV tables and markdown skeleton for the comparison report.

Pure functions over typed `EvaluationReport` objects. No matplotlib, no
model loading, no rollout computation. All inputs come from already-loaded
metrics.json files; outputs are deterministic files under the report's
`tables/` directory or the report root (for `report.md`).

The public surface centres on three reports: EGNN, HGNN, and the
constant-velocity baseline. Baseline-normalised rollout scores remain in
the trainer's metric stack but are intentionally absent from the report
artifacts produced here.

References:
    - Source schema: evaluation/_types.py (EvaluationReport)
    - Orchestrator:  evaluation/report.py (Reporter.run)
"""

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from evaluation._types import (
    EncounterBinReport,
    EncounterBinsReport,
    EvaluationReport,
    RolloutStepMetrics,
)

PER_BIN_SUMMARY_COLUMNS = (
    "bin",
    "bin_id",
    "count",
    "d_min_median",
    "egnn_final_median_state_mse",
    "hgnn_final_median_state_mse",
    "baseline_final_median_state_mse",
    "egnn_final_finite_fraction",
    "hgnn_final_finite_fraction",
    "baseline_final_finite_fraction",
    "egnn_energy_drift_median",
    "hgnn_energy_drift_median",
    "baseline_energy_drift_median",
)

KEY_TIMESTEP_SUMMARY_COLUMNS = (
    "bin",
    "step",
    "egnn_state_mse_median",
    "hgnn_state_mse_median",
    "baseline_state_mse_median",
    "egnn_state_mse_p95",
    "hgnn_state_mse_p95",
    "baseline_state_mse_p95",
    "egnn_finite_fraction",
    "hgnn_finite_fraction",
    "baseline_finite_fraction",
)


def write_per_bin_summary_csv(
    egnn: EvaluationReport,
    hgnn: EvaluationReport,
    baseline: EvaluationReport,
    path: Path,
) -> None:
    """Write one row per encounter bin summarising all three reports side-by-side.

    Bin order follows `egnn.encounter_bins.bins` (the canonical schedule
    written by the evaluator). The caller is expected to have validated
    that EGNN, HGNN, and the baseline share this layout.
    """
    bins = _require_bins(egnn)
    hgnn_bins = _require_bins(hgnn)
    baseline_bins = _require_bins(baseline)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_BIN_SUMMARY_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for bin_def in bins.bins:
            writer.writerow(
                _per_bin_row(
                    bin_def.id,
                    bin_def.name,
                    bins.by_name[bin_def.name],
                    hgnn_bins.by_name[bin_def.name],
                    baseline_bins.by_name[bin_def.name],
                )
            )


def write_key_timestep_summary_csv(
    egnn: EvaluationReport,
    hgnn: EvaluationReport,
    baseline: EvaluationReport,
    path: Path,
) -> None:
    """Write one row per (bin, anchor step) with state MSE summaries for all three models.

    The step universe is the intersection of `rollout.steps` keys at the
    top level across all three reports; in practice all come from the same
    evaluator codepath, but the intersection guards against schema drift.
    """
    egnn_bins = _require_bins(egnn)
    hgnn_bins = _require_bins(hgnn)
    baseline_bins = _require_bins(baseline)
    steps = _shared_anchor_steps(egnn, hgnn, baseline)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=KEY_TIMESTEP_SUMMARY_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for bin_def in egnn_bins.bins:
            egnn_bin = egnn_bins.by_name[bin_def.name]
            hgnn_bin = hgnn_bins.by_name[bin_def.name]
            baseline_bin = baseline_bins.by_name[bin_def.name]
            for step in steps:
                writer.writerow(_key_step_row(bin_def.name, step, egnn_bin, hgnn_bin, baseline_bin))


def write_report_markdown(
    egnn: EvaluationReport,
    hgnn: EvaluationReport,
    baseline: EvaluationReport,
    output_dir: Path,
    *,
    figures: Iterable[str],
    tables: Iterable[str],
) -> None:
    """Render `output_dir/report.md` from the three reports plus artifact pointers.

    `figures` and `tables` are relative basenames; the markdown prefixes
    the appropriate sub-directory. The markdown index lists a single flat
    figures section for the public presentation plots.
    """
    egnn_bins = _require_bins(egnn)
    hgnn_bins = _require_bins(hgnn)
    baseline_bins = _require_bins(baseline)
    sections = [
        "# EGNN vs HGNN vs Constant Velocity - Evaluation Comparison Report",
        "",
        "## Provenance",
        "",
        _markdown_table(
            ["Field", "EGNN", "HGNN", "Baseline"],
            [
                [
                    "run_id",
                    _str(egnn.metadata.run_id),
                    _str(hgnn.metadata.run_id),
                    _str(baseline.metadata.run_id),
                ],
                [
                    "checkpoint_epoch",
                    _str(egnn.metadata.checkpoint_epoch),
                    _str(hgnn.metadata.checkpoint_epoch),
                    _str(baseline.metadata.checkpoint_epoch),
                ],
                [
                    "checkpoint_val_loss",
                    _fmt_float(egnn.metadata.checkpoint_val_loss),
                    _fmt_float(hgnn.metadata.checkpoint_val_loss),
                    _fmt_float(baseline.metadata.checkpoint_val_loss),
                ],
                [
                    "test_path",
                    egnn.metadata.test_path,
                    hgnn.metadata.test_path,
                    baseline.metadata.test_path,
                ],
                [
                    "n_trajectories",
                    _str(egnn.metadata.n_trajectories),
                    _str(hgnn.metadata.n_trajectories),
                    _str(baseline.metadata.n_trajectories),
                ],
                [
                    "n_frames",
                    _str(egnn.metadata.n_frames),
                    _str(hgnn.metadata.n_frames),
                    _str(baseline.metadata.n_frames),
                ],
                [
                    "device",
                    egnn.metadata.device,
                    hgnn.metadata.device,
                    baseline.metadata.device,
                ],
            ],
        ),
        "",
        "## Encounter Bin Layout",
        "",
        _markdown_table(
            ["Bin", "lo", "hi", "Count"],
            [
                [b.name, _fmt_float(b.lo), _fmt_inf(b.hi), _str(egnn_bins.by_name[b.name].count)]
                for b in egnn_bins.bins
            ],
        ),
        "",
        "## Headline: Per-bin Physical Metrics",
        "",
        "Final-step median state MSE and final relative energy drift, per encounter bin. Lower is better for both; raw physical units, no normalisation. See the rollout MSE and energy drift figures for the full time dynamics.",
        "",
        _markdown_table(
            [
                "Bin",
                "n",
                "EGNN final MSE",
                "HGNN final MSE",
                "Baseline final MSE",
                "EGNN energy drift",
                "HGNN energy drift",
                "Baseline energy drift",
            ],
            [
                [
                    b.name,
                    _str(egnn_bins.by_name[b.name].count),
                    _fmt_float(_final_median_state_mse(egnn_bins.by_name[b.name])),
                    _fmt_float(_final_median_state_mse(hgnn_bins.by_name[b.name])),
                    _fmt_float(_final_median_state_mse(baseline_bins.by_name[b.name])),
                    _fmt_float(_final_energy_drift(egnn_bins.by_name[b.name])),
                    _fmt_float(_final_energy_drift(hgnn_bins.by_name[b.name])),
                    _fmt_float(_final_energy_drift(baseline_bins.by_name[b.name])),
                ]
                for b in egnn_bins.bins
            ],
        ),
        "",
        "## Artifacts",
        "",
        "### Figures",
        "",
        _markdown_bullets(f"figures/{name}" for name in figures) or "_no figures listed_",
        "",
        "### Tables",
        "",
        _markdown_bullets(f"tables/{name}" for name in tables) or "_no tables listed_",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(sections))


def _require_bins(report: EvaluationReport) -> EncounterBinsReport:
    """Narrow EvaluationReport.encounter_bins out of None; validated upstream."""
    if report.encounter_bins is None:
        msg = "report has no encounter_bins block; orchestrator should have rejected this earlier"
        raise ValueError(msg)
    return report.encounter_bins


def _final_median_state_mse(bin_report: EncounterBinReport) -> float | None:
    """Last entry of the per-bin median state MSE curve."""
    return bin_report.rollout.curves.state_mse.median[-1]


def _final_energy_drift(bin_report: EncounterBinReport) -> float | None:
    """Median relative drift of physical energy at the final rollout step."""
    return bin_report.energy.physical.final_relative_drift.median


def _per_bin_row(
    bin_id: int,
    bin_name: str,
    egnn_bin: EncounterBinReport,
    hgnn_bin: EncounterBinReport,
    baseline_bin: EncounterBinReport,
) -> dict[str, Any]:
    """Build a single per-bin row keyed by `PER_BIN_SUMMARY_COLUMNS`."""
    return {
        "bin": bin_name,
        "bin_id": bin_id,
        "count": egnn_bin.count,
        "d_min_median": egnn_bin.d_min.median,
        "egnn_final_median_state_mse": _final_median_state_mse(egnn_bin),
        "hgnn_final_median_state_mse": _final_median_state_mse(hgnn_bin),
        "baseline_final_median_state_mse": _final_median_state_mse(baseline_bin),
        "egnn_final_finite_fraction": egnn_bin.rollout.state_final_finite_fraction,
        "hgnn_final_finite_fraction": hgnn_bin.rollout.state_final_finite_fraction,
        "baseline_final_finite_fraction": baseline_bin.rollout.state_final_finite_fraction,
        "egnn_energy_drift_median": egnn_bin.energy.physical.final_relative_drift.median,
        "hgnn_energy_drift_median": hgnn_bin.energy.physical.final_relative_drift.median,
        "baseline_energy_drift_median": baseline_bin.energy.physical.final_relative_drift.median,
    }


def _shared_anchor_steps(
    egnn: EvaluationReport,
    hgnn: EvaluationReport,
    baseline: EvaluationReport,
) -> list[int]:
    """Intersection of top-level rollout.steps keys across all three reports, sorted asc."""
    common = (
        set(egnn.rollout.steps.keys())
        & set(hgnn.rollout.steps.keys())
        & set(baseline.rollout.steps.keys())
    )
    return sorted(int(k) for k in common)


def _step_metrics(bin_report: EncounterBinReport, step: int) -> RolloutStepMetrics | None:
    """Pull per-bin step metrics, returning None when the step is missing."""
    return bin_report.rollout.steps.get(str(step))


def _key_step_row(
    bin_name: str,
    step: int,
    egnn_bin: EncounterBinReport,
    hgnn_bin: EncounterBinReport,
    baseline_bin: EncounterBinReport,
) -> dict[str, Any]:
    """Build a single (bin, step) row keyed by `KEY_TIMESTEP_SUMMARY_COLUMNS`."""
    egnn_step = _step_metrics(egnn_bin, step)
    hgnn_step = _step_metrics(hgnn_bin, step)
    baseline_step = _step_metrics(baseline_bin, step)
    return {
        "bin": bin_name,
        "step": step,
        "egnn_state_mse_median": _safe_median(egnn_step),
        "hgnn_state_mse_median": _safe_median(hgnn_step),
        "baseline_state_mse_median": _safe_median(baseline_step),
        "egnn_state_mse_p95": _safe_p95(egnn_step),
        "hgnn_state_mse_p95": _safe_p95(hgnn_step),
        "baseline_state_mse_p95": _safe_p95(baseline_step),
        "egnn_finite_fraction": _safe_finite_fraction(egnn_step),
        "hgnn_finite_fraction": _safe_finite_fraction(hgnn_step),
        "baseline_finite_fraction": _safe_finite_fraction(baseline_step),
    }


def _safe_median(step: RolloutStepMetrics | None) -> float | None:
    """Return median state MSE for an anchor step, or None when the step is absent."""
    return step.state_mse.median if step is not None else None


def _safe_p95(step: RolloutStepMetrics | None) -> float | None:
    """Return p95 state MSE for an anchor step, or None when the step is absent."""
    return step.state_mse.p95 if step is not None else None


def _safe_finite_fraction(step: RolloutStepMetrics | None) -> float | None:
    """Return finite-fraction for an anchor step, or None when the step is absent."""
    return step.state_mse.finite_fraction if step is not None else None


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a github-flavored markdown table."""
    sep = "|" + "|".join("---" for _ in headers) + "|"
    header_line = "| " + " | ".join(headers) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, sep, *body_lines])


def _markdown_bullets(items: Iterable[str]) -> str:
    """Render an unordered list, or empty string when there are no items."""
    rendered = [f"- {item}" for item in items]
    return "\n".join(rendered)


def _str(value: str | int | float | None) -> str:
    """Stringify None as 'n/a' so markdown cells never contain the empty string."""
    return "n/a" if value is None else str(value)


def _fmt_float(value: float | None) -> str:
    """Format a possibly-None float for markdown cells; ~4 sig figs."""
    return f"{value:.4g}" if value is not None else "n/a"


def _fmt_inf(value: float) -> str:
    """Render +inf as the explicit string so markdown reads naturally."""
    return "+inf" if value == float("inf") else _fmt_float(value)
