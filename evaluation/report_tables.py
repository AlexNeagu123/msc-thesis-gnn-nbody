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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation._types import (
    EncounterBinReport,
    EncounterBinsReport,
    EvaluationReport,
    RolloutStepMetrics,
)
from evaluation.evaluate_chunked import (
    USABLE_K_POSITION_MSE_THRESHOLD,
    ChunkedEndpointsRow,
)

CHUNKED_FIGURE_NAME = "chunked_endpoint_position_mse_by_bin.png"
CHUNKED_SUMMARY_CSV_NAME = "chunked_summary.csv"
CHUNKED_ENDPOINTS_CSV_NAME = "chunked_endpoints.csv"
CHUNKED_MARKDOWN_NAME = "chunked_report.md"
_CHUNKED_MODEL_LABELS: tuple[tuple[str, str], ...] = (
    ("egnn", "EGNN"),
    ("hgnn", "HGNN"),
    ("baseline_constant_velocity", "constant velocity"),
)


@dataclass(frozen=True)
class ChunkedReportSection:
    """Pre-loaded data passed to `write_report_markdown` when chunked/ exists.

    Built by `evaluation.report.Reporter` after detecting the chunked
    sub-directory. Carries the path prefix to use in markdown links plus
    the typed endpoint rows so the section can render the usable-K table
    without re-parsing the CSV inside the writer.
    """

    rel_dir: str
    endpoint_rows: list[ChunkedEndpointsRow]
    usable_k: dict[tuple[str, str], int | None]


PER_BIN_SUMMARY_COLUMNS = (
    "bin",
    "bin_id",
    "count",
    "d_min_median",
    "egnn_final_median_position_mse",
    "hgnn_final_median_position_mse",
    "baseline_final_median_position_mse",
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
    "egnn_position_mse_median",
    "hgnn_position_mse_median",
    "baseline_position_mse_median",
    "egnn_position_mse_p95",
    "hgnn_position_mse_p95",
    "baseline_position_mse_p95",
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
    """Write one row per (bin, anchor step) with position and state MSE summaries for all three models.

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
    chunked: ChunkedReportSection | None = None,
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
        "Final-step median position MSE and final relative energy drift, per encounter bin. Lower is better for both; raw physical units, no normalisation. Position MSE is the audience-facing forecast quality metric (state and velocity MSE remain available in metrics.json and the technical CSVs). See the rollout position MSE and energy drift figures for the full time dynamics.",
        "",
        _markdown_table(
            [
                "Bin",
                "n",
                "EGNN final position MSE",
                "HGNN final position MSE",
                "Baseline final position MSE",
                "EGNN energy drift",
                "HGNN energy drift",
                "Baseline energy drift",
            ],
            [
                [
                    b.name,
                    _str(egnn_bins.by_name[b.name].count),
                    _fmt_float(_final_median_position_mse(egnn_bins.by_name[b.name])),
                    _fmt_float(_final_median_position_mse(hgnn_bins.by_name[b.name])),
                    _fmt_float(_final_median_position_mse(baseline_bins.by_name[b.name])),
                    _fmt_float(_final_energy_drift(egnn_bins.by_name[b.name])),
                    _fmt_float(_final_energy_drift(hgnn_bins.by_name[b.name])),
                    _fmt_float(_final_energy_drift(baseline_bins.by_name[b.name])),
                ]
                for b in egnn_bins.bins
            ],
        ),
        "",
        *(_chunked_section_lines(chunked) if chunked is not None else []),
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


def _chunked_section_lines(section: ChunkedReportSection) -> list[str]:
    """Render the optional chunked-forecasting section that lives between Headline and Artifacts.

    Periodic correction, not autonomous simulation: this is restated as a
    bold blockquote so a reader skimming the report cannot miss it. The
    figure embed and the artifact links use `section.rel_dir` as the
    in-markdown prefix so the same renderer works whether the chunked
    directory sits at `chunked/` or somewhere else under the report root.
    """
    rmse_threshold = USABLE_K_POSITION_MSE_THRESHOLD**0.5
    rel = section.rel_dir.rstrip("/")
    return [
        "## Short-horizon corrected forecasting",
        "",
        (
            "> **This is not autonomous simulation.** "
            "EGNN and HGNN are re-anchored to the ground truth every K rollout steps; "
            "the metric below is the median endpoint position MSE across trajectories "
            "in each encounter bin."
        ),
        "",
        (
            f"A chunk size K is *usable* when its median endpoint position RMSE stays "
            f"at most {rmse_threshold:.2f} coordinate units "
            f"(equivalently, median endpoint position MSE <= {USABLE_K_POSITION_MSE_THRESHOLD})."
        ),
        "",
        "### Largest usable K per bin and model",
        "",
        _chunked_usable_k_table(section),
        "",
        "### Endpoint position MSE figure",
        "",
        f"![chunked endpoint position MSE]({rel}/{CHUNKED_FIGURE_NAME})",
        "",
        "### Chunked artifacts",
        "",
        _markdown_bullets(
            [
                f"[{rel}/{CHUNKED_SUMMARY_CSV_NAME}]({rel}/{CHUNKED_SUMMARY_CSV_NAME})",
                f"[{rel}/{CHUNKED_ENDPOINTS_CSV_NAME}]({rel}/{CHUNKED_ENDPOINTS_CSV_NAME})",
                f"[{rel}/{CHUNKED_MARKDOWN_NAME}]({rel}/{CHUNKED_MARKDOWN_NAME})",
            ]
        ),
        "",
    ]


def _chunked_usable_k_table(section: ChunkedReportSection) -> str:
    """Render the usable-K matrix as `bin x model` markdown."""
    bin_names = list(dict.fromkeys(r.bin for r in section.endpoint_rows))
    headers = ["bin", *[label for _name, label in _CHUNKED_MODEL_LABELS]]
    rows = [
        [
            bin_name,
            *[
                _fmt_usable_k(section.usable_k.get((bin_name, name)))
                for name, _label in _CHUNKED_MODEL_LABELS
            ],
        ]
        for bin_name in bin_names
    ]
    return _markdown_table(headers, rows)


def _fmt_usable_k(value: int | None) -> str:
    """Render `largest_usable_k` cells; missing or non-qualifying becomes 'none'."""
    return "none" if value is None else f"K={value}"


def _require_bins(report: EvaluationReport) -> EncounterBinsReport:
    """Narrow EvaluationReport.encounter_bins out of None; validated upstream."""
    if report.encounter_bins is None:
        msg = "report has no encounter_bins block; orchestrator should have rejected this earlier"
        raise ValueError(msg)
    return report.encounter_bins


def _final_median_state_mse(bin_report: EncounterBinReport) -> float | None:
    """Last entry of the per-bin median state MSE curve."""
    return bin_report.rollout.curves.state_mse.median[-1]


def _final_median_position_mse(bin_report: EncounterBinReport) -> float | None:
    """Last entry of the per-bin median position MSE curve (audience-facing headline)."""
    return bin_report.rollout.curves.position_mse.median[-1]


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
        "egnn_final_median_position_mse": _final_median_position_mse(egnn_bin),
        "hgnn_final_median_position_mse": _final_median_position_mse(hgnn_bin),
        "baseline_final_median_position_mse": _final_median_position_mse(baseline_bin),
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
        "egnn_position_mse_median": _safe_position_median(egnn_step),
        "hgnn_position_mse_median": _safe_position_median(hgnn_step),
        "baseline_position_mse_median": _safe_position_median(baseline_step),
        "egnn_position_mse_p95": _safe_position_p95(egnn_step),
        "hgnn_position_mse_p95": _safe_position_p95(hgnn_step),
        "baseline_position_mse_p95": _safe_position_p95(baseline_step),
        "egnn_state_mse_median": _safe_state_median(egnn_step),
        "hgnn_state_mse_median": _safe_state_median(hgnn_step),
        "baseline_state_mse_median": _safe_state_median(baseline_step),
        "egnn_state_mse_p95": _safe_state_p95(egnn_step),
        "hgnn_state_mse_p95": _safe_state_p95(hgnn_step),
        "baseline_state_mse_p95": _safe_state_p95(baseline_step),
        "egnn_finite_fraction": _safe_finite_fraction(egnn_step),
        "hgnn_finite_fraction": _safe_finite_fraction(hgnn_step),
        "baseline_finite_fraction": _safe_finite_fraction(baseline_step),
    }


def _safe_state_median(step: RolloutStepMetrics | None) -> float | None:
    """Return median state MSE for an anchor step, or None when the step is absent."""
    return step.state_mse.median if step is not None else None


def _safe_state_p95(step: RolloutStepMetrics | None) -> float | None:
    """Return p95 state MSE for an anchor step, or None when the step is absent."""
    return step.state_mse.p95 if step is not None else None


def _safe_position_median(step: RolloutStepMetrics | None) -> float | None:
    """Return median position MSE for an anchor step, or None when the step is absent."""
    return step.position_mse.median if step is not None else None


def _safe_position_p95(step: RolloutStepMetrics | None) -> float | None:
    """Return p95 position MSE for an anchor step, or None when the step is absent."""
    return step.position_mse.p95 if step is not None else None


def _safe_finite_fraction(step: RolloutStepMetrics | None) -> float | None:
    """Return finite-fraction for an anchor step, or None when the step is absent.

    Reads from `position_mse` because the table now leads with position MSE
    columns; `finite_fraction` is identical across the metric subfields by
    construction (same trajectories drive all three), so the choice of source
    does not change the value.
    """
    return step.position_mse.finite_fraction if step is not None else None


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
