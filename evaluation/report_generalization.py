"""Generalization report for one body count N: report.py with the distance bins dropped.

Every figure and table reads top-level aggregate metrics instead of per-bin slices, reusing
the figure primitives from report_figures.py. One artifact directory per N.
"""

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from evaluation._types import EvaluationReport
from evaluation.report_figures import (
    _DRIFT_REFERENCE_LEVEL,
    BASELINE_COLOR,
    BASELINE_LINESTYLE,
    BASELINE_MARKER,
    EGNN_COLOR,
    EGNN_MARKER,
    HGNN_COLOR,
    HGNN_MARKER,
    _annotate_bars,
    _apply_figure_style,
    _curve_to_array,
    _finite_max,
    _HorizonMetric,
    _pad_final_bar_log_ylim,
    _save_and_close,
    _value_at_horizon,
)
from utils import get_logger

logger = get_logger(__name__)

BASELINE_MODEL_NAME = "baseline_constant_velocity"
SUMMARY_CSV_NAME = "summary_by_model.csv"
KEY_STEPS: tuple[int, ...] = (1, 10, 50, 100, 199)
SUMMARY_COLUMNS: tuple[str, ...] = (
    "model_name",
    "n_particles",
    "n_trajectories",
    "checkpoint_epoch",
    "single_step_position_mse_median",
    "single_step_state_mse_median",
    *(f"rollout_position_mse_median_{step}" for step in KEY_STEPS),
    "rollout_state_final_finite_fraction",
    "energy_final_drift_median",
    "energy_max_drift_median",
)

# (report-attr, color, legend label, linewidth, linestyle) per model curve.
_SOLID = "-"

# rollout horizon takes the faceting role distance clusters had in the 3-body report;
# split across two figures so neither is too tall (mirrors the snapshot rows there).
_HORIZON_SNAPSHOT_EARLY: tuple[int, ...] = (1, 3, 5, 10, 25)
_HORIZON_SNAPSHOT_LATE: tuple[int, ...] = (50, 100, 150, 199)


class GeneralizationReporter:
    """Generate a comparison artifact for one generalization set (one N), without distance bins."""

    def __init__(
        self,
        egnn_path: Path,
        hgnn_path: Path,
        baseline_path: Path,
        output_dir: Path,
        *,
        label: str | None = None,
        include_egnn: bool = True,
    ) -> None:
        """Store input paths, output dir, and label.

        EGNN is always kept in the CSV/markdown; `include_egnn` only controls whether it is
        drawn (its off-distribution explosion otherwise dominates the axes).
        """
        self.egnn_path = egnn_path
        self.hgnn_path = hgnn_path
        self.baseline_path = baseline_path
        self.output_dir = output_dir
        self.figures_dir = output_dir / "figures"
        self.tables_dir = output_dir / "tables"
        self.label = label
        self.include_egnn = include_egnn

    def run(self) -> None:
        """Load the three reports, validate, and write figures, tables, and markdown."""
        egnn, hgnn, baseline = self._load_reports()
        self._validate_compatible(egnn, hgnn, baseline)
        label = self.label or f"N={egnn.metadata.n_particles}"
        self._setup_output()

        logger.info(
            "generalization report | %s | egnn epoch=%s | hgnn epoch=%s | baseline=%s | n=%d, traj=%d",
            label,
            egnn.metadata.checkpoint_epoch,
            hgnn.metadata.checkpoint_epoch,
            baseline.metadata.run_id,
            egnn.metadata.n_particles,
            egnn.metadata.n_trajectories,
        )

        figures = self._write_figures(egnn, hgnn, baseline, label)
        tables = self._write_tables(egnn, hgnn, baseline)
        write_generalization_markdown(
            egnn,
            hgnn,
            baseline,
            self.output_dir,
            figures=figures,
            tables=tables,
            label=label,
        )
        logger.info(
            "wrote %d figures, %d tables, and report.md to %s",
            len(figures),
            len(tables),
            self.output_dir,
        )

    def _write_figures(
        self,
        egnn: EvaluationReport,
        hgnn: EvaluationReport,
        baseline: EvaluationReport,
        label: str,
    ) -> list[str]:
        """Render the aggregate figures and return their basenames."""
        self._clear_figure_artifacts()
        names: list[str] = []

        include_egnn = self.include_egnn
        rollout = (
            self.figures_dir / "01_rollout_position_mse.png",
            self.figures_dir / "01_rollout_position_mse.pdf",
        )
        plot_rollout_mse_aggregate(
            egnn, hgnn, baseline, rollout, label=label, include_egnn=include_egnn
        )

        drift = (self.figures_dir / "02_energy_drift.png", self.figures_dir / "02_energy_drift.pdf")
        plot_energy_drift_aggregate(
            egnn, hgnn, baseline, drift, label=label, include_egnn=include_egnn
        )

        early = (
            self.figures_dir / "03_horizon_snapshots_early.png",
            self.figures_dir / "03_horizon_snapshots_early.pdf",
        )
        plot_horizon_snapshots_aggregate(
            egnn,
            hgnn,
            baseline,
            early,
            label=label,
            horizons=_HORIZON_SNAPSHOT_EARLY,
            include_egnn=include_egnn,
        )

        late = (
            self.figures_dir / "04_horizon_snapshots_late.png",
            self.figures_dir / "04_horizon_snapshots_late.pdf",
        )
        plot_horizon_snapshots_aggregate(
            egnn,
            hgnn,
            baseline,
            late,
            label=label,
            horizons=_HORIZON_SNAPSHOT_LATE,
            include_egnn=include_egnn,
        )

        for paths in (rollout, drift, early, late):
            names.extend(p.name for p in paths)
        return names

    def _write_tables(
        self,
        egnn: EvaluationReport,
        hgnn: EvaluationReport,
        baseline: EvaluationReport,
    ) -> list[str]:
        """Write the per-model summary CSV and return its basename."""
        write_summary_by_model_csv(egnn, hgnn, baseline, self.tables_dir / SUMMARY_CSV_NAME)
        return [SUMMARY_CSV_NAME]

    def _clear_figure_artifacts(self) -> None:
        """Remove stale PNG/PDF figures so re-runs mirror the current manifest."""
        for path in self.figures_dir.glob("*"):
            if path.suffix in {".png", ".pdf"}:
                path.unlink()

    def _load_reports(self) -> tuple[EvaluationReport, EvaluationReport, EvaluationReport]:
        """Read all three metrics.json files into typed EvaluationReports."""
        return (
            _load_report(self.egnn_path),
            _load_report(self.hgnn_path),
            _load_report(self.baseline_path),
        )

    def _validate_compatible(
        self,
        egnn: EvaluationReport,
        hgnn: EvaluationReport,
        baseline: EvaluationReport,
    ) -> None:
        """Require matching test population and rollout steps, and a constant-velocity baseline."""
        self._require_constant_velocity_baseline(baseline)
        for field in ("n_particles", "n_trajectories"):
            self._require_metadata_field_matches(field, egnn, hgnn, baseline)
        self._require_matching_rollout_steps(egnn, hgnn, baseline)

    def _require_constant_velocity_baseline(self, baseline: EvaluationReport) -> None:
        """Reject any baseline other than the official constant-velocity one."""
        actual = baseline.metadata.model_name
        if actual != BASELINE_MODEL_NAME:
            msg = (
                f"baseline report must come from the {BASELINE_MODEL_NAME!r} baseline; "
                f"got model_name={actual!r} from {self.baseline_path}"
            )
            raise ValueError(msg)

    def _require_metadata_field_matches(
        self,
        field: str,
        egnn: EvaluationReport,
        hgnn: EvaluationReport,
        baseline: EvaluationReport,
    ) -> None:
        """Require all three reports agree on one EvaluationMetadata field."""
        values = {
            "egnn": getattr(egnn.metadata, field),
            "hgnn": getattr(hgnn.metadata, field),
            "baseline": getattr(baseline.metadata, field),
        }
        if len(set(values.values())) > 1:
            msg = (
                f"metadata.{field} differs between reports; "
                f"egnn={values['egnn']} hgnn={values['hgnn']} baseline={values['baseline']}"
            )
            raise ValueError(msg)

    def _require_matching_rollout_steps(
        self,
        egnn: EvaluationReport,
        hgnn: EvaluationReport,
        baseline: EvaluationReport,
    ) -> None:
        """Require the rollout curve step axis is identical across reports."""
        egnn_steps = tuple(egnn.rollout.curves.step)
        hgnn_steps = tuple(hgnn.rollout.curves.step)
        baseline_steps = tuple(baseline.rollout.curves.step)
        if egnn_steps != hgnn_steps or egnn_steps != baseline_steps:
            msg = (
                "rollout step values differ across reports; "
                f"egnn={egnn_steps} hgnn={hgnn_steps} baseline={baseline_steps}"
            )
            raise ValueError(msg)

    def _setup_output(self) -> None:
        """Create the figures/ and tables/ subdirectories under output_dir."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)
        self.tables_dir.mkdir(exist_ok=True)


def plot_rollout_mse_aggregate(
    egnn: EvaluationReport,
    hgnn: EvaluationReport,
    baseline: EvaluationReport,
    output_paths: Iterable[Path],
    *,
    label: str,
    include_egnn: bool = True,
) -> None:
    """Single-panel median position MSE vs rollout step for the active models."""
    _apply_figure_style()
    series = _model_series(egnn, hgnn, baseline, include_egnn=include_egnn)
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    for report, color, name, lw, ls, _marker in series:
        steps = np.asarray(report.rollout.curves.step, dtype=int)
        curve = _curve_to_array(report.rollout.curves.position_mse.median)
        ax.plot(steps[1:], curve[1:], color=color, linewidth=lw, linestyle=ls, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("median position MSE")
    ax.set_title(f"Rollout position MSE ({label})")
    ax.legend(handles=_legend_handles(series, include_markers=False), frameon=False)
    fig.tight_layout()
    _save_and_close(fig, output_paths)


def plot_energy_drift_aggregate(
    egnn: EvaluationReport,
    hgnn: EvaluationReport,
    baseline: EvaluationReport,
    output_paths: Iterable[Path],
    *,
    label: str,
    include_egnn: bool = True,
) -> None:
    """Single-panel median relative energy drift vs step (log-log) with a y=1 reference."""
    _apply_figure_style()
    series = _model_series(egnn, hgnn, baseline, include_egnn=include_egnn)
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.axhline(_DRIFT_REFERENCE_LEVEL, color="grey", linestyle="--", linewidth=1.0, alpha=0.45)
    for report, color, name, lw, ls, _marker in series:
        steps = np.asarray(report.energy.physical.curves.step, dtype=int)
        curve = _curve_to_array(report.energy.physical.curves.median)
        ax.plot(steps[1:], curve[1:], color=color, linewidth=lw, linestyle=ls, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rollout step (log scale)")
    ax.set_ylabel(r"median $|\Delta E / E_0|$")
    ax.set_title(f"Relative energy drift ({label})")
    ax.legend(handles=_legend_handles(series, include_markers=False), frameon=False)
    fig.tight_layout()
    _save_and_close(fig, output_paths)


def plot_horizon_snapshots_aggregate(
    egnn: EvaluationReport,
    hgnn: EvaluationReport,
    baseline: EvaluationReport,
    output_paths: Iterable[Path],
    *,
    label: str,
    horizons: tuple[int, ...],
    include_egnn: bool = True,
) -> None:
    """Per-horizon snapshot bars: one row per horizon, position MSE (left) and energy drift (right)."""
    _apply_figure_style()
    series = _model_series(egnn, hgnn, baseline, include_egnn=include_egnn)
    reports = [report for report, *_rest in series]
    colors = [color for _report, color, *_rest in series]
    fig, axes = plt.subplots(len(horizons), 2, figsize=(14.5, 2.8 * len(horizons)), squeeze=False)
    for row, horizon in enumerate(horizons):
        mse_values = _aggregate_values_at_horizon(
            reports, metric=_HorizonMetric.POSITION_MSE, horizon=horizon
        )
        drift_values = _aggregate_values_at_horizon(
            reports, metric=_HorizonMetric.ENERGY_DRIFT, horizon=horizon
        )
        _render_snapshot_bars(
            axes[row, 0],
            mse_values,
            colors,
            metric=_HorizonMetric.POSITION_MSE,
            row_label=f"h={horizon}",
        )
        _render_snapshot_bars(
            axes[row, 1],
            drift_values,
            colors,
            metric=_HorizonMetric.ENERGY_DRIFT,
            row_label=None,
        )
    axes[0, 0].set_title("median position MSE")
    axes[0, 1].set_title(r"median $|\Delta E / E_0|$")
    fig.legend(
        handles=_legend_handles(series, include_markers=True),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(series),
        frameon=False,
        fontsize=15,
        title="Models",
        title_fontsize=17,
    )
    fig.suptitle(f"Accuracy and energy by rollout horizon ({label})", y=0.995)
    fig.tight_layout(rect=(0.02, 0.0, 1.0, 0.925), h_pad=1.25, w_pad=1.8)
    _save_and_close(fig, output_paths)


def write_summary_by_model_csv(
    egnn: EvaluationReport,
    hgnn: EvaluationReport,
    baseline: EvaluationReport,
    path: Path,
) -> None:
    """Write one aggregate row per model (EGNN, HGNN, baseline)."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for report in (egnn, hgnn, baseline):
            writer.writerow(_summary_row(report))


def write_generalization_markdown(
    egnn: EvaluationReport,
    hgnn: EvaluationReport,
    baseline: EvaluationReport,
    output_dir: Path,
    *,
    figures: Iterable[str],
    tables: Iterable[str],
    label: str,
) -> None:
    """Render `output_dir/report.md` for one generalization set."""
    sections = [
        f"# Generalization report ({label}): EGNN vs HGNN vs Constant Velocity",
        "",
        "## Provenance",
        "",
        _md_table(
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
                    "test_path",
                    egnn.metadata.test_path,
                    hgnn.metadata.test_path,
                    baseline.metadata.test_path,
                ],
                [
                    "n_particles",
                    _str(egnn.metadata.n_particles),
                    _str(hgnn.metadata.n_particles),
                    _str(baseline.metadata.n_particles),
                ],
                [
                    "n_trajectories",
                    _str(egnn.metadata.n_trajectories),
                    _str(hgnn.metadata.n_trajectories),
                    _str(baseline.metadata.n_trajectories),
                ],
            ],
        ),
        "",
        "## Headline (aggregate, no distance clusters)",
        "",
        "Final-step median position MSE and final relative energy drift across all trajectories. Lower is better.",
        "",
        _md_table(
            ["Model", "final position MSE", "final energy drift", "final finite fraction"],
            [
                _headline_row("EGNN", egnn),
                _headline_row("HGNN", hgnn),
                _headline_row("constant velocity", baseline),
            ],
        ),
        "",
        "## Artifacts",
        "",
        "### Figures",
        "",
        _md_bullets(f"figures/{name}" for name in figures) or "_no figures listed_",
        "",
        "### Tables",
        "",
        _md_bullets(f"tables/{name}" for name in tables) or "_no tables listed_",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(sections))


def _model_series(
    egnn: EvaluationReport,
    hgnn: EvaluationReport,
    baseline: EvaluationReport,
    *,
    include_egnn: bool,
) -> list[tuple[EvaluationReport, str, str, float, object, str]]:
    """Active model series as (report, color, label, linewidth, linestyle, marker).

    EGNN is dropped when `include_egnn` is False; order is EGNN (if present), HGNN, baseline.
    """
    series: list[tuple[EvaluationReport, str, str, float, object, str]] = []
    if include_egnn:
        series.append((egnn, EGNN_COLOR, "EGNN", 2.2, _SOLID, EGNN_MARKER))
    series.append((hgnn, HGNN_COLOR, "HGNN", 2.2, _SOLID, HGNN_MARKER))
    series.append(
        (baseline, BASELINE_COLOR, "constant velocity", 1.8, BASELINE_LINESTYLE, BASELINE_MARKER)
    )
    return series


def _legend_handles(
    series: list[tuple[EvaluationReport, str, str, float, object, str]],
    *,
    include_markers: bool,
) -> list[plt.Line2D]:
    """Build legend handles for exactly the active model series (no fixed three-model assumption)."""
    return [
        plt.Line2D(
            [],
            [],
            color=color,
            linewidth=2.4,
            linestyle="-" if linestyle == _SOLID else linestyle,
            marker=marker if include_markers else None,
            markersize=8,
            label=label,
        )
        for _report, color, label, _lw, linestyle, marker in series
    ]


def _render_snapshot_bars(
    ax: plt.Axes,
    values: np.ndarray,
    colors: list[str],
    *,
    metric: _HorizonMetric,
    row_label: str | None,
) -> None:
    """Render one snapshot row of per-model bars, for a variable model count."""
    x = np.arange(len(values))
    bars = ax.bar(x, values, width=0.62, color=colors)
    if metric is _HorizonMetric.ENERGY_DRIFT:
        ax.set_yscale("log")
        ax.axhline(_DRIFT_REFERENCE_LEVEL, color="grey", linestyle="--", linewidth=1.0, alpha=0.5)
        _pad_final_bar_log_ylim(ax, (values,))
    else:
        ax.set_ylim(0.0, _finite_max((values,)) * 1.30)
    for bar, color in zip(bars, colors, strict=True):
        _annotate_bars(ax, [bar], color=color)
    ax.set_xticks([])
    if row_label is not None:
        ax.set_ylabel(row_label, rotation=0, ha="right", va="center", labelpad=58)


def _aggregate_values_at_horizon(
    reports: list[EvaluationReport],
    *,
    metric: _HorizonMetric,
    horizon: int,
) -> np.ndarray:
    """Median value at one horizon for each report, in the order given."""

    def value(report: EvaluationReport) -> float:
        if metric is _HorizonMetric.POSITION_MSE:
            steps = report.rollout.curves.step
            values = report.rollout.curves.position_mse.median
        else:
            steps = report.energy.physical.curves.step
            values = report.energy.physical.curves.median
        return _value_at_horizon(steps, values, horizon, label=f"{metric.value} h={horizon}")

    return np.array([value(report) for report in reports], dtype=float)


def _summary_row(report: EvaluationReport) -> dict[str, Any]:
    """Build one aggregate CSV row for a single report."""
    steps = list(report.rollout.curves.step)
    medians = list(report.rollout.curves.position_mse.median)
    row: dict[str, Any] = {
        "model_name": report.metadata.model_name,
        "n_particles": report.metadata.n_particles,
        "n_trajectories": report.metadata.n_trajectories,
        "checkpoint_epoch": report.metadata.checkpoint_epoch,
        "single_step_position_mse_median": report.single_step.position_mse.median,
        "single_step_state_mse_median": report.single_step.state_mse.median,
        "rollout_state_final_finite_fraction": report.rollout.state_final_finite_fraction,
        "energy_final_drift_median": report.energy.physical.final_relative_drift.median,
        "energy_max_drift_median": report.energy.physical.max_relative_drift.median,
    }
    for step in KEY_STEPS:
        row[f"rollout_position_mse_median_{step}"] = _curve_value_at_step(steps, medians, step)
    return row


def _headline_row(model_label: str, report: EvaluationReport) -> list[str]:
    """One markdown headline row: final position MSE, final energy drift, final finite fraction."""
    final_mse = report.rollout.curves.position_mse.median[-1]
    final_drift = report.energy.physical.final_relative_drift.median
    finite_fraction = report.rollout.state_final_finite_fraction
    return [
        model_label,
        _fmt_float(final_mse),
        _fmt_float(final_drift),
        _fmt_float(finite_fraction),
    ]


def _curve_value_at_step(steps: list[int], values: list[float | None], step: int) -> float | None:
    """Return the curve value at `step`, or None when that step is absent."""
    try:
        idx = steps.index(step)
    except ValueError:
        return None
    return values[idx]


def _load_report(path: Path) -> EvaluationReport:
    """Load a metrics.json file into a typed EvaluationReport."""
    with path.open() as f:
        return EvaluationReport.from_dict(json.load(f))


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a github-flavored markdown table."""
    sep = "|" + "|".join("---" for _ in headers) + "|"
    header_line = "| " + " | ".join(headers) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, sep, *body_lines])


def _md_bullets(items: Iterable[str]) -> str:
    """Render an unordered list, or empty string when there are no items."""
    return "\n".join(f"- {item}" for item in items)


def _str(value: str | int | float | None) -> str:
    """Stringify None as 'n/a' so markdown cells are never empty."""
    return "n/a" if value is None else str(value)


def _fmt_float(value: float | None) -> str:
    """Format a possibly-None float for markdown cells; ~4 sig figs."""
    return f"{value:.4g}" if value is not None else "n/a"


def main() -> None:
    """Generate one generalization report from CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a generalization comparison report for one body count (no distance groups).",
    )
    parser.add_argument("--egnn", type=str, required=True, help="Path to EGNN metrics.json.")
    parser.add_argument("--hgnn", type=str, required=True, help="Path to HGNN metrics.json.")
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to constant-velocity baseline metrics.json.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for figures/, tables/, report.md.",
    )
    parser.add_argument(
        "--label", type=str, default=None, help="Facet label (default: 'N=<n_particles>')."
    )
    parser.add_argument(
        "--no-egnn",
        action="store_true",
        help="Drop EGNN from the figures (kept in the CSV/markdown record).",
    )
    args = parser.parse_args()

    GeneralizationReporter(
        egnn_path=Path(args.egnn),
        hgnn_path=Path(args.hgnn),
        baseline_path=Path(args.baseline),
        output_dir=Path(args.output),
        label=args.label,
        include_egnn=not args.no_egnn,
    ).run()


if __name__ == "__main__":
    main()
