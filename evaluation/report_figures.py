"""Presentation-grade figures for the comparison report.

Public plotters, each taking the EGNN, HGNN, and baseline reports:
    - plot_rollout_mse_presentation  : position MSE curve per bin
    - plot_energy_drift_presentation : |dE/E_0| curve per bin
    - plot_horizon_snapshot_by_bin   : position MSE and energy bars at one horizon
"""

from collections.abc import Iterable
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.container import BarContainer
from matplotlib.patches import Rectangle

from evaluation._types import EncounterBinReport, EvaluationReport
from evaluation.report_tables import _require_bins


class _HorizonMetric(Enum):
    """Which per-bin curve a horizon panel reads from."""

    POSITION_MSE = "position_mse"
    ENERGY_DRIFT = "energy_drift"


_STYLE_APPLIED = False

EGNN_COLOR = "#1f77b4"
HGNN_COLOR = "#ff7f0e"
BASELINE_COLOR = "#555555"
BASELINE_LINESTYLE = (0, (4, 2))
EGNN_MARKER = "o"
HGNN_MARKER = "s"
BASELINE_MARKER = "D"

HORIZON_ANCHORS: tuple[int, ...] = (1, 3, 5, 10, 25, 50, 100, 150, 199)
"""Rollout steps the horizon figures sample. Fixed across MSE and energy drift."""

_DRIFT_REFERENCE_LEVEL = 1.0
"""Horizontal reference: |dE / E_0| = 1 means full-order-of-magnitude energy violation."""


def _apply_figure_style() -> None:
    """Pin the seaborn whitegrid/talk theme for slides and PDF, once per process (idempotent)."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )
    _STYLE_APPLIED = True


def plot_rollout_mse_presentation(
    egnn: EvaluationReport,
    hgnn: EvaluationReport,
    baseline: EvaluationReport,
    output_paths: Iterable[Path],
) -> None:
    """Headline rollout figure: per-bin grid of median position MSE vs step (log y)."""
    _apply_figure_style()
    bin_names, panels = _per_bin_panels(egnn, hgnn, baseline)
    n_bins = len(bin_names)

    fig, axes, cols = _make_grid(n_bins)
    for idx, name in enumerate(bin_names):
        egnn_bin, hgnn_bin, baseline_bin = panels[name]
        _render_mse_panel(
            axes[idx],
            egnn_bin,
            hgnn_bin,
            baseline_bin,
            bin_name=name,
            is_leftmost=_is_leftmost(idx, cols),
            is_bottom=_is_bottom_panel(idx, n_bins, cols),
        )

    _render_legend_panel(axes[n_bins])
    _hide_unused_panels(axes, used=n_bins + 1)
    _suptitle_with_padding(fig, "Rollout position MSE by distance cluster")
    _save_and_close(fig, output_paths)


def plot_energy_drift_presentation(
    egnn: EvaluationReport,
    hgnn: EvaluationReport,
    baseline: EvaluationReport,
    output_paths: Iterable[Path],
) -> None:
    """Headline energy-drift figure: per-bin grid of median |dE/E_0| vs step (log-log)."""
    _apply_figure_style()
    bin_names, panels = _per_bin_panels(egnn, hgnn, baseline)
    n_bins = len(bin_names)

    fig, axes, cols = _make_grid(n_bins)
    for idx, name in enumerate(bin_names):
        egnn_bin, hgnn_bin, baseline_bin = panels[name]
        _render_drift_panel(
            axes[idx],
            egnn_bin,
            hgnn_bin,
            baseline_bin,
            bin_name=name,
            is_leftmost=_is_leftmost(idx, cols),
            is_bottom=_is_bottom_panel(idx, n_bins, cols),
        )

    _render_legend_panel(axes[n_bins])
    _hide_unused_panels(axes, used=n_bins + 1)
    _suptitle_with_padding(fig, "Relative energy drift by distance cluster")
    _save_and_close(fig, output_paths)


def plot_final_mse_by_bin(
    egnn: EvaluationReport,
    hgnn: EvaluationReport,
    baseline: EvaluationReport,
    output_paths: Iterable[Path],
) -> None:
    """Final-step comparison figure: median position MSE as grouped bars per bin."""
    _plot_metric_bars_at_step(
        egnn,
        hgnn,
        baseline,
        output_paths,
        metric=_HorizonMetric.POSITION_MSE,
        step=None,
    )


def plot_final_energy_drift_by_bin(
    egnn: EvaluationReport,
    hgnn: EvaluationReport,
    baseline: EvaluationReport,
    output_paths: Iterable[Path],
) -> None:
    """Final-step comparison figure: median |dE/E_0| as grouped bars per bin."""
    _plot_metric_bars_at_step(
        egnn,
        hgnn,
        baseline,
        output_paths,
        metric=_HorizonMetric.ENERGY_DRIFT,
        step=None,
    )


def plot_mse_bars_at_horizon(
    egnn: EvaluationReport,
    hgnn: EvaluationReport,
    baseline: EvaluationReport,
    output_paths: Iterable[Path],
    *,
    horizon: int,
) -> None:
    """Grouped position MSE bars by encounter bin at a selected rollout horizon."""
    _plot_metric_bars_at_step(
        egnn,
        hgnn,
        baseline,
        output_paths,
        metric=_HorizonMetric.POSITION_MSE,
        step=horizon,
    )


def plot_horizon_snapshot_by_bin(
    egnn: EvaluationReport,
    hgnn: EvaluationReport,
    baseline: EvaluationReport,
    output_paths: Iterable[Path],
    *,
    horizon: int,
) -> None:
    """Fixed-horizon snapshot: MSE and energy-drift bars side by side for each bin."""
    _apply_figure_style()
    bin_names, panels = _per_bin_panels(egnn, hgnn, baseline)
    fig, axes = plt.subplots(
        len(bin_names),
        2,
        figsize=(14.5, 2.8 * len(bin_names)),
        squeeze=False,
    )
    for row, name in enumerate(bin_names):
        egnn_bin, hgnn_bin, baseline_bin = panels[name]
        count = egnn_bin.count
        mse_values = _model_metric_values_at_step(
            egnn_bin,
            hgnn_bin,
            baseline_bin,
            metric=_HorizonMetric.POSITION_MSE,
            step=horizon,
            label=name,
        )
        drift_values = _model_metric_values_at_step(
            egnn_bin,
            hgnn_bin,
            baseline_bin,
            metric=_HorizonMetric.ENERGY_DRIFT,
            step=horizon,
            label=name,
        )

        _render_horizon_snapshot_panel(
            axes[row, 0],
            mse_values,
            metric=_HorizonMetric.POSITION_MSE,
            row_label=_bin_label(name, count),
        )
        _render_horizon_snapshot_panel(
            axes[row, 1],
            drift_values,
            metric=_HorizonMetric.ENERGY_DRIFT,
            row_label=None,
        )

    axes[0, 0].set_title("median position MSE")
    axes[0, 1].set_title(r"median $|\Delta E / E_0|$")
    fig.legend(
        handles=_model_legend_handles(include_markers=True),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
        fontsize=15,
        title="Models",
        title_fontsize=17,
    )
    fig.suptitle(f"Rollout horizon {horizon}: accuracy and energy by distance cluster", y=0.995)
    fig.tight_layout(rect=(0.02, 0.0, 1.0, 0.925), h_pad=1.25, w_pad=1.8)
    _save_and_close(fig, output_paths)


def plot_horizon_mse_by_bin(
    egnn: EvaluationReport,
    hgnn: EvaluationReport,
    baseline: EvaluationReport,
    output_paths: Iterable[Path],
) -> None:
    """Headline horizon-MSE figure: median position MSE at fixed horizons (rows=bins, cols=models)."""
    _plot_horizon_metric_grid(
        egnn,
        hgnn,
        baseline,
        output_paths,
        metric=_HorizonMetric.POSITION_MSE,
        title="Rollout position MSE at fixed horizons by distance cluster",
    )


def plot_horizon_energy_drift_by_bin(
    egnn: EvaluationReport,
    hgnn: EvaluationReport,
    baseline: EvaluationReport,
    output_paths: Iterable[Path],
) -> None:
    """Headline horizon-drift figure: exact median |dE/E_0| at fixed horizons."""
    _plot_horizon_metric_grid(
        egnn,
        hgnn,
        baseline,
        output_paths,
        metric=_HorizonMetric.ENERGY_DRIFT,
        title="Relative energy drift at fixed horizons by distance cluster",
    )


def _plot_metric_bars_at_step(
    egnn: EvaluationReport,
    hgnn: EvaluationReport,
    baseline: EvaluationReport,
    output_paths: Iterable[Path],
    *,
    metric: _HorizonMetric,
    step: int | None,
) -> None:
    """Render a grouped bar chart for one metric at one rollout step."""
    _apply_figure_style()
    bin_names, panels = _per_bin_panels(egnn, hgnn, baseline)
    x = np.arange(len(bin_names))
    width = 0.24

    egnn_values, hgnn_values, baseline_values = _metric_values_at_step(
        bin_names, panels, metric=metric, step=step
    )
    fig, ax = plt.subplots(figsize=(14.5, 7.0))
    bars_egnn = ax.bar(x - width, egnn_values, width, color=EGNN_COLOR, label="EGNN")
    bars_hgnn = ax.bar(x, hgnn_values, width, color=HGNN_COLOR, label="HGNN")
    bars_baseline = ax.bar(
        x + width,
        baseline_values,
        width,
        color=BASELINE_COLOR,
        alpha=0.82,
        label="constant velocity",
    )

    if metric is _HorizonMetric.ENERGY_DRIFT:
        ax.set_yscale("log")
        ax.axhline(_DRIFT_REFERENCE_LEVEL, color="grey", linestyle="--", linewidth=1.1, alpha=0.6)
        ax.text(
            0.01,
            _DRIFT_REFERENCE_LEVEL,
            r"$|\Delta E/E_0|=1$",
            transform=ax.get_yaxis_transform(),
            color="grey",
            fontsize=11,
            va="bottom",
        )
        _pad_final_bar_log_ylim(ax, (egnn_values, hgnn_values, baseline_values))
    else:
        ax.set_ylim(0.0, _finite_max((egnn_values, hgnn_values, baseline_values)) * 1.22)

    _annotate_bars(ax, bars_egnn, color=EGNN_COLOR)
    _annotate_bars(ax, bars_hgnn, color=HGNN_COLOR)
    _annotate_bars(ax, bars_baseline, color=BASELINE_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels([_bin_label(name, panels[name][0].count) for name in bin_names])
    ax.set_ylabel(_horizon_metric_y_label(metric))
    ax.set_title(_metric_bar_title(metric, step=step))
    legend_loc = "upper right" if metric is _HorizonMetric.ENERGY_DRIFT else "upper left"
    ax.legend(loc=legend_loc, frameon=False, ncol=3)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.14)
    _save_and_close(fig, output_paths)


def _metric_values_at_step(
    bin_names: list[str],
    panels: dict[str, tuple[EncounterBinReport, EncounterBinReport, EncounterBinReport]],
    *,
    metric: _HorizonMetric,
    step: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect metric values in canonical bin order at `step` or at the final step."""
    egnn_values = []
    hgnn_values = []
    baseline_values = []
    for name in bin_names:
        egnn_bin, hgnn_bin, baseline_bin = panels[name]
        egnn_values.append(_metric_value_at_step(egnn_bin, metric=metric, step=step, label=name))
        hgnn_values.append(_metric_value_at_step(hgnn_bin, metric=metric, step=step, label=name))
        baseline_values.append(
            _metric_value_at_step(baseline_bin, metric=metric, step=step, label=name)
        )
    return (
        np.array(egnn_values, dtype=float),
        np.array(hgnn_values, dtype=float),
        np.array(baseline_values, dtype=float),
    )


def _model_metric_values_at_step(
    egnn_bin: EncounterBinReport,
    hgnn_bin: EncounterBinReport,
    baseline_bin: EncounterBinReport,
    *,
    metric: _HorizonMetric,
    step: int | None,
    label: str,
) -> np.ndarray:
    """Collect EGNN/HGNN/baseline values for one bin at a selected step."""
    return np.array(
        [
            _metric_value_at_step(egnn_bin, metric=metric, step=step, label=f"{label} EGNN"),
            _metric_value_at_step(hgnn_bin, metric=metric, step=step, label=f"{label} HGNN"),
            _metric_value_at_step(
                baseline_bin,
                metric=metric,
                step=step,
                label=f"{label} constant_velocity",
            ),
        ],
        dtype=float,
    )


def _metric_value_at_step(
    bin_report: EncounterBinReport,
    *,
    metric: _HorizonMetric,
    step: int | None,
    label: str,
) -> float:
    """Return the median value for one bin and metric at `step` or final step."""
    if metric is _HorizonMetric.POSITION_MSE:
        steps = list(bin_report.rollout.curves.step)
        values = list(bin_report.rollout.curves.position_mse.median)
    else:
        steps = list(bin_report.energy.physical.curves.step)
        values = list(bin_report.energy.physical.curves.median)
    if step is None:
        return _nan_if_none(values[-1])
    return _value_at_horizon(steps, values, step, label=f"bin={label!r} {metric.value}")


def _metric_bar_title(metric: _HorizonMetric, *, step: int | None) -> str:
    """Human-readable title for grouped metric bar charts."""
    prefix = "Final-step" if step is None else f"Rollout step {step}"
    if metric is _HorizonMetric.POSITION_MSE:
        return f"{prefix} median position MSE by distance cluster"
    return f"{prefix} relative energy drift by distance cluster"


def _bin_label(name: str, count: int) -> str:
    """Compact x-axis label with bin name and sample count."""
    return f"{name}\n(n={count})"


def _finite_max(series: tuple[np.ndarray, ...]) -> float:
    """Maximum finite value across arrays, with a defensive positive fallback."""
    values = [float(v) for arr in series for v in arr if np.isfinite(v)]
    return max(values) if values else 1.0


def _finite_positive_min_max(series: tuple[np.ndarray, ...]) -> tuple[float, float]:
    """Positive finite min/max across arrays, suitable for log y-limits."""
    values = [float(v) for arr in series for v in arr if np.isfinite(v) and v > 0]
    if not values:
        return 1e-3, 1.0
    return min(values), max(values)


def _pad_final_bar_log_ylim(ax: plt.Axes, series: tuple[np.ndarray, ...]) -> None:
    """Set log bar limits with room for labels above the tallest bar."""
    y_min, y_max = _finite_positive_min_max(series)
    ax.set_ylim(y_min / 2.0, y_max * 6.0)


def _annotate_bars(ax: plt.Axes, bars: Iterable[Rectangle] | BarContainer, *, color: str) -> None:
    """Place compact value labels above finite bars."""
    for bar in bars:
        value = float(bar.get_height())
        if not np.isfinite(value) or value <= 0:
            continue
        ax.annotate(
            _format_horizon_value(value),
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=color,
            fontsize=10,
            clip_on=False,
        )


def _render_horizon_snapshot_panel(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    metric: _HorizonMetric,
    row_label: str | None,
) -> None:
    """Render the three-model bars for one metric inside a fixed-horizon row."""
    x = np.arange(3)
    bars = ax.bar(
        x,
        values,
        width=0.62,
        color=[EGNN_COLOR, HGNN_COLOR, BASELINE_COLOR],
    )
    bars[2].set_alpha(0.82)
    if metric is _HorizonMetric.ENERGY_DRIFT:
        ax.set_yscale("log")
        ax.axhline(_DRIFT_REFERENCE_LEVEL, color="grey", linestyle="--", linewidth=1.0, alpha=0.5)
        _pad_final_bar_log_ylim(ax, (values,))
    else:
        ax.set_ylim(0.0, _finite_max((values,)) * 1.30)

    for bar, color in zip(bars, [EGNN_COLOR, HGNN_COLOR, BASELINE_COLOR], strict=True):
        _annotate_bars(ax, [bar], color=color)
    ax.set_xticks([])
    if row_label is not None:
        ax.set_ylabel(row_label, rotation=0, ha="right", va="center", labelpad=58)


def _plot_horizon_metric_grid(
    egnn: EvaluationReport,
    hgnn: EvaluationReport,
    baseline: EvaluationReport,
    output_paths: Iterable[Path],
    *,
    metric: _HorizonMetric,
    title: str,
) -> None:
    """Render one wide horizon panel per encounter bin for one metric."""
    _apply_figure_style()
    bin_names, panels = _per_bin_panels(egnn, hgnn, baseline)
    fig, axes = _make_horizon_grid(len(bin_names))
    for idx, name in enumerate(bin_names):
        egnn_bin, hgnn_bin, baseline_bin = panels[name]
        _render_horizon_panel(
            axes[idx],
            egnn_bin,
            hgnn_bin,
            baseline_bin,
            bin_name=name,
            metric=metric,
            show_x_label=idx == len(bin_names) - 1,
        )
    fig.legend(
        handles=_model_legend_handles(include_markers=True),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
        fontsize=16,
        title="Models",
        title_fontsize=18,
    )
    fig.supylabel(_horizon_metric_y_label(metric), x=0.012)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0.035, 0.0, 1.0, 0.91))
    _save_and_close(fig, output_paths)


def _per_bin_panels(
    egnn: EvaluationReport,
    hgnn: EvaluationReport,
    baseline: EvaluationReport,
) -> tuple[
    list[str],
    dict[str, tuple[EncounterBinReport, EncounterBinReport, EncounterBinReport]],
]:
    """Return canonical bin order plus a (egnn, hgnn, baseline) triple per bin."""
    egnn_bins = _require_bins(egnn)
    hgnn_bins = _require_bins(hgnn)
    baseline_bins = _require_bins(baseline)
    names = [b.name for b in egnn_bins.bins]
    panels = {
        name: (
            egnn_bins.by_name[name],
            hgnn_bins.by_name[name],
            baseline_bins.by_name[name],
        )
        for name in names
    }
    return names, panels


def _make_grid(n_bins: int, *, scale: float = 1.0) -> tuple[plt.Figure, list[plt.Axes], int]:
    """Build a 2-row shared-axis grid for `n_bins` panels plus one legend cell.

    Columns are sized so the legend slots into the bottom row; returns the column count.
    """
    cols = max(2, (n_bins + 2) // 2)
    rows = 2
    fig, grid = plt.subplots(
        rows,
        cols,
        figsize=(5.3 * cols * scale, 4.5 * rows * scale),
        sharex=True,
        sharey=True,
    )
    return fig, list(grid.flat), cols


def _make_horizon_grid(n_bins: int) -> tuple[plt.Figure, list[plt.Axes]]:
    """Build one wide row per encounter bin for fixed-horizon value plots."""
    fig, grid = plt.subplots(
        n_bins,
        1,
        figsize=(16.5, 3.1 * n_bins),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    return fig, list(grid[:, 0])


def _is_leftmost(panel_idx: int, cols: int) -> bool:
    """Panels in column 0 own the y-axis label for their row."""
    return panel_idx % cols == 0


def _is_bottom_panel(panel_idx: int, n_bins: int, cols: int) -> bool:
    """A panel owns the x-axis label when no later bin panel shares its column."""
    col = panel_idx % cols
    return all(later % cols != col for later in range(panel_idx + 1, n_bins))


def _hide_unused_panels(axes: list[plt.Axes], *, used: int) -> None:
    """Turn off any axes past the last used cell to keep the figure clean."""
    for ax in axes[used:]:
        ax.axis("off")


def _apply_outer_labels(
    ax: plt.Axes,
    *,
    x_label: str,
    y_label: str,
    is_leftmost: bool,
    is_bottom: bool,
) -> None:
    """Restrict axis labels and ticks to the grid's outer edge (legend cell breaks the default)."""
    if is_leftmost:
        ax.set_ylabel(y_label)
        ax.tick_params(axis="y", labelleft=True)
    else:
        ax.tick_params(axis="y", labelleft=False)
    if is_bottom:
        ax.set_xlabel(x_label)
        ax.tick_params(axis="x", labelbottom=True)
    else:
        ax.tick_params(axis="x", labelbottom=False)


def _render_mse_panel(
    ax: plt.Axes,
    egnn_bin: EncounterBinReport,
    hgnn_bin: EncounterBinReport,
    baseline_bin: EncounterBinReport,
    *,
    bin_name: str,
    is_leftmost: bool,
    is_bottom: bool,
) -> None:
    """Plot one MSE panel: three curves on log-y, linear x; labels only on outer-edge panels."""
    steps = np.asarray(egnn_bin.rollout.curves.step, dtype=int)
    egnn_curve = _curve_to_array(egnn_bin.rollout.curves.position_mse.median)
    hgnn_curve = _curve_to_array(hgnn_bin.rollout.curves.position_mse.median)
    baseline_curve = _curve_to_array(baseline_bin.rollout.curves.position_mse.median)

    ax.plot(steps[1:], egnn_curve[1:], color=EGNN_COLOR, linewidth=2.2, label="EGNN")
    ax.plot(steps[1:], hgnn_curve[1:], color=HGNN_COLOR, linewidth=2.2, label="HGNN")
    ax.plot(
        steps[1:],
        baseline_curve[1:],
        color=BASELINE_COLOR,
        linewidth=1.8,
        linestyle=BASELINE_LINESTYLE,
        label="constant velocity",
    )
    ax.set_yscale("log")
    ax.set_title(f"{bin_name} (n={egnn_bin.count})")
    _apply_outer_labels(
        ax,
        x_label="rollout step",
        y_label="median position MSE",
        is_leftmost=is_leftmost,
        is_bottom=is_bottom,
    )


def _render_drift_panel(
    ax: plt.Axes,
    egnn_bin: EncounterBinReport,
    hgnn_bin: EncounterBinReport,
    baseline_bin: EncounterBinReport,
    *,
    bin_name: str,
    is_leftmost: bool,
    is_bottom: bool,
) -> None:
    """Plot one drift panel: three curves on log-log with a y=1 reference; outer-edge labels only."""
    steps = np.asarray(egnn_bin.energy.physical.curves.step, dtype=int)
    egnn_curve = _curve_to_array(egnn_bin.energy.physical.curves.median)
    hgnn_curve = _curve_to_array(hgnn_bin.energy.physical.curves.median)
    baseline_curve = _curve_to_array(baseline_bin.energy.physical.curves.median)

    ax.axhline(_DRIFT_REFERENCE_LEVEL, color="grey", linestyle="--", linewidth=1.0, alpha=0.45)
    ax.plot(steps[1:], egnn_curve[1:], color=EGNN_COLOR, linewidth=2.2, label="EGNN")
    ax.plot(steps[1:], hgnn_curve[1:], color=HGNN_COLOR, linewidth=2.2, label="HGNN")
    ax.plot(
        steps[1:],
        baseline_curve[1:],
        color=BASELINE_COLOR,
        linewidth=1.8,
        linestyle=BASELINE_LINESTYLE,
        label="constant velocity",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{bin_name} (n={egnn_bin.count})")
    _apply_outer_labels(
        ax,
        x_label="rollout step (log scale)",
        y_label=r"median $|\Delta E / E_0|$",
        is_leftmost=is_leftmost,
        is_bottom=is_bottom,
    )


def _render_legend_panel(ax: plt.Axes, *, include_markers: bool = False) -> None:
    """Fill the spare grid cell with a borderless legend keyed to the bin panels."""
    ax.axis("off")
    ax.legend(
        handles=_model_legend_handles(include_markers=include_markers),
        loc="center",
        frameon=False,
        fontsize=18,
        title="Models",
        title_fontsize=20,
    )


def _model_legend_handles(*, include_markers: bool) -> list[plt.Line2D]:
    """Legend handles shared by continuous and horizon figures."""
    return [
        plt.Line2D(
            [],
            [],
            color=EGNN_COLOR,
            linewidth=2.6,
            marker=EGNN_MARKER if include_markers else None,
            markersize=8,
            label="EGNN",
        ),
        plt.Line2D(
            [],
            [],
            color=HGNN_COLOR,
            linewidth=2.6,
            marker=HGNN_MARKER if include_markers else None,
            markersize=8,
            label="HGNN",
        ),
        plt.Line2D(
            [],
            [],
            color=BASELINE_COLOR,
            linewidth=2.2,
            linestyle=BASELINE_LINESTYLE,
            marker=BASELINE_MARKER if include_markers else None,
            markersize=7,
            label="constant velocity",
        ),
    ]


def _suptitle_with_padding(fig: plt.Figure, text: str) -> None:
    """Apply the figure-level title with a small top margin so it does not clip."""
    fig.suptitle(text, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))


def _save_and_close(fig: plt.Figure, output_paths: Iterable[Path]) -> None:
    """Persist the figure to every output path then release the matplotlib handle."""
    for path in output_paths:
        fig.savefig(path)
    plt.close(fig)


def _render_horizon_panel(
    ax: plt.Axes,
    egnn_bin: EncounterBinReport,
    hgnn_bin: EncounterBinReport,
    baseline_bin: EncounterBinReport,
    *,
    bin_name: str,
    metric: _HorizonMetric,
    show_x_label: bool,
) -> None:
    """Plot all three model series for one bin over categorical horizons (MSE linear, drift log)."""
    x = np.arange(len(HORIZON_ANCHORS))
    egnn_y = _horizon_values(egnn_bin, metric=metric, label=f"EGNN bin={bin_name!r}")
    hgnn_y = _horizon_values(hgnn_bin, metric=metric, label=f"HGNN bin={bin_name!r}")
    baseline_y = _horizon_values(baseline_bin, metric=metric, label=f"baseline bin={bin_name!r}")

    ax.plot(
        x,
        egnn_y,
        color=EGNN_COLOR,
        linewidth=2.2,
        marker=EGNN_MARKER,
        markersize=7,
        label="EGNN",
    )
    ax.plot(
        x,
        hgnn_y,
        color=HGNN_COLOR,
        linewidth=2.2,
        marker=HGNN_MARKER,
        markersize=7,
        label="HGNN",
    )
    ax.plot(
        x,
        baseline_y,
        color=BASELINE_COLOR,
        linewidth=1.8,
        linestyle=BASELINE_LINESTYLE,
        marker=BASELINE_MARKER,
        markersize=6,
        label="constant velocity",
    )
    if metric is _HorizonMetric.POSITION_MSE:
        _annotate_horizon_values(ax, x, egnn_y, color=EGNN_COLOR, y_offset=8, va="bottom")
        _annotate_horizon_values(ax, x, hgnn_y, color=HGNN_COLOR, y_offset=18, va="bottom")
        _annotate_horizon_values(
            ax,
            x,
            baseline_y,
            color=BASELINE_COLOR,
            y_offset=28,
            va="bottom",
        )
    else:
        _annotate_horizon_values(ax, x, egnn_y, color=EGNN_COLOR, y_offset=9, va="bottom")
        _annotate_horizon_values(ax, x, hgnn_y, color=HGNN_COLOR, y_offset=-12, va="top")
        _annotate_horizon_values(
            ax,
            x,
            baseline_y,
            color=BASELINE_COLOR,
            y_offset=17,
            va="bottom",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in HORIZON_ANCHORS])
    ax.set_title(f"{bin_name} (n={egnn_bin.count})")
    if metric is _HorizonMetric.ENERGY_DRIFT:
        ax.set_yscale("log")
        ax.axhline(_DRIFT_REFERENCE_LEVEL, color="grey", linestyle="--", linewidth=1.0, alpha=0.45)
    else:
        ax.set_ylim(bottom=0.0)
    _pad_horizon_panel_y_limits(ax, (egnn_y, hgnn_y, baseline_y), metric=metric)
    if show_x_label:
        ax.set_xlabel("rollout horizon")
        ax.tick_params(axis="x", labelbottom=True)
    else:
        ax.tick_params(axis="x", labelbottom=False)


def _horizon_metric_y_label(metric: _HorizonMetric) -> str:
    """Figure-level y label for the model-column horizon grids."""
    if metric is _HorizonMetric.ENERGY_DRIFT:
        return r"median $|\Delta E / E_0|$"
    return "median position MSE"


def _pad_horizon_panel_y_limits(
    ax: plt.Axes,
    series: tuple[np.ndarray, ...],
    *,
    metric: _HorizonMetric,
) -> None:
    """Give value annotations breathing room within a horizon panel."""
    finite_positive = [y for arr in series for y in arr if np.isfinite(y) and y > 0]
    if not finite_positive:
        return
    y_min = min(finite_positive)
    y_max = max(finite_positive)
    if metric is _HorizonMetric.ENERGY_DRIFT:
        ax.set_ylim(y_min / 2.0, y_max * 3.0)
    else:
        ax.set_ylim(0.0, y_max * 1.35)


def _annotate_horizon_values(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    y_offset: int,
    va: str,
) -> None:
    """Place compact numeric labels near each finite horizon marker."""
    for x_i, y_i in zip(x, y, strict=True):
        if not np.isfinite(y_i) or y_i <= 0:
            continue
        ax.annotate(
            _format_horizon_value(float(y_i)),
            xy=(x_i, y_i),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            va=va,
            color=color,
            fontsize=8.5,
            clip_on=False,
        )


def _format_horizon_value(value: float) -> str:
    """Return a compact label for values spanning tiny MSEs to huge drifts."""
    if not np.isfinite(value):
        return "n/a"
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude < 1e-3 or magnitude >= 1e4:
        return _compact_scientific(value)
    if magnitude < 1e-2:
        return f"{value:.4f}"
    if magnitude < 1.0:
        return f"{value:.3f}"
    return f"{value:.2f}"


def _compact_scientific(value: float) -> str:
    """Format scientific notation without visual clutter like e-04 or 7.0e."""
    mantissa, exponent = f"{value:.1e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def _horizon_values(
    bin_report: EncounterBinReport,
    *,
    metric: _HorizonMetric,
    label: str,
) -> np.ndarray:
    """Pluck the HORIZON_ANCHORS values out of a per-bin curve (exact match; None -> NaN)."""
    if metric is _HorizonMetric.POSITION_MSE:
        step = list(bin_report.rollout.curves.step)
        median = list(bin_report.rollout.curves.position_mse.median)
    else:
        step = list(bin_report.energy.physical.curves.step)
        median = list(bin_report.energy.physical.curves.median)
    return np.array(
        [
            _value_at_horizon(step, median, h, label=f"{label} {metric.value}")
            for h in HORIZON_ANCHORS
        ],
        dtype=float,
    )


def _value_at_horizon(
    curves_step: list[int],
    curves_values: list[float | None],
    horizon: int,
    *,
    label: str,
) -> float:
    """Return the curve value at the exact `horizon` step, or raise (named via `label`)."""
    try:
        idx = curves_step.index(horizon)
    except ValueError as exc:
        msg = (
            f"horizon {horizon} not present in {label} curves.step (available steps: {curves_step})"
        )
        raise ValueError(msg) from exc
    return _nan_if_none(curves_values[idx])


def _curve_to_array(values: list[float | None]) -> np.ndarray:
    """Convert a metric curve (list[float | None]) to a numpy array with NaN for None."""
    return np.array([_nan_if_none(v) for v in values], dtype=float)


def _nan_if_none(value: float | None) -> float:
    """Map None -> NaN so plotting libraries treat the entry as a gap, not zero."""
    return float("nan") if value is None else value
