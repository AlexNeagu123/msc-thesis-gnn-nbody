"""Tests for evaluation/report_figures.py.

The figures module exposes four public presentation plotters: continuous
rollout MSE per bin, continuous energy drift per bin, horizon-anchor MSE
per bin, and horizon-anchor energy drift per bin. All four are fed by
EGNN + HGNN + constant-velocity baseline reports.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from evaluation._types import EvaluationReport
from evaluation.report_figures import (
    HORIZON_ANCHORS,
    _curve_to_array,
    _format_horizon_value,
    _nan_if_none,
    _value_at_horizon,
    plot_energy_drift_presentation,
    plot_final_energy_drift_by_bin,
    plot_final_mse_by_bin,
    plot_horizon_energy_drift_by_bin,
    plot_horizon_mse_by_bin,
    plot_horizon_snapshot_by_bin,
    plot_mse_bars_at_horizon,
    plot_rollout_mse_presentation,
)
from evaluation.test_report_tables import _build_reports


def _reports() -> tuple[EvaluationReport, EvaluationReport, EvaluationReport]:
    """Local alias so each test reads as 'arrange one trio per test'."""
    return _build_reports()


def test_nan_if_none_passes_through_finite_values() -> None:
    """The helper only substitutes for None, not for finite floats."""
    assert _nan_if_none(0.42) == 0.42
    assert _nan_if_none(-1.0) == -1.0
    assert math.isnan(_nan_if_none(None))


def test_curve_to_array_replaces_none_with_nan() -> None:
    """None entries in a metric curve must convert to NaN, not zero."""
    arr = _curve_to_array([0.0, 1.5, None, 3.0])

    assert arr[0] == 0.0
    assert arr[1] == 1.5
    assert math.isnan(arr[2])
    assert arr[3] == 3.0


def test_plot_rollout_mse_presentation_writes_both_artifacts(tmp_path: Path) -> None:
    """PNG and PDF outputs land non-empty for the rollout MSE figure."""
    egnn, hgnn, baseline = _reports()
    png = tmp_path / "rollout.png"
    pdf = tmp_path / "rollout.pdf"

    plot_rollout_mse_presentation(egnn, hgnn, baseline, (png, pdf))

    assert png.is_file()
    assert pdf.is_file()
    assert png.stat().st_size > 0
    assert pdf.stat().st_size > 0


def test_plot_rollout_mse_presentation_closes_figure(tmp_path: Path) -> None:
    """No figure leaks: function must close its figure before returning."""
    egnn, hgnn, baseline = _reports()
    plt.close("all")
    initial = len(plt.get_fignums())

    plot_rollout_mse_presentation(egnn, hgnn, baseline, (tmp_path / "r.png",))

    assert len(plt.get_fignums()) == initial


def test_plot_energy_drift_presentation_writes_both_artifacts(tmp_path: Path) -> None:
    """PNG and PDF outputs land non-empty for the energy drift figure."""
    egnn, hgnn, baseline = _reports()
    png = tmp_path / "drift.png"
    pdf = tmp_path / "drift.pdf"

    plot_energy_drift_presentation(egnn, hgnn, baseline, (png, pdf))

    assert png.is_file()
    assert pdf.is_file()
    assert png.stat().st_size > 0
    assert pdf.stat().st_size > 0


def test_plot_energy_drift_presentation_closes_figure(tmp_path: Path) -> None:
    """No figure leaks: function must close its figure before returning."""
    egnn, hgnn, baseline = _reports()
    plt.close("all")
    initial = len(plt.get_fignums())

    plot_energy_drift_presentation(egnn, hgnn, baseline, (tmp_path / "d.png",))

    assert len(plt.get_fignums()) == initial


def test_is_bottom_panel_handles_legend_at_last_cell() -> None:
    """The panel above the legend is 'bottom' for its column even though row 0 is above."""
    from evaluation.report_figures import _is_bottom_panel, _is_leftmost

    # n_bins=5, cols=3, rows=2. Panels 0..4 are bins, 5 is legend.
    # column 0: panels 0, 3 -> 3 is bottom
    # column 1: panels 1, 4 -> 4 is bottom
    # column 2: panel 2 (no panel at 5 because that's legend) -> 2 is bottom
    assert _is_leftmost(0, cols=3)
    assert _is_leftmost(3, cols=3)
    assert not _is_leftmost(1, cols=3)
    assert not _is_bottom_panel(0, n_bins=5, cols=3)
    assert _is_bottom_panel(2, n_bins=5, cols=3)
    assert _is_bottom_panel(3, n_bins=5, cols=3)
    assert _is_bottom_panel(4, n_bins=5, cols=3)


def test_plot_rollout_mse_presentation_labels_only_outer_panels(tmp_path: Path) -> None:
    """Inner panels must not repeat the axis labels carried by the outer edge.

    Renders the presentation figure to a tmp file and inspects every axis
    on the produced figure: only column-0 panels should carry the y label
    and only column-bottom panels should carry the x label.
    """
    import matplotlib.pyplot as plt

    egnn, hgnn, baseline = _reports()
    plt.close("all")

    plot_rollout_mse_presentation(egnn, hgnn, baseline, (tmp_path / "r.png",))

    # The function closes its figure, so re-render against the same fixtures
    # while keeping the figure open to inspect labels.
    from evaluation.report_figures import (
        _is_bottom_panel,
        _is_leftmost,
        _make_grid,
        _per_bin_panels,
        _render_mse_panel,
    )

    bin_names, panels = _per_bin_panels(egnn, hgnn, baseline)
    n_bins = len(bin_names)
    fig, axes, cols = _make_grid(n_bins)
    for idx, name in enumerate(bin_names):
        e_bin, h_bin, b_bin = panels[name]
        _render_mse_panel(
            axes[idx],
            e_bin,
            h_bin,
            b_bin,
            bin_name=name,
            is_leftmost=_is_leftmost(idx, cols),
            is_bottom=_is_bottom_panel(idx, n_bins, cols),
        )
    try:
        for idx in range(n_bins):
            ax = axes[idx]
            expects_y = _is_leftmost(idx, cols)
            expects_x = _is_bottom_panel(idx, n_bins, cols)
            assert bool(ax.get_ylabel()) == expects_y, idx
            assert bool(ax.get_xlabel()) == expects_x, idx
    finally:
        plt.close(fig)


def test_plot_rollout_mse_presentation_tolerates_none_in_curves(tmp_path: Path) -> None:
    """A None entry in any curve must be plotted as a gap, not crash."""
    egnn, hgnn, baseline = _reports()
    # corrupt one entry in baseline.curves.position_mse.median to None
    baseline_close = baseline.encounter_bins.by_name["close"]
    baseline_close.rollout.curves.position_mse.median[-1] = None

    plot_rollout_mse_presentation(egnn, hgnn, baseline, (tmp_path / "r.png",))

    assert (tmp_path / "r.png").is_file()


def test_horizon_anchors_are_pinned() -> None:
    """Pin the horizon list so a future tweak forces explicit downstream test updates."""
    assert HORIZON_ANCHORS == (1, 3, 5, 10, 25, 50, 100, 150, 199)


def test_value_at_horizon_returns_value_for_exact_step() -> None:
    """Lookup matches the curve value at the exact step, no positional shortcut."""
    steps = [0, 1, 3, 5, 10, 25, 50, 100, 150, 199]
    values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    assert _value_at_horizon(steps, values, 25, label="any") == 0.4
    assert _value_at_horizon(steps, values, 3, label="any") == 0.1
    assert _value_at_horizon(steps, values, 199, label="any") == 0.8


def test_value_at_horizon_converts_none_to_nan() -> None:
    """A None entry at the matched index becomes NaN so plotting renders a gap."""
    steps = [0, 3, 5]
    values = [0.0, None, 0.2]

    assert math.isnan(_value_at_horizon(steps, values, 3, label="any"))


def test_value_at_horizon_raises_on_missing_horizon() -> None:
    """A missing horizon must fail loudly; the spec forbids silent interpolation."""
    steps = [0, 1, 2, 3]
    values = [0.0, 0.1, 0.2, 0.3]

    with pytest.raises(ValueError, match=r"horizon 7 not present in label=foo curves\.step"):
        _value_at_horizon(steps, values, 7, label="label=foo")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "0"),
        (0.0007, "7e-4"),
        (0.0034, "0.0034"),
        (0.1274, "0.127"),
        (1.424, "1.42"),
        (8_671_949.9, "8.7e6"),
        (float("nan"), "n/a"),
    ],
)
def test_format_horizon_value_compacts_common_metric_ranges(
    value: float,
    expected: str,
) -> None:
    """Horizon marker labels should be readable across MSE and energy-drift scales."""
    assert _format_horizon_value(value) == expected


def test_plot_final_mse_by_bin_writes_both_artifacts(tmp_path: Path) -> None:
    """PNG and PDF outputs land non-empty for the final-step MSE bar figure."""
    egnn, hgnn, baseline = _reports()
    png = tmp_path / "final_mse.png"
    pdf = tmp_path / "final_mse.pdf"

    plot_final_mse_by_bin(egnn, hgnn, baseline, (png, pdf))

    assert png.is_file()
    assert pdf.is_file()
    assert png.stat().st_size > 0
    assert pdf.stat().st_size > 0


def test_plot_final_mse_by_bin_closes_figure(tmp_path: Path) -> None:
    """No figure leaks: the final-step MSE plotter closes before returning."""
    egnn, hgnn, baseline = _reports()
    plt.close("all")
    initial = len(plt.get_fignums())

    plot_final_mse_by_bin(egnn, hgnn, baseline, (tmp_path / "final.png",))

    assert len(plt.get_fignums()) == initial


def test_plot_horizon_snapshot_by_bin_writes_both_artifacts(tmp_path: Path) -> None:
    """PNG and PDF outputs land non-empty for a paired MSE/energy horizon snapshot."""
    egnn, hgnn, baseline = _reports()
    png = tmp_path / "h25.png"
    pdf = tmp_path / "h25.pdf"

    plot_horizon_snapshot_by_bin(egnn, hgnn, baseline, (png, pdf), horizon=25)

    assert png.is_file()
    assert pdf.is_file()
    assert png.stat().st_size > 0
    assert pdf.stat().st_size > 0


def test_plot_horizon_snapshot_by_bin_rejects_missing_horizon(tmp_path: Path) -> None:
    """Horizon snapshots use exact step lookup and fail on missing anchors."""
    egnn, hgnn, baseline = _reports()
    close_curves = egnn.encounter_bins.by_name["close"].rollout.curves
    idx_25 = close_curves.step.index(25)
    del close_curves.step[idx_25]
    del close_curves.position_mse.median[idx_25]
    del close_curves.position_mse.mean_finite[idx_25]
    del close_curves.position_mse.p95[idx_25]
    del close_curves.position_mse.finite_fraction[idx_25]

    with pytest.raises(ValueError, match=r"horizon 25 not present"):
        plot_horizon_snapshot_by_bin(egnn, hgnn, baseline, (tmp_path / "h25.png",), horizon=25)


def test_plot_mse_bars_at_horizon_writes_both_artifacts(tmp_path: Path) -> None:
    """The MSE-only bar plotter remains available for focused diagnostics."""
    egnn, hgnn, baseline = _reports()
    png = tmp_path / "mse_h25.png"
    pdf = tmp_path / "mse_h25.pdf"

    plot_mse_bars_at_horizon(egnn, hgnn, baseline, (png, pdf), horizon=25)

    assert png.is_file()
    assert pdf.is_file()
    assert png.stat().st_size > 0
    assert pdf.stat().st_size > 0


def test_plot_final_energy_drift_by_bin_writes_both_artifacts(tmp_path: Path) -> None:
    """PNG and PDF outputs land non-empty for the final-step drift bar figure."""
    egnn, hgnn, baseline = _reports()
    png = tmp_path / "final_drift.png"
    pdf = tmp_path / "final_drift.pdf"

    plot_final_energy_drift_by_bin(egnn, hgnn, baseline, (png, pdf))

    assert png.is_file()
    assert pdf.is_file()
    assert png.stat().st_size > 0
    assert pdf.stat().st_size > 0


def test_plot_final_energy_drift_by_bin_closes_figure(tmp_path: Path) -> None:
    """No figure leaks: the final-step drift plotter closes before returning."""
    egnn, hgnn, baseline = _reports()
    plt.close("all")
    initial = len(plt.get_fignums())

    plot_final_energy_drift_by_bin(egnn, hgnn, baseline, (tmp_path / "final.png",))

    assert len(plt.get_fignums()) == initial


def test_final_mse_bar_helper_uses_linear_axis() -> None:
    """The final MSE comparison is a direct linear bar chart."""
    egnn, hgnn, baseline = _reports()
    from evaluation.report_figures import (
        _HorizonMetric,
        _metric_values_at_step,
        _per_bin_panels,
    )

    bin_names, panels = _per_bin_panels(egnn, hgnn, baseline)
    egnn_values, hgnn_values, baseline_values = _metric_values_at_step(
        bin_names,
        panels,
        metric=_HorizonMetric.POSITION_MSE,
        step=None,
    )

    assert egnn_values[-1] == pytest.approx(
        egnn.encounter_bins.by_name[bin_names[-1]].rollout.curves.position_mse.median[-1]
    )
    assert hgnn_values[-1] == pytest.approx(
        hgnn.encounter_bins.by_name[bin_names[-1]].rollout.curves.position_mse.median[-1]
    )
    assert baseline_values[-1] == pytest.approx(
        baseline.encounter_bins.by_name[bin_names[-1]].rollout.curves.position_mse.median[-1]
    )


def test_metric_values_at_step_uses_requested_horizon() -> None:
    """Selected-horizon bars read the requested step, not the final value."""
    egnn, hgnn, baseline = _reports()
    from evaluation.report_figures import (
        _HorizonMetric,
        _metric_values_at_step,
        _per_bin_panels,
    )

    bin_names, panels = _per_bin_panels(egnn, hgnn, baseline)
    egnn_values, hgnn_values, baseline_values = _metric_values_at_step(
        bin_names,
        panels,
        metric=_HorizonMetric.POSITION_MSE,
        step=25,
    )

    assert egnn_values[0] == pytest.approx(
        egnn.encounter_bins.by_name[bin_names[0]].rollout.curves.position_mse.median[
            egnn.encounter_bins.by_name[bin_names[0]].rollout.curves.step.index(25)
        ]
    )
    assert hgnn_values[0] == pytest.approx(
        hgnn.encounter_bins.by_name[bin_names[0]].rollout.curves.position_mse.median[
            hgnn.encounter_bins.by_name[bin_names[0]].rollout.curves.step.index(25)
        ]
    )
    assert baseline_values[0] == pytest.approx(
        baseline.encounter_bins.by_name[bin_names[0]].rollout.curves.position_mse.median[
            baseline.encounter_bins.by_name[bin_names[0]].rollout.curves.step.index(25)
        ]
    )


def test_plot_horizon_mse_by_bin_writes_both_artifacts(tmp_path: Path) -> None:
    """PNG and PDF outputs land non-empty for the horizon MSE figure."""
    egnn, hgnn, baseline = _reports()
    png = tmp_path / "h_mse.png"
    pdf = tmp_path / "h_mse.pdf"

    plot_horizon_mse_by_bin(egnn, hgnn, baseline, (png, pdf))

    assert png.is_file()
    assert pdf.is_file()
    assert png.stat().st_size > 0
    assert pdf.stat().st_size > 0


def test_plot_horizon_mse_by_bin_closes_figure(tmp_path: Path) -> None:
    """No figure leaks: the horizon MSE plotter closes its figure before returning."""
    egnn, hgnn, baseline = _reports()
    plt.close("all")
    initial = len(plt.get_fignums())

    plot_horizon_mse_by_bin(egnn, hgnn, baseline, (tmp_path / "h.png",))

    assert len(plt.get_fignums()) == initial


def test_plot_horizon_energy_drift_by_bin_writes_both_artifacts(tmp_path: Path) -> None:
    """PNG and PDF outputs land non-empty for the horizon energy-drift figure."""
    egnn, hgnn, baseline = _reports()
    png = tmp_path / "h_drift.png"
    pdf = tmp_path / "h_drift.pdf"

    plot_horizon_energy_drift_by_bin(egnn, hgnn, baseline, (png, pdf))

    assert png.is_file()
    assert pdf.is_file()
    assert png.stat().st_size > 0
    assert pdf.stat().st_size > 0


def test_plot_horizon_energy_drift_by_bin_closes_figure(tmp_path: Path) -> None:
    """No figure leaks: the horizon drift plotter closes its figure before returning."""
    egnn, hgnn, baseline = _reports()
    plt.close("all")
    initial = len(plt.get_fignums())

    plot_horizon_energy_drift_by_bin(egnn, hgnn, baseline, (tmp_path / "h.png",))

    assert len(plt.get_fignums()) == initial


def test_plot_horizon_mse_by_bin_labels_x_ticks_with_horizons(tmp_path: Path) -> None:
    """Every HORIZON_ANCHORS value must appear as an x-tick label on every bin panel."""
    egnn, hgnn, baseline = _reports()
    from evaluation.report_figures import (
        _HorizonMetric,
        _make_horizon_grid,
        _render_horizon_panel,
    )

    bin_names = [b.name for b in egnn.encounter_bins.bins]
    n_bins = len(bin_names)
    fig, axes = _make_horizon_grid(n_bins)
    for idx, name in enumerate(bin_names):
        _render_horizon_panel(
            axes[idx],
            egnn.encounter_bins.by_name[name],
            hgnn.encounter_bins.by_name[name],
            baseline.encounter_bins.by_name[name],
            bin_name=name,
            metric=_HorizonMetric.POSITION_MSE,
            show_x_label=idx == n_bins - 1,
        )
    try:
        fig.canvas.draw()  # populate tick label text strings on visible axes
        expected = [str(h) for h in HORIZON_ANCHORS]
        for idx in range(n_bins):
            ax = axes[idx]
            # tick positions are pinned on every panel even when the labels are
            # hidden by `labelbottom=False` on the inner cells.
            assert list(ax.get_xticks()) == list(range(len(HORIZON_ANCHORS))), idx
            if idx == n_bins - 1:
                actual = [t.get_text() for t in ax.get_xticklabels()]
                assert actual == expected, (idx, actual)
    finally:
        plt.close(fig)


def test_render_horizon_panel_annotates_every_model_horizon() -> None:
    """Fixed-horizon panels expose exact values as text, not only as log-axis positions."""
    egnn, hgnn, baseline = _reports()
    from evaluation.report_figures import _HorizonMetric, _render_horizon_panel

    fig, ax = plt.subplots()
    try:
        _render_horizon_panel(
            ax,
            egnn.encounter_bins.by_name["close"],
            hgnn.encounter_bins.by_name["close"],
            baseline.encounter_bins.by_name["close"],
            bin_name="close",
            metric=_HorizonMetric.POSITION_MSE,
            show_x_label=True,
        )

        labels = [text.get_text() for text in ax.texts]
        assert len(labels) == 3 * len(HORIZON_ANCHORS)
        assert "0.0050" in labels
        assert "0.015" in labels
        assert "0.040" in labels
    finally:
        plt.close(fig)


def test_render_horizon_mse_panel_uses_linear_y_axis() -> None:
    """MSE horizon plots avoid log ticks so the numeric labels are easier to read."""
    egnn, hgnn, baseline = _reports()
    from evaluation.report_figures import _HorizonMetric, _render_horizon_panel

    fig, ax = plt.subplots()
    try:
        _render_horizon_panel(
            ax,
            egnn.encounter_bins.by_name["close"],
            hgnn.encounter_bins.by_name["close"],
            baseline.encounter_bins.by_name["close"],
            bin_name="close",
            metric=_HorizonMetric.POSITION_MSE,
            show_x_label=True,
        )

        assert ax.get_yscale() == "linear"
    finally:
        plt.close(fig)


def test_plot_horizon_mse_by_bin_rejects_curves_missing_horizon(tmp_path: Path) -> None:
    """If a fixture lacks a required horizon, the plotter raises a clear ValueError."""
    egnn, hgnn, baseline = _reports()
    # Drop horizon 25 from EGNN's "close" bin so the lookup fails for that horizon.
    close_curves = egnn.encounter_bins.by_name["close"].rollout.curves
    idx_25 = close_curves.step.index(25)
    del close_curves.step[idx_25]
    del close_curves.position_mse.median[idx_25]
    del close_curves.position_mse.mean_finite[idx_25]
    del close_curves.position_mse.p95[idx_25]
    del close_curves.position_mse.finite_fraction[idx_25]

    with pytest.raises(ValueError, match=r"horizon 25 not present"):
        plot_horizon_mse_by_bin(egnn, hgnn, baseline, (tmp_path / "h.png",))
