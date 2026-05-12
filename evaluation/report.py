"""Comparison report generator: turn three metrics.json files into thesis artifacts.

The report layer is a pure post-processor on the JSON produced by
`evaluation.evaluate` and `evaluation.evaluate_baseline`. It loads three
`EvaluationReport`s (EGNN, HGNN, constant-velocity baseline), validates
that they share the same encounter-bin layout and rollout horizon, and
orchestrates figure + table + markdown generation.

No model loading, no rollout recomputation, no GPU. This keeps the report
fast to iterate on and decoupled from the evaluator's expensive path.

References:
    - Source schema: evaluation/_types.py (EvaluationReport, EncounterBinsReport)
    - Producers:     evaluation/evaluate.py, evaluation/evaluate_baseline.py
    - Figures:       evaluation/report_figures.py
    - Tables:        evaluation/report_tables.py
"""

import argparse
import json
from pathlib import Path

from evaluation._types import EncounterBinReport, EncounterBinsReport, EvaluationReport
from evaluation.evaluate_chunked import (
    largest_usable_k,
    read_endpoint_rows,
)
from evaluation.report_figures import (
    HORIZON_ANCHORS,
    plot_energy_drift_presentation,
    plot_horizon_snapshot_by_bin,
    plot_rollout_mse_presentation,
)
from evaluation.report_tables import (
    CHUNKED_ENDPOINTS_CSV_NAME,
    CHUNKED_FIGURE_NAME,
    CHUNKED_MARKDOWN_NAME,
    CHUNKED_SUMMARY_CSV_NAME,
    ChunkedReportSection,
    write_key_timestep_summary_csv,
    write_per_bin_summary_csv,
    write_report_markdown,
)
from utils import get_logger

logger = get_logger(__name__)


class Reporter:
    """Generate a comparison report from EGNN, HGNN, and baseline evaluation metrics."""

    def __init__(
        self,
        egnn_path: Path,
        hgnn_path: Path,
        baseline_path: Path,
        output_dir: Path,
    ) -> None:
        """Store input paths and derive figures/tables sub-directories under output_dir."""
        self.egnn_path = egnn_path
        self.hgnn_path = hgnn_path
        self.baseline_path = baseline_path
        self.output_dir = output_dir
        self.figures_dir = output_dir / "figures"
        self.tables_dir = output_dir / "tables"

    def run(self) -> None:
        """Load reports, validate, emit the output skeleton, and write tables + markdown."""
        egnn, hgnn, baseline = self._load_reports()
        self._validate_compatible(egnn, hgnn, baseline)
        self._setup_output()

        bin_names = (
            [b.name for b in egnn.encounter_bins.bins] if egnn.encounter_bins is not None else []
        )
        logger.info(
            "loaded reports | egnn epoch=%s val=%s | hgnn epoch=%s val=%s "
            "| baseline=%s | bins=%s | rollout_steps=%d",
            egnn.metadata.checkpoint_epoch,
            _fmt_optional_float(egnn.metadata.checkpoint_val_loss),
            hgnn.metadata.checkpoint_epoch,
            _fmt_optional_float(hgnn.metadata.checkpoint_val_loss),
            baseline.metadata.run_id,
            bin_names,
            len(egnn.rollout.curves.step),
        )

        tables = self._write_tables(egnn, hgnn, baseline)
        figures = self._write_figures(egnn, hgnn, baseline)
        chunked = self._load_chunked_section()
        write_report_markdown(
            egnn,
            hgnn,
            baseline,
            self.output_dir,
            figures=figures,
            tables=tables,
            chunked=chunked,
        )
        logger.info(
            "wrote %d figures, %d tables, and report.md to %s (chunked: %s)",
            len(figures),
            len(tables),
            self.output_dir,
            "yes" if chunked is not None else "no",
        )

    def _write_tables(
        self,
        egnn: EvaluationReport,
        hgnn: EvaluationReport,
        baseline: EvaluationReport,
    ) -> list[str]:
        """Write all CSV tables and return their basenames (for the markdown index)."""
        per_bin_name = "per_bin_summary.csv"
        key_step_name = "key_timestep_summary.csv"
        write_per_bin_summary_csv(egnn, hgnn, baseline, self.tables_dir / per_bin_name)
        write_key_timestep_summary_csv(egnn, hgnn, baseline, self.tables_dir / key_step_name)
        return [per_bin_name, key_step_name]

    def _write_figures(
        self,
        egnn: EvaluationReport,
        hgnn: EvaluationReport,
        baseline: EvaluationReport,
    ) -> list[str]:
        """Render every presentation figure, returning the artifact basenames."""
        self._clear_figure_artifacts()
        rollout_png = self.figures_dir / "01_rollout_position_mse_by_bin.png"
        rollout_pdf = self.figures_dir / "01_rollout_position_mse_by_bin.pdf"
        plot_rollout_mse_presentation(egnn, hgnn, baseline, (rollout_png, rollout_pdf))

        drift_png = self.figures_dir / "02_energy_drift_by_bin.png"
        drift_pdf = self.figures_dir / "02_energy_drift_by_bin.pdf"
        plot_energy_drift_presentation(egnn, hgnn, baseline, (drift_png, drift_pdf))

        figure_names = [
            rollout_png.name,
            rollout_pdf.name,
            drift_png.name,
            drift_pdf.name,
        ]

        for idx, horizon in enumerate(HORIZON_ANCHORS, start=3):
            horizon_png = self.figures_dir / f"{idx:02d}_h{horizon}_by_bin.png"
            horizon_pdf = self.figures_dir / f"{idx:02d}_h{horizon}_by_bin.pdf"
            plot_horizon_snapshot_by_bin(
                egnn,
                hgnn,
                baseline,
                (horizon_png, horizon_pdf),
                horizon=horizon,
            )
            figure_names.extend([horizon_png.name, horizon_pdf.name])

        return figure_names

    def _clear_figure_artifacts(self) -> None:
        """Remove stale PNG/PDF figures so re-runs mirror the current manifest."""
        for path in self.figures_dir.glob("*"):
            if path.suffix in {".png", ".pdf"}:
                path.unlink()

    def _load_reports(
        self,
    ) -> tuple[EvaluationReport, EvaluationReport, EvaluationReport]:
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
        """Reject report triples that cannot be plotted side-by-side.

        Each contract is a separate method so a failure pinpoints the
        exact mismatch. Order matters: structural prerequisites first
        (bins block present, bin layout identical), then identity-of-test-
        population (same n_traj/n_frames/n_particles, same per-bin counts
        and d_min summaries), then step-axis equality used by the figures.
        """
        self._require_bins_present(egnn, hgnn, baseline)
        self._require_baseline_is_constant_velocity(baseline)
        self._require_matching_bin_definitions(egnn, hgnn, baseline)
        self._require_matching_test_population(egnn, hgnn, baseline)
        self._require_matching_rollout_steps(egnn, hgnn, baseline)
        self._require_matching_per_bin_step_arrays(egnn, hgnn, baseline)

    def _require_bins_present(
        self,
        egnn: EvaluationReport,
        hgnn: EvaluationReport,
        baseline: EvaluationReport,
    ) -> None:
        """Fail fast when any report is missing an `encounter_bins` block."""
        missing = []
        if egnn.encounter_bins is None:
            missing.append(f"egnn ({self.egnn_path})")
        if hgnn.encounter_bins is None:
            missing.append(f"hgnn ({self.hgnn_path})")
        if baseline.encounter_bins is None:
            missing.append(f"baseline ({self.baseline_path})")
        if missing:
            msg = (
                "comparison report requires stratified evaluations on all models; "
                f"encounter_bins missing in: {', '.join(missing)}"
            )
            raise ValueError(msg)

    def _require_baseline_is_constant_velocity(self, baseline: EvaluationReport) -> None:
        """Reject any baseline other than the official constant-velocity one.

        The legend and markdown headline label the third curve "constant
        velocity" verbatim, so a mean_state or persistence report would
        produce a misleading artifact. Caught at the orchestrator boundary
        because the producer (evaluate_baseline.py) tags model_name with
        the exact baseline kind it ran.
        """
        actual = baseline.metadata.model_name
        expected = "baseline_constant_velocity"
        if actual != expected:
            msg = (
                f"baseline report must come from the {expected!r} baseline; "
                f"got model_name={actual!r} from {self.baseline_path}"
            )
            raise ValueError(msg)

    def _require_matching_bin_definitions(
        self,
        egnn: EvaluationReport,
        hgnn: EvaluationReport,
        baseline: EvaluationReport,
    ) -> None:
        """Fail fast when the three reports disagree on bin layout."""
        egnn_sig = _bins_signature(egnn)
        hgnn_sig = _bins_signature(hgnn)
        baseline_sig = _bins_signature(baseline)
        if egnn_sig != hgnn_sig or egnn_sig != baseline_sig:
            msg = (
                "encounter_bins definitions differ between reports; "
                f"egnn={egnn_sig} hgnn={hgnn_sig} baseline={baseline_sig}"
            )
            raise ValueError(msg)

    def _require_matching_test_population(
        self,
        egnn: EvaluationReport,
        hgnn: EvaluationReport,
        baseline: EvaluationReport,
    ) -> None:
        """Reject report triples evaluated against different test populations.

        Bin edges can be identical across regenerated test sets, so a
        layout-only check is not enough. This validator compares the
        observable per-population signals that the evaluator records:
        metadata n_trajectories / n_frames / n_particles, the per-bin
        trajectory count, and the per-bin d_min summary statistics.
        """
        for field in ("n_trajectories", "n_frames", "n_particles"):
            self._require_metadata_field_matches(field, egnn, hgnn, baseline)

        egnn_bins = _require_bins_block(egnn)
        hgnn_bins = _require_bins_block(hgnn)
        baseline_bins = _require_bins_block(baseline)
        for bin_def in egnn_bins.bins:
            egnn_bin = egnn_bins.by_name[bin_def.name]
            hgnn_bin = hgnn_bins.by_name[bin_def.name]
            baseline_bin = baseline_bins.by_name[bin_def.name]
            self._require_bin_count_matches(bin_def.name, egnn_bin, hgnn_bin, baseline_bin)
            self._require_bin_d_min_matches(bin_def.name, egnn_bin, hgnn_bin, baseline_bin)

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
        unique = set(values.values())
        if len(unique) > 1:
            msg = (
                f"metadata.{field} differs between reports; "
                f"egnn={values['egnn']} hgnn={values['hgnn']} baseline={values['baseline']}"
            )
            raise ValueError(msg)

    def _require_bin_count_matches(
        self,
        bin_name: str,
        egnn_bin: EncounterBinReport,
        hgnn_bin: EncounterBinReport,
        baseline_bin: EncounterBinReport,
    ) -> None:
        """Require all three reports observe the same per-bin trajectory count."""
        counts = (egnn_bin.count, hgnn_bin.count, baseline_bin.count)
        if len(set(counts)) > 1:
            msg = (
                f"per-bin count differs for {bin_name!r}; "
                f"egnn={counts[0]} hgnn={counts[1]} baseline={counts[2]}"
            )
            raise ValueError(msg)

    def _require_bin_d_min_matches(
        self,
        bin_name: str,
        egnn_bin: EncounterBinReport,
        hgnn_bin: EncounterBinReport,
        baseline_bin: EncounterBinReport,
    ) -> None:
        """Require the d_min summary statistics to match exactly across reports.

        Identity comparison is safe: when the test bundle is the same, the
        evaluator's summary code produces bit-identical floats; any drift
        means the test populations differ, which is what we want to catch.
        """
        egnn_sig = _d_min_signature(egnn_bin)
        hgnn_sig = _d_min_signature(hgnn_bin)
        baseline_sig = _d_min_signature(baseline_bin)
        if egnn_sig != hgnn_sig or egnn_sig != baseline_sig:
            msg = (
                f"per-bin d_min summary differs for {bin_name!r}; "
                f"egnn={egnn_sig} hgnn={hgnn_sig} baseline={baseline_sig}"
            )
            raise ValueError(msg)

    def _require_matching_rollout_steps(
        self,
        egnn: EvaluationReport,
        hgnn: EvaluationReport,
        baseline: EvaluationReport,
    ) -> None:
        """Require the top-level rollout step axis is identical, not just same-length.

        Plotting indexes EGNN's step array against HGNN and baseline data;
        a length-only check would let two reports with the same horizon but
        different step values silently misplot.
        """
        egnn_steps = tuple(egnn.rollout.curves.step)
        hgnn_steps = tuple(hgnn.rollout.curves.step)
        baseline_steps = tuple(baseline.rollout.curves.step)
        if egnn_steps != hgnn_steps or egnn_steps != baseline_steps:
            msg = (
                "rollout step values differ across reports; "
                f"egnn={egnn_steps} hgnn={hgnn_steps} baseline={baseline_steps}"
            )
            raise ValueError(msg)

    def _require_matching_per_bin_step_arrays(
        self,
        egnn: EvaluationReport,
        hgnn: EvaluationReport,
        baseline: EvaluationReport,
    ) -> None:
        """Require per-bin rollout and energy step arrays match across reports.

        The figures sample EGNN's per-bin step array for HGNN and baseline
        curves; same-length-different-values would silently misplot, just
        like the top-level case.
        """
        egnn_bins = _require_bins_block(egnn)
        hgnn_bins = _require_bins_block(hgnn)
        baseline_bins = _require_bins_block(baseline)
        for bin_def in egnn_bins.bins:
            egnn_bin = egnn_bins.by_name[bin_def.name]
            hgnn_bin = hgnn_bins.by_name[bin_def.name]
            baseline_bin = baseline_bins.by_name[bin_def.name]
            self._require_step_tuple_matches(
                f"rollout.curves.step in bin {bin_def.name!r}",
                tuple(egnn_bin.rollout.curves.step),
                tuple(hgnn_bin.rollout.curves.step),
                tuple(baseline_bin.rollout.curves.step),
            )
            self._require_step_tuple_matches(
                f"energy.physical.curves.step in bin {bin_def.name!r}",
                tuple(egnn_bin.energy.physical.curves.step),
                tuple(hgnn_bin.energy.physical.curves.step),
                tuple(baseline_bin.energy.physical.curves.step),
            )

    def _require_step_tuple_matches(
        self,
        label: str,
        egnn_steps: tuple[int, ...],
        hgnn_steps: tuple[int, ...],
        baseline_steps: tuple[int, ...],
    ) -> None:
        """Raise a labelled error when the three step tuples disagree."""
        if egnn_steps != hgnn_steps or egnn_steps != baseline_steps:
            msg = (
                f"{label} differs across reports; "
                f"egnn={egnn_steps} hgnn={hgnn_steps} baseline={baseline_steps}"
            )
            raise ValueError(msg)

    def _setup_output(self) -> None:
        """Create the figures/ and tables/ subdirectories under output_dir."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)
        self.tables_dir.mkdir(exist_ok=True)

    def _load_chunked_section(self) -> ChunkedReportSection | None:
        """Detect `<output_dir>/chunked/`; build the report-md section if the full manifest is present.

        The chunked sub-directory is optional. We attach the section only
        when every artifact the section links to is present, so the main
        report never points at a stale or missing chunked figure / CSV /
        markdown. A partial manifest logs a warning and falls back to a
        report without the section. A present-but-malformed endpoints CSV
        causes `read_endpoint_rows` to raise so we fail loudly instead of
        emitting a hollow section.
        """
        chunked_dir = self.output_dir / "chunked"
        required = {
            CHUNKED_ENDPOINTS_CSV_NAME: chunked_dir / CHUNKED_ENDPOINTS_CSV_NAME,
            CHUNKED_SUMMARY_CSV_NAME: chunked_dir / CHUNKED_SUMMARY_CSV_NAME,
            CHUNKED_FIGURE_NAME: chunked_dir / CHUNKED_FIGURE_NAME,
            CHUNKED_MARKDOWN_NAME: chunked_dir / CHUNKED_MARKDOWN_NAME,
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if len(missing) == len(required):
            return None
        if missing:
            logger.warning(
                "chunked sub-directory is partial; skipping the report.md section. "
                "Missing files under %s: %s. Re-run `python -m evaluation.evaluate_chunked` "
                "to regenerate the full manifest.",
                chunked_dir,
                missing,
            )
            return None
        rows = read_endpoint_rows(required[CHUNKED_ENDPOINTS_CSV_NAME])
        return ChunkedReportSection(
            rel_dir="chunked",
            endpoint_rows=rows,
            usable_k=largest_usable_k(rows),
        )


def _load_report(path: Path) -> EvaluationReport:
    """Load a metrics.json file into a typed EvaluationReport."""
    with path.open() as f:
        return EvaluationReport.from_dict(json.load(f))


def _bins_signature(report: EvaluationReport) -> tuple[tuple[str, float, float], ...]:
    """Build a comparable signature for encounter_bins (name, lo, hi tuple per bin)."""
    if report.encounter_bins is None:
        return ()
    return tuple((b.name, b.lo, b.hi) for b in report.encounter_bins.bins)


def _require_bins_block(report: EvaluationReport) -> EncounterBinsReport:
    """Narrow `encounter_bins` out of None; presence is validated upstream."""
    if report.encounter_bins is None:
        msg = "report has no encounter_bins block; orchestrator should have rejected this earlier"
        raise ValueError(msg)
    return report.encounter_bins


def _d_min_signature(
    bin_report: EncounterBinReport,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """Compact tuple representation of a bin's d_min summary for equality checks."""
    summary = bin_report.d_min
    return (summary.mean, summary.median, summary.max, summary.p5, summary.p50)


def _fmt_optional_float(value: float | None) -> str:
    """Format a possibly-None float for log lines."""
    return f"{value:.4f}" if value is not None else "n/a"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a thesis-grade comparison report from three metrics.json files.",
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
        help="Output directory for figures/, tables/, and report.md.",
    )
    args = parser.parse_args()

    Reporter(
        egnn_path=Path(args.egnn),
        hgnn_path=Path(args.hgnn),
        baseline_path=Path(args.baseline),
        output_dir=Path(args.output),
    ).run()
