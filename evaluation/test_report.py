"""Tests for the comparison report orchestrator (`evaluation/report.py`).

The orchestrator now consumes three reports (EGNN, HGNN, constant-velocity
baseline) and validates that all three share the same bin layout and
rollout horizon. These tests exercise loading, validation, and the CLI
entrypoint.
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation.report import Reporter, _bins_signature, _load_report
from evaluation.test_report_tables import horizonify
from evaluation.test_types import _hgnn_report_dict, _stratified_report_dict


def _baseline_stratified_report_dict() -> dict:
    """Baseline-shape stratified report sharing the EGNN fixture's bin layout."""
    base = _stratified_report_dict()
    base["metadata"]["model_name"] = "baseline_constant_velocity"
    base["metadata"]["run_id"] = "baseline_constant_velocity"
    base["metadata"]["checkpoint_path"] = None
    return base


def _hgnn_stratified_report_dict() -> dict:
    """HGNN-shape stratified report sharing the same bin layout as the EGNN fixture."""
    egnn = _stratified_report_dict()
    hgnn = _hgnn_report_dict()
    hgnn["encounter_bins"] = copy.deepcopy(egnn["encounter_bins"])
    hgnn["metadata"]["model_name"] = "hgnn"
    hgnn["metadata"]["checkpoint_path"] = "runs/hgnn/x/best.pt"
    return hgnn


def _write_json(path: Path, payload: dict) -> Path:
    """Persist a fixture dict to a JSON file under `path`."""
    path.write_text(json.dumps(payload))
    return path


def _build_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Stage matching trio of horizon-aware metrics.json files plus an empty output dir."""
    egnn_path = _write_json(tmp_path / "egnn.json", horizonify(_stratified_report_dict()))
    hgnn_path = _write_json(tmp_path / "hgnn.json", horizonify(_hgnn_stratified_report_dict()))
    baseline_path = _write_json(
        tmp_path / "baseline.json", horizonify(_baseline_stratified_report_dict())
    )
    output_dir = tmp_path / "report"
    return egnn_path, hgnn_path, baseline_path, output_dir


def _build_reporter(tmp_path: Path) -> Reporter:
    """Create a Reporter against staged matching reports under `tmp_path`."""
    egnn_path, hgnn_path, baseline_path, output_dir = _build_paths(tmp_path)
    return Reporter(
        egnn_path=egnn_path,
        hgnn_path=hgnn_path,
        baseline_path=baseline_path,
        output_dir=output_dir,
    )


def test_load_report_parses_typed_report(tmp_path: Path) -> None:
    """A metrics.json file is decoded into a typed EvaluationReport with encounter_bins."""
    path = _write_json(tmp_path / "egnn.json", _stratified_report_dict())

    report = _load_report(path)

    assert report.metadata.model_name == "egnn"
    assert report.encounter_bins is not None
    assert [b.name for b in report.encounter_bins.bins] == ["extreme", "close", "smooth"]


def test_bins_signature_includes_inf_sentinel_as_float(tmp_path: Path) -> None:
    """The signature collapses bin defs into a comparable tuple, with +inf preserved."""
    path = _write_json(tmp_path / "egnn.json", _stratified_report_dict())
    report = _load_report(path)

    sig = _bins_signature(report)

    assert sig[-1] == ("smooth", 0.05, float("inf"))


def test_run_smoke_against_matching_reports(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Three compatible reports complete the run, creating figures/ and tables/."""
    reporter = _build_reporter(tmp_path)

    with caplog.at_level("INFO", logger="evaluation.report"):
        reporter.run()

    assert reporter.figures_dir.is_dir()
    assert reporter.tables_dir.is_dir()
    messages = [r.getMessage() for r in caplog.records]
    assert any("loaded reports" in m for m in messages), messages


def test_run_writes_horizon_bar_figure_files(tmp_path: Path) -> None:
    """Report emits continuous curves plus paired MSE/energy snapshots per horizon."""
    reporter = _build_reporter(tmp_path)

    reporter.run()

    png_files = sorted(reporter.figures_dir.glob("*.png"))
    pdf_files = sorted(reporter.figures_dir.glob("*.pdf"))
    assert len(png_files) == 11, [p.name for p in png_files]
    assert len(pdf_files) == 11, [p.name for p in pdf_files]
    expected_stems = {
        "01_rollout_position_mse_by_bin",
        "02_energy_drift_by_bin",
        "03_h1_by_bin",
        "04_h3_by_bin",
        "05_h5_by_bin",
        "06_h10_by_bin",
        "07_h25_by_bin",
        "08_h50_by_bin",
        "09_h100_by_bin",
        "10_h150_by_bin",
        "11_h199_by_bin",
    }
    assert {p.stem for p in png_files} == expected_stems
    assert {p.stem for p in pdf_files} == expected_stems


def test_run_rejects_missing_encounter_bins_on_egnn(tmp_path: Path) -> None:
    """An EGNN report without encounter_bins fails fast at validation."""
    from evaluation.test_types import _egnn_report_dict

    egnn_path = _write_json(tmp_path / "egnn.json", _egnn_report_dict())
    hgnn_path = _write_json(tmp_path / "hgnn.json", _hgnn_stratified_report_dict())
    baseline_path = _write_json(tmp_path / "baseline.json", _baseline_stratified_report_dict())
    output_dir = tmp_path / "report"

    with pytest.raises(ValueError, match="encounter_bins missing in: egnn"):
        Reporter(
            egnn_path=egnn_path,
            hgnn_path=hgnn_path,
            baseline_path=baseline_path,
            output_dir=output_dir,
        ).run()


def test_run_rejects_missing_encounter_bins_on_hgnn(tmp_path: Path) -> None:
    """An HGNN report without encounter_bins fails fast at validation."""
    egnn_path = _write_json(tmp_path / "egnn.json", _stratified_report_dict())
    hgnn_path = _write_json(tmp_path / "hgnn.json", _hgnn_report_dict())
    baseline_path = _write_json(tmp_path / "baseline.json", _baseline_stratified_report_dict())
    output_dir = tmp_path / "report"

    with pytest.raises(ValueError, match="encounter_bins missing in: hgnn"):
        Reporter(
            egnn_path=egnn_path,
            hgnn_path=hgnn_path,
            baseline_path=baseline_path,
            output_dir=output_dir,
        ).run()


def test_run_rejects_missing_encounter_bins_on_baseline(tmp_path: Path) -> None:
    """A baseline report without encounter_bins fails fast at validation."""
    from evaluation.test_types import _egnn_report_dict

    egnn_path = _write_json(tmp_path / "egnn.json", _stratified_report_dict())
    hgnn_path = _write_json(tmp_path / "hgnn.json", _hgnn_stratified_report_dict())
    unstratified_baseline = _egnn_report_dict()
    unstratified_baseline["metadata"]["model_name"] = "baseline_constant_velocity"
    baseline_path = _write_json(tmp_path / "baseline.json", unstratified_baseline)
    output_dir = tmp_path / "report"

    with pytest.raises(ValueError, match="encounter_bins missing in: baseline"):
        Reporter(
            egnn_path=egnn_path,
            hgnn_path=hgnn_path,
            baseline_path=baseline_path,
            output_dir=output_dir,
        ).run()


def test_run_rejects_mismatched_bin_definitions(tmp_path: Path) -> None:
    """Differing bin lo/hi between any two reports must be a hard error."""
    egnn_dict = _stratified_report_dict()
    hgnn_dict = _hgnn_stratified_report_dict()
    baseline_dict = _baseline_stratified_report_dict()
    hgnn_dict["encounter_bins"]["bins"][0]["hi"] = 0.02  # was 0.01

    egnn_path = _write_json(tmp_path / "egnn.json", egnn_dict)
    hgnn_path = _write_json(tmp_path / "hgnn.json", hgnn_dict)
    baseline_path = _write_json(tmp_path / "baseline.json", baseline_dict)
    output_dir = tmp_path / "report"

    with pytest.raises(ValueError, match="encounter_bins definitions differ"):
        Reporter(
            egnn_path=egnn_path,
            hgnn_path=hgnn_path,
            baseline_path=baseline_path,
            output_dir=output_dir,
        ).run()


def test_run_rejects_mismatched_rollout_step_lengths(tmp_path: Path) -> None:
    """Any disagreement on rollout step axis (length) must fail fast."""
    egnn_dict = _stratified_report_dict()
    hgnn_dict = _hgnn_stratified_report_dict()
    baseline_dict = _baseline_stratified_report_dict()
    baseline_dict["rollout"]["curves"]["step"] = [0, 1, 2]  # was [0, 1, 2, 3]

    egnn_path = _write_json(tmp_path / "egnn.json", egnn_dict)
    hgnn_path = _write_json(tmp_path / "hgnn.json", hgnn_dict)
    baseline_path = _write_json(tmp_path / "baseline.json", baseline_dict)
    output_dir = tmp_path / "report"

    with pytest.raises(ValueError, match="rollout step values differ"):
        Reporter(
            egnn_path=egnn_path,
            hgnn_path=hgnn_path,
            baseline_path=baseline_path,
            output_dir=output_dir,
        ).run()


def test_run_rejects_mismatched_rollout_step_values(tmp_path: Path) -> None:
    """Same-length step arrays with different values must still fail fast.

    Without exact comparison the figures would index EGNN's step array
    against HGNN/baseline curve data, silently misplotting.
    """
    egnn_dict = _stratified_report_dict()
    hgnn_dict = _hgnn_stratified_report_dict()
    baseline_dict = _baseline_stratified_report_dict()
    baseline_dict["rollout"]["curves"]["step"] = [0, 1, 2, 9]  # same length, different values

    egnn_path = _write_json(tmp_path / "egnn.json", egnn_dict)
    hgnn_path = _write_json(tmp_path / "hgnn.json", hgnn_dict)
    baseline_path = _write_json(tmp_path / "baseline.json", baseline_dict)
    output_dir = tmp_path / "report"

    with pytest.raises(ValueError, match="rollout step values differ"):
        Reporter(
            egnn_path=egnn_path,
            hgnn_path=hgnn_path,
            baseline_path=baseline_path,
            output_dir=output_dir,
        ).run()


def test_run_rejects_per_bin_rollout_step_mismatch(tmp_path: Path) -> None:
    """A per-bin rollout step axis must also match across reports."""
    egnn_dict = _stratified_report_dict()
    hgnn_dict = _hgnn_stratified_report_dict()
    baseline_dict = _baseline_stratified_report_dict()
    # Per-bin curves are deep-copies of the top-level ones; mutate just one bin's array.
    bin_block = baseline_dict["encounter_bins"]["by_name"]["close"]
    bin_block["rollout"] = copy.deepcopy(bin_block["rollout"])
    bin_block["rollout"]["curves"]["step"] = [0, 1, 2, 9]

    egnn_path = _write_json(tmp_path / "egnn.json", egnn_dict)
    hgnn_path = _write_json(tmp_path / "hgnn.json", hgnn_dict)
    baseline_path = _write_json(tmp_path / "baseline.json", baseline_dict)
    output_dir = tmp_path / "report"

    with pytest.raises(ValueError, match=r"rollout.curves.step in bin 'close' differs"):
        Reporter(
            egnn_path=egnn_path,
            hgnn_path=hgnn_path,
            baseline_path=baseline_path,
            output_dir=output_dir,
        ).run()


def test_run_rejects_per_bin_energy_step_mismatch(tmp_path: Path) -> None:
    """A per-bin energy.physical step axis must also match across reports."""
    egnn_dict = _stratified_report_dict()
    hgnn_dict = _hgnn_stratified_report_dict()
    baseline_dict = _baseline_stratified_report_dict()
    bin_block = baseline_dict["encounter_bins"]["by_name"]["close"]
    bin_block["energy"] = copy.deepcopy(bin_block["energy"])
    bin_block["energy"]["physical"]["curves"]["step"] = [9]  # was [0]

    egnn_path = _write_json(tmp_path / "egnn.json", egnn_dict)
    hgnn_path = _write_json(tmp_path / "hgnn.json", hgnn_dict)
    baseline_path = _write_json(tmp_path / "baseline.json", baseline_dict)
    output_dir = tmp_path / "report"

    with pytest.raises(ValueError, match=r"energy.physical.curves.step in bin 'close' differs"):
        Reporter(
            egnn_path=egnn_path,
            hgnn_path=hgnn_path,
            baseline_path=baseline_path,
            output_dir=output_dir,
        ).run()


def test_run_rejects_baseline_other_than_constant_velocity(tmp_path: Path) -> None:
    """Only the official constant-velocity baseline is allowed; legend would lie otherwise."""
    egnn_dict = _stratified_report_dict()
    hgnn_dict = _hgnn_stratified_report_dict()
    baseline_dict = _baseline_stratified_report_dict()
    baseline_dict["metadata"]["model_name"] = "baseline_mean_state"

    egnn_path = _write_json(tmp_path / "egnn.json", egnn_dict)
    hgnn_path = _write_json(tmp_path / "hgnn.json", hgnn_dict)
    baseline_path = _write_json(tmp_path / "baseline.json", baseline_dict)
    output_dir = tmp_path / "report"

    with pytest.raises(ValueError, match="baseline_constant_velocity"):
        Reporter(
            egnn_path=egnn_path,
            hgnn_path=hgnn_path,
            baseline_path=baseline_path,
            output_dir=output_dir,
        ).run()


def test_run_rejects_n_trajectories_mismatch(tmp_path: Path) -> None:
    """Different n_trajectories across reports means they evaluated different test sets."""
    egnn_dict = _stratified_report_dict()
    hgnn_dict = _hgnn_stratified_report_dict()
    baseline_dict = _baseline_stratified_report_dict()
    baseline_dict["metadata"]["n_trajectories"] = baseline_dict["metadata"]["n_trajectories"] + 1

    egnn_path = _write_json(tmp_path / "egnn.json", egnn_dict)
    hgnn_path = _write_json(tmp_path / "hgnn.json", hgnn_dict)
    baseline_path = _write_json(tmp_path / "baseline.json", baseline_dict)
    output_dir = tmp_path / "report"

    with pytest.raises(ValueError, match=r"metadata\.n_trajectories differs"):
        Reporter(
            egnn_path=egnn_path,
            hgnn_path=hgnn_path,
            baseline_path=baseline_path,
            output_dir=output_dir,
        ).run()


def test_run_rejects_per_bin_count_mismatch(tmp_path: Path) -> None:
    """Same bin edges but different per-bin counts means the test populations differ."""
    egnn_dict = _stratified_report_dict()
    hgnn_dict = _hgnn_stratified_report_dict()
    baseline_dict = _baseline_stratified_report_dict()
    baseline_dict["encounter_bins"]["by_name"]["close"]["count"] = 99

    egnn_path = _write_json(tmp_path / "egnn.json", egnn_dict)
    hgnn_path = _write_json(tmp_path / "hgnn.json", hgnn_dict)
    baseline_path = _write_json(tmp_path / "baseline.json", baseline_dict)
    output_dir = tmp_path / "report"

    with pytest.raises(ValueError, match=r"per-bin count differs for 'close'"):
        Reporter(
            egnn_path=egnn_path,
            hgnn_path=hgnn_path,
            baseline_path=baseline_path,
            output_dir=output_dir,
        ).run()


def test_run_rejects_per_bin_d_min_mismatch(tmp_path: Path) -> None:
    """Mismatched d_min summaries pin down that the test trajectories themselves differ."""
    egnn_dict = _stratified_report_dict()
    hgnn_dict = _hgnn_stratified_report_dict()
    baseline_dict = _baseline_stratified_report_dict()
    baseline_dict["encounter_bins"]["by_name"]["close"]["d_min"]["median"] = 0.999

    egnn_path = _write_json(tmp_path / "egnn.json", egnn_dict)
    hgnn_path = _write_json(tmp_path / "hgnn.json", hgnn_dict)
    baseline_path = _write_json(tmp_path / "baseline.json", baseline_dict)
    output_dir = tmp_path / "report"

    with pytest.raises(ValueError, match=r"per-bin d_min summary differs for 'close'"):
        Reporter(
            egnn_path=egnn_path,
            hgnn_path=hgnn_path,
            baseline_path=baseline_path,
            output_dir=output_dir,
        ).run()


def test_setup_output_is_idempotent(tmp_path: Path) -> None:
    """Re-running against an existing output directory does not fail."""
    reporter = _build_reporter(tmp_path)
    reporter.output_dir.mkdir()
    (reporter.output_dir / "figures").mkdir()

    reporter.run()
    reporter.run()

    assert reporter.figures_dir.is_dir()
    assert reporter.tables_dir.is_dir()


_CHUNKED_ENDPOINTS_CSV_FIXTURE = (
    "chunk_size,bin,model,median_end_state_mse,p95_end_state_mse,finite_fraction,"
    "median_end_position_mse,p95_end_position_mse,median_end_velocity_mse,p95_end_velocity_mse\n"
    "1,extreme,egnn,0.01,0.02,1.0,0.005,0.009,0.015,0.02\n"
    "1,extreme,hgnn,0.01,0.02,1.0,0.04,0.09,0.015,0.02\n"
    "1,extreme,baseline_constant_velocity,0.5,0.6,1.0,0.5,0.6,0.5,0.6\n"
    "3,extreme,egnn,0.03,0.05,1.0,0.10,0.15,0.04,0.06\n"
    "3,extreme,hgnn,0.03,0.05,1.0,0.30,0.40,0.04,0.06\n"
    "3,extreme,baseline_constant_velocity,1.0,1.2,1.0,1.0,1.2,1.0,1.2\n"
    "1,close,egnn,0.01,0.02,1.0,0.20,0.25,0.015,0.02\n"
    "1,close,hgnn,0.01,0.02,1.0,0.50,0.55,0.015,0.02\n"
    "1,close,baseline_constant_velocity,0.5,0.6,1.0,0.5,0.6,0.5,0.6\n"
    "1,smooth,egnn,0.01,0.02,1.0,0.005,0.009,0.015,0.02\n"
    "1,smooth,hgnn,0.01,0.02,1.0,0.01,0.02,0.015,0.02\n"
    "1,smooth,baseline_constant_velocity,0.5,0.6,1.0,0.5,0.6,0.5,0.6\n"
)


def _seed_chunked_dir(reporter: Reporter) -> Path:
    """Drop a minimal chunked sub-directory under the reporter's output dir."""
    chunked_dir = reporter.output_dir / "chunked"
    chunked_dir.mkdir(parents=True, exist_ok=True)
    (chunked_dir / "chunked_endpoints.csv").write_text(_CHUNKED_ENDPOINTS_CSV_FIXTURE)
    # only the CSV is required for the section; create empty siblings so the markdown
    # links to existing files (a stricter check than relying on the loader alone).
    (chunked_dir / "chunked_summary.csv").write_text("")
    (chunked_dir / "chunked_report.md").write_text("")
    (chunked_dir / "chunked_endpoint_position_mse_by_bin.png").write_bytes(b"PNG")
    return chunked_dir


def test_run_omits_chunked_section_when_chunked_dir_absent(tmp_path: Path) -> None:
    """No chunked/ -> report.md must not advertise the short-horizon section."""
    reporter = _build_reporter(tmp_path)

    reporter.run()

    md = (reporter.output_dir / "report.md").read_text()
    assert "Short-horizon corrected forecasting" not in md
    assert "chunked/" not in md


def test_run_emits_chunked_section_when_chunked_dir_present(tmp_path: Path) -> None:
    """A chunked sub-directory wires the short-horizon section into report.md."""
    reporter = _build_reporter(tmp_path)
    # Reporter._setup_output() will create output_dir; seed chunked AFTER setup so the
    # directory exists when _load_chunked_section runs. Easiest path: ensure the parent
    # exists, then drop chunked/ before invoking run().
    reporter.output_dir.mkdir(parents=True, exist_ok=True)
    _seed_chunked_dir(reporter)

    reporter.run()

    md = (reporter.output_dir / "report.md").read_text()
    assert "## Short-horizon corrected forecasting" in md
    assert "**This is not autonomous simulation.**" in md
    assert "median endpoint position MSE <= 0.25" in md
    assert "### Largest usable K per bin and model" in md
    # extreme bin: EGNN k=1 (0.005) and k=3 (0.10) both qualify -> largest is K=3.
    # HGNN k=1 (0.04) qualifies, k=3 (0.30) does not -> largest is K=1.
    # baseline never qualifies -> none.
    assert "| extreme | K=3 | K=1 | none |" in md
    # smooth: only K=1 evaluated; EGNN+HGNN both qualify, baseline does not.
    assert "| smooth | K=1 | K=1 | none |" in md
    assert "chunked/chunked_endpoint_position_mse_by_bin.png" in md
    assert "chunked/chunked_summary.csv" in md
    assert "chunked/chunked_endpoints.csv" in md
    assert "chunked/chunked_report.md" in md


def test_run_chunked_section_loader_raises_on_malformed_csv(tmp_path: Path) -> None:
    """A chunked/ with a corrupt endpoints CSV must fail loudly, not emit an empty section."""
    reporter = _build_reporter(tmp_path)
    reporter.output_dir.mkdir(parents=True, exist_ok=True)
    chunked_dir = reporter.output_dir / "chunked"
    chunked_dir.mkdir()
    # Stage the full manifest so the partial-manifest fallback does not kick in.
    (chunked_dir / "chunked_summary.csv").write_text("")
    (chunked_dir / "chunked_report.md").write_text("")
    (chunked_dir / "chunked_endpoint_position_mse_by_bin.png").write_bytes(b"PNG")
    (chunked_dir / "chunked_endpoints.csv").write_text(
        "chunk_size,bin,model\nnot-an-int,extreme,egnn\n"
    )

    with pytest.raises(ValueError, match="incompatible header"):
        reporter.run()


def test_run_skips_chunked_section_when_manifest_is_partial(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An endpoints CSV alone is not enough; the section attaches only with the full manifest."""
    reporter = _build_reporter(tmp_path)
    reporter.output_dir.mkdir(parents=True, exist_ok=True)
    chunked_dir = reporter.output_dir / "chunked"
    chunked_dir.mkdir()
    # only the endpoints CSV is present; the figure / summary CSV / report MD are missing
    (chunked_dir / "chunked_endpoints.csv").write_text(_CHUNKED_ENDPOINTS_CSV_FIXTURE)

    with caplog.at_level("WARNING", logger="evaluation.report"):
        reporter.run()

    md = (reporter.output_dir / "report.md").read_text()
    assert "Short-horizon corrected forecasting" not in md
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("chunked sub-directory is partial" in m for m in warnings), warnings


def test_cli_invokes_reporter_end_to_end(tmp_path: Path) -> None:
    """`python -m evaluation.report` runs the scaffold and creates the output skeleton."""
    egnn_path, hgnn_path, baseline_path, output_dir = _build_paths(tmp_path)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.report",
            "--egnn",
            str(egnn_path),
            "--hgnn",
            str(hgnn_path),
            "--baseline",
            str(baseline_path),
            "--output",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert (output_dir / "figures").is_dir()
    assert (output_dir / "tables").is_dir()


def test_cli_requires_baseline_flag(tmp_path: Path) -> None:
    """Omitting --baseline must surface argparse's required-flag error."""
    egnn_path, hgnn_path, _baseline_path, output_dir = _build_paths(tmp_path)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.report",
            "--egnn",
            str(egnn_path),
            "--hgnn",
            str(hgnn_path),
            "--output",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
        check=False,
    )

    assert proc.returncode != 0
    assert "--baseline" in proc.stderr
