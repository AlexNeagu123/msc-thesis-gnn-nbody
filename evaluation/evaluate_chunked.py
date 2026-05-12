"""Chunked forecast evaluation: short-horizon prediction under periodic truth resets.

The model is asked to predict the next `chunk_size` frames starting from a
true observation, then the next chunk begins again from the next true
observation, and so on. The metric is forecast accuracy *between*
observations, not the autonomous-simulation drift that the main rollout
evaluation tracks.

Three predictors are compared on the official stratified test set:
    - EGNN checkpoint
    - HGNN checkpoint
    - constant-velocity baseline

The module emits two CSVs, one markdown, and one figure pair:
    - chunked_summary.csv               (median over all predicted frames except 0)
    - chunked_endpoints.csv             (state + position + velocity MSE over chunk-endpoint frames)
    - chunked_report.md
    - chunked_endpoint_position_mse_by_bin.{png,pdf}

Endpoint position MSE is the audience-facing metric: a K is called "usable"
when its median endpoint position MSE stays at most `USABLE_K_POSITION_MSE_THRESHOLD`
(corresponding to a 0.5 coordinate-unit RMSE). State and velocity MSE remain in
the CSV as technical backup.

This is intentionally separate from `evaluation/evaluate.py`: the standard
evaluator runs one model end-to-end against a stratified test set; this
runner runs three models against the same dataset under a different
predictor contract and writes a self-contained artifact triple.

References:
    - evaluation/_loader.py    : load_trained_model
    - evaluation/metrics.py    : rollout (single-trajectory autoregressive helper)
    - evaluation/_binning.py   : trajectory_masks
    - models/baselines.py      : ConstantVelocityBaseline
"""

import argparse
import csv
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from data._io import read_trajectories
from data._types import Trajectories
from evaluation._binning import trajectory_masks
from evaluation._loader import load_trained_model
from evaluation.metrics import rollout
from models.baselines import ConstantVelocityBaseline
from training._io import load_config
from utils import get_logger

logger = get_logger(__name__)

DEFAULT_CHUNKS: tuple[int, ...] = (1, 3, 5, 10, 25)
MODEL_NAMES: tuple[str, str, str] = ("egnn", "hgnn", "baseline_constant_velocity")
SUMMARY_COLUMNS = (
    "chunk_size",
    "bin",
    "model",
    "median_state_mse",
    "p95_state_mse",
    "finite_fraction",
    "median_position_mse",
    "median_velocity_mse",
)
ENDPOINTS_COLUMNS = (
    "chunk_size",
    "bin",
    "model",
    "median_end_state_mse",
    "p95_end_state_mse",
    "finite_fraction",
    "median_end_position_mse",
    "p95_end_position_mse",
    "median_end_velocity_mse",
    "p95_end_velocity_mse",
)
USABLE_K_POSITION_MSE_THRESHOLD = 0.25
"""Endpoint position-MSE threshold for "usable K"; equivalent to RMSE = 0.5 coord units."""

_DISCLAIMER = (
    "This is not autonomous simulation. It measures short-horizon forecasting "
    "under periodic correction from true observations."
)


@dataclass(frozen=True)
class ChunkedSummaryRow:
    """One row of chunked_summary.csv: per (chunk_size, bin, model)."""

    chunk_size: int
    bin: str
    model: str
    median_state_mse: float | None
    p95_state_mse: float | None
    finite_fraction: float | None
    median_position_mse: float | None
    median_velocity_mse: float | None


@dataclass(frozen=True)
class ChunkedEndpointsRow:
    """One row of chunked_endpoints.csv: chunk-endpoint frame metrics per cell.

    Position MSE is the audience-facing metric for the headline figure and the
    usable-K rule; state and velocity MSE are kept as technical backup.
    """

    chunk_size: int
    bin: str
    model: str
    median_end_state_mse: float | None
    p95_end_state_mse: float | None
    finite_fraction: float | None
    median_end_position_mse: float | None
    p95_end_position_mse: float | None
    median_end_velocity_mse: float | None
    p95_end_velocity_mse: float | None


@dataclass(frozen=True)
class ChunkedRun:
    """Bundle of artifacts a single ChunkedEvaluator.run produces."""

    summary_rows: list[ChunkedSummaryRow]
    endpoint_rows: list[ChunkedEndpointsRow]
    summary_csv: Path
    endpoints_csv: Path
    markdown: Path
    figure_paths: list[Path]


def run_chunked_rollout(
    model: nn.Module,
    test_traj: npt.NDArray[np.floating],
    chunk_size: int,
    device: torch.device,
) -> npt.NDArray[np.floating]:
    """Predict each trajectory in chunks of `chunk_size`, resetting to truth between chunks.

    For each trajectory the algorithm is:
        - initialise predicted[0] = truth[0]
        - for each start in 0, K, 2K, ...:
            steps = min(K, n_frames - 1 - start)
            run an autoregressive rollout of `steps` steps from truth[start]
            write the resulting predictions into predicted[start+1 : start+1+steps]

    The final chunk is allowed to be partial (i.e. shorter than `chunk_size`)
    when `n_frames - 1` is not divisible by `chunk_size`. Output shape matches
    `test_traj`: `(n_traj, n_frames, n_particles, state_dim)`.

    `chunk_size` must be at least 1. The model is invoked with `torch.no_grad`
    inside `rollout`, so callers do not need to set inference mode.
    """
    if chunk_size < 1:
        msg = f"chunk_size must be >= 1; got {chunk_size}"
        raise ValueError(msg)

    n_traj, n_frames, _n_particles, _state_dim = test_traj.shape
    predicted = np.empty_like(test_traj)
    for i in range(n_traj):
        predicted[i, 0] = test_traj[i, 0]
        for start in range(0, n_frames - 1, chunk_size):
            steps = min(chunk_size, n_frames - 1 - start)
            initial = torch.from_numpy(test_traj[i, start]).float()
            chunk_pred = rollout(model, initial, steps, device)
            predicted[i, start + 1 : start + 1 + steps] = chunk_pred[1:]
    return predicted


def chunk_endpoint_frames(n_frames: int, chunk_size: int) -> list[int]:
    """Return the trailing-frame index of every chunk over a horizon of `n_frames`.

    With `n_frames = 200` and `chunk_size = 10` this yields `[10, 20, ..., 199]`;
    the last entry is the partial-chunk tail when `n_frames - 1` is not a
    multiple of `chunk_size`. Indices are 1-based against the predicted array
    (frame 0 is truth and never an endpoint).

    `chunk_size` must be at least 1. Without this guard a non-positive value
    would never advance the loop and the function would hang; the orchestrator
    rejects bad chunk sizes upstream, but this helper is part of the public
    surface so external callers need their own catch.
    """
    if chunk_size < 1:
        msg = f"chunk_size must be >= 1; got {chunk_size}"
        raise ValueError(msg)
    endpoints: list[int] = []
    start = 0
    while start < n_frames - 1:
        steps = min(chunk_size, n_frames - 1 - start)
        endpoints.append(start + steps)
        start += chunk_size
    return endpoints


def aggregate_summary(
    truth: npt.NDArray[np.floating],
    predicted: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
) -> dict[str, float | None]:
    """Aggregate per-frame MSE across (trajectories-in-bin, frames > 0).

    Returns None for every field when the bin is empty (mask all False).
    Median / p95 are computed over the finite entries only; the
    `finite_fraction` field reports the share of (traj, frame) pairs that
    produced a finite MSE.
    """
    if not mask.any():
        return _empty_summary()

    pred = predicted[mask, 1:]
    tru = truth[mask, 1:]
    state_mse = ((pred[..., :4] - tru[..., :4]) ** 2).mean(axis=(2, 3))
    pos_mse = ((pred[..., :2] - tru[..., :2]) ** 2).mean(axis=(2, 3))
    vel_mse = ((pred[..., 2:4] - tru[..., 2:4]) ** 2).mean(axis=(2, 3))

    state_flat = state_mse.ravel()
    return {
        "median_state_mse": _median(state_flat),
        "p95_state_mse": _percentile(state_flat, 95.0),
        "finite_fraction": float(np.isfinite(state_flat).mean()),
        "median_position_mse": _median(pos_mse.ravel()),
        "median_velocity_mse": _median(vel_mse.ravel()),
    }


def aggregate_endpoints(
    truth: npt.NDArray[np.floating],
    predicted: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    chunk_size: int,
) -> dict[str, float | None]:
    """Aggregate state, position, and velocity MSE across (trajectories-in-bin, chunk-endpoint frames).

    Position MSE drives the headline figure and the usable-K rule. State and
    velocity MSE accompany it in the CSV for technical reference.
    """
    if not mask.any():
        return _empty_endpoints()

    n_frames = truth.shape[1]
    endpoint_frames = chunk_endpoint_frames(n_frames, chunk_size)
    pred = predicted[mask][:, endpoint_frames]
    tru = truth[mask][:, endpoint_frames]
    state_mse = ((pred[..., :4] - tru[..., :4]) ** 2).mean(axis=(2, 3)).ravel()
    position_mse = ((pred[..., :2] - tru[..., :2]) ** 2).mean(axis=(2, 3)).ravel()
    velocity_mse = ((pred[..., 2:4] - tru[..., 2:4]) ** 2).mean(axis=(2, 3)).ravel()
    return {
        "median_end_state_mse": _median(state_mse),
        "p95_end_state_mse": _percentile(state_mse, 95.0),
        "finite_fraction": float(np.isfinite(state_mse).mean()),
        "median_end_position_mse": _median(position_mse),
        "p95_end_position_mse": _percentile(position_mse, 95.0),
        "median_end_velocity_mse": _median(velocity_mse),
        "p95_end_velocity_mse": _percentile(velocity_mse, 95.0),
    }


def _empty_endpoints() -> dict[str, float | None]:
    """Empty-bin sentinel for the endpoint aggregator."""
    return {
        "median_end_state_mse": None,
        "p95_end_state_mse": None,
        "finite_fraction": None,
        "median_end_position_mse": None,
        "p95_end_position_mse": None,
        "median_end_velocity_mse": None,
        "p95_end_velocity_mse": None,
    }


def largest_usable_k(
    endpoint_rows: list[ChunkedEndpointsRow],
    *,
    threshold: float = USABLE_K_POSITION_MSE_THRESHOLD,
) -> dict[tuple[str, str], int | None]:
    """For each (bin, model), return the largest K with median endpoint position MSE <= threshold.

    `None` means no evaluated K met the threshold for that cell. Cells that
    produced no finite estimate (empty bin, all non-finite predictions) are
    treated the same way as cells that failed the threshold.
    """
    by_cell: dict[tuple[str, str], int | None] = {}
    for row in endpoint_rows:
        key = (row.bin, row.model)
        candidate = row.median_end_position_mse
        qualifies = candidate is not None and np.isfinite(candidate) and candidate <= threshold
        if not qualifies:
            by_cell.setdefault(key, None)
            continue
        previous = by_cell.get(key)
        if previous is None or row.chunk_size > previous:
            by_cell[key] = row.chunk_size
    return by_cell


class ChunkedEvaluator:
    """Load models, run chunked rollouts at every chunk size, and emit artifacts."""

    def __init__(
        self,
        *,
        egnn_checkpoint: Path,
        hgnn_checkpoint: Path,
        egnn_config: Path,
        hgnn_config: Path,
        test_path: Path,
        train_path: Path,
        output_dir: Path,
        chunks: Iterable[int] = DEFAULT_CHUNKS,
        device: str = "auto",
    ) -> None:
        """Store every input path needed to reproduce the experiment from scratch."""
        self.egnn_checkpoint = egnn_checkpoint
        self.hgnn_checkpoint = hgnn_checkpoint
        self.egnn_config = egnn_config
        self.hgnn_config = hgnn_config
        self.test_path = test_path
        self.train_path = train_path
        self.output_dir = output_dir
        self.chunks = tuple(chunks)
        self.device = device

    def run(self) -> ChunkedRun:
        """Execute the full pipeline; return paths to the produced artifacts."""
        if not self.chunks:
            msg = "chunks must be non-empty"
            raise ValueError(msg)

        torch_device = self._resolve_device()
        test_bundle = self._read_trajectories()
        self._require_stratified(test_bundle)
        assert test_bundle.encounter_bins is not None  # narrowed by _require_stratified
        assert test_bundle.encounter_bin_id is not None
        bin_defs = test_bundle.encounter_bins
        bin_masks = trajectory_masks(test_bundle.encounter_bin_id, len(bin_defs))

        models = self._load_models(torch_device)

        summary_rows: list[ChunkedSummaryRow] = []
        endpoint_rows: list[ChunkedEndpointsRow] = []
        for chunk_size in self.chunks:
            for model_name, model in models.items():
                logger.info(
                    "chunked rollout: model=%s K=%d n_traj=%d",
                    model_name,
                    chunk_size,
                    test_bundle.states.shape[0],
                )
                predicted = run_chunked_rollout(model, test_bundle.states, chunk_size, torch_device)
                for bin_def, mask in zip(bin_defs, bin_masks, strict=True):
                    summary = aggregate_summary(test_bundle.states, predicted, mask)
                    endpoint = aggregate_endpoints(test_bundle.states, predicted, mask, chunk_size)
                    summary_rows.append(
                        ChunkedSummaryRow(
                            chunk_size=chunk_size,
                            bin=bin_def.name,
                            model=model_name,
                            **summary,  # type: ignore[arg-type]
                        )
                    )
                    endpoint_rows.append(
                        ChunkedEndpointsRow(
                            chunk_size=chunk_size,
                            bin=bin_def.name,
                            model=model_name,
                            **endpoint,  # type: ignore[arg-type]
                        )
                    )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        _clear_chunked_figure_artifacts(self.output_dir)
        summary_csv = self.output_dir / "chunked_summary.csv"
        endpoints_csv = self.output_dir / "chunked_endpoints.csv"
        markdown = self.output_dir / "chunked_report.md"
        _write_csv(summary_csv, SUMMARY_COLUMNS, summary_rows)
        _write_csv(endpoints_csv, ENDPOINTS_COLUMNS, endpoint_rows)
        _write_markdown(
            markdown,
            summary_rows,
            endpoint_rows,
            list(self.chunks),
            provenance=self._provenance(),
        )
        figure_paths = _plot_endpoint_position_mse_grouped(
            self.output_dir / "chunked_endpoint_position_mse_by_bin",
            endpoint_rows,
            list(self.chunks),
        )
        logger.info(
            "wrote %d summary rows, %d endpoint rows, and %d figure files to %s",
            len(summary_rows),
            len(endpoint_rows),
            len(figure_paths),
            self.output_dir,
        )
        return ChunkedRun(
            summary_rows=summary_rows,
            endpoint_rows=endpoint_rows,
            summary_csv=summary_csv,
            endpoints_csv=endpoints_csv,
            markdown=markdown,
            figure_paths=figure_paths,
        )

    def _resolve_device(self) -> torch.device:
        """Resolve `device` to a concrete torch.device, honouring 'auto'."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)

    def _read_trajectories(self) -> Trajectories:
        """Hook: subclasses override for tests that inject a fake bundle."""
        return read_trajectories(self.test_path)

    def _require_stratified(self, bundle: Trajectories) -> None:
        """The chunked report is per-bin; an un-stratified bundle has no story to tell."""
        if bundle.encounter_bins is None or bundle.encounter_bin_id is None:
            msg = (
                "chunked evaluation requires a stratified test set; "
                "encounter_bins and encounter_bin_id are both required"
            )
            raise ValueError(msg)

    def _load_models(self, torch_device: torch.device) -> dict[str, nn.Module]:
        """Hook: subclasses override to inject pre-built models for tests."""
        egnn = load_trained_model(self.egnn_config, self.egnn_checkpoint, torch_device).model
        hgnn = load_trained_model(self.hgnn_config, self.hgnn_checkpoint, torch_device).model
        dt = self._require_matching_dt()
        baseline = ConstantVelocityBaseline(dt=dt).to(torch_device)
        baseline.eval()
        return {"egnn": egnn, "hgnn": hgnn, "baseline_constant_velocity": baseline}

    def _require_matching_dt(self) -> float:
        """Both configs must agree on dt so the baseline integrates the right step."""
        egnn_dt = load_config(self.egnn_config).data.dt
        hgnn_dt = load_config(self.hgnn_config).data.dt
        if egnn_dt != hgnn_dt:
            msg = f"egnn_config.dt ({egnn_dt}) differs from hgnn_config.dt ({hgnn_dt})"
            raise ValueError(msg)
        return float(egnn_dt)

    def _provenance(self) -> dict[str, str]:
        """Input-path map embedded verbatim in the report's provenance section."""
        return {
            "egnn_config": str(self.egnn_config),
            "egnn_checkpoint": str(self.egnn_checkpoint),
            "hgnn_config": str(self.hgnn_config),
            "hgnn_checkpoint": str(self.hgnn_checkpoint),
            "test_path": str(self.test_path),
            "train_path": str(self.train_path),
        }


def _empty_summary() -> dict[str, float | None]:
    """Empty-bin sentinel row for the summary aggregator."""
    return {
        "median_state_mse": None,
        "p95_state_mse": None,
        "finite_fraction": None,
        "median_position_mse": None,
        "median_velocity_mse": None,
    }


def _median(values: npt.NDArray[np.floating]) -> float | None:
    """Median of finite entries, or None when nothing is finite."""
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else None


def _percentile(values: npt.NDArray[np.floating], q: float) -> float | None:
    """Percentile of finite entries, or None when nothing is finite."""
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, q)) if finite.size else None


def _clear_chunked_figure_artifacts(output_dir: Path) -> None:
    """Drop stale chunked PNG/PDF figures so a re-run mirrors the current manifest.

    Mirrors `evaluation.report.Reporter._clear_figure_artifacts` so renaming
    the headline figure (state -> position MSE) does not leave the previous
    `chunked_state_mse_by_bin.{png,pdf}` next to the new files.
    """
    for path in output_dir.glob("chunked_*.png"):
        path.unlink()
    for path in output_dir.glob("chunked_*.pdf"):
        path.unlink()


def read_endpoint_rows(path: Path) -> list[ChunkedEndpointsRow]:
    """Load `chunked_endpoints.csv` back into typed rows for cross-report consumers.

    The header must match `ENDPOINTS_COLUMNS` exactly. A legacy six-column
    CSV (state MSE only, no position / velocity columns) is rejected up
    front because silently filling the missing fields with `None` would
    make `largest_usable_k` render every cell as "none" instead of
    surfacing the schema drift.

    The main report's chunked section reuses these rows to render the
    usable-K summary; doing the parse here keeps the schema knowledge in
    one module. Empty CSV cells round-trip to `None`, matching how
    `_write_csv` emits them.
    """
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        _require_endpoint_header(reader.fieldnames, path)
        rows: list[ChunkedEndpointsRow] = []
        for raw in reader:
            rows.append(
                ChunkedEndpointsRow(
                    chunk_size=int(raw["chunk_size"]),
                    bin=raw["bin"],
                    model=raw["model"],
                    median_end_state_mse=_opt_float(raw["median_end_state_mse"]),
                    p95_end_state_mse=_opt_float(raw["p95_end_state_mse"]),
                    finite_fraction=_opt_float(raw["finite_fraction"]),
                    median_end_position_mse=_opt_float(raw["median_end_position_mse"]),
                    p95_end_position_mse=_opt_float(raw["p95_end_position_mse"]),
                    median_end_velocity_mse=_opt_float(raw["median_end_velocity_mse"]),
                    p95_end_velocity_mse=_opt_float(raw["p95_end_velocity_mse"]),
                )
            )
    return rows


def _require_endpoint_header(fieldnames: list[str] | None, path: Path) -> None:
    """Fail loud when the CSV header drops the new position / velocity columns."""
    actual = tuple(fieldnames or ())
    if actual != ENDPOINTS_COLUMNS:
        missing = [c for c in ENDPOINTS_COLUMNS if c not in actual]
        extra = [c for c in actual if c not in ENDPOINTS_COLUMNS]
        msg = (
            f"chunked endpoints CSV at {path} has incompatible header. "
            f"Expected {ENDPOINTS_COLUMNS}, got {actual}. "
            f"Missing columns: {missing}; unexpected columns: {extra}. "
            "Regenerate the chunked artifacts with the current "
            "`evaluation.evaluate_chunked` so the position / velocity columns are present."
        )
        raise ValueError(msg)


def _opt_float(value: str | None) -> float | None:
    """Coerce a CSV cell to float, returning None for empty / 'None' / missing."""
    if value is None or value == "" or value == "None":
        return None
    return float(value)


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[Any]) -> None:
    """Persist dataclass rows to a CSV in the declared column order."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_markdown(
    path: Path,
    summary_rows: list[ChunkedSummaryRow],
    endpoint_rows: list[ChunkedEndpointsRow],
    chunks: list[int],
    *,
    provenance: dict[str, str],
) -> None:
    """Render the human-readable report: disclaimer, usable-K table, per-K tables."""
    usable = largest_usable_k(endpoint_rows)
    sections = [
        "# Chunked Forecast Evaluation",
        "",
        f"> **{_DISCLAIMER}**",
        "",
        "## Provenance",
        "",
        _provenance_table(provenance),
        "",
        (
            "`train_path` is not used for the chunked constant-velocity baseline; "
            "it is recorded for dataset provenance only."
        ),
        "",
        f"Chunk sizes evaluated: {', '.join(str(c) for c in chunks)}.",
        "",
        "## Usable K per bin and model",
        "",
        (
            "A chunk size K is *usable* when its median endpoint position RMSE "
            f"stays at most {USABLE_K_POSITION_MSE_THRESHOLD**0.5} coordinate units "
            f"(equivalently, median endpoint position MSE <= {USABLE_K_POSITION_MSE_THRESHOLD}). "
            "The cell below reports the largest evaluated K that satisfies the rule; "
            "`none` means no evaluated K qualifies for that (bin, model) pair."
        ),
        "",
        _usable_k_table(usable, endpoint_rows),
        "",
        "## Median state MSE by chunk size and bin",
        "",
        _summary_table(summary_rows),
        "",
        "## Chunk-endpoint position MSE (headline)",
        "",
        (
            "Median endpoint position MSE is the audience-facing forecasting metric. "
            "Endpoint state and velocity MSE remain in `chunked_endpoints.csv` as "
            "technical backup."
        ),
        "",
        _endpoint_table(endpoint_rows),
        "",
    ]
    path.write_text("\n".join(sections))


def _provenance_table(provenance: dict[str, str]) -> str:
    """Markdown table listing every input path that drove the experiment."""
    headers = ["Field", "Value"]
    body = [[field, value] for field, value in provenance.items()]
    return _md_table(headers, body)


def _summary_table(rows: list[ChunkedSummaryRow]) -> str:
    """Markdown table over the summary rows ordered by (chunk_size, bin, model)."""
    headers = ["K", "bin", "model", "median state MSE", "p95 state MSE", "finite fraction"]
    body = [
        [
            str(r.chunk_size),
            r.bin,
            r.model,
            _fmt(r.median_state_mse),
            _fmt(r.p95_state_mse),
            _fmt(r.finite_fraction),
        ]
        for r in rows
    ]
    return _md_table(headers, body)


def _endpoint_table(rows: list[ChunkedEndpointsRow]) -> str:
    """Markdown endpoint table: position MSE leads, state and velocity follow as backup."""
    headers = [
        "K",
        "bin",
        "model",
        "median end position MSE",
        "p95 end position MSE",
        "median end state MSE",
        "median end velocity MSE",
        "finite fraction",
    ]
    body = [
        [
            str(r.chunk_size),
            r.bin,
            r.model,
            _fmt(r.median_end_position_mse),
            _fmt(r.p95_end_position_mse),
            _fmt(r.median_end_state_mse),
            _fmt(r.median_end_velocity_mse),
            _fmt(r.finite_fraction),
        ]
        for r in rows
    ]
    return _md_table(headers, body)


def _usable_k_table(
    usable: dict[tuple[str, str], int | None],
    endpoint_rows: list[ChunkedEndpointsRow],
) -> str:
    """Render the largest usable K per (bin, model) as a markdown table.

    Bin order follows first occurrence in `endpoint_rows` (which the orchestrator
    emits in canonical encounter-bin order). Model order follows `MODEL_NAMES`.
    """
    bins = list(dict.fromkeys(r.bin for r in endpoint_rows))
    headers = ["bin", *[_legend_label(name) for name in MODEL_NAMES]]
    body = [
        [
            bin_name,
            *[_fmt_usable_k(usable.get((bin_name, model))) for model in MODEL_NAMES],
        ]
        for bin_name in bins
    ]
    return _md_table(headers, body)


def _fmt_usable_k(value: int | None) -> str:
    """Render `largest_usable_k` cells; missing or non-qualifying becomes 'none'."""
    return "none" if value is None else f"K={value}"


def _md_table(headers: list[str], body: list[list[str]]) -> str:
    """Tiny github-flavored markdown table renderer."""
    sep = "|" + "|".join("---" for _ in headers) + "|"
    header_line = "| " + " | ".join(headers) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join([header_line, sep, *body_lines])


def _fmt(value: float | None) -> str:
    """Format a possibly-None float at ~4 sig figs; mirrors the report tables."""
    return f"{value:.4g}" if value is not None else "n/a"


_MODEL_COLOR = {
    "egnn": "#1f77b4",
    "hgnn": "#ff7f0e",
    "baseline_constant_velocity": "#555555",
}


def _plot_endpoint_position_mse_grouped(
    output_stem: Path,
    endpoint_rows: list[ChunkedEndpointsRow],
    chunks: list[int],
) -> list[Path]:
    """Render the headline chunked figure: median endpoint position MSE per bin.

    One panel per encounter bin, three grouped bars per chunk size (EGNN, HGNN,
    constant-velocity baseline). Position MSE is the audience-facing metric so
    audience-side comparisons match the threshold used by the usable-K table.
    """
    bins = list(dict.fromkeys(r.bin for r in endpoint_rows))
    n_bins = len(bins)
    if n_bins == 0:
        return []

    cols = max(2, (n_bins + 1 + 1) // 2)
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.0 * rows), sharey=True)
    flat_axes = list(axes.flat)

    bar_width = 0.27
    x = np.arange(len(chunks))
    for ax_idx, bin_name in enumerate(bins):
        ax = flat_axes[ax_idx]
        per_model = {
            name: [
                _lookup_endpoint_position_median(endpoint_rows, chunk, bin_name, name)
                for chunk in chunks
            ]
            for name in MODEL_NAMES
        }
        for offset, model_name in enumerate(MODEL_NAMES):
            ax.bar(
                x + (offset - 1) * bar_width,
                _safe_for_bar(per_model[model_name]),
                bar_width,
                color=_MODEL_COLOR[model_name],
                label=_legend_label(model_name),
            )
        ax.set_title(bin_name)
        ax.set_xticks(x)
        ax.set_xticklabels([f"K={k}" for k in chunks])
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(
            USABLE_K_POSITION_MSE_THRESHOLD,
            color="grey",
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
        )
        if _needs_log_axis(per_model):
            ax.set_yscale("log")
        if ax_idx % cols == 0:
            ax.set_ylabel("median endpoint position MSE")

    # fill the spare cell with the figure legend
    legend_ax = flat_axes[n_bins]
    legend_ax.axis("off")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_MODEL_COLOR[name], label=_legend_label(name))
        for name in MODEL_NAMES
    ]
    handles.append(
        plt.Line2D(
            [],
            [],
            color="grey",
            linestyle="--",
            linewidth=1.0,
            label=f"usable threshold = {USABLE_K_POSITION_MSE_THRESHOLD}",
        )
    )
    legend_ax.legend(handles=handles, loc="center", fontsize=11, frameon=False)
    for ax in flat_axes[n_bins + 1 :]:
        ax.axis("off")

    fig.suptitle("Chunked forecast: median endpoint position MSE by encounter bin", y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    for path in (png_path, pdf_path):
        fig.savefig(path)
    plt.close(fig)
    return [png_path, pdf_path]


def _lookup_endpoint_position_median(
    endpoint_rows: list[ChunkedEndpointsRow],
    chunk_size: int,
    bin_name: str,
    model: str,
) -> float | None:
    """Find median endpoint position MSE for one (K, bin, model); None when missing."""
    for r in endpoint_rows:
        if r.chunk_size == chunk_size and r.bin == bin_name and r.model == model:
            return r.median_end_position_mse
    return None


def _safe_for_bar(values: list[float | None]) -> list[float]:
    """Replace None with 0 for matplotlib's bar() input; log scale handles outliers."""
    return [v if v is not None and np.isfinite(v) else 0.0 for v in values]


def _legend_label(model_name: str) -> str:
    """Map internal model ids to human-readable legend strings."""
    return {"egnn": "EGNN", "hgnn": "HGNN", "baseline_constant_velocity": "constant velocity"}[
        model_name
    ]


def _needs_log_axis(per_model: dict[str, list[float | None]]) -> bool:
    """Switch to log y when the data spans more than two decades; linear otherwise."""
    flat = [v for series in per_model.values() for v in series if v is not None and v > 0]
    if len(flat) < 2:
        return False
    return max(flat) / min(flat) > 100.0


def main() -> None:
    """CLI entrypoint: parse paths and delegate to ChunkedEvaluator.run."""
    parser = argparse.ArgumentParser(
        description="Chunked forecast evaluation with periodic truth resets.",
    )
    parser.add_argument("--egnn-checkpoint", type=str, required=True)
    parser.add_argument("--hgnn-checkpoint", type=str, required=True)
    parser.add_argument("--egnn-config", type=str, required=True)
    parser.add_argument("--hgnn-config", type=str, required=True)
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument(
        "--train-path",
        type=str,
        required=True,
        help=(
            "Recorded in the report's provenance section. Not consumed by the chunked "
            "constant-velocity baseline, but kept so the artifact pins the dataset that "
            "the EGNN and HGNN checkpoints were trained against."
        ),
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--chunks",
        type=int,
        nargs="+",
        default=list(DEFAULT_CHUNKS),
        help="Chunk sizes to evaluate (default: 1 3 5 10 25).",
    )
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # train_path is provenance-only here, so the evaluator never reads it; validate at
    # the CLI boundary so a typo is caught before the orchestrator burns time on
    # rollouts that would still succeed.
    train_path = Path(args.train_path)
    if not train_path.exists():
        msg = (
            f"--train-path does not exist: {train_path}. The chunked experiment does not "
            "consume this file, but it must point at the real training dataset so the "
            "report's provenance section is honest."
        )
        raise FileNotFoundError(msg)

    ChunkedEvaluator(
        egnn_checkpoint=Path(args.egnn_checkpoint),
        hgnn_checkpoint=Path(args.hgnn_checkpoint),
        egnn_config=Path(args.egnn_config),
        hgnn_config=Path(args.hgnn_config),
        test_path=Path(args.test_path),
        train_path=train_path,
        output_dir=Path(args.output_dir),
        chunks=args.chunks,
        device=args.device,
    ).run()


if __name__ == "__main__":
    main()
