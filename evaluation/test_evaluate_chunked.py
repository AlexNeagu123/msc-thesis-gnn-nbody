"""Tests for evaluation/evaluate_chunked.py.

Selector + aggregator tests build small synthetic Trajectories and a toy
deterministic model so the chunked-rollout contract is verifiable without
ever touching a checkpoint. The orchestrator is exercised via a DI subclass
that swaps the rollout and model-loading hooks.
"""

import csv
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch import nn

from data._types import EncounterBin, Trajectories
from evaluation.evaluate_chunked import (
    ENDPOINTS_COLUMNS,
    SUMMARY_COLUMNS,
    ChunkedEvaluator,
    aggregate_endpoints,
    aggregate_summary,
    chunk_endpoint_frames,
    run_chunked_rollout,
)
from evaluation.metrics import rollout


class _ShiftModel(nn.Module):
    """Toy model whose prediction is `state + delta` along the position axes.

    Used to exercise the chunked rollout without involving real checkpoints.
    Compounding error is `delta * step_count`, so chunked rollouts produce
    measurably smaller per-frame error than autonomous ones.
    """

    def __init__(self, delta: float = 0.1) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        out = state.clone()
        out[..., 0] = state[..., 0] + self.delta
        out[..., 1] = state[..., 1] + self.delta
        return out


def _toy_trajectories(
    n_traj: int = 2, n_frames: int = 7, n_particles: int = 2
) -> npt.NDArray[np.floating]:
    """Deterministic ground-truth array; identity dynamics make hand-checking easy."""
    truth = np.zeros((n_traj, n_frames, n_particles, 5), dtype=np.float64)
    truth[..., 4] = 1.0  # mass
    return truth


def _bundle_for(states: npt.NDArray[np.floating], bin_ids: list[int]) -> Trajectories:
    """Compose a stratified Trajectories with the given per-trajectory bin assignments."""
    n_traj, n_frames, _, _ = states.shape
    bins = (
        EncounterBin(name="close", lo=0.0, hi=0.05),
        EncounterBin(name="far", lo=0.05, hi=float("inf")),
    )
    return Trajectories(
        states=states,
        energies=np.zeros((n_traj, n_frames)),
        metadata=None,
        encounter_bin_id=np.array(bin_ids, dtype=np.int64),
        encounter_bin_name=np.array([bins[i].name for i in bin_ids]),
        min_pairwise_distance=np.full(n_traj, 0.04, dtype=np.float64),
        encounter_bins=bins,
    )


def test_run_chunked_rollout_k1_uses_truth_every_step() -> None:
    """K=1 means every prediction starts from the next truth frame.

    With a +0.1 shift model and an all-zero ground truth, every predicted
    frame should equal truth + delta exactly once (no error compounding).
    """
    truth = _toy_trajectories(n_traj=1, n_frames=4)
    model = _ShiftModel(delta=0.1)
    device = torch.device("cpu")

    predicted = run_chunked_rollout(model, truth, chunk_size=1, device=device)

    assert predicted[0, 0, 0, 0] == 0.0  # frame 0 is truth
    # frames 1..3 each predicted as truth + delta (truth==0 so each is exactly delta)
    np.testing.assert_allclose(predicted[0, 1:, :, 0], 0.1)
    np.testing.assert_allclose(predicted[0, 1:, :, 1], 0.1)


def test_run_chunked_rollout_k3_resets_at_chunk_boundaries() -> None:
    """K=3 should autoregress within a chunk then reset to truth at the next boundary."""
    truth = _toy_trajectories(n_traj=1, n_frames=7)
    model = _ShiftModel(delta=0.1)
    device = torch.device("cpu")

    predicted = run_chunked_rollout(model, truth, chunk_size=3, device=device)

    # Chunk 1: starts at truth[0]=0, predicts frames 1..3 as 0.1, 0.2, 0.3.
    np.testing.assert_allclose(predicted[0, 1, :, 0], 0.1)
    np.testing.assert_allclose(predicted[0, 2, :, 0], 0.2)
    np.testing.assert_allclose(predicted[0, 3, :, 0], 0.3)
    # Chunk 2: starts at truth[3]=0, predicts frames 4..6 as 0.1, 0.2, 0.3.
    np.testing.assert_allclose(predicted[0, 4, :, 0], 0.1)
    np.testing.assert_allclose(predicted[0, 5, :, 0], 0.2)
    np.testing.assert_allclose(predicted[0, 6, :, 0], 0.3)


def test_run_chunked_rollout_handles_partial_final_chunk() -> None:
    """When n_frames - 1 is not divisible by K, the last chunk shortens cleanly."""
    truth = _toy_trajectories(n_traj=1, n_frames=8)  # 7 transitions, K=3 -> chunks of 3,3,1
    model = _ShiftModel(delta=0.1)
    device = torch.device("cpu")

    predicted = run_chunked_rollout(model, truth, chunk_size=3, device=device)

    # Final partial chunk: starts at truth[6]=0, predicts only frame 7 as 0.1.
    np.testing.assert_allclose(predicted[0, 7, :, 0], 0.1)


def test_run_chunked_rollout_equals_autonomous_when_chunk_geq_steps() -> None:
    """K >= n_frames - 1 collapses to a single autonomous rollout."""
    truth = _toy_trajectories(n_traj=1, n_frames=5)
    model = _ShiftModel(delta=0.1)
    device = torch.device("cpu")

    chunked = run_chunked_rollout(model, truth, chunk_size=10, device=device)
    autonomous_initial = torch.from_numpy(truth[0, 0]).float()
    autonomous = rollout(model, autonomous_initial, n_steps=4, device=device)

    np.testing.assert_allclose(chunked[0], autonomous)


def test_run_chunked_rollout_differs_from_autonomous_when_chunk_lt_steps() -> None:
    """K < n_frames - 1 must produce a different sequence because of the resets."""
    truth = _toy_trajectories(n_traj=1, n_frames=10)
    model = _ShiftModel(delta=0.1)
    device = torch.device("cpu")

    chunked = run_chunked_rollout(model, truth, chunk_size=3, device=device)
    autonomous_initial = torch.from_numpy(truth[0, 0]).float()
    autonomous = rollout(model, autonomous_initial, n_steps=9, device=device)

    assert not np.allclose(chunked[0], autonomous)


def test_run_chunked_rollout_rejects_zero_chunk_size() -> None:
    """K must be at least 1; a non-positive K is caught at the helper entry."""
    truth = _toy_trajectories(n_traj=1, n_frames=4)
    model = _ShiftModel()

    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        run_chunked_rollout(model, truth, chunk_size=0, device=torch.device("cpu"))


def test_chunk_endpoint_frames_includes_partial_chunk_tail() -> None:
    """The final partial chunk contributes its trailing frame index to the endpoints list."""
    # n_frames=8 (-> 7 transitions), K=3 -> chunks at 0,3,6 yield endpoints 3, 6, 7
    assert chunk_endpoint_frames(n_frames=8, chunk_size=3) == [3, 6, 7]
    # K equal to or larger than the horizon yields a single endpoint at the final frame.
    assert chunk_endpoint_frames(n_frames=5, chunk_size=10) == [4]


def test_chunk_endpoint_frames_rejects_zero_chunk_size() -> None:
    """A direct call with chunk_size=0 must raise instead of hanging on the while loop."""
    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        chunk_endpoint_frames(n_frames=10, chunk_size=0)


def test_aggregate_summary_returns_nones_for_empty_bin() -> None:
    """An all-False mask short-circuits to a None-filled summary; no division by zero."""
    truth = _toy_trajectories(n_traj=3, n_frames=4)
    predicted = truth.copy()
    mask = np.zeros(3, dtype=bool)

    out = aggregate_summary(truth, predicted, mask)

    assert out == {
        "median_state_mse": None,
        "p95_state_mse": None,
        "finite_fraction": None,
        "median_position_mse": None,
        "median_velocity_mse": None,
    }


def test_aggregate_summary_respects_bin_mask() -> None:
    """Aggregation uses only trajectories whose mask entry is True."""
    truth = _toy_trajectories(n_traj=2, n_frames=4)
    # Trajectory 0: predicted == truth (zero error).
    # Trajectory 1: predicted offset by 1 in every position component (large error).
    predicted = truth.copy()
    predicted[1, 1:, :, :2] += 1.0

    in_bin_only_0 = np.array([True, False])
    in_bin_only_1 = np.array([False, True])

    out_zero = aggregate_summary(truth, predicted, in_bin_only_0)
    out_high = aggregate_summary(truth, predicted, in_bin_only_1)

    assert out_zero["median_state_mse"] == 0.0
    assert out_high["median_state_mse"] is not None
    assert out_high["median_state_mse"] > 0.0


def test_aggregate_endpoints_reads_chunk_tail_frames() -> None:
    """Endpoint aggregation pulls only the trailing frame of each chunk."""
    truth = _toy_trajectories(n_traj=1, n_frames=7)
    predicted = truth.copy()
    # Make endpoint frames (3 and 6 with K=3) carry a large error; the inner frames stay clean.
    predicted[0, 3, :, :2] = 1.0
    predicted[0, 6, :, :2] = 1.0
    mask = np.array([True])

    out = aggregate_endpoints(truth, predicted, mask, chunk_size=3)

    assert out["median_end_state_mse"] is not None
    # Per-frame state MSE = mean over (particles, [x, y, vx, vy]) of squared diff.
    # With positions shifted by 1.0 and velocities clean, that is (1+1+0+0)/4 = 0.5.
    assert out["median_end_state_mse"] == pytest.approx(0.5)


class _StubEvaluator(ChunkedEvaluator):
    """Evaluator subclass that bypasses checkpoint/Bundle loading for tests."""

    def __init__(
        self,
        *,
        bundle: Trajectories,
        models: dict[str, nn.Module],
        output_dir: Path,
        chunks: tuple[int, ...] = (1, 2),
    ) -> None:
        """Skip the parent __init__: real paths are unused under DI."""
        self.egnn_checkpoint = Path("unused")
        self.hgnn_checkpoint = Path("unused")
        self.egnn_config = Path("unused")
        self.hgnn_config = Path("unused")
        self.test_path = Path("unused")
        self.train_path = Path("unused")
        self.output_dir = output_dir
        self.chunks = chunks
        self.device = "cpu"
        self._bundle = bundle
        self._models = models

    def _read_trajectories(self) -> Trajectories:  # type: ignore[override]
        return self._bundle

    def _load_models(self, torch_device: torch.device) -> dict[str, nn.Module]:  # type: ignore[override]
        for m in self._models.values():
            m.to(torch_device)
            m.eval()
        return self._models


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Return (column_order, list of row dicts) for a CSV."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def test_chunked_evaluator_writes_csvs_and_markdown(tmp_path: Path) -> None:
    """End-to-end DI smoke: orchestrator emits both CSVs, markdown, and a figure pair."""
    truth = _toy_trajectories(n_traj=4, n_frames=5)
    bundle = _bundle_for(truth, bin_ids=[0, 0, 1, 1])
    models = {
        "egnn": _ShiftModel(delta=0.1),
        "hgnn": _ShiftModel(delta=0.05),
        "baseline_constant_velocity": _ShiftModel(delta=0.2),
    }
    evaluator = _StubEvaluator(bundle=bundle, models=models, output_dir=tmp_path, chunks=(1, 2))

    run = evaluator.run()

    assert run.summary_csv.is_file()
    assert run.endpoints_csv.is_file()
    assert run.markdown.is_file()

    summary_cols, summary_rows = _read_csv_rows(run.summary_csv)
    endpoint_cols, endpoint_rows = _read_csv_rows(run.endpoints_csv)
    assert tuple(summary_cols) == SUMMARY_COLUMNS
    assert tuple(endpoint_cols) == ENDPOINTS_COLUMNS
    # 2 chunks * 3 models * 2 bins = 12 rows per CSV
    assert len(summary_rows) == 12
    assert len(endpoint_rows) == 12

    md = run.markdown.read_text()
    assert "Chunked Forecast Evaluation" in md
    assert "not autonomous simulation" in md
    assert "close" in md
    assert "far" in md
    # Provenance section pins every input path and explains why train_path is recorded.
    assert "## Provenance" in md
    assert "egnn_config" in md
    assert "hgnn_config" in md
    assert "test_path" in md
    assert "train_path" in md
    assert "not used for the chunked constant-velocity baseline" in md

    # at least one figure file was produced
    assert any(p.is_file() and p.stat().st_size > 0 for p in run.figure_paths)


def test_chunked_evaluator_smaller_k_has_smaller_median_for_compounding_model(
    tmp_path: Path,
) -> None:
    """A linearly-compounding error model should track better with smaller chunks."""
    truth = _toy_trajectories(n_traj=2, n_frames=10)
    bundle = _bundle_for(truth, bin_ids=[0, 0])
    models = {
        "egnn": _ShiftModel(delta=0.1),
        "hgnn": _ShiftModel(delta=0.1),
        "baseline_constant_velocity": _ShiftModel(delta=0.1),
    }
    evaluator = _StubEvaluator(bundle=bundle, models=models, output_dir=tmp_path, chunks=(1, 9))

    run = evaluator.run()

    egnn_rows = {
        r.chunk_size: r.median_state_mse
        for r in run.summary_rows
        if r.model == "egnn" and r.bin == "close"
    }
    assert egnn_rows[1] is not None
    assert egnn_rows[9] is not None
    assert egnn_rows[1] < egnn_rows[9]


def test_chunked_evaluator_rejects_unstratified_bundle(tmp_path: Path) -> None:
    """An un-stratified bundle is rejected up front; no rollouts get run."""
    truth = _toy_trajectories(n_traj=2, n_frames=4)
    unstratified = Trajectories(states=truth, energies=np.zeros((2, 4)))
    evaluator = _StubEvaluator(
        bundle=unstratified,
        models={
            "egnn": _ShiftModel(),
            "hgnn": _ShiftModel(),
            "baseline_constant_velocity": _ShiftModel(),
        },
        output_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="requires a stratified test set"):
        evaluator.run()


def test_chunked_evaluator_rejects_empty_chunks(tmp_path: Path) -> None:
    """An empty `chunks` argument fails fast with a clear message."""
    truth = _toy_trajectories(n_traj=2, n_frames=4)
    bundle = _bundle_for(truth, bin_ids=[0, 1])
    evaluator = _StubEvaluator(
        bundle=bundle,
        models={
            "egnn": _ShiftModel(),
            "hgnn": _ShiftModel(),
            "baseline_constant_velocity": _ShiftModel(),
        },
        output_dir=tmp_path,
        chunks=(),
    )

    with pytest.raises(ValueError, match="chunks must be non-empty"):
        evaluator.run()
