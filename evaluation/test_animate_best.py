"""Tests for evaluation/animate_best.py."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest

from data._types import EncounterBin, Trajectories
from evaluation.animate_best import (
    AnimationOutputs,
    BestTrajectory,
    BestTrajectoryAnimator,
    _per_trajectory_mean_position_mse,
    _shared_axis_limits,
    render_three_panel_animation,
    select_best_trajectories,
    select_trajectories_from_file,
)


def _bins_fixture() -> tuple[EncounterBin, ...]:
    """Three encounter bins covering the synthetic distance range."""
    return (
        EncounterBin(name="close", lo=0.0, hi=0.03),
        EncounterBin(name="mid", lo=0.03, hi=0.1),
        EncounterBin(name="far", lo=0.1, hi=float("inf")),
    )


def _build_bundle(
    states: npt.NDArray[np.floating],
    bin_ids: list[int],
    d_mins: list[float],
) -> Trajectories:
    """Compose a stratified Trajectories object for selector tests."""
    n_traj, n_frames, _, _ = states.shape
    return Trajectories(
        states=states,
        energies=np.zeros((n_traj, n_frames)),
        metadata=None,
        encounter_bin_id=np.array(bin_ids, dtype=np.int64),
        encounter_bin_name=np.array([_bins_fixture()[i].name for i in bin_ids]),
        min_pairwise_distance=np.array(d_mins, dtype=np.float64),
        encounter_bins=_bins_fixture(),
    )


def _states(n_traj: int, n_frames: int = 4, n_particles: int = 3) -> npt.NDArray[np.floating]:
    """Build deterministic ground-truth states; final frame fixed so MSE math is easy."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((n_traj, n_frames, n_particles, 5))


def _predicted_with_offsets(
    truth: npt.NDArray[np.floating],
    final_offsets: list[float],
) -> npt.NDArray[np.floating]:
    """Mirror `truth` exactly except for an additive offset on the final frame.

    Per-trajectory mean MSE scales with offset**2, so the per-bin rank order is preserved
    and tests assert on winners, not absolute values.
    """
    predicted = truth.copy()
    for i, offset in enumerate(final_offsets):
        predicted[i, -1, :, :4] += offset
    return predicted


def test_select_best_trajectories_picks_lowest_joint_mean_mse() -> None:
    """Across each bin, the lowest-joint-mean-MSE trajectory wins."""
    truth = _states(n_traj=6)
    egnn = _predicted_with_offsets(truth, [0.5, 0.1, 0.4, 0.2, 1.0, 0.3])
    hgnn = _predicted_with_offsets(truth, [0.2, 0.5, 0.1, 0.6, 0.5, 0.9])
    # joint mean MSE is a constant rescale of (egnn_offset^2 + hgnn_offset^2):
    # 0: 0.25+0.04=0.29  1: 0.01+0.25=0.26 (close winner)
    # 2: 0.16+0.01=0.17 (mid winner)  3: 0.04+0.36=0.40
    # 4: 1.0+0.25=1.25  5: 0.09+0.81=0.90 (far winner)
    bundle = _build_bundle(
        truth,
        bin_ids=[0, 0, 1, 1, 2, 2],
        d_mins=[0.01, 0.02, 0.05, 0.07, 0.5, 0.7],
    )

    selections = select_best_trajectories(bundle, egnn, hgnn)

    winners = {s.bin_name: s.traj_index for s in selections}
    assert winners == {"close": 1, "mid": 2, "far": 5}


def test_select_best_trajectories_ignores_non_finite_candidates() -> None:
    """A trajectory with NaN/inf anywhere in the rollout is skipped from the selection."""
    truth = _states(n_traj=6)
    egnn = _predicted_with_offsets(truth, [0.1, 0.5, 0.3, 0.4, 0.2, 0.3])
    hgnn = _predicted_with_offsets(truth, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    # Force trajectory 0 to have a non-finite mid-rollout entry on the EGNN side.
    egnn[0, 1, 0, 0] = float("nan")
    bundle = _build_bundle(
        truth,
        bin_ids=[0, 0, 1, 1, 2, 2],
        d_mins=[0.01, 0.02, 0.05, 0.07, 0.5, 0.7],
    )

    selections = select_best_trajectories(bundle, egnn, hgnn)

    winners = {s.bin_name: s.traj_index for s in selections}
    # trajectory 0 has the lowest joint offset but a poisoned rollout; trajectory 1 wins "close"
    assert winners["close"] == 1


def test_select_best_trajectories_raises_when_bin_has_no_finite_candidate() -> None:
    """A bin where every trajectory is divergent must raise instead of silently dropping."""
    truth = _states(n_traj=3)
    egnn = _predicted_with_offsets(truth, [0.1, 0.2, 0.3])
    hgnn = _predicted_with_offsets(truth, [0.1, 0.2, 0.3])
    # All trajectories in bin 0 are non-finite somewhere in the rollout; bin 1 stays fine.
    egnn[0, 2, 0, 0] = float("inf")
    egnn[1, 1, 0, 0] = float("nan")
    bundle = _build_bundle(
        truth,
        bin_ids=[0, 0, 1],
        d_mins=[0.01, 0.02, 0.05],
    )

    with pytest.raises(
        ValueError, match="bin 'close' has no trajectories with finite mean rollout position MSE"
    ):
        select_best_trajectories(bundle, egnn, hgnn)


def test_select_best_trajectories_raises_for_unstratified_bundle() -> None:
    """An un-stratified test bundle is rejected with a clear message."""
    truth = _states(n_traj=2)
    egnn = _predicted_with_offsets(truth, [0.1, 0.2])
    hgnn = _predicted_with_offsets(truth, [0.1, 0.2])
    bundle = Trajectories(states=truth, energies=np.zeros((2, 4)))

    with pytest.raises(ValueError, match="requires a stratified test set"):
        select_best_trajectories(bundle, egnn, hgnn)


def test_select_best_trajectories_rejects_wrong_n_traj() -> None:
    """Prediction arrays with wrong leading dimension fail fast, before any MSE arithmetic."""
    truth = _states(n_traj=4)
    egnn = _predicted_with_offsets(truth, [0.1, 0.2, 0.3, 0.4])
    bundle = _build_bundle(
        truth,
        bin_ids=[0, 0, 1, 2],
        d_mins=[0.01, 0.02, 0.05, 0.5],
    )
    hgnn_wrong = _predicted_with_offsets(_states(n_traj=3), [0.1, 0.2, 0.3])  # n_traj mismatch

    with pytest.raises(ValueError, match="hgnn_predicted shape does not match"):
        select_best_trajectories(bundle, egnn, hgnn_wrong)


def test_select_best_trajectories_rejects_wrong_n_frames() -> None:
    """Prediction arrays with the wrong frame count fail fast as well."""
    truth = _states(n_traj=4)
    bundle = _build_bundle(
        truth,
        bin_ids=[0, 0, 1, 2],
        d_mins=[0.01, 0.02, 0.05, 0.5],
    )
    egnn = _predicted_with_offsets(truth, [0.1, 0.2, 0.3, 0.4])
    hgnn = _predicted_with_offsets(truth, [0.1, 0.2, 0.3, 0.4])
    short = hgnn[:, :-1]  # drop the final frame from one prediction array

    with pytest.raises(ValueError, match="hgnn_predicted shape does not match"):
        select_best_trajectories(bundle, egnn, short)


def test_best_trajectory_basename_is_deterministic() -> None:
    """The MP4 and GIF share a stable stem keyed by bin and trajectory index."""
    sel = BestTrajectory(
        bin_id=2,
        bin_name="mid",
        traj_index=42,
        d_min=0.04,
        egnn_mean_position_mse=0.05,
        hgnn_mean_position_mse=0.03,
    )

    assert sel.basename == "mid_traj_42"


def test_per_trajectory_mean_position_mse_propagates_nan_from_any_frame() -> None:
    """A single non-finite entry anywhere in the rollout poisons that trajectory's mean MSE."""
    truth = _states(n_traj=2)
    predicted = truth.copy()
    predicted[0, 1, 0, 0] = float("nan")  # mid-rollout NaN, not just final frame

    mse = _per_trajectory_mean_position_mse(truth, predicted)

    assert np.isnan(mse[0])
    assert mse[1] == 0.0


def test_shared_axis_limits_uses_finite_positions_only() -> None:
    """Inf/NaN positions must not poison the shared axis limits."""
    a = np.zeros((2, 1, 5))
    a[0, 0, 0] = 0.0
    a[0, 0, 1] = 0.0
    a[1, 0, 0] = 1.0
    a[1, 0, 1] = 2.0
    b = np.full_like(a, float("nan"))
    c = np.zeros_like(a)
    c[0, 0, 0] = -3.0
    c[0, 0, 1] = -1.0

    x_lim, y_lim = _shared_axis_limits([a, b, c])

    assert x_lim[0] < -2.9
    assert x_lim[1] > 0.9
    assert y_lim[0] < -0.9
    assert y_lim[1] > 1.9


def test_render_three_panel_animation_writes_gif_and_closes_figure(tmp_path: Path) -> None:
    """A short synthetic trajectory renders a non-empty GIF; no matplotlib figures leak."""
    truth = _states(n_traj=1, n_frames=4)[0]
    egnn = truth.copy()
    egnn[-1, :, :4] += 0.1
    hgnn = truth.copy()
    hgnn[-1, :, :4] += 0.2
    sel = BestTrajectory(
        bin_id=0,
        bin_name="close",
        traj_index=0,
        d_min=0.01,
        egnn_mean_position_mse=0.01,
        hgnn_mean_position_mse=0.04,
    )
    plt.close("all")
    initial = len(plt.get_fignums())

    outputs = render_three_panel_animation(
        truth,
        egnn,
        hgnn,
        sel,
        mp4_path=tmp_path / "close_traj_0.mp4",
        gif_path=tmp_path / "close_traj_0.gif",
        fps=10,
    )

    assert outputs.gif_path is not None
    assert outputs.gif_path.is_file()
    assert outputs.gif_path.stat().st_size > 0
    assert len(plt.get_fignums()) == initial


def test_render_three_panel_animation_rejects_non_positive_fps(tmp_path: Path) -> None:
    """Reject fps < 1 at the renderer entry; matplotlib would otherwise crash on 1000 // 0."""
    truth = _states(n_traj=1, n_frames=4)[0]
    sel = BestTrajectory(
        bin_id=0,
        bin_name="close",
        traj_index=0,
        d_min=0.01,
        egnn_mean_position_mse=0.0,
        hgnn_mean_position_mse=0.0,
    )

    with pytest.raises(ValueError, match="fps must be >= 1"):
        render_three_panel_animation(
            truth,
            truth.copy(),
            truth.copy(),
            sel,
            mp4_path=tmp_path / "x.mp4",
            gif_path=tmp_path / "x.gif",
            fps=0,
        )


class _FakeAnimator(BestTrajectoryAnimator):
    """Animator subclass that bypasses model loading for orchestrator tests."""

    def __init__(
        self,
        *,
        bundle: Trajectories,
        egnn_predicted: npt.NDArray[np.floating],
        hgnn_predicted: npt.NDArray[np.floating],
        output_dir: Path,
        selection_file: Path | None = None,
    ) -> None:
        """Skip the parent __init__: the disk-loading hooks are stubbed below."""
        self.egnn_checkpoint = Path("unused")
        self.hgnn_checkpoint = Path("unused")
        self.egnn_config = Path("unused")
        self.hgnn_config = Path("unused")
        self.test_path = Path("unused")
        self.output_dir = output_dir
        self.device = "cpu"
        self.fps = 10
        self.selection_file = selection_file
        self._bundle = bundle
        self._egnn_predicted = egnn_predicted
        self._hgnn_predicted = hgnn_predicted

    def _read_trajectories(self) -> Trajectories:  # type: ignore[override]
        return self._bundle

    def _rollout_for_model(
        self,
        model_name: str,
        config_path: Path,
        checkpoint_path: Path,
        test_states: npt.NDArray[np.floating],
        torch_device: object,
    ) -> npt.NDArray[np.floating]:
        """Return the pre-baked rollout for the requested model name."""
        if model_name == "egnn":
            return self._egnn_predicted
        return self._hgnn_predicted


def test_animator_orchestrator_renders_one_animation_per_bin(tmp_path: Path) -> None:
    """End-to-end DI smoke: one (bin, traj) selection -> one GIF output per bin."""
    truth = _states(n_traj=6)
    egnn = _predicted_with_offsets(truth, [0.1, 0.5, 0.3, 0.4, 0.2, 0.6])
    hgnn = _predicted_with_offsets(truth, [0.2, 0.4, 0.1, 0.5, 0.3, 0.4])
    bundle = _build_bundle(
        truth,
        bin_ids=[0, 0, 1, 1, 2, 2],
        d_mins=[0.01, 0.02, 0.05, 0.07, 0.5, 0.7],
    )
    animator = _FakeAnimator(
        bundle=bundle,
        egnn_predicted=egnn,
        hgnn_predicted=hgnn,
        output_dir=tmp_path,
    )

    outputs = animator.run()

    assert isinstance(outputs, list)
    assert len(outputs) == 3
    bin_names = [o.selection.bin_name for o in outputs]
    assert bin_names == ["close", "mid", "far"]
    for out in outputs:
        assert isinstance(out, AnimationOutputs)
        assert out.gif_path is not None
        assert out.gif_path.is_file()
        assert out.gif_path.name == f"{out.selection.basename}.gif"


def _write_selection_yaml(tmp_path: Path, mapping: dict[str, int]) -> Path:
    """Persist a small selection mapping for the manual-selection tests."""
    path = tmp_path / "animation_selection.yaml"
    path.write_text("\n".join(f"{k}: {v}" for k, v in mapping.items()) + "\n")
    return path


def _manual_selector_inputs() -> tuple[
    Trajectories, npt.NDArray[np.floating], npt.NDArray[np.floating]
]:
    """Six-trajectory bundle covering all three bins, finite predictions across the board."""
    truth = _states(n_traj=6)
    egnn = _predicted_with_offsets(truth, [0.1, 0.5, 0.3, 0.4, 0.2, 0.6])
    hgnn = _predicted_with_offsets(truth, [0.2, 0.4, 0.1, 0.5, 0.3, 0.4])
    bundle = _build_bundle(
        truth,
        bin_ids=[0, 0, 1, 1, 2, 2],
        d_mins=[0.01, 0.02, 0.05, 0.07, 0.5, 0.7],
    )
    return bundle, egnn, hgnn


def test_select_trajectories_from_file_returns_requested_indices_in_bin_order(
    tmp_path: Path,
) -> None:
    """A valid file picks the named indices and preserves canonical bin order."""
    bundle, egnn, hgnn = _manual_selector_inputs()
    selection_path = _write_selection_yaml(tmp_path, {"close": 0, "mid": 3, "far": 4})

    selections = select_trajectories_from_file(bundle, egnn, hgnn, selection_path)

    assert [s.bin_name for s in selections] == ["close", "mid", "far"]
    assert [s.traj_index for s in selections] == [0, 3, 4]
    # Numeric fields are populated, mirroring the automatic selector.
    assert selections[0].d_min == bundle.min_pairwise_distance[0]
    assert np.isfinite(selections[0].egnn_mean_position_mse)
    assert np.isfinite(selections[0].hgnn_mean_position_mse)


def test_select_trajectories_from_file_rejects_missing_bin(tmp_path: Path) -> None:
    """A YAML that omits a test-set bin fails fast and names the missing entry."""
    bundle, egnn, hgnn = _manual_selector_inputs()
    selection_path = _write_selection_yaml(tmp_path, {"close": 0, "mid": 3})  # 'far' missing

    with pytest.raises(ValueError, match=r"missing bins: \['far'\]"):
        select_trajectories_from_file(bundle, egnn, hgnn, selection_path)


def test_select_trajectories_from_file_rejects_unknown_bin(tmp_path: Path) -> None:
    """A YAML with a bin name absent from the test set fails fast."""
    bundle, egnn, hgnn = _manual_selector_inputs()
    selection_path = _write_selection_yaml(tmp_path, {"close": 0, "mid": 3, "far": 4, "extreme": 1})

    with pytest.raises(ValueError, match=r"unknown bin names: \['extreme'\]"):
        select_trajectories_from_file(bundle, egnn, hgnn, selection_path)


def test_select_trajectories_from_file_rejects_out_of_range_index(tmp_path: Path) -> None:
    """An index past the test set's row count is caught with a specific message."""
    bundle, egnn, hgnn = _manual_selector_inputs()
    selection_path = _write_selection_yaml(tmp_path, {"close": 0, "mid": 3, "far": 999})

    with pytest.raises(ValueError, match="out of range"):
        select_trajectories_from_file(bundle, egnn, hgnn, selection_path)


def test_select_trajectories_from_file_rejects_wrong_bin_assignment(tmp_path: Path) -> None:
    """An index that exists but lives in a different bin must be rejected loudly."""
    bundle, egnn, hgnn = _manual_selector_inputs()
    # Trajectory 4 is in bin 2 ('far'); assigning it to 'close' must fail.
    selection_path = _write_selection_yaml(tmp_path, {"close": 4, "mid": 3, "far": 5})

    with pytest.raises(ValueError, match="lives in bin id=2"):
        select_trajectories_from_file(bundle, egnn, hgnn, selection_path)


def test_select_trajectories_from_file_rejects_non_finite_prediction(tmp_path: Path) -> None:
    """A manual pick whose rollout diverged for either model is refused at validation time."""
    bundle, egnn, hgnn = _manual_selector_inputs()
    # Poison trajectory 0's EGNN rollout with a mid-frame NaN; mean MSE becomes NaN.
    egnn[0, 1, 0, 0] = float("nan")
    selection_path = _write_selection_yaml(tmp_path, {"close": 0, "mid": 3, "far": 4})

    with pytest.raises(ValueError, match="non-finite mean rollout position MSE for EGNN"):
        select_trajectories_from_file(bundle, egnn, hgnn, selection_path)


def test_select_trajectories_from_file_rejects_non_integer_value(tmp_path: Path) -> None:
    """The loader must reject malformed YAML values (floats, strings, lists, booleans)."""
    bundle, egnn, hgnn = _manual_selector_inputs()
    path = tmp_path / "bad.yaml"
    path.write_text("close: 0.5\nmid: 3\nfar: 4\n")

    with pytest.raises(ValueError, match="must be an integer"):
        select_trajectories_from_file(bundle, egnn, hgnn, path)


def test_animator_uses_manual_selection_when_file_provided(tmp_path: Path) -> None:
    """End-to-end DI smoke: with selection_file set, run() takes the manual path."""
    bundle, egnn, hgnn = _manual_selector_inputs()
    # Manual picks differ from what the automatic selector would choose.
    selection_path = _write_selection_yaml(tmp_path, {"close": 0, "mid": 3, "far": 4})
    animator = _FakeAnimator(
        bundle=bundle,
        egnn_predicted=egnn,
        hgnn_predicted=hgnn,
        output_dir=tmp_path,
        selection_file=selection_path,
    )

    outputs = animator.run()

    assert [o.selection.traj_index for o in outputs] == [0, 3, 4]
    for out in outputs:
        assert out.gif_path is not None
        assert out.gif_path.is_file()
