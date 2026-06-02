"""Render one 3-panel (truth | EGNN | HGNN) animation per encounter bin for the report."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import yaml
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter, writers
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from data._io import read_trajectories
from data._types import Trajectories
from evaluation._loader import load_trained_model
from evaluation.metrics import run_all_rollouts
from utils import get_logger

logger = get_logger(__name__)

DEFAULT_FPS = 20
_PARTICLE_PALETTE = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#17becf")
_AXIS_PADDING = 0.5


@dataclass(frozen=True)
class BestTrajectory:
    """One per-bin trajectory chosen for animation, with the MSE that picked it."""

    bin_id: int
    bin_name: str
    traj_index: int
    d_min: float
    egnn_mean_position_mse: float
    hgnn_mean_position_mse: float

    @property
    def basename(self) -> str:
        """File stem shared between the MP4 and GIF outputs."""
        return f"{self.bin_name}_traj_{self.traj_index}"


@dataclass(frozen=True)
class AnimationOutputs:
    """Resolved on-disk artifacts for one rendered animation."""

    selection: BestTrajectory
    mp4_path: Path | None
    gif_path: Path | None


def select_best_trajectories(
    test_bundle: Trajectories,
    egnn_predicted: npt.NDArray[np.floating],
    hgnn_predicted: npt.NDArray[np.floating],
) -> list[BestTrajectory]:
    """Pick, per bin, the trajectory minimising EGNN+HGNN mean rollout position MSE.

    Only trajectories finite under both models are eligible; a bin with none raises.
    """
    _require_stratified_bundle(test_bundle)
    _require_matching_shapes(test_bundle.states, egnn_predicted, hgnn_predicted)
    assert test_bundle.encounter_bins is not None  # narrowed by _require_stratified_bundle
    assert test_bundle.encounter_bin_id is not None
    assert test_bundle.min_pairwise_distance is not None

    egnn_mean_mse = _per_trajectory_mean_position_mse(test_bundle.states, egnn_predicted)
    hgnn_mean_mse = _per_trajectory_mean_position_mse(test_bundle.states, hgnn_predicted)
    joint = egnn_mean_mse + hgnn_mean_mse
    finite = np.isfinite(egnn_mean_mse) & np.isfinite(hgnn_mean_mse)

    selections: list[BestTrajectory] = []
    for bin_id, bin_def in enumerate(test_bundle.encounter_bins):
        bin_mask = test_bundle.encounter_bin_id == bin_id
        eligible = bin_mask & finite
        if not eligible.any():
            msg = (
                f"bin {bin_def.name!r} has no trajectories with finite mean rollout position MSE "
                "for both models; cannot select a representative animation"
            )
            raise ValueError(msg)
        candidate_indices = np.flatnonzero(eligible)
        winner = int(candidate_indices[np.argmin(joint[candidate_indices])])
        selections.append(
            BestTrajectory(
                bin_id=bin_id,
                bin_name=bin_def.name,
                traj_index=winner,
                d_min=float(test_bundle.min_pairwise_distance[winner]),
                egnn_mean_position_mse=float(egnn_mean_mse[winner]),
                hgnn_mean_position_mse=float(hgnn_mean_mse[winner]),
            )
        )
    return selections


def select_trajectories_from_file(
    test_bundle: Trajectories,
    egnn_predicted: npt.NDArray[np.floating],
    hgnn_predicted: npt.NDArray[np.floating],
    selection_path: Path,
) -> list[BestTrajectory]:
    """Resolve a hand-edited YAML `{bin_name: traj_index}` into validated selections.

    Every bin must appear once, each index must live in its declared bin, and both
    models must be finite there; returned in canonical bin order.
    """
    _require_stratified_bundle(test_bundle)
    _require_matching_shapes(test_bundle.states, egnn_predicted, hgnn_predicted)
    assert test_bundle.encounter_bins is not None
    assert test_bundle.encounter_bin_id is not None
    assert test_bundle.min_pairwise_distance is not None

    raw = _load_selection_yaml(selection_path)
    bins = test_bundle.encounter_bins
    bin_names = [b.name for b in bins]
    _require_complete_bin_coverage(raw, bin_names, selection_path)

    egnn_mse = _per_trajectory_mean_position_mse(test_bundle.states, egnn_predicted)
    hgnn_mse = _per_trajectory_mean_position_mse(test_bundle.states, hgnn_predicted)
    n_traj = test_bundle.states.shape[0]

    selections: list[BestTrajectory] = []
    for bin_id, bin_def in enumerate(bins):
        idx = int(raw[bin_def.name])
        _require_index_in_range(idx, bin_def.name, n_traj)
        _require_index_in_bin(idx, bin_def.name, bin_id, test_bundle.encounter_bin_id)
        _require_finite_predictions(idx, bin_def.name, egnn_mse, hgnn_mse)
        selections.append(
            BestTrajectory(
                bin_id=bin_id,
                bin_name=bin_def.name,
                traj_index=idx,
                d_min=float(test_bundle.min_pairwise_distance[idx]),
                egnn_mean_position_mse=float(egnn_mse[idx]),
                hgnn_mean_position_mse=float(hgnn_mse[idx]),
            )
        )
    return selections


def render_three_panel_animation(
    true_traj: npt.NDArray[np.floating],
    egnn_pred: npt.NDArray[np.floating],
    hgnn_pred: npt.NDArray[np.floating],
    selection: BestTrajectory,
    *,
    mp4_path: Path,
    gif_path: Path,
    fps: int = DEFAULT_FPS,
) -> AnimationOutputs:
    """Render the 3-panel animation and save MP4 and GIF.

    MP4 is skipped (returns `mp4_path=None`) when ffmpeg is unavailable; the GIF
    still renders. `fps` must be >= 1.
    """
    if fps < 1:
        msg = f"fps must be >= 1; got {fps}"
        raise ValueError(msg)
    n_frames, n_particles, _state_dim = true_traj.shape
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    try:
        x_lim, y_lim = _shared_axis_limits([true_traj, egnn_pred, hgnn_pred])
        panel_data = [true_traj, egnn_pred, hgnn_pred]
        artists = [
            _setup_panel(ax, label, n_particles, x_lim, y_lim)
            for ax, label in zip(axes, ("ground truth", "EGNN", "HGNN"), strict=True)
        ]
        fig.suptitle(_supertitle(selection), fontsize=13)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

        def _update(frame: int) -> list[Line2D]:
            updated: list[Line2D] = []
            for data, (trails, dots) in zip(panel_data, artists, strict=True):
                for p in range(n_particles):
                    trails[p].set_data(data[: frame + 1, p, 0], data[: frame + 1, p, 1])
                    dots[p].set_data([data[frame, p, 0]], [data[frame, p, 1]])
                    updated.extend([trails[p], dots[p]])
            return updated

        anim = FuncAnimation(fig, _update, frames=n_frames, interval=1000 // fps, blit=False)

        mp4_written = _save_mp4_if_possible(anim, mp4_path, fps=fps, selection=selection)
        gif_written = _save_gif(anim, gif_path, fps=fps, selection=selection)
        return AnimationOutputs(
            selection=selection,
            mp4_path=mp4_path if mp4_written else None,
            gif_path=gif_path if gif_written else None,
        )
    finally:
        plt.close(fig)


class BestTrajectoryAnimator:
    """Orchestrate model loading, rollouts, selection, and rendering."""

    def __init__(
        self,
        *,
        egnn_checkpoint: Path,
        hgnn_checkpoint: Path,
        egnn_config: Path,
        hgnn_config: Path,
        test_path: Path,
        output_dir: Path,
        device: str = "auto",
        fps: int = DEFAULT_FPS,
        selection_file: Path | None = None,
    ) -> None:
        """Store input paths; `selection_file` overrides the automatic selector when set."""
        self.egnn_checkpoint = egnn_checkpoint
        self.hgnn_checkpoint = hgnn_checkpoint
        self.egnn_config = egnn_config
        self.hgnn_config = hgnn_config
        self.test_path = test_path
        self.output_dir = output_dir
        self.device = device
        self.fps = fps
        self.selection_file = selection_file

    def run(self) -> list[AnimationOutputs]:
        """Execute the full pipeline; return one AnimationOutputs per encounter bin."""
        torch_device = self._resolve_device()
        test_bundle = self._read_trajectories()
        egnn_predicted = self._rollout_for_model(
            "egnn", self.egnn_config, self.egnn_checkpoint, test_bundle.states, torch_device
        )
        hgnn_predicted = self._rollout_for_model(
            "hgnn", self.hgnn_config, self.hgnn_checkpoint, test_bundle.states, torch_device
        )
        if self.selection_file is None:
            selections = select_best_trajectories(test_bundle, egnn_predicted, hgnn_predicted)
        else:
            logger.info("using manual trajectory selection from %s", self.selection_file)
            selections = select_trajectories_from_file(
                test_bundle, egnn_predicted, hgnn_predicted, self.selection_file
            )
        return self._render_all(test_bundle, egnn_predicted, hgnn_predicted, selections)

    def _resolve_device(self) -> torch.device:
        """Resolve 'auto' to the strongest backend, otherwise honour the literal string."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)

    def _read_trajectories(self) -> Trajectories:
        """Hook: override in tests to inject a fake test bundle."""
        return read_trajectories(self.test_path)

    def _rollout_for_model(
        self,
        model_name: str,
        config_path: Path,
        checkpoint_path: Path,
        test_states: npt.NDArray[np.floating],
        torch_device: torch.device,
    ) -> npt.NDArray[np.floating]:
        """Load one model and run autoregressive rollouts on the test set."""
        model = load_trained_model(config_path, checkpoint_path, torch_device).model
        logger.info("running %s rollouts on %d trajectories", model_name, test_states.shape[0])
        return run_all_rollouts(model, test_states, torch_device)

    def _render_all(
        self,
        test_bundle: Trajectories,
        egnn_predicted: npt.NDArray[np.floating],
        hgnn_predicted: npt.NDArray[np.floating],
        selections: list[BestTrajectory],
    ) -> list[AnimationOutputs]:
        """Render one animation per selection and collect on-disk artifact paths."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[AnimationOutputs] = []
        for sel in selections:
            mp4_path = self.output_dir / f"{sel.basename}.mp4"
            gif_path = self.output_dir / f"{sel.basename}.gif"
            outputs.append(
                render_three_panel_animation(
                    test_bundle.states[sel.traj_index],
                    egnn_predicted[sel.traj_index],
                    hgnn_predicted[sel.traj_index],
                    sel,
                    mp4_path=mp4_path,
                    gif_path=gif_path,
                    fps=self.fps,
                )
            )
        logger.info("rendered %d animations to %s", len(outputs), self.output_dir)
        return outputs


def _require_stratified_bundle(test_bundle: Trajectories) -> None:
    """Fail fast when the test bundle lacks stratification metadata."""
    if (
        test_bundle.encounter_bins is None
        or test_bundle.encounter_bin_id is None
        or test_bundle.min_pairwise_distance is None
    ):
        msg = (
            "best-trajectory animation requires a stratified test set; "
            "encounter_bins / encounter_bin_id / min_pairwise_distance are all required"
        )
        raise ValueError(msg)


def _per_trajectory_mean_position_mse(
    test_traj: npt.NDArray[np.floating],
    predicted: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Rollout-averaged position MSE per trajectory, used as the selection signal."""
    diff = predicted[..., :2] - test_traj[..., :2]
    # non-finite steps propagate, so divergent rollouts surface as a non-finite mean
    return (diff**2).mean(axis=(1, 2, 3))


def _require_matching_shapes(
    truth_states: npt.NDArray[np.floating],
    egnn_predicted: npt.NDArray[np.floating],
    hgnn_predicted: npt.NDArray[np.floating],
) -> None:
    """Reject prediction arrays whose shape disagrees with the test bundle."""
    if egnn_predicted.shape != truth_states.shape:
        msg = (
            "egnn_predicted shape does not match test_bundle.states; "
            f"got {egnn_predicted.shape}, expected {truth_states.shape}"
        )
        raise ValueError(msg)
    if hgnn_predicted.shape != truth_states.shape:
        msg = (
            "hgnn_predicted shape does not match test_bundle.states; "
            f"got {hgnn_predicted.shape}, expected {truth_states.shape}"
        )
        raise ValueError(msg)


def _load_selection_yaml(path: Path) -> dict[str, int]:
    """Parse `{bin_name: traj_index}` from disk; reject malformed shapes up front."""
    with path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        msg = f"selection file {path} must be a mapping; got {type(raw).__name__}"
        raise ValueError(msg)
    parsed: dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            msg = f"selection key {key!r} in {path} must be a string"
            raise ValueError(msg)
        # bool is a subclass of int in Python; reject explicitly so `true`/`false` are not coerced.
        if isinstance(value, bool) or not isinstance(value, int):
            msg = (
                f"selection value for {key!r} in {path} must be an integer; "
                f"got {type(value).__name__}"
            )
            raise ValueError(msg)
        parsed[key] = value
    return parsed


def _require_complete_bin_coverage(
    raw: dict[str, int], bin_names: list[str], selection_path: Path
) -> None:
    """Every test-set bin must be named exactly once; no extras allowed."""
    yaml_names = set(raw.keys())
    expected = set(bin_names)
    missing = sorted(expected - yaml_names)
    unknown = sorted(yaml_names - expected)
    if missing:
        msg = (
            f"selection file {selection_path} is missing bins: {missing}. "
            f"Expected one entry per encounter bin: {bin_names}"
        )
        raise ValueError(msg)
    if unknown:
        msg = (
            f"selection file {selection_path} has unknown bin names: {unknown}. "
            f"Expected one entry per encounter bin: {bin_names}"
        )
        raise ValueError(msg)


def _require_index_in_range(idx: int, bin_name: str, n_traj: int) -> None:
    """Trajectory indices must point at a real row of the test bundle."""
    if idx < 0 or idx >= n_traj:
        msg = (
            f"trajectory index {idx} for bin {bin_name!r} is out of range (n_trajectories={n_traj})"
        )
        raise ValueError(msg)


def _require_index_in_bin(
    idx: int,
    bin_name: str,
    bin_id: int,
    encounter_bin_id: npt.NDArray[np.integer],
) -> None:
    """The selected index must actually live in the bin the YAML assigns it to."""
    actual_bin_id = int(encounter_bin_id[idx])
    if actual_bin_id != bin_id:
        msg = (
            f"trajectory {idx} is assigned to bin {bin_name!r} (id={bin_id}) in the "
            f"selection file but lives in bin id={actual_bin_id} in the test set"
        )
        raise ValueError(msg)


def _require_finite_predictions(
    idx: int,
    bin_name: str,
    egnn_mse: npt.NDArray[np.floating],
    hgnn_mse: npt.NDArray[np.floating],
) -> None:
    """Reject manual picks whose rollout diverged; the animation would be a NaN swarm."""
    egnn_ok = bool(np.isfinite(egnn_mse[idx]))
    hgnn_ok = bool(np.isfinite(hgnn_mse[idx]))
    if not (egnn_ok and hgnn_ok):
        missing_models = [name for name, ok in (("EGNN", egnn_ok), ("HGNN", hgnn_ok)) if not ok]
        msg = (
            f"trajectory {idx} in bin {bin_name!r} has non-finite mean rollout position MSE for "
            f"{', '.join(missing_models)}; manual selection cannot use it"
        )
        raise ValueError(msg)


def _shared_axis_limits(
    panels: list[npt.NDArray[np.floating]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute common (xlim, ylim) from finite positions across all three panels."""
    xs: list[float] = []
    ys: list[float] = []
    for panel in panels:
        finite = np.isfinite(panel[..., 0]) & np.isfinite(panel[..., 1])
        if not finite.any():
            continue
        xs.append(float(np.min(panel[..., 0][finite])))
        xs.append(float(np.max(panel[..., 0][finite])))
        ys.append(float(np.min(panel[..., 1][finite])))
        ys.append(float(np.max(panel[..., 1][finite])))
    if not xs or not ys:
        return (-1.0, 1.0), (-1.0, 1.0)
    return (
        (min(xs) - _AXIS_PADDING, max(xs) + _AXIS_PADDING),
        (min(ys) - _AXIS_PADDING, max(ys) + _AXIS_PADDING),
    )


def _setup_panel(
    ax: Axes,
    label: str,
    n_particles: int,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
) -> tuple[list[Line2D], list[Line2D]]:
    """Configure one panel's axes and return its (trails, dots) artists."""
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(label)
    trails = [
        ax.plot([], [], color=_particle_color(p), alpha=0.45, linewidth=1.1)[0]
        for p in range(n_particles)
    ]
    dots = [
        ax.plot([], [], "o", color=_particle_color(p), markersize=9)[0] for p in range(n_particles)
    ]
    return trails, dots


def _particle_color(index: int) -> str:
    """Stable per-particle color, cycled if more than the palette length."""
    return _PARTICLE_PALETTE[index % len(_PARTICLE_PALETTE)]


def _supertitle(selection: BestTrajectory) -> str:
    """Compact one-line title with bin, trajectory index, d_min, and mean rollout position MSE."""
    return (
        f"bin={selection.bin_name} | trajectory {selection.traj_index} | "
        f"d_min={selection.d_min:.4g} | "
        f"EGNN mean position MSE={selection.egnn_mean_position_mse:.4g} | "
        f"HGNN mean position MSE={selection.hgnn_mean_position_mse:.4g}"
    )


def _save_mp4_if_possible(
    anim: FuncAnimation, path: Path, *, fps: int, selection: BestTrajectory
) -> bool:
    """Try to save the MP4; return False with a clear log when ffmpeg is missing."""
    if not writers.is_available("ffmpeg"):
        logger.error(
            "skipping MP4 for %s: ffmpeg is not available. Install it via "
            "`brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux); "
            "GIF export will still be attempted.",
            selection.basename,
        )
        return False
    try:
        anim.save(str(path), writer=FFMpegWriter(fps=fps))
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        logger.error("failed to write %s via ffmpeg: %s", path, exc)
        return False
    return True


def _save_gif(anim: FuncAnimation, path: Path, *, fps: int, selection: BestTrajectory) -> bool:
    """Save the GIF via Pillow; return False on failure so the caller can log scope."""
    try:
        anim.save(str(path), writer=PillowWriter(fps=fps))
    except (RuntimeError, OSError) as exc:
        logger.error("failed to write GIF %s for %s: %s", path, selection.basename, exc)
        return False
    return True


def main() -> None:
    """CLI entrypoint: parse paths and delegate to BestTrajectoryAnimator.run."""
    parser = argparse.ArgumentParser(
        description="Render per-bin 3-panel animations for the official report.",
    )
    parser.add_argument("--egnn-checkpoint", type=str, required=True)
    parser.add_argument("--hgnn-checkpoint", type=str, required=True)
    parser.add_argument("--egnn-config", type=str, required=True)
    parser.add_argument("--hgnn-config", type=str, required=True)
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--selection-file",
        type=str,
        default=None,
        help=(
            "Optional YAML mapping {bin_name: traj_index}; when set, render those exact "
            "trajectories instead of running the automatic lowest-mean-MSE selector."
        ),
    )
    args = parser.parse_args()

    BestTrajectoryAnimator(
        egnn_checkpoint=Path(args.egnn_checkpoint),
        hgnn_checkpoint=Path(args.hgnn_checkpoint),
        egnn_config=Path(args.egnn_config),
        hgnn_config=Path(args.hgnn_config),
        test_path=Path(args.test_path),
        output_dir=Path(args.output_dir),
        device=args.device,
        fps=args.fps,
        selection_file=Path(args.selection_file) if args.selection_file else None,
    ).run()


if __name__ == "__main__":
    main()
