"""Official numeric evaluation runner for trained checkpoints."""

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from data._io import read_trajectories
from data._types import Trajectories
from evaluation._binning import expand_trajectory_mask_to_transitions, trajectory_masks
from evaluation._io import write_evaluation_report, write_summary_csv
from evaluation._loader import load_trained_model
from evaluation._types import (
    DistanceSummary,
    DivergenceMetrics,
    DriftSummary,
    EncounterBinDefinition,
    EncounterBinReport,
    EncounterBinsReport,
    EnergyDriftCurves,
    EnergyDriftReport,
    EnergyDriftStepSummary,
    EnergyReport,
    EvaluationMetadata,
    EvaluationReport,
    MseSummary,
    PerBinBaselineRatios,
    RolloutCurves,
    RolloutMetricCurves,
    RolloutMetricSeries,
    RolloutMetricSummary,
    RolloutMSE,
    RolloutReport,
    RolloutStepMetrics,
    SingleStepMetrics,
    SingleStepReport,
)
from evaluation.metrics import (
    compute_energy,
    compute_rollout_mse,
    compute_single_step_metrics,
    run_all_rollouts,
    subset_rollout_mse,
)
from evaluation.rollout_score import BaselineEnvelopeComputer
from models.hgnn import HGNN
from training._io import load_config
from training._types import TrainConfig
from training.rollout_score import DEFAULT_ANCHOR_STEPS, compute_rollout_score
from utils import get_logger

logger = get_logger(__name__)

DEFAULT_TEST_PATH = Path("data/output/test.h5")
DIVERGENCE_THRESHOLDS = [1.0, 10.0, 100.0, 1000.0]


def evaluate_checkpoint(
    cfg: TrainConfig,
    checkpoint_path: str | Path,
    *,
    config_path: str | Path,
    test_path: str | Path = DEFAULT_TEST_PATH,
    output_dir: str | Path | None = None,
    device: str = "auto",
) -> EvaluationReport:
    """Evaluate one checkpoint and write JSON/CSV artifacts."""
    checkpoint_path = Path(checkpoint_path)
    config_path = Path(config_path)
    test_path = Path(test_path)

    torch_device = _resolve_device(device)
    loaded = load_trained_model(config_path, checkpoint_path, torch_device)
    checkpoint, model = loaded.checkpoint, loaded.model
    pos_std, vel_std = loaded.pos_std, loaded.vel_std

    test_bundle = read_trajectories(test_path)
    test_traj = test_bundle.states

    single_step_metrics = compute_single_step_metrics(model, str(test_path), torch_device)
    predicted = run_all_rollouts(model, test_traj, torch_device)
    rollout_mse = compute_rollout_mse(test_traj, predicted)

    envelope_computer: BaselineEnvelopeComputer | None = None
    if test_bundle.encounter_bin_id is not None:
        envelope_computer = BaselineEnvelopeComputer(
            train_path=cfg.data.train_path,
            dt=cfg.data.dt,
            device=torch_device,
        )
        envelope_computer.fit(test_traj)

    n_traj, n_frames, n_particles, _state_dim = test_traj.shape
    metadata = EvaluationMetadata(
        model_name=cfg.model.name,
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        test_path=str(test_path),
        device=str(torch_device),
        checkpoint_epoch=checkpoint.epoch,
        checkpoint_val_loss=checkpoint.val_loss,
        run_id=checkpoint.run_id or checkpoint_path.parent.name,
        git_commit=checkpoint.git_commit,
        pos_std=pos_std,
        vel_std=vel_std,
        n_trajectories=n_traj,
        n_frames=n_frames,
        n_transitions=n_frames - 1,
        n_particles=n_particles,
    )
    report = build_evaluation_report(
        model=model,
        test_traj=test_traj,
        predicted=predicted,
        single_step_metrics=single_step_metrics,
        rollout_mse=rollout_mse,
        metadata=metadata,
        device=torch_device,
        test_bundle=test_bundle,
        envelope_computer=envelope_computer,
    )

    target_dir = _output_dir(output_dir, cfg.model.name, checkpoint_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    write_evaluation_report(target_dir / "metrics.json", report)
    write_summary_csv(target_dir / "summary.csv", report)

    logger.info("wrote evaluation report to %s", target_dir)
    return report


def build_evaluation_report(
    *,
    model: nn.Module,
    test_traj: np.ndarray,
    predicted: np.ndarray,
    single_step_metrics: SingleStepMetrics,
    rollout_mse: RolloutMSE,
    metadata: EvaluationMetadata,
    device: torch.device,
    test_bundle: Trajectories | None = None,
    envelope_computer: BaselineEnvelopeComputer | None = None,
    steps: list[int] | None = None,
) -> EvaluationReport:
    """Assemble the typed report from precomputed metrics and metadata.

    A stratified `test_bundle` adds the per-bin `encounter_bins` block; an
    `envelope_computer` adds baseline-normalized rollout scores there.
    """
    if steps is None:
        steps = _summary_steps(test_traj.shape[1] - 1)

    single_step = SingleStepReport(
        state_mse=_summarize_mse(single_step_metrics.state_mse),
        position_mse=_summarize_mse(single_step_metrics.position_mse),
        velocity_mse=_summarize_mse(single_step_metrics.velocity_mse),
        min_pairwise_distance=_summarize_distance(single_step_metrics.min_pairwise_distance),
    )

    rollout = _build_rollout_report(rollout_mse, steps)
    energy = _build_energy_report(model, predicted, device, steps)

    encounter_bins = None
    if test_bundle is not None and test_bundle.encounter_bin_id is not None:
        encounter_bins = _build_encounter_bins_report(
            bundle=test_bundle,
            model=model,
            predicted=predicted,
            single_step_metrics=single_step_metrics,
            rollout_mse=rollout_mse,
            steps=steps,
            device=device,
            n_transitions=metadata.n_transitions,
            envelope_computer=envelope_computer,
        )

    return EvaluationReport(
        metadata=metadata,
        single_step=single_step,
        rollout=rollout,
        energy=energy,
        encounter_bins=encounter_bins,
    )


def _build_rollout_report(rollout_mse: RolloutMSE, steps: list[int]) -> RolloutReport:
    """Assemble a RolloutReport from a (possibly subsetted) RolloutMSE."""
    return RolloutReport(
        steps=_rollout_steps(rollout_mse, steps),
        curves=_rollout_curves(rollout_mse),
        first_nonfinite_step=_first_nonfinite_steps(rollout_mse.state.per_trajectory),
        state_mse_thresholds=_divergence_report(
            rollout_mse.state.per_trajectory,
            DIVERGENCE_THRESHOLDS,
        ),
        position_mse_thresholds=_divergence_report(
            rollout_mse.position.per_trajectory,
            DIVERGENCE_THRESHOLDS,
        ),
        state_final_finite_fraction=_float(rollout_mse.state.finite_fraction[-1]),
    )


def _build_energy_report(
    model: nn.Module,
    predicted: np.ndarray,
    device: torch.device,
    anchor_steps: list[int],
) -> EnergyReport:
    """Assemble an EnergyReport with optional learned-Hamiltonian drift."""
    return EnergyReport(
        physical=_energy_drift_report(predicted, anchor_steps),
        learned_hamiltonian=(
            _learned_hamiltonian_drift(model, predicted, device, anchor_steps)
            if isinstance(model, HGNN)
            else None
        ),
    )


def _build_encounter_bins_report(
    *,
    bundle: Trajectories,
    model: nn.Module,
    predicted: np.ndarray,
    single_step_metrics: SingleStepMetrics,
    rollout_mse: RolloutMSE,
    steps: list[int],
    device: torch.device,
    n_transitions: int,
    envelope_computer: BaselineEnvelopeComputer | None,
) -> EncounterBinsReport:
    """Build the per-bin encounter report from already-computed metric arrays.

    The bundle's stratification fields are required to be populated; the
    caller (build_evaluation_report) gates entry on `encounter_bin_id is
    not None`, and `data/_io.py:_validate_stratification` guarantees the
    other three fields are populated together.
    """
    bin_id = bundle.encounter_bin_id
    d_mins = bundle.min_pairwise_distance
    bins = bundle.encounter_bins
    assert bin_id is not None
    assert d_mins is not None
    assert bins is not None

    masks = trajectory_masks(bin_id, len(bins))

    bin_definitions = [
        EncounterBinDefinition(id=i, name=b.name, lo=b.lo, hi=b.hi) for i, b in enumerate(bins)
    ]

    by_name: dict[str, EncounterBinReport] = {}
    for b, mask in zip(bins, masks, strict=True):
        by_name[b.name] = _build_per_bin_report(
            mask=mask,
            d_mins=d_mins,
            model=model,
            predicted=predicted,
            single_step_metrics=single_step_metrics,
            rollout_mse=rollout_mse,
            steps=steps,
            device=device,
            n_transitions=n_transitions,
            envelope_computer=envelope_computer,
        )

    return EncounterBinsReport(bins=bin_definitions, by_name=by_name)


def _build_per_bin_report(
    *,
    mask: npt.NDArray[np.bool_],
    d_mins: npt.NDArray[np.floating],
    model: nn.Module,
    predicted: np.ndarray,
    single_step_metrics: SingleStepMetrics,
    rollout_mse: RolloutMSE,
    steps: list[int],
    device: torch.device,
    n_transitions: int,
    envelope_computer: BaselineEnvelopeComputer | None,
) -> EncounterBinReport:
    """Build one per-bin report by slicing already-computed metric arrays."""
    count = int(mask.sum())
    expanded_mask = expand_trajectory_mask_to_transitions(mask, n_transitions)

    d_min_summary = _summarize_distance(d_mins[mask])

    single_step = SingleStepReport(
        state_mse=_summarize_mse(single_step_metrics.state_mse[expanded_mask]),
        position_mse=_summarize_mse(single_step_metrics.position_mse[expanded_mask]),
        velocity_mse=_summarize_mse(single_step_metrics.velocity_mse[expanded_mask]),
        min_pairwise_distance=_summarize_distance(
            single_step_metrics.min_pairwise_distance[expanded_mask]
        ),
    )

    bin_rollout_mse = subset_rollout_mse(rollout_mse, mask)
    rollout = _build_rollout_report(bin_rollout_mse, steps)

    bin_predicted = predicted[mask]
    energy = _build_energy_report(model, bin_predicted, device, steps)

    baseline_ratios = None
    if count > 0 and envelope_computer is not None:
        baseline_ratios = _build_baseline_ratios(
            bin_rollout_mse=bin_rollout_mse,
            mask=mask,
            envelope_computer=envelope_computer,
        )

    return EncounterBinReport(
        count=count,
        d_min=d_min_summary,
        single_step=single_step,
        rollout=rollout,
        energy=energy,
        baseline_ratios=baseline_ratios,
    )


def _build_baseline_ratios(
    *,
    bin_rollout_mse: RolloutMSE,
    mask: npt.NDArray[np.bool_],
    envelope_computer: BaselineEnvelopeComputer,
) -> PerBinBaselineRatios:
    """Score the per-bin model curve against the per-bin baseline envelope.

    Reuses training/rollout_score.py:compute_rollout_score so the
    per-bin score has the same semantics as the validation-side score
    (geometric-mean log ratio, dominance horizon, fraction beating
    baseline). Anchor steps default to DEFAULT_ANCHOR_STEPS so this
    block stays consistent with the validation-time score in Block 4.
    """
    model_curve = bin_rollout_mse.state.median[1:]
    envelope = envelope_computer.envelope_for_mask(mask)
    score = compute_rollout_score(model_curve, envelope, anchor_steps=DEFAULT_ANCHOR_STEPS)

    return PerBinBaselineRatios(
        score=_float(score.score),
        state_mse_ratios={s: _float(r) for s, r in score.ratios_at_step.items()},
        dominance_horizon=int(score.dominance_horizon),
        fraction_beating_baseline=_float(score.fraction_beating_baseline),
        final_ratio=_float(score.final_ratio),
    )


def _summary_steps(n_steps: int) -> list[int]:
    """Return standard rollout summary steps within trajectory bounds."""
    candidates = [1, 10, 50, 100, n_steps]
    return list(dict.fromkeys(step for step in candidates if 0 < step <= n_steps))


def _rollout_steps(
    rollout_mse: RolloutMSE,
    steps: list[int],
) -> dict[str, RolloutStepMetrics]:
    """Summarize rollout MSE at selected steps."""
    return {
        str(step): RolloutStepMetrics(
            state_mse=_rollout_metric_summary(rollout_mse.state, step),
            position_mse=_rollout_metric_summary(rollout_mse.position, step),
            velocity_mse=_rollout_metric_summary(rollout_mse.velocity, step),
        )
        for step in steps
    }


def _rollout_metric_summary(series: RolloutMetricSeries, step: int) -> RolloutMetricSummary:
    """Summarize one rollout MSE series at a selected step."""
    return RolloutMetricSummary(
        mean_finite=_float(series.mean[step]),
        median=_float(series.median[step]),
        p95=_optional_float(_finite_percentile(series.per_trajectory[:, step], 95)),
        finite_fraction=_float(series.finite_fraction[step]),
    )


def _rollout_curves(rollout_mse: RolloutMSE) -> RolloutCurves:
    """Return full per-step rollout curves for crossover analysis."""
    return RolloutCurves(
        step=list(range(len(rollout_mse.state.mean))),
        state_mse=_rollout_metric_curves(rollout_mse.state),
        position_mse=_rollout_metric_curves(rollout_mse.position),
        velocity_mse=_rollout_metric_curves(rollout_mse.velocity),
    )


def _rollout_metric_curves(series: RolloutMetricSeries) -> RolloutMetricCurves:
    """Return full per-step curves for one rollout MSE series."""
    steps = range(len(series.mean))
    return RolloutMetricCurves(
        mean_finite=[_float(value) for value in series.mean],
        median=[_float(value) for value in series.median],
        p95=[
            _optional_float(_finite_percentile(series.per_trajectory[:, step], 95))
            for step in steps
        ],
        finite_fraction=[_float(value) for value in series.finite_fraction],
    )


def _first_nonfinite_steps(per_trajectory: np.ndarray) -> list[int | None]:
    """Find the first non-finite MSE step for each trajectory."""
    result = []
    finite = np.isfinite(per_trajectory)

    for row in finite:
        bad = np.flatnonzero(~row)
        result.append(int(bad[0]) if len(bad) else None)

    return result


def _divergence_report(
    per_trajectory: np.ndarray,
    thresholds: list[float],
) -> dict[str, DivergenceMetrics]:
    """Summarize rollout divergence by MSE threshold."""
    return {
        _threshold_key(threshold): DivergenceMetrics(
            first_step=_first_threshold_steps(per_trajectory, threshold),
            final_fraction_below=_fraction_below_at_final(per_trajectory, threshold),
        )
        for threshold in thresholds
    }


def _first_threshold_steps(per_trajectory: np.ndarray, threshold: float) -> list[int | None]:
    """Find first step where each trajectory reaches a threshold."""
    result = []

    for row in per_trajectory:
        bad = np.flatnonzero(np.isfinite(row) & (row >= threshold))
        result.append(int(bad[0]) if len(bad) else None)

    return result


def _fraction_below_at_final(per_trajectory: np.ndarray, threshold: float) -> float | None:
    """Return fraction of trajectories finite and below threshold at final step."""
    final = per_trajectory[:, -1]
    if final.size == 0:
        return None
    return _float(np.mean(np.isfinite(final) & (final < threshold)))


def _energy_drift_report(trajectories: np.ndarray, anchor_steps: list[int]) -> EnergyDriftReport:
    """Summarize relative drift in the known physical energy."""
    drift_matrix = _physical_drift_matrix(trajectories)
    return _build_energy_drift_report(drift_matrix, anchor_steps)


def _learned_hamiltonian_drift(
    model: HGNN,
    trajectories: np.ndarray,
    device: torch.device,
    anchor_steps: list[int],
) -> EnergyDriftReport:
    """Summarize drift in the learned Hamiltonian."""
    drift_matrix = _learned_drift_matrix(model, trajectories, device)
    return _build_energy_drift_report(drift_matrix, anchor_steps)


def _physical_drift_matrix(trajectories: np.ndarray) -> np.ndarray:
    """Compute the (n_traj, n_steps) physical relative-drift matrix."""
    rows = [_relative_drift(compute_energy(traj)) for traj in trajectories]
    if not rows:
        return np.empty((0, 0), dtype=float)
    return np.asarray(rows, dtype=float)


def _learned_drift_matrix(
    model: HGNN, trajectories: np.ndarray, device: torch.device
) -> np.ndarray:
    """Compute the (n_traj, n_steps) learned-Hamiltonian relative-drift matrix."""
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for traj in trajectories:
            state = torch.from_numpy(traj).float().to(device)
            x = state[..., :2] / model.pos_std
            v = state[..., 2:4] / model.vel_std
            mass = state[..., 4:]
            hamiltonian = model.hamiltonian(x, v, mass).detach().cpu().numpy()
            rows.append(_relative_drift(hamiltonian))
    if not rows:
        return np.empty((0, 0), dtype=float)
    return np.asarray(rows, dtype=float)


def _build_energy_drift_report(
    drift_matrix: np.ndarray, anchor_steps: list[int]
) -> EnergyDriftReport:
    """Build the full EnergyDriftReport from a (n_traj, n_steps) drift matrix.

    Handles empty bins (n_traj == 0) by returning a report whose summary
    fields are None and whose per-step / per-traj containers are length-
    consistent with the (zero-trajectory) input.
    """
    n_traj = drift_matrix.shape[0]
    if n_traj == 0:
        final_drifts: list[float | None] = []
        max_drifts: list[float | None] = []
    else:
        final_drifts = [_optional_float(v) for v in drift_matrix[:, -1]]
        max_drifts = [_optional_float(_nanmax(row)) for row in drift_matrix]

    return EnergyDriftReport(
        final_relative_drift=_summarize_drift(np.asarray(final_drifts, dtype=float)),
        max_relative_drift=_summarize_drift(np.asarray(max_drifts, dtype=float)),
        per_trajectory_final=final_drifts,
        per_trajectory_max=max_drifts,
        steps={
            str(step): _energy_drift_step_summary(drift_matrix[:, step] if n_traj else np.empty(0))
            for step in anchor_steps
        },
        curves=_energy_drift_curves(drift_matrix),
        per_trajectory_at_steps={
            str(step): ([_optional_float(v) for v in drift_matrix[:, step]] if n_traj else [])
            for step in anchor_steps
        },
    )


def _energy_drift_step_summary(values: np.ndarray) -> EnergyDriftStepSummary:
    """Summarize drift at one step across trajectories: mean_finite/median/p95/finite_fraction."""
    if values.size == 0:
        return EnergyDriftStepSummary(mean_finite=None, median=None, p95=None, finite_fraction=None)
    finite = values[np.isfinite(values)]
    return EnergyDriftStepSummary(
        mean_finite=_optional_float(np.mean(finite)) if finite.size else None,
        median=_optional_float(np.median(finite)) if finite.size else None,
        p95=_optional_float(np.percentile(finite, 95)) if finite.size else None,
        finite_fraction=_float(np.mean(np.isfinite(values))),
    )


def _energy_drift_curves(drift_matrix: np.ndarray) -> EnergyDriftCurves:
    """Build the per-step curves block from the full drift matrix."""
    n_traj, n_steps = drift_matrix.shape
    if n_traj == 0:
        return EnergyDriftCurves(step=[], mean_finite=[], median=[], p95=[], finite_fraction=[])
    means: list[float | None] = []
    medians: list[float | None] = []
    p95s: list[float | None] = []
    finite_fractions: list[float | None] = []
    for step in range(n_steps):
        summary = _energy_drift_step_summary(drift_matrix[:, step])
        means.append(summary.mean_finite)
        medians.append(summary.median)
        p95s.append(summary.p95)
        finite_fractions.append(summary.finite_fraction)
    return EnergyDriftCurves(
        step=list(range(n_steps)),
        mean_finite=means,
        median=medians,
        p95=p95s,
        finite_fraction=finite_fractions,
    )


def _relative_drift(values: np.ndarray) -> np.ndarray:
    """Compute absolute relative drift against the first value."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.abs((values - values[0]) / values[0])


def _summarize_mse(values: np.ndarray) -> MseSummary:
    """Summarize single-step MSE values (mean, median, max, p95, p99)."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return MseSummary(mean=None, median=None, max=None, p95=None, p99=None)
    return MseSummary(
        mean=_float(np.mean(finite)),
        median=_float(np.median(finite)),
        max=_float(np.max(finite)),
        p95=_float(np.percentile(finite, 95)),
        p99=_float(np.percentile(finite, 99)),
    )


def _summarize_distance(values: np.ndarray) -> DistanceSummary:
    """Summarize minimum pairwise distance values (mean, median, max, p5, p50)."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return DistanceSummary(mean=None, median=None, max=None, p5=None, p50=None)
    return DistanceSummary(
        mean=_float(np.mean(finite)),
        median=_float(np.median(finite)),
        max=_float(np.max(finite)),
        p5=_float(np.percentile(finite, 5)),
        p50=_float(np.percentile(finite, 50)),
    )


def _summarize_drift(values: np.ndarray) -> DriftSummary:
    """Summarize energy drift values (mean, median, max, p95)."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return DriftSummary(mean=None, median=None, max=None, p95=None)
    return DriftSummary(
        mean=_float(np.mean(finite)),
        median=_float(np.median(finite)),
        max=_float(np.max(finite)),
        p95=_float(np.percentile(finite, 95)),
    )


def _finite_percentile(values: np.ndarray, percentile: int) -> float | None:
    """Compute a percentile over finite values only."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return None
    return _float(np.percentile(finite, percentile))


def _threshold_key(threshold: float) -> str:
    """Format threshold values as stable JSON/CSV key fragments."""
    if threshold.is_integer():
        return str(int(threshold))
    return str(threshold).replace(".", "p")


def _nanmax(values: np.ndarray) -> float:
    """Return nan when no finite max exists."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(np.max(finite))


def _float(value: object) -> float | None:
    """Convert scalar values to JSON-safe floats."""
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _optional_float(value: object | None) -> float | None:
    """Convert optional scalar values to JSON-safe floats."""
    if value is None:
        return None
    return _float(value)


def _output_dir(output_dir: str | Path | None, model_name: str, checkpoint_path: Path) -> Path:
    """Resolve the report output directory.

    Resolution order:
        1. Explicit `output_dir` always wins.
        2. If the checkpoint sits under any `runs/` ancestor (canonical layout),
           default to `<run_dir>/evaluation/` so artifacts stay self-contained.

    Checkpoints that match neither path raise rather than silently scattering
    reports outside the canonical layout. The `model_name` argument is kept for
    callers that resolve output dirs from training config (signature parity).
    """
    del model_name  # reserved for future custom resolution
    if output_dir is not None:
        return Path(output_dir)
    if "runs" in checkpoint_path.parts:
        return checkpoint_path.parent / "evaluation"
    msg = (
        f"cannot infer output directory for checkpoint at {checkpoint_path}: "
        "checkpoint is not under a `runs/` ancestor and no --output-dir was given"
    )
    raise ValueError(msg)


def _resolve_device(device_cfg: str) -> torch.device:
    """Resolve device string to a torch.device."""
    if device_cfg != "auto":
        return torch.device(device_cfg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    """Run evaluation from CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-path", type=str, default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate_checkpoint(
        cfg,
        args.checkpoint,
        config_path=args.config,
        test_path=args.test_path,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
