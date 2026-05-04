"""Official numeric evaluation runner for trained checkpoints."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data._io import read_states
from data.dataset import NBodyDataset
from evaluation._io import write_evaluation_report, write_summary_csv
from evaluation._types import (
    DistanceSummary,
    DivergenceMetrics,
    DriftSummary,
    EnergyDriftReport,
    EnergyReport,
    EvaluationMetadata,
    EvaluationReport,
    MseSummary,
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
)
from models.hgnn import HGNN
from training._io import load_checkpoint, load_config
from training._types import Checkpoint, TrainConfig
from training.train import build_model
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
    checkpoint = load_checkpoint(checkpoint_path, torch_device)
    pos_std, vel_std = _normalization_stats(cfg, checkpoint)
    model = _load_model(cfg, checkpoint, pos_std, vel_std, torch_device)

    test_traj = read_states(test_path)

    single_step_metrics = compute_single_step_metrics(model, str(test_path), torch_device)
    predicted = run_all_rollouts(model, test_traj, torch_device)
    rollout_mse = compute_rollout_mse(test_traj, predicted)

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
    steps: list[int] | None = None,
) -> EvaluationReport:
    """Build the typed evaluation report from precomputed metrics and caller metadata."""
    if steps is None:
        steps = _summary_steps(test_traj.shape[1] - 1)

    single_step = SingleStepReport(
        state_mse=_summarize_mse(single_step_metrics.state_mse),
        position_mse=_summarize_mse(single_step_metrics.position_mse),
        velocity_mse=_summarize_mse(single_step_metrics.velocity_mse),
        min_pairwise_distance=_summarize_distance(single_step_metrics.min_pairwise_distance),
    )

    rollout = RolloutReport(
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

    energy = EnergyReport(
        physical=_energy_drift_report(predicted),
        learned_hamiltonian=(
            _learned_hamiltonian_drift(model, predicted, device)
            if isinstance(model, HGNN)
            else None
        ),
    )

    return EvaluationReport(
        metadata=metadata,
        single_step=single_step,
        rollout=rollout,
        energy=energy,
    )


def _load_model(
    cfg: TrainConfig,
    checkpoint: Checkpoint,
    pos_std: float,
    vel_std: float,
    device: torch.device,
) -> nn.Module:
    """Build the configured model and load checkpoint weights."""
    model = build_model(cfg, pos_std=pos_std, vel_std=vel_std).to(device)
    model.load_state_dict(checkpoint.model)
    model.eval()
    return model


def _normalization_stats(
    cfg: TrainConfig,
    checkpoint: Checkpoint,
) -> tuple[float, float]:
    """Load checkpoint normalization stats, falling back to train data."""
    if checkpoint.pos_std is not None and checkpoint.vel_std is not None:
        return checkpoint.pos_std, checkpoint.vel_std

    train_path = Path(cfg.data.train_path)
    if train_path.exists():
        train_set = NBodyDataset(str(train_path))
        return (
            float(train_set.inputs[..., :2].std()),
            float(train_set.inputs[..., 2:4].std()),
        )

    msg = f"Missing checkpoint normalization stats and train data: {train_path}"
    raise FileNotFoundError(msg)


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
    return _float(np.mean(np.isfinite(final) & (final < threshold)))


def _energy_drift_report(trajectories: np.ndarray) -> EnergyDriftReport:
    """Summarize relative drift in the known physical energy."""
    final_drifts = []
    max_drifts = []

    for traj in trajectories:
        energy = compute_energy(traj)
        drift = _relative_drift(energy)
        final_drifts.append(drift[-1])
        max_drifts.append(_nanmax(drift))

    return EnergyDriftReport(
        final_relative_drift=_summarize_drift(np.asarray(final_drifts)),
        max_relative_drift=_summarize_drift(np.asarray(max_drifts)),
        per_trajectory_final=final_drifts,
        per_trajectory_max=max_drifts,
    )


def _learned_hamiltonian_drift(
    model: HGNN,
    trajectories: np.ndarray,
    device: torch.device,
) -> EnergyDriftReport:
    """Summarize drift in the learned Hamiltonian."""
    final_drifts = []
    max_drifts = []

    with torch.no_grad():
        for traj in trajectories:
            state = torch.from_numpy(traj).float().to(device)
            x = state[..., :2] / model.pos_std
            v = state[..., 2:4] / model.vel_std
            mass = state[..., 4:]
            hamiltonian = model.hamiltonian(x, v, mass).detach().cpu().numpy()
            drift = _relative_drift(hamiltonian)
            final_drifts.append(drift[-1])
            max_drifts.append(_nanmax(drift))

    return EnergyDriftReport(
        final_relative_drift=_summarize_drift(np.asarray(final_drifts)),
        max_relative_drift=_summarize_drift(np.asarray(max_drifts)),
        per_trajectory_final=final_drifts,
        per_trajectory_max=max_drifts,
    )


def _relative_drift(values: np.ndarray) -> np.ndarray:
    """Compute absolute relative drift against the first value."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.abs((values - values[0]) / values[0])


def _summarize_mse(values: np.ndarray) -> MseSummary:
    """Summarize single-step MSE values (mean/median/max + p95, p99)."""
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
    """Summarize minimum pairwise distance values (mean/median/max + p5, p50)."""
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
    """Summarize energy drift values (mean/median/max + p95)."""
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
        3. Otherwise (legacy `checkpoints/<model>/<run_id>/`), fall back to
           `results/evaluation/<model>/<run_id>/` for ad-hoc evaluation.
    """
    if output_dir is not None:
        return Path(output_dir)
    if "runs" in checkpoint_path.parts:
        return checkpoint_path.parent / "evaluation"
    return Path("results") / "evaluation" / model_name / checkpoint_path.parent.name


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
