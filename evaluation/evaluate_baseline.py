"""Run the official evaluation pipeline against a deterministic baseline.

Mirrors evaluation/evaluate.py without depending on a checkpoint or training
config. Writes the same metrics.json and summary.csv artifacts so baseline
results sit alongside trained-model results in the thesis tables.

References:
    - models/baselines.py for the baseline implementations.
    - evaluation/evaluate.py for the metric pipeline this script reuses.
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from data._io import read_trajectories
from evaluation._io import write_evaluation_report, write_summary_csv
from evaluation._types import EvaluationMetadata, EvaluationReport
from evaluation.evaluate import build_evaluation_report
from evaluation.metrics import (
    compute_rollout_mse,
    compute_single_step_metrics,
    run_all_rollouts,
)
from models.baselines import (
    ConstantVelocityBaseline,
    MeanStateBaseline,
    MeanVelocityBaseline,
    PersistenceBaseline,
)
from utils import get_logger

logger = get_logger(__name__)

BASELINES = ("persistence", "constant_velocity", "mean_velocity", "mean_state")
FITTED_BASELINES = frozenset({"mean_velocity", "mean_state"})
DEFAULT_DT = 0.05


def evaluate_baseline(
    *,
    baseline: str,
    test_path: str | Path,
    train_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    dt: float = DEFAULT_DT,
    device: str = "cpu",
) -> EvaluationReport:
    """Evaluate a deterministic baseline and write the standard report artifacts."""
    if baseline not in BASELINES:
        msg = f"unknown baseline {baseline!r}; choose from {BASELINES}"
        raise ValueError(msg)
    if baseline in FITTED_BASELINES and train_path is None:
        msg = f"baseline {baseline!r} requires --train-path to fit"
        raise ValueError(msg)

    test_path = Path(test_path)
    torch_device = torch.device(device)

    model = _build_baseline(baseline, dt=dt, train_path=train_path).to(torch_device)
    model.eval()

    test_bundle = read_trajectories(test_path)
    test_traj = test_bundle.states
    n_traj, n_frames, n_particles, _state_dim = test_traj.shape

    single_step_metrics = compute_single_step_metrics(model, str(test_path), torch_device)
    predicted = run_all_rollouts(model, test_traj, torch_device)
    rollout_mse = compute_rollout_mse(test_traj, predicted)

    metadata = EvaluationMetadata(
        model_name=f"baseline_{baseline}",
        checkpoint_path=None,
        config_path=None,
        test_path=str(test_path),
        device=str(torch_device),
        checkpoint_epoch=None,
        checkpoint_val_loss=None,
        run_id=f"baseline_{baseline}",
        git_commit=None,
        pos_std=float(test_traj[..., :2].std()),
        vel_std=float(test_traj[..., 2:4].std()),
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
    )

    target_dir = _resolve_output_dir(output_dir, baseline)
    target_dir.mkdir(parents=True, exist_ok=True)
    write_evaluation_report(target_dir / "metrics.json", report)
    write_summary_csv(target_dir / "summary.csv", report)
    logger.info("wrote baseline %s report to %s", baseline, target_dir)
    return report


def _build_baseline(
    baseline: str,
    *,
    dt: float,
    train_path: str | Path | None,
) -> nn.Module:
    """Construct the baseline, fitting from train_path when needed."""
    if baseline == "persistence":
        return PersistenceBaseline()
    if baseline == "constant_velocity":
        return ConstantVelocityBaseline(dt=dt)
    if baseline == "mean_velocity":
        return MeanVelocityBaseline.fit(str(train_path), dt=dt)
    if baseline == "mean_state":
        return MeanStateBaseline.fit(str(train_path))
    msg = f"unknown baseline {baseline!r}"
    raise ValueError(msg)


def _resolve_output_dir(output_dir: str | Path | None, baseline: str) -> Path:
    """Default to runs/baselines/<kind>/evaluation when not specified."""
    if output_dir is not None:
        return Path(output_dir)
    return Path("runs/baselines") / baseline / "evaluation"


def main() -> None:
    """Run baseline evaluation from CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a deterministic baseline.")
    parser.add_argument("--baseline", type=str, choices=BASELINES, required=True)
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="required for fitted baselines (mean_velocity, mean_state).",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    evaluate_baseline(
        baseline=args.baseline,
        test_path=args.test_path,
        train_path=args.train_path,
        output_dir=args.output_dir,
        dt=args.dt,
        device=args.device,
    )


if __name__ == "__main__":
    main()
