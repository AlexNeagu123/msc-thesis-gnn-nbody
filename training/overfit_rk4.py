"""Train-only, fixed-window canonical HGNN overfit diagnostic (not validation).

Uses the production trainer's loss, optimizer and gradient-handling loop on one
repeated fixed batch. Never accesses validation/test files or selects a scientific
checkpoint. Run the bounded pilot only after inspecting this diagnostic.
"""

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation._io import _json_safe
from models.baselines import ConstantVelocityBaseline, PersistenceBaseline
from models.hgnn_canonical import MODEL_ID
from training._io import load_config, save_checkpoint
from training._types import Checkpoint
from training.fixed_validation import file_sha256
from training.train import Trainer


def _metrics(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> dict:
    """Measure all fixed-window futures, restoring caller mode and keeping failures visible."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            state = inputs
            predictions = []
            for _ in range(targets.shape[1]):
                state = model(state)
                predictions.append(state)
            predicted = torch.stack(predictions, dim=1).double()
            error = predicted[..., :4] - targets[..., :4].double()
            finite = bool(torch.isfinite(predicted).all())
            return {
                "state4_mse": float(error.square().mean()),
                "position_mae": float(error[..., :2].abs().mean()),
                "velocity_mae": float(error[..., 2:4].abs().mean()),
                "finite": finite,
            }
    finally:
        model.train(was_training)


def run_overfit(
    config_path: Path,
    output_dir: Path,
    *,
    steps: int = 500,
    horizon: int = 5,
    windows: int = 32,
    trajectories: int = 4,
    log_every: int = 50,
) -> dict:
    """Fit one frozen training batch; save provenance, progress and a diagnostic checkpoint."""
    for name, value in (
        ("steps", steps),
        ("horizon", horizon),
        ("windows", windows),
        ("trajectories", trajectories),
        ("log_every", log_every),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if windows < trajectories:
        raise ValueError("windows must be >= trajectories")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    cfg = load_config(config_path)
    if cfg.model.name != MODEL_ID:
        raise ValueError("the overfit diagnostic requires hgnn_canonical_v1")
    # Both constructor data paths deliberately point at TRAIN. No genuine validation
    # file is opened. Fit the same full-training-input scales as the initial pilot.
    cfg = replace(
        cfg,
        data=replace(cfg.data, val_path=cfg.data.train_path, n_train_trajectories=None),
        training=replace(
            cfg.training,
            epochs=steps,
            multi_step_horizon=1,
            multi_step_gamma=1.0,
            noise_factor=0.0,
            checkpoint_metric="val_loss",
            curriculum_horizons=None,
            curriculum_epochs=None,
            curriculum_lrs=None,
            curriculum_gradient_clip_norms=None,
            max_train_batches_per_epoch=None,
        ),
        scheduler=replace(cfg.scheduler, enabled=False),
        checkpointing=replace(cfg.checkpointing, enabled=False),
        logging=replace(cfg.logging, enabled=False),
    )
    begin = time.perf_counter()
    trainer = Trainer(cfg)
    states = trainer.train_loader.dataset.states
    if horizon >= states.shape[1] or trajectories > len(states):
        raise ValueError("requested horizon/trajectory count exceeds training data")
    starts_available = states.shape[1] - horizon
    if (windows + trajectories - 1) // trajectories > starts_available:
        raise ValueError("too many distinct windows for the selected training simulations")
    rng = np.random.default_rng(cfg.training.seed)
    chosen = rng.choice(len(states), size=trajectories, replace=False)
    pairs = []
    for index, trajectory in enumerate(chosen):
        count = windows // trajectories + int(index < windows % trajectories)
        starts = rng.choice(starts_available, size=count, replace=False)
        pairs.extend((int(trajectory), int(start)) for start in starts)
    pairs = [pairs[index] for index in rng.permutation(windows)]
    inputs = torch.stack([states[trajectory, start] for trajectory, start in pairs])
    targets = torch.stack(
        [states[trajectory, start + 1 : start + 1 + horizon] for trajectory, start in pairs]
    )
    with h5py.File(cfg.data.train_path, "r") as handle:
        ids = handle["simulation_ids"][:] if "simulation_ids" in handle else np.arange(len(states))
    if len(ids) != len(states):
        raise ValueError("training simulation IDs disagree with trajectory count")
    trainer.current_horizon = horizon
    cfg.training.multi_step_horizon = horizon
    trainer.train_loader = DataLoader(
        TensorDataset(inputs, targets[:, 0] if horizon == 1 else targets),
        batch_size=windows,
        shuffle=False,
    )
    trainer.diagnostics.dataset = None
    inputs_device, targets_device = inputs.to(trainer.device), targets.to(trainer.device)
    # Reserve a NEW output directory before any optimizer update; never overwrite.
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez(
        output_dir / "fixed_windows.npz",
        inputs=inputs.numpy(),
        targets=targets.numpy(),
        trajectory_indices=np.array([pair[0] for pair in pairs]),
        simulation_ids=np.array([ids[pair[0]] for pair in pairs]),
        start_frames=np.array([pair[1] for pair in pairs]),
    )
    baselines = {
        "persistence": _metrics(
            PersistenceBaseline().to(trainer.device), inputs_device, targets_device
        ),
        "constant_velocity": _metrics(
            ConstantVelocityBaseline(cfg.data.dt).to(trainer.device), inputs_device, targets_device
        ),
    }
    metadata = {
        "protocol": "daniela-fixed-training-overfit-v1",
        "purpose": "memorization_diagnostic_not_validation",
        "source_config": str(config_path.resolve()),
        "source_config_sha256": file_sha256(config_path),
        "effective_config": asdict(cfg),
        "train_sha256": trainer.data_hashes["train"],
        "source_sha256": {
            **trainer.source_hashes,
            "training/overfit_rk4.py": file_sha256(Path(__file__)),
        },
        "normalization_scope": "all_training_inputs_horizon1",
        "normalization_trajectories": len(states),
        "pos_std": trainer.pos_std,
        "momentum_std": trainer.momentum_std,
        "seed": cfg.training.seed,
        "horizon": horizon,
        "windows": windows,
        "selected_simulation_ids": ids[chosen].tolist(),
        "fixed_windows_sha256": file_sha256(output_dir / "fixed_windows.npz"),
        "torch_version": torch.__version__,
        "device": str(trainer.device),
        "inference_dtype": "float32",
        "metric_dtype": "float64",
        "validation_or_test_accessed": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(_json_safe(metadata), indent=2, allow_nan=False) + "\n"
    )
    print(
        f"Frozen {windows} windows from {trajectories} TRAINING simulations; horizon={horizon}; no validation/test access",
        flush=True,
    )
    for name, values in baselines.items():
        print(f"baseline {name}: {json.dumps(values)}", flush=True)
    history = []

    def probe(attempt: int, grad_norm: float | None = None) -> None:
        values = _metrics(trainer.model, inputs_device, targets_device)
        row = {
            "attempt": attempt,
            "optimizer_steps": trainer.optimizer_steps,
            "grad_norm": grad_norm,
            **values,
        }
        history.append(row)
        with (output_dir / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(_json_safe(row), allow_nan=False) + "\n")
        print(
            f"attempt={attempt}/{steps} updates={trainer.optimizer_steps} "
            f"state_mse={values['state4_mse']:.6g} pos_mae={values['position_mae']:.6g} "
            f"vel_mae={values['velocity_mae']:.6g} grad_norm={grad_norm}",
            flush=True,
        )

    probe(0)
    for attempt in range(1, steps + 1):
        summary = trainer._run_epoch(training=True, verbose=False)
        if attempt == 1 or attempt % log_every == 0 or attempt == steps:
            probe(attempt, summary.grad_norm_mean)
    initial, final = history[0], history[-1]
    report = {
        "metadata": metadata,
        "baselines": baselines,
        "history": history,
        "attempted_batches": trainer.attempted_train_batches,
        "optimizer_steps": trainer.optimizer_steps,
        "checks": {
            "finite_final_predictions": final["finite"],
            "training_state_mse_reduced": bool(
                final["finite"] and final["state4_mse"] < initial["state4_mse"]
            ),
            "beats_cv_state_mse": bool(
                final["finite"]
                and final["state4_mse"] < baselines["constant_velocity"]["state4_mse"]
            ),
            "beats_cv_position_mae": bool(
                final["finite"]
                and final["position_mae"] < baselines["constant_velocity"]["position_mae"]
            ),
        },
        "wall_seconds_before_final_artifact_write": time.perf_counter() - begin,
        "note": "Fixed training-window memorization only; not generalization or a final model/runtime comparison.",
    }
    save_checkpoint(
        output_dir / "overfit.pt",
        Checkpoint(
            epoch=steps,
            model=trainer.model.state_dict(),
            optimizer=trainer.optimizer.state_dict(),
            val_loss=float("nan"),
            config=asdict(cfg),
            model_name=MODEL_ID,
            run_id=trainer.run_id,
            pos_std=trainer.pos_std,
            vel_std=trainer.vel_std,
            git_commit=trainer._git_commit(),
            selected_metric="training_subset_state4_mse",
            selected_score=final["state4_mse"],
        ),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n"
    )
    print(f"Saved {output_dir / 'summary.json'}; checks={report['checks']}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/hgnn-pilot.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--windows", type=int, default=32)
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    run_overfit(
        args.config,
        args.output_dir,
        steps=args.steps,
        horizon=args.horizon,
        windows=args.windows,
        trajectories=args.trajectories,
        log_every=args.log_every,
    )
