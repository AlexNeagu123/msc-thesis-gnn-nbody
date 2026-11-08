"""Explicit CPU probe or bounded fresh/resumed development training; never auto-run."""

import argparse
import copy
import json
import math
import os
import resource
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from training._io import load_checkpoint, load_config
from training._types import TrainConfig
from training.colab_runtime import backup_round, verify_backup
from training.cpu_microbatch import CpuMicrobatchTrainer
from training.resume import environment

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs/daniela-cpu-long-horizons-v1"


def write_status(path: Path, value: dict) -> None:
    """Publish small status atomically, without touching checkpoints."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def setup_runtime() -> None:
    """Use the tested single-thread float32 CPU backend; preserve no CUDA assumptions."""
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def probe(cfg: TrainConfig, *, micro_batch_size: int, repeats: int) -> None:
    """Training-only matched-window timings with zero optimizer updates."""
    cfg = replace(
        cfg,
        data=replace(cfg.data, val_path=cfg.data.train_path),
        training=replace(
            cfg.training,
            epochs=1,
            multi_step_horizon=1,
            curriculum_horizons=None,
            curriculum_epochs=None,
        ),
        checkpointing=replace(cfg.checkpointing, enabled=False),
        logging=replace(cfg.logging, enabled=False),
    )
    trainer = CpuMicrobatchTrainer(cfg, micro_batch_size=micro_batch_size)
    states = trainer.train_loader.dataset.states
    rng = np.random.default_rng(cfg.training.seed)
    trajectories = rng.integers(0, len(states), cfg.training.batch_size)
    starts = rng.integers(0, states.shape[1] - 100, cfg.training.batch_size)
    inputs = torch.stack([states[i, t] for i, t in zip(trajectories, starts, strict=True)])
    before = copy.deepcopy(trainer.model.state_dict())
    profiles = []
    for horizon in (20, 50, 100):
        targets = torch.stack(
            [states[i, t + 1 : t + 1 + horizon] for i, t in zip(trajectories, starts, strict=True)]
        )
        trainer.current_horizon = horizon
        samples = []
        for iteration in range(repeats + 1):
            begin = time.perf_counter()
            result = trainer._backward_logical_batch(inputs, targets)
            if result is None:
                raise RuntimeError("nonfinite probe loss")
            norm = torch.nn.utils.clip_grad_norm_(
                trainer.model.parameters(), trainer.current_clip_norm
            )
            if not torch.isfinite(norm):
                raise RuntimeError("nonfinite probe gradients")
            elapsed = time.perf_counter() - begin
            del result
            if iteration:
                samples.append(elapsed)
            print(
                json.dumps(
                    {
                        "horizon": horizon,
                        "iteration": iteration,
                        "seconds": elapsed,
                        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    }
                ),
                flush=True,
            )
        profiles.append(
            {"horizon": horizon, "seconds": samples, "median_seconds": float(np.median(samples))}
        )
    for name, value in before.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, trainer.model.state_dict()[name])
    assert trainer.optimizer_steps == trainer.attempted_train_batches == 0
    output = ROOT / "runs/daniela-cpu-microbatch-probe-v1"
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"probe-{time.time_ns()}.json").open("x") as handle:
        report = {
            "profiles": profiles,
            "environment": environment(trainer.device),
            "execution": trainer._execution_identity(cfg),
            "optimizer_steps": 0,
            "parameters_unchanged": True,
            "whole_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            "source_sha256": trainer.source_hashes,
            "scope": "warmup excluded; logical64=weighted microbatches+backward+one clip; excludes optimizer/data loading/validation/checkpoint IO; RSS includes entire process/setup",
        }
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)


def train(cfg: TrainConfig, *, micro_batch_size: int, rounds: int | None) -> None:
    """Resume a verified local CPU round or start a separate fresh CPU experiment."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    identity = {
        "protocol": "canonical-cpu-microbatch-v1",
        "micro_batch_size": micro_batch_size,
        "effective_batch_size": cfg.training.batch_size,
    }
    plan = {"config": asdict(cfg), "cpu_execution": identity}
    plan_path = RUN_DIR / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("Saved CPU experiment has a different recipe; do not overwrite it")
    else:
        with plan_path.open("x") as handle:
            json.dump(plan, handle, indent=2)
    backups = RUN_DIR / "backups"
    verified = [(verify_backup(p.parent), p.parent) for p in backups.glob("*/*/COMMITTED.json")]
    if len({p.parent.name for _, p in verified}) > 1:
        raise ValueError("Multiple logical CPU runs found; choose explicitly rather than silently")
    latest_path = None
    total_rounds = sum(cfg.training.curriculum_epochs)
    if verified:
        marker, snapshot = max(verified, key=lambda pair: pair[0]["epoch"])
        if marker["epoch"] == total_rounds:
            print("Already complete; no training repeated. Final snapshot:", snapshot, flush=True)
            return
        if marker["epoch"] > total_rounds:
            raise ValueError("Backup progress exceeds frozen recipe")
        latest_path = snapshot / "latest.pt"
        print("Resuming verified CPU snapshot:", snapshot, flush=True)
    else:
        print(
            "FRESH CPU EXPERIMENT: h1/5/20/50/100, 500 attempts each; logical batch64, microbatch16 by default.",
            flush=True,
        )
    trainer = CpuMicrobatchTrainer(
        cfg, micro_batch_size=micro_batch_size, resume_checkpoint=latest_path
    )
    with (trainer.ckpt_dir / "session_metadata.json").open("x") as handle:
        json.dump(
            {
                "purpose": "development_only_NOT_final_benchmark",
                "cpu_execution": identity,
                "config": asdict(cfg),
                "environment": environment(trainer.device),
                "source_sha256": trainer.source_hashes,
                "resume_checkpoint": str(latest_path) if latest_path else None,
            },
            handle,
            indent=2,
        )
    status_path = RUN_DIR / "status.json"
    write_status(
        status_path,
        {
            "state": "running",
            "run_id": trainer.run_id,
            "segment": str(trainer.ckpt_dir),
            "committed_round": len(trainer._progress.get("train_history", [])),
            "optimizer_steps": trainer.optimizer_steps,
        },
    )
    local_rounds = 0
    while len(trainer._progress.get("train_history", [])) < total_rounds:
        result = trainer.run(max_rounds=1)
        snapshot = backup_round(trainer.ckpt_dir, backups)
        verify_backup(snapshot)
        local_rounds += 1
        complete = len(result.train_history) == total_rounds
        write_status(
            status_path,
            {
                "state": "complete" if complete else "running",
                "run_id": trainer.run_id,
                "snapshot": str(snapshot),
                "committed_round": len(result.train_history),
                "total_rounds": total_rounds,
                "optimizer_steps": trainer.optimizer_steps,
                "attempted_batches": trainer.attempted_train_batches,
                "best_score": trainer._progress["best_selected_score"]
                if math.isfinite(trainer._progress["best_selected_score"])
                else None,
                "best_round": result.best_epoch,
            },
        )
        print(
            "COMMITTED:",
            snapshot,
            "updates",
            trainer.optimizer_steps,
            "attempts",
            trainer.attempted_train_batches,
            flush=True,
        )
        if trainer.optimizer_steps != trainer.attempted_train_batches:
            write_status(
                status_path, {"state": "stopped_nonfinite_update", "snapshot": str(snapshot)}
            )
            raise RuntimeError(
                "CPU run encountered a skipped update; stopped after preserving snapshot"
            )
        if not trainer.validation_history[-1]["model"]["eligible"]:
            write_status(
                status_path, {"state": "stopped_nonfinite_validation", "snapshot": str(snapshot)}
            )
            raise RuntimeError(
                "CPU validation produced nonfinite predictions; stopped after preserving snapshot"
            )
        if rounds is not None and local_rounds >= rounds:
            if not complete:
                write_status(
                    status_path, {"state": "paused_at_boundary", "snapshot": str(snapshot)}
                )
            break
    best = load_checkpoint(trainer.ckpt_dir / "best.pt", torch.device("cpu"))
    print("CPU SESSION FINISHED. Best:", best.selected_score, "round:", best.epoch, flush=True)


def main() -> None:
    """Require an explicit probe or training action."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--probe", action="store_true")
    action.add_argument("--train", action="store_true")
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--rounds-this-invocation", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or (
        args.rounds_this_invocation is not None and args.rounds_this_invocation < 1
    ):
        parser.error("repeats/round limit must be positive")
    os.chdir(ROOT)
    setup_runtime()
    cfg = load_config(ROOT / "configs/hgnn-cpu-long-horizons.yaml")
    if args.probe:
        probe(cfg, micro_batch_size=args.micro_batch_size, repeats=args.repeats)
    else:
        train(cfg, micro_batch_size=args.micro_batch_size, rounds=args.rounds_this_invocation)


if __name__ == "__main__":
    main()
