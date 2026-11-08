"""Explicit local-to-Drive committed snapshots and no-update device timing probes."""

import copy
import json
import shutil
import time
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from training._io import load_checkpoint
from training._types import TrainConfig
from training.fixed_validation import file_sha256
from training.resume import environment
from training.train import Trainer

ARTIFACTS = (
    "latest.pt",
    "best.pt",
    "metrics.csv",
    "fixed_validation.jsonl",
    "diagnostics.log",
    "session_metadata.json",
)


def backup_round(run_dir: Path, destination: Path) -> Path:
    """Copy a stopped-at-boundary run; verify hashes and publish a commit marker LAST."""
    checkpoint = load_checkpoint(run_dir / "latest.pt", torch.device("cpu"))
    if not getattr(checkpoint, "resume_state", None):
        raise ValueError("backup requires a resumable latest.pt")
    snapshot = (
        destination / checkpoint.run_id / f"round-{checkpoint.epoch:06d}-{uuid.uuid4().hex[:8]}"
    )
    snapshot.mkdir(parents=True, exist_ok=False)
    files = {}
    for name in ARTIFACTS:
        source = run_dir / name
        if source.exists():
            if source.is_symlink():
                raise ValueError("backup source cannot be a symlink")
            before = file_sha256(source)
            shutil.copyfile(source, snapshot / name)
            if file_sha256(snapshot / name) != before or file_sha256(source) != before:
                raise RuntimeError(
                    "backup changed or failed verification; snapshot is NOT committed"
                )
            files[name] = before
    marker = {
        "protocol": "canonical-drive-backup-v1",
        "epoch": checkpoint.epoch,
        "optimizer_steps": checkpoint.resume_state["optimizer_steps"],
        "files": files,
    }
    with (snapshot / "COMMITTED.json").open("x") as handle:
        json.dump(marker, handle, indent=2)
    return snapshot


def verify_backup(snapshot: Path) -> dict:
    """Accept only complete, hash-verified backups; untrusted pickle files remain unsafe."""
    marker = json.loads((snapshot / "COMMITTED.json").read_text())
    if marker.get("protocol") != "canonical-drive-backup-v1" or "latest.pt" not in marker["files"]:
        raise ValueError("invalid backup commit marker")
    for name, digest in marker["files"].items():
        if name not in ARTIFACTS or (snapshot / name).is_symlink():
            raise ValueError("invalid backup file")
        if file_sha256(snapshot / name) != digest:
            raise ValueError(f"backup hash mismatch: {name}")
    return marker


def restore_backup(snapshot: Path, destination: Path) -> Path:
    """Verify a committed Drive snapshot and copy to a NEW local directory."""
    marker = verify_backup(snapshot)
    destination.mkdir(parents=True, exist_ok=False)
    for name in marker["files"]:
        shutil.copyfile(snapshot / name, destination / name)
        if file_sha256(destination / name) != marker["files"][name]:
            raise ValueError("local restore hash mismatch")
    return destination / "latest.pt"


def benchmark_training(
    cfg: TrainConfig,
    *,
    device: str,
    horizons: tuple[int, ...] = (1, 5, 20),
    batch_size: int = 64,
    repeats: int = 3,
) -> dict:
    """Time synchronized loss/backward/clipping only; perform ZERO optimizer steps."""
    if repeats < 1 or batch_size < 1 or not horizons or min(horizons) < 1:
        raise ValueError("benchmark counts/horizons must be positive")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not silently benchmark CPU")
    cfg = replace(
        cfg,
        data=replace(cfg.data, val_path=cfg.data.train_path),
        training=replace(
            cfg.training,
            device=device,
            epochs=1,
            multi_step_horizon=1,
            curriculum_horizons=None,
            curriculum_epochs=None,
            curriculum_lrs=None,
            curriculum_gradient_clip_norms=None,
        ),
        checkpointing=replace(cfg.checkpointing, enabled=False),
        logging=replace(cfg.logging, enabled=False),
    )
    trainer = Trainer(cfg)
    states = trainer.train_loader.dataset.states
    if max(horizons) >= states.shape[1]:
        raise ValueError("benchmark horizon exceeds saved training frames")
    rng = np.random.default_rng(cfg.training.seed)
    trajectories = rng.integers(0, len(states), batch_size)
    starts = rng.integers(0, states.shape[1] - max(horizons), batch_size)
    inputs = torch.stack([states[i, t] for i, t in zip(trajectories, starts, strict=True)]).to(
        trainer.device
    )
    before = copy.deepcopy(trainer.model.state_dict())
    results = []
    for horizon in horizons:
        targets = torch.stack(
            [states[i, t + 1 : t + 1 + horizon] for i, t in zip(trajectories, starts, strict=True)]
        ).to(trainer.device)
        trainer.current_horizon = horizon
        samples = []
        loss = None
        try:
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for iteration in range(repeats + 1):
                trainer.model.zero_grad(set_to_none=True)
                if device == "cuda":
                    torch.cuda.synchronize()
                begin = time.perf_counter()
                _, loss, _ = trainer._compute_loss(
                    inputs, targets[:, 0] if horizon == 1 else targets
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("benchmark loss is nonfinite")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    trainer.model.parameters(), cfg.training.gradient_clip_norm
                )
                if not torch.isfinite(norm):
                    raise RuntimeError("benchmark gradients are nonfinite")
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - begin
                if iteration:
                    samples.append(elapsed)
            results.append(
                {
                    "horizon": horizon,
                    "batch_size": batch_size,
                    "status": "ok",
                    "seconds": samples,
                    "median_seconds": float(np.median(samples)),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated()
                    if device == "cuda"
                    else None,
                }
            )
        except torch.cuda.OutOfMemoryError:
            results.append({"horizon": horizon, "batch_size": batch_size, "status": "oom"})
            del loss
            trainer.model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            break
    for key, value in before.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, trainer.model.state_dict()[key])
    assert trainer.optimizer_steps == 0
    return {
        "environment": environment(trainer.device),
        "profiles": results,
        "scope": "warmup excluded; synchronized forward/loss/backward/clipping; excludes data transfer, optimizer, validation and checkpoint IO",
        "optimizer_steps": 0,
        "parameters_unchanged": True,
        "warning": "Timing/peak excludes optimizer-state initialization; not a full training speedup guarantee.",
    }
