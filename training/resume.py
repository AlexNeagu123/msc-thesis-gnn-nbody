"""Round-boundary continuation for canonical HGNN; never a weights-only warm start."""

from __future__ import annotations

import copy
import os
import platform
import random
from dataclasses import asdict
from typing import TYPE_CHECKING

import numpy as np
import torch

from models.hgnn_canonical import MODEL_ID

if TYPE_CHECKING:
    from training._types import Checkpoint, TrainConfig
    from training.train import Trainer

VERSION = "canonical-round-resume-v1"


def environment(device: torch.device) -> dict:
    """Record runtime settings relevant to numerical replay, not credentials."""
    return {
        "python": platform.python_version(),
        "cuda_build": torch.version.cuda,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device_type": device.type,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "threads": torch.get_num_threads(),
    }


def comparable_config(cfg: dict) -> dict:
    """Allow file relocation and artifact destinations, not a changed training recipe."""
    value = copy.deepcopy(cfg)
    value.pop("checkpointing", None)
    value.pop("logging", None)
    for key in ("train_path", "val_path"):
        value["data"].pop(key, None)
    value["training"].pop("device", None)
    return value


def validate(checkpoint: Checkpoint, cfg: TrainConfig) -> dict:
    """Reject old inference-only weights, incompatible recipes and malformed progress."""
    state = getattr(checkpoint, "resume_state", None)
    if (
        checkpoint.model_name != MODEL_ID
        or not isinstance(state, dict)
        or state.get("version") != VERSION
    ):
        raise ValueError(
            "resume requires a canonical latest.pt with full round-boundary resume state"
        )
    if cfg.model.name != MODEL_ID or cfg.training.checkpoint_metric not in (
        "val_loss",
        "fixed_position_mae",
    ):
        raise ValueError("resume supports canonical val_loss/fixed_position_mae only")
    if checkpoint.config is None or comparable_config(checkpoint.config) != comparable_config(
        asdict(cfg)
    ):
        raise ValueError("resume config changes the saved training recipe")
    if not cfg.checkpointing.enabled:
        raise ValueError("resume requires checkpointing enabled")
    if not checkpoint.run_id or len(state["progress"]["train_history"]) != checkpoint.epoch:
        raise ValueError("resume checkpoint has inconsistent completed-round progress")
    if len(state["progress"]["val_history"]) != checkpoint.epoch:
        raise ValueError("resume checkpoint has inconsistent validation progress")
    if checkpoint.pos_std != state["pos_std"] or checkpoint.vel_std != state["vel_std"]:
        raise ValueError("resume normalization metadata disagrees")
    return state


def restore(trainer: Trainer, checkpoint: Checkpoint, *, allow_runtime_change: bool) -> None:
    """Restore complete committed-round state; initialization RNG consumption is discarded."""
    state = validate(checkpoint, trainer.cfg)
    if state["data_hashes"] != trainer.data_hashes:
        raise ValueError("resume data hashes disagree")
    if state["source_hashes"] != trainer.source_hashes:
        raise ValueError("resume source hashes disagree; fork a new experiment instead")
    runtime = environment(trainer.device)
    changed = state["environment"] != runtime
    if changed and not allow_runtime_change:
        raise ValueError(
            "resume runtime changed; --allow-runtime-change explicitly permits non-bitwise continuation"
        )
    trainer.model.load_state_dict(checkpoint.model)
    if (
        float(trainer.model.pos_std) != trainer.pos_std
        or float(trainer.model.momentum_std) != trainer.momentum_std
    ):
        raise ValueError("resume saved normalization buffers disagree with metadata")
    trainer.optimizer.load_state_dict(checkpoint.optimizer)
    saved_scheduler = state["scheduler"]
    if (saved_scheduler is None) != (trainer.scheduler is None):
        raise ValueError("resume scheduler mismatch")
    if trainer.scheduler is not None:
        trainer.scheduler.load_state_dict(saved_scheduler)
    horizon = state["horizon"]
    if horizon != trainer.current_horizon:
        trainer.train_loader, trainer.val_loader = trainer._setup_loaders_for_horizon(horizon)
        trainer.current_horizon = horizon
        trainer.diagnostics.dataset = trainer.train_loader.dataset
    trainer.optimizer_steps = state["optimizer_steps"]
    trainer.attempted_train_batches = state["attempted_train_batches"]
    trainer.validation_history = copy.deepcopy(state["validation_history"])
    trainer._progress = copy.deepcopy(state["progress"])
    trainer._best_checkpoint = copy.deepcopy(state["best_checkpoint"])
    trainer._resume_lineage = [
        *state["lineage"],
        {
            "completed_round": checkpoint.epoch,
            "runtime_changed": changed,
            "from_environment": state["environment"],
            "to_environment": runtime,
        },
    ]
    trainer._train_generator.set_state(state["rng"]["sampler"].cpu())
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"].cpu())
    if trainer.device.type == "cuda" and state["rng"]["cuda"] is not None:
        if len(state["rng"]["cuda"]) != torch.cuda.device_count():
            raise ValueError("resume CUDA device count changed")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["rng"]["cuda"]])


def capture(trainer: Trainer) -> dict:
    """Snapshot the committed round, sampler and all stochastic streams."""
    return {
        "version": VERSION,
        "environment": environment(trainer.device),
        "data_hashes": trainer.data_hashes,
        "source_hashes": trainer.source_hashes,
        "horizon": trainer.current_horizon,
        "pos_std": trainer.pos_std,
        "vel_std": trainer.vel_std,
        "momentum_std": trainer.momentum_std,
        "optimizer_steps": trainer.optimizer_steps,
        "attempted_train_batches": trainer.attempted_train_batches,
        "scheduler": trainer.scheduler.state_dict() if trainer.scheduler is not None else None,
        "progress": copy.deepcopy(trainer._progress),
        "validation_history": copy.deepcopy(trainer.validation_history),
        "best_checkpoint": trainer._best_checkpoint,
        "lineage": trainer._resume_lineage,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "sampler": trainer._train_generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if trainer.device.type == "cuda" else None,
        },
    }
