"""Load config and checkpoint into a ready-to-eval model, shared across evaluation/."""

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from data.dataset import NBodyDataset
from models.hgnn_canonical import MODEL_ID as CANONICAL_MODEL_ID
from training._io import load_checkpoint, load_config
from training._types import Checkpoint, TrainConfig
from training.train import build_model


@dataclass(frozen=True)
class LoadedModel:
    """Artifacts produced by `load_trained_model`."""

    cfg: TrainConfig
    checkpoint: Checkpoint
    model: nn.Module
    pos_std: float
    vel_std: float


def load_trained_model(
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> LoadedModel:
    """Build the model on `device` from config and checkpoint."""
    cfg = load_config(config_path)
    checkpoint = load_checkpoint(checkpoint_path, device)
    pos_std, vel_std = _resolve_normalization_stats(cfg, checkpoint)
    if cfg.model.name == CANONICAL_MODEL_ID:
        if checkpoint.model_name != CANONICAL_MODEL_ID:
            raise ValueError(
                "canonical config requires an explicitly identified canonical checkpoint"
            )
        momentum_std = checkpoint.model.get("momentum_std")
        if not isinstance(momentum_std, torch.Tensor) or momentum_std.numel() != 1:
            raise ValueError("canonical checkpoint is missing its scalar momentum_std buffer")
        saved_dt = checkpoint.model.get("dt")
        if (
            not isinstance(saved_dt, torch.Tensor)
            or saved_dt.numel() != 1
            or not math.isclose(float(saved_dt), cfg.data.dt, rel_tol=1e-7, abs_tol=0)
        ):
            raise ValueError("canonical checkpoint timestep disagrees with config")
        saved_pos_std = checkpoint.model.get("pos_std")
        if (
            not isinstance(saved_pos_std, torch.Tensor)
            or saved_pos_std.numel() != 1
            or not math.isclose(float(saved_pos_std), pos_std, rel_tol=1e-7, abs_tol=0)
        ):
            raise ValueError("canonical checkpoint position scale disagrees with its metadata")
        model = build_model(
            cfg,
            pos_std=pos_std,
            vel_std=vel_std,
            momentum_std=float(momentum_std),
        ).to(device)
    else:
        model = build_model(cfg, pos_std=pos_std, vel_std=vel_std).to(device)
    model.load_state_dict(checkpoint.model)
    model.eval()
    return LoadedModel(
        cfg=cfg,
        checkpoint=checkpoint,
        model=model,
        pos_std=pos_std,
        vel_std=vel_std,
    )


def _resolve_normalization_stats(
    cfg: TrainConfig,
    checkpoint: Checkpoint,
) -> tuple[float, float]:
    """Return (pos_std, vel_std) from the checkpoint, else refit from train data."""
    if checkpoint.pos_std is not None and checkpoint.vel_std is not None:
        return checkpoint.pos_std, checkpoint.vel_std

    if cfg.model.name == CANONICAL_MODEL_ID:
        raise ValueError(
            "canonical checkpoint must preserve normalization stats; refitting during evaluation is forbidden"
        )

    train_path = Path(cfg.data.train_path)
    if train_path.exists():
        train_set = NBodyDataset(str(train_path))
        return (
            float(train_set.inputs[..., :2].std()),
            float(train_set.inputs[..., 2:4].std()),
        )

    msg = f"Missing checkpoint normalization stats and train data: {train_path}"
    raise FileNotFoundError(msg)
