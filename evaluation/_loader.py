"""Shared loader: turn config + checkpoint paths into a ready-to-eval model.

Centralises the recipe `evaluation/evaluate.py` and `evaluation/animate_best.py`
have been duplicating: read the YAML config, load the checkpoint, resolve
normalisation stats (falling back to train data when the checkpoint omits
them), build the model, copy weights, switch to eval mode.

New callers in `evaluation/` should depend on this module instead of
re-implementing the dance. `evaluate.py` and `animate_best.py` keep their
existing private helpers for now; a follow-up refactor can route them
through `load_trained_model` once their test surfaces are aligned.

References:
    - training/_io.py    (load_config, load_checkpoint)
    - training/train.py  (build_model)
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from data.dataset import NBodyDataset
from training._io import load_checkpoint, load_config
from training._types import Checkpoint, TrainConfig
from training.train import build_model


@dataclass(frozen=True)
class LoadedModel:
    """Bundle of artifacts produced by `load_trained_model`."""

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
    """Load config + checkpoint, build the model on `device`, return all five artifacts.

    Normalisation stats follow the same precedence as `evaluation/evaluate.py`:
    use the checkpoint's `pos_std`/`vel_std` if present, otherwise refit from
    the train HDF5 referenced by the config. Missing in both is a hard error.
    """
    cfg = load_config(config_path)
    checkpoint = load_checkpoint(checkpoint_path, device)
    pos_std, vel_std = _resolve_normalization_stats(cfg, checkpoint)
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
    """Return (pos_std, vel_std), preferring checkpoint values, falling back to train data."""
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
