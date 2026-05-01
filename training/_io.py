"""Serialization for training artifacts: configs, checkpoints, metrics CSV.

Centralizes all on-disk I/O for the training pipeline so producers and
consumers (train.py, evaluate.py) never touch raw torch.save / file writes.

References:
    - Schemas: training/_types.py (TrainConfig, Checkpoint, EpochMetrics)
"""

from pathlib import Path

import torch
import yaml

from training._types import Checkpoint, EpochMetrics, TrainConfig


def load_config(path: str | Path) -> TrainConfig:
    """Load a YAML config file into a typed TrainConfig."""
    with Path(path).open() as f:
        raw = yaml.safe_load(f)
    return TrainConfig.from_dict(raw)


def save_checkpoint(path: Path, checkpoint: Checkpoint) -> None:
    """Persist a Checkpoint to disk via torch.save."""
    torch.save(checkpoint, path)


def load_checkpoint(path: Path, device: torch.device) -> Checkpoint:
    """Load a checkpoint, normalizing legacy dict checkpoints into Checkpoint."""
    raw = torch.load(path, weights_only=False, map_location=device)
    if isinstance(raw, Checkpoint):
        return raw
    if isinstance(raw, dict):
        return Checkpoint(
            epoch=raw.get("epoch", 0),
            model=raw.get("model", {}),
            optimizer=raw.get("optimizer", {}),
            val_loss=raw.get("val_loss", float("inf")),
            config=raw.get("config"),
            model_name=raw.get("model_name"),
            run_id=raw.get("run_id"),
            pos_std=raw.get("pos_std"),
            vel_std=raw.get("vel_std"),
            git_commit=raw.get("git_commit"),
        )
    msg = f"Unsupported checkpoint type: {type(raw).__name__}"
    raise TypeError(msg)


def init_metrics_csv(path: Path) -> None:
    """Create a metrics CSV with the EpochMetrics header row."""
    path.write_text(EpochMetrics.csv_header() + "\n")


def append_metrics(path: Path, row: EpochMetrics) -> None:
    """Append one EpochMetrics row to an existing metrics CSV."""
    with path.open("a") as f:
        f.write(row.to_csv_row() + "\n")
