"""Serialization for training artifacts: configs, checkpoints, metrics CSV."""

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
    """Load a checkpoint, normalising dict-shaped payloads into Checkpoint."""
    raw = torch.load(path, weights_only=False, map_location=device)
    if isinstance(raw, Checkpoint):
        return raw
    if isinstance(raw, dict):
        model_state = raw.get("model")
        optimizer_state = raw.get("optimizer")
        if not isinstance(model_state, dict):
            msg = f"Checkpoint at {path} is missing a dict-shaped 'model' state"
            raise ValueError(msg)
        if not isinstance(optimizer_state, dict):
            msg = f"Checkpoint at {path} is missing a dict-shaped 'optimizer' state"
            raise ValueError(msg)
        return Checkpoint(
            epoch=raw.get("epoch", 0),
            model=model_state,
            optimizer=optimizer_state,
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


def init_metrics_csv(path: Path, bin_names: tuple[str, ...] = ()) -> None:
    """Create a metrics CSV with the EpochMetrics header; `bin_names` widens it per bin."""
    path.write_text(EpochMetrics.csv_header(bin_names) + "\n")


def append_metrics(
    path: Path,
    row: EpochMetrics,
    bin_names: tuple[str, ...] = (),
) -> None:
    """Append one EpochMetrics row; `bin_names` must match init_metrics_csv for this file."""
    with path.open("a") as f:
        f.write(row.to_csv_row(bin_names) + "\n")
