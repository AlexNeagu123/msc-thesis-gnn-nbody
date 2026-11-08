"""Atomic local checkpoint bundles for this deterministic, no-dropout step engine.

Restores model, full Adam, explicit sampler/permutation/RNG and caller progress.
No claim to general stochastic-model TF/Keras RNG restoration. No pickle loading.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import tensorflow as tf

from competitors.gnode_pignn_training.data import PairSampler
from competitors.gnode_pignn_training.engine import StepEngine


def sha(path: Path) -> str:
    """Hash exact local file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_checkpoint(
    destination: Path, engine: StepEngine, sampler: PairSampler, identity: dict, progress: dict
) -> None:
    """Commit a new bundle only; never overwrite original or prior checkpoints."""
    if destination.exists():
        raise FileExistsError(destination)
    if progress.get("updates") != int(engine.optimizer.iterations):
        raise ValueError("progress update counter disagrees with Adam")
    temporary = destination.with_name(destination.name + ".pending-" + uuid.uuid4().hex)
    temporary.mkdir(parents=True)
    checkpoint = tf.train.Checkpoint(model=engine.model, optimizer=engine.optimizer)
    prefix = checkpoint.save(str(temporary / "state"))
    metadata = {
        "protocol": "daniela-deterministic-step-resume-v1",
        "recipe": engine.recipe.record(),
        "identity": identity,
        "sampler": sampler.state(),
        "progress": progress,
        "checkpoint_prefix": Path(prefix).name,
        "rng_boundary": "explicit sampler Generator; model has no stochastic training operations",
    }
    with (temporary / "metadata.json").open("x") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
    hashes = {p.name: sha(p) for p in temporary.iterdir() if p.is_file()}
    with (temporary / "COMMITTED.json").open("x") as handle:
        json.dump(hashes, handle, indent=2)
    for path in temporary.iterdir():
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    os.replace(temporary, destination)
    # Local hash-verified completion, not an off-device/power-loss durability claim.


def restore_checkpoint(
    directory: Path, engine: StepEngine, sampler: PairSampler, identity: dict
) -> dict:
    """Require complete matching identity; consume every TF checkpoint object."""
    hashes = json.loads((directory / "COMMITTED.json").read_text())
    for name, digest in hashes.items():
        if Path(name).name != name or sha(directory / name) != digest:
            raise ValueError("checkpoint bundle hash mismatch")
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["protocol"] != "daniela-deterministic-step-resume-v1":
        raise ValueError("unsupported checkpoint protocol")
    if metadata["recipe"] != engine.recipe.record() or metadata["identity"] != identity:
        raise ValueError("checkpoint scientific/runtime identity mismatch")
    sampler.restore(metadata["sampler"])
    prefix = metadata["checkpoint_prefix"]
    if Path(prefix).name != prefix:
        raise ValueError("invalid checkpoint prefix")
    checkpoint = tf.train.Checkpoint(model=engine.model, optimizer=engine.optimizer)
    checkpoint.restore(str(directory / prefix)).assert_consumed()
    if metadata["progress"].get("updates") != int(engine.optimizer.iterations):
        raise ValueError("restored progress update counter disagrees with Adam")
    return metadata["progress"]
