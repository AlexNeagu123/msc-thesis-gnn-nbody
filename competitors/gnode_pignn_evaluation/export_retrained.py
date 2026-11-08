"""Export frozen retrained-checkpoint rollouts in the shared future-state layout.

Mirrors the original-checkpoint exporter (same reference, batch64, 1999 steps, same
non-XLA graph rollout and export conversion). Reads no target, selects nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from competitors.gnode_pignn_tf.runtime import configure


def digest(path: Path) -> str:
    """Hash file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Export one role (primary or secondary) of a previously frozen retrained run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--role", choices=["primary", "secondary"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    root = Path(__file__).resolve().parents[2]
    for section in ("source_sha256", "input_sha256"):
        for relative, expected in freeze[section].items():
            if digest(root / relative) != expected:
                raise ValueError(f"{section} changed after freeze: {relative}")
    if args.output.exists():
        raise FileExistsError(args.output)
    role = freeze["roles"][args.role]
    environment = configure()
    if environment["packages"] != freeze["environment"]["packages"]:
        raise ValueError("frozen package versions mismatch")
    import numpy as np
    import tensorflow as tf

    from competitors.gnode_pignn_tf.contracts import to_mass_first, to_shared_layout
    from competitors.gnode_pignn_tf.models import compiled_rollout
    from competitors.gnode_pignn_training.checkpoint import restore_checkpoint
    from competitors.gnode_pignn_training.data import PairSampler
    from competitors.gnode_pignn_training.engine import StepEngine
    from competitors.gnode_pignn_training.recipes import RECIPES

    recipe = RECIPES[freeze["recipe_id"]]
    directory = root / role["checkpoint"]
    metadata = json.loads((directory / "metadata.json").read_text())
    identity = metadata["identity"]
    # Explicit identity checks before using the checkpoint's own recorded identity.
    if identity["environment"] != environment:
        raise ValueError("runtime differs from the training/checkpoint identity")
    if identity["recipe_id"] != recipe.model_id or metadata["recipe"] != recipe.record():
        raise ValueError("recipe identity mismatch")
    if identity["scope"] != "production_shared320_80_from_scratch":
        raise ValueError("checkpoint is not a production shared-split run")
    for relative, expected in identity["sources"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"training source differs from checkpoint identity: {relative}")
    engine = StepEngine(recipe)
    sampler = PairSampler(639680, recipe.batch_size, recipe.kind == "gnode", recipe.seed)
    progress = restore_checkpoint(directory, engine, sampler, identity)
    if progress["updates"] != role["updates"] or progress["epochs_done"] != role["epoch"]:
        raise ValueError("checkpoint progress differs from frozen role")
    execute = compiled_rollout(engine.inference_view(), recipe.kind)
    execute(np.asarray(freeze["train_warmup_mass_first"], dtype=np.float32), tf.constant(1)).numpy()
    with np.load(root / freeze["reference"], allow_pickle=False) as handle:
        reference, ids = handle["reference"], handle["run_ids"]
    if reference.shape != (100, 2000, 3, 5) or ids.tolist() != freeze["ordered_run_ids"]:
        raise ValueError("shared reference shape/ordered IDs mismatch")
    initial = reference[:, 0].copy()
    del reference  # only initial states are consumed by inference
    state = to_mass_first(initial)
    steps, batch_size = freeze["steps"], freeze["batch_size"]
    if (steps, batch_size) != (1999, 64):
        raise ValueError("unexpected evaluation recipe")
    predicted = np.empty((100, steps, 3, 5), dtype=np.float32)
    started = time.perf_counter()
    for start in range(0, len(state), batch_size):
        stop = min(start + batch_size, len(state))
        result = execute(state[start:stop], tf.constant(steps)).numpy()
        predicted[start:stop] = to_shared_layout(result, initial[start:stop])
    elapsed = time.perf_counter() - started
    failed = ~np.isfinite(predicted[..., :4]).all(axis=(1, 2, 3))
    info = {
        "model_label": f"Retrained {freeze['recipe_id']} shared320/80, {args.role} ({role['name']})",
        "role": args.role,
        "role_detail": role,
        "checkpoint_metadata_sha256": digest(directory / "metadata.json"),
        "environment": environment,
        "freeze_sha256": digest(args.freeze),
        "recipe": freeze,
        "inference_wall_seconds": elapsed,
        "timing_scope": "warm non-XLA graph rollout batches + host export conversion; excludes load/trace/IO/scoring",
        "optimizer_updates": 0,
        "mass_export": "float32 cast once, retained exactly; same convention as frozen HGNN",
        "historical_validation_reuse": True,
        "failed_run_ids": ids[failed].tolist(),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    temporary = args.output / "predictions.pending.npz"
    with temporary.open("xb") as handle:
        np.savez(
            handle,
            predicted=predicted,
            run_ids=ids,
            metadata_json=np.array(json.dumps(info, allow_nan=False)),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output / "predictions.npz")
    with (args.output / "inference.json").open("x") as handle:
        json.dump(info, handle, indent=2, allow_nan=False)
    with (args.output / "INFERENCE_COMMITTED.json").open("x") as handle:
        json.dump(
            {
                "prediction_sha256": digest(args.output / "predictions.npz"),
                "metadata_sha256": digest(args.output / "inference.json"),
                "finite_runs": int((~failed).sum()),
                "total_runs": len(ids),
            },
            handle,
            indent=2,
        )
    print(f"{args.role}: {elapsed:.3f}s; {int(failed.sum())} failed; output={args.output}")


if __name__ == "__main__":
    main()
