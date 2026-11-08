"""Frozen original-checkpoint inference only; no fitting, scoring or selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from competitors.gnode_pignn_tf.runtime import configure


def digest(path: Path) -> str:
    """Hash file bytes without deserialisation."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Export exact shared future states using a previously frozen recipe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["gnode", "pignn"], required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    root = Path(__file__).resolve().parents[2]
    for relative, expected in freeze["source_sha256"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"source changed after freeze: {relative}")
    for relative, expected in freeze["input_sha256"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"input changed after freeze: {relative}")
    if args.output.exists():
        raise FileExistsError(args.output)
    environment = configure()
    if environment["packages"] != freeze["environment"]["packages"]:
        raise ValueError("frozen package versions mismatch")
    import numpy as np
    import tensorflow as tf

    from competitors.gnode_pignn_tf.contracts import to_mass_first, to_shared_layout
    from competitors.gnode_pignn_tf.models import build_model, compiled_rollout, load_public_weights

    kind = args.kind
    checkpoint = root / freeze["models"][kind]["checkpoint"]
    model = build_model(kind)
    weights = load_public_weights(model, kind, checkpoint)
    execute = compiled_rollout(model, kind)
    # Trace/warm up from TRAIN state saved in freeze, never from comparison labels.
    execute(np.asarray(freeze["train_warmup_mass_first"], dtype=np.float32), tf.constant(1)).numpy()
    with np.load(root / freeze["reference"], allow_pickle=False) as handle:
        reference, ids = handle["reference"], handle["run_ids"]
    if reference.shape != (100, 2000, 3, 5) or ids.tolist() != freeze["ordered_run_ids"]:
        raise ValueError("shared reference shape/ordered IDs mismatch")
    if not np.isfinite(reference).all() or not np.all(reference[..., 4] == reference[:, :1, :, 4]):
        raise ValueError("invalid shared reference")
    initial = reference[:, 0].copy()
    del reference  # only initial states are needed; no target is consumed by inference
    state = to_mass_first(initial)
    steps, batch_size = freeze["steps"], freeze["batch_size"]
    if steps != 1999 or batch_size != 64:
        raise ValueError("unexpected original-weight evaluation recipe")
    predicted = np.empty((100, steps, 3, 5), dtype=np.float32)
    started = time.perf_counter()
    for start in range(0, len(state), batch_size):
        stop = min(start + batch_size, len(state))
        result = execute(state[start:stop], tf.constant(steps)).numpy()
        predicted[start:stop] = to_shared_layout(result, initial[start:stop])
    elapsed = time.perf_counter() - started
    failed = ~np.isfinite(predicted[..., :4]).all(axis=(1, 2, 3))
    metadata = {
        "model_label": f"Daniela public {kind.upper()} checkpoint; repository quantitative-test semantics",
        "checkpoint": weights,
        "environment": environment,
        "freeze_sha256": digest(args.freeze),
        "recipe": freeze,
        "inference_wall_seconds": elapsed,
        "timing_scope": "two warm non-XLA graph rollout batches + host export conversion; excludes graph trace/model load/reference IO/scoring",
        "optimizer_updates": 0,
        "weights_only": True,
        "mass_export": "float32 cast once, retained exactly; same convention as frozen HGNN",
        "historical_validation_reuse": True,
        "certified_thesis_checkpoint": False,
        "failed_run_ids": ids[failed].tolist(),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    temporary = args.output / "predictions.pending.npz"
    with temporary.open("xb") as handle:
        np.savez(
            handle,
            predicted=predicted,
            run_ids=ids,
            metadata_json=np.array(json.dumps(metadata, allow_nan=False)),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output / "predictions.npz")
    with (args.output / "inference.json").open("x") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
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
    print(
        f"{kind}: {elapsed:.3f}s warm inference; {int(failed.sum())} failed runs; output={args.output}"
    )


if __name__ == "__main__":
    main()
