"""TRAIN-only timing/parity observations; never opens comparison trajectories."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from competitors.gnode_pignn_tf.runtime import configure


def main() -> None:
    """Measure reviewed public-weight rollout on bounded TRAIN batches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    environment = configure()
    import h5py
    import numpy as np
    import tensorflow as tf

    from competitors.gnode_pignn_tf.contracts import to_mass_first
    from competitors.gnode_pignn_tf.models import (
        build_model,
        compiled_rollout,
        load_public_weights,
        rollout,
    )

    root = Path(__file__).resolve().parents[2]
    audit = root / "runs/daniela-model-audit-v1/checkpoints-public"
    with h5py.File(root / "data/output/daniela-rk4/train.h5") as handle:
        assert handle.attrs["split"] == "train"
        initial = handle["trajectories"][:64, 0]
        ids = handle["simulation_ids"][:64]
    x = to_mass_first(initial)
    observations = {
        "environment": environment,
        "training_ids": ids.tolist(),
        "test_records_read": False,
        "optimizer_updates": 0,
        "models": {},
    }
    for kind, filename in [
        ("gnode", "model_GNODE_nbody.keras"),
        ("pignn", "model_PIGNN_nbody.keras"),
    ]:
        model = build_model(kind)
        weights = load_public_weights(model, kind, audit / filename)
        execute = compiled_rollout(model, kind)
        started = time.perf_counter()
        graph = execute(x[:4], tf.constant(32)).numpy()
        cold = time.perf_counter() - started
        eager = rollout(model, x[:4], kind, 32).numpy()
        np.testing.assert_allclose(graph, eager, rtol=1e-5, atol=1e-6)
        times = []
        for _ in range(3):
            started = time.perf_counter()
            predicted = execute(x, tf.constant(100)).numpy()
            times.append(time.perf_counter() - started)
            assert predicted.shape == (64, 100, 3, 5)
            assert np.isfinite(predicted).all()
        observations["models"][kind] = {
            "weights": weights,
            "eager_graph_train_maxabs_32_steps": float(np.max(np.abs(graph - eager))),
            "cold_graph_4x32_seconds": cold,
            "warm_graph_64x100_seconds": times,
            "projected_2_batches_x1999_steps_seconds": float(np.median(times) * 2 * 1999 / 100),
            "projection_caveat": "linear projection from TRAIN batch64; excludes startup/IO/scoring; final batch36",
        }
    observations["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    with args.output.open("x") as handle:
        json.dump(observations, handle, indent=2, allow_nan=False)
    print(json.dumps(observations, indent=2))


if __name__ == "__main__":
    main()
