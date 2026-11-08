"""Disposable TRAIN-only Block A probe: exactly84 updates, no production entry point.

Four cells x (one trace/warmup update +20 timed repeated-batch updates).
No validation/test reads, rollout claims, checkpoints or reusable weights.
"""

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

from competitors.gnode_pignn_training import production as P  # noqa: N812


def main() -> None:
    """Run the bounded disposable probe and write evidence, never model weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    if not P.power_ok():
        raise RuntimeError("Mains required")
    env = P.runtime()
    import numpy as np
    import tensorflow as tf

    from competitors.gnode_pignn_training.data import load_train_pairs
    from competitors.gnode_pignn_training.engine import StepEngine
    from competitors.pignn_ablation.recipes import CELLS, scientific_delta

    sources = {
        str(p.relative_to(P.ROOT)): P.sha(p)
        for directory in ["gnode_pignn_tf", "gnode_pignn_training", "pignn_ablation"]
        for p in sorted((P.ROOT / "competitors" / directory).glob("*.py"))
    }
    train = P.ROOT / "data/output/daniela-rk4/train.h5"
    expected = "b1237f9e0831758b2cf56a82167cfbd5ba84e6eb52e428ea7e53e5c9187695bf"
    assert P.sha(train) == expected
    x, y, ids = load_train_pairs(train)
    # One common full batch spread deterministically across TRAIN; disposable
    # repeated-batch learning probe, NOT the production no-shuffle trajectory.
    indices = np.linspace(0, len(x) - 1, 512, dtype=np.int64)
    xb, yb = x[indices], y[indices]
    del x, y
    baseline = None
    results = {}

    def guard() -> None:
        if not P.power_ok():
            raise RuntimeError("Mains lost; stop disposable probe")
        if time.monotonic() - started > 150:
            raise RuntimeError("150s internal probe cap")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 2500 * 1024**2:
            raise RuntimeError("2.5GiB RSS guard")

    for cell, recipe in CELLS.items():
        guard()
        engine = StepEngine(recipe)
        arrays = engine.arrays()
        if baseline is None:
            baseline = arrays
        for a, b in zip(baseline, arrays, strict=True):
            np.testing.assert_array_equal(a, b)
        before = {k: float(v.numpy()) for k, v in engine.loss_parts(xb, yb).items()}
        with tf.GradientTape(persistent=True) as tape:
            parts = engine.loss_parts(xb, yb)
        norms = {}
        gradients = {}
        for name in ["data", "physics", "total"]:
            g = tape.gradient(parts[name], engine.model.trainable_variables)
            assert all(v is not None for v in g)
            flat = np.concatenate([v.numpy().ravel().astype(np.float64) for v in g])
            assert np.isfinite(flat).all()
            gradients[name] = flat
            norms[name] = float(np.linalg.norm(flat))
        del tape
        cosine = float(
            np.dot(gradients["data"], gradients["physics"]) / (norms["data"] * norms["physics"])
        )
        warm = engine.step(xb, yb)
        assert all(np.isfinite(float(v)) for v in warm.values())
        times = []
        for _ in range(20):
            guard()
            t = time.perf_counter()
            step = engine.step(xb, yb)
            assert all(np.isfinite(float(v)) for v in step.values())
            times.append(time.perf_counter() - t)
        after = {k: float(v.numpy()) for k, v in engine.loss_parts(xb, yb).items()}
        assert int(engine.optimizer.iterations) == 21
        assert all(np.isfinite(a).all() for a in engine.arrays())
        assert after["total"] < before["total"], (cell, before, after)
        results[cell] = {
            "recipe": recipe.record(),
            "scientific_delta_from_A00": scientific_delta(CELLS["A00"], recipe),
            "initial_arrays_exactly_paired": True,
            "before": before,
            "after": after,
            "initial_gradient_norms": norms,
            "initial_data_physics_gradient_cosine": cosine,
            "updates": 21,
            "timed_updates": 20,
            "median_update_seconds": float(np.median(times)),
            "projected_125000_update_hours_excluding_validation_IO": float(
                np.median(times) * 125000 / 3600
            ),
        }
        del engine
        tf.keras.backend.clear_session()
    for rel, h in sources.items():
        assert P.sha(P.ROOT / rel) == h
    report = {
        "scope": "disposable TRAIN-only repeated-batch probe; not convergence/stability validation or production initialisation",
        "environment": env,
        "sources_sha256": sources,
        "train_sha256": expected,
        "train_systems": len(ids),
        "probe_pair_indices": indices.tolist(),
        "probe_batch_sha256": hashlib.sha256(xb.tobytes() + yb.tobytes()).hexdigest(),
        "cells": results,
        "total_updates": 84,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "validation_or_test_reads": False,
        "weights_saved": False,
        "full_runs_authorised": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                c: {
                    k: v
                    for k, v in r.items()
                    if k
                    in [
                        "before",
                        "after",
                        "median_update_seconds",
                        "projected_125000_update_hours_excluding_validation_IO",
                    ]
                }
                for c, r in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
