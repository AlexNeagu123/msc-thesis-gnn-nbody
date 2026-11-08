"""Explicit bounded TRAIN-only disposable probes, not a full-training launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
import traceback
from pathlib import Path

from competitors.gnode_pignn_tf.runtime import configure


def sha(path: Path) -> str:
    """Hash input/source identities without altering files."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, data: dict) -> None:
    """Preserve evidence by refusing to overwrite existing results."""
    with path.open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)


def main() -> None:
    """Run exactly declared small updates and timing; never read VAL/TEST states."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-disposable-probes", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    environment = configure()
    import numpy as np
    import tensorflow as tf

    from competitors.gnode_pignn_tf.models import compiled_rollout
    from competitors.gnode_pignn_training.checkpoint import restore_checkpoint, save_checkpoint
    from competitors.gnode_pignn_training.data import PairSampler, load_train_pairs
    from competitors.gnode_pignn_training.engine import StepEngine
    from competitors.gnode_pignn_training.recipes import RECIPES

    root = Path(__file__).resolve().parents[2]
    train = root / "data/output/daniela-rk4/train.h5"
    old = json.loads(
        (root / "runs/daniela-public-replay-v1/FROZEN_BEFORE_COMPARISON.json").read_text()
    )
    expected = old["input_sha256"][str(train.relative_to(root))]
    if sha(train) != expected:
        raise ValueError("training file differs from frozen shared split")
    source = {
        str(p.relative_to(root)): sha(p)
        for folder in ("gnode_pignn_tf", "gnode_pignn_training")
        for p in sorted((root / "competitors" / folder).glob("*.py"))
    }
    identity = {
        "environment": environment,
        "train_sha256": expected,
        "sources": source,
        "scope": "disposable_preflight_not_production_initialization",
    }
    plan = {
        "identity": identity,
        "recipes": {k: v.record() for k, v in RECIPES.items()},
        "per_recipe": {
            "timing_batch": 512,
            "timing_warmup_updates": 2,
            "timed_updates": 5,
            "learning_batch": 16,
            "learning_updates": 200,
            "learning_gate": "finite and final_total < 0.9*initial_total",
            "standin_rollout_train_systems": 80,
            "standin_future_steps": 1999,
        },
        "max_disposable_probe_updates": 621,
        "validation_or_test_states_read": False,
        "full_training_authorized": False,
    }
    save(args.output / "FROZEN_PROBE_PLAN.json", plan)
    try:
        x, y, ids = load_train_pairs(train)
        rng = np.random.default_rng(2026)
        systems = rng.choice(320, size=4, replace=False)
        indices = np.concatenate(
            [system * 1999 + rng.choice(1999, size=4, replace=False) for system in systems]
        )
        benchmark_indices = rng.choice(len(x), size=512, replace=False)
        small_x, small_y = x[indices], y[indices]
        results = {
            "plan": plan,
            "selected_system_ids": ids[systems].tolist(),
            "selected_flat_pair_indices": indices.tolist(),
            "models": {},
        }
        for key, recipe in RECIPES.items():
            print(f"START {key}", flush=True)
            benchmark = StepEngine(recipe)
            bx, by = x[benchmark_indices], y[benchmark_indices]
            cold = time.perf_counter()
            benchmark.step(bx, by)
            cold = time.perf_counter() - cold
            benchmark.step(bx, by)
            times = []
            for _ in range(5):
                started = time.perf_counter()
                step = benchmark.step(bx, by)
                _ = float(step["total"])  # synchronize timing boundary
                times.append(time.perf_counter() - started)
            assert int(benchmark.optimizer.iterations) == 7
            engine = StepEngine(recipe)  # fresh seed42; benchmark weights discarded
            initial = {
                k: float(v)
                for k, v in engine.loss_parts(tf.constant(small_x), tf.constant(small_y)).items()
            }
            sampler = PairSampler(16, 16, shuffle=False)
            curve = [{"updates": 0, **initial}]
            for update in range(1, 201):
                local = sampler.next_indices()
                engine.step(small_x[local], small_y[local])
                if update in (10, 50, 100, 200):
                    curve.append(
                        {
                            "updates": update,
                            **{
                                k: float(v)
                                for k, v in engine.loss_parts(
                                    tf.constant(small_x), tf.constant(small_y)
                                ).items()
                            },
                        }
                    )
            final = curve[-1]
            assert int(engine.optimizer.iterations) == 200
            checkpoint_identity = {
                **identity,
                "selected_flat_pair_indices": indices.tolist(),
                "effective_probe_batch": 16,
            }
            destination = args.output / key
            save_checkpoint(
                destination,
                engine,
                sampler,
                checkpoint_identity,
                {"updates": 200, "full_training": False},
            )
            reloaded = StepEngine(recipe)
            sampler2 = PairSampler(16, 16, False)
            restore_checkpoint(destination, reloaded, sampler2, checkpoint_identity)
            for a, b in zip(engine.arrays(), reloaded.arrays(), strict=True):
                np.testing.assert_array_equal(a, b)
            # Compute-only stand-in for full validation, using TRAIN initial states.
            # No reference labels consumed; not a validation/generalisation estimate.
            model = engine.inference_view()
            execute = compiled_rollout(model, recipe.kind)
            execute(x[:1], tf.constant(1)).numpy()
            initial80 = x[np.arange(80) * 1999]
            started = time.perf_counter()
            finite_rollouts = 0
            for start in range(0, 80, 64):
                predicted = execute(initial80[start : start + 64], tf.constant(1999)).numpy()
                finite_rollouts += int(np.isfinite(predicted).all(axis=(1, 2, 3)).sum())
            standin_time = time.perf_counter() - started
            median = float(np.median(times))
            finite = all(
                np.isfinite(row[name]) for row in curve for name in ("total", "data", "physics")
            )
            passed = finite and final["total"] < 0.9 * initial["total"]
            row = {
                "cold_batch512_update_seconds": cold,
                "warm_batch512_update_seconds": times,
                "median_batch512_update_seconds": median,
                "learning_curve": curve,
                "learning_gate_pass": bool(passed),
                "final_over_initial_total": final["total"] / initial["total"],
                "restored_model_and_adam_exact": True,
                "train80_standin_rollout_seconds": standin_time,
                "train80_standin_finite_rollouts": finite_rollouts,
                "projection100_epochs_plus25_rollouts_hours": (125000 * median + 25 * standin_time)
                / 3600,
                "projection_caveat": "compute-only based on5 measured updates and one TRAIN stand-in pass; excludes epoch data loading/shuffling, scoring, backup/IO, thermal changes; not a job-time guarantee",
                "disposable_updates": 207,
            }
            results["models"][key] = row
            save(args.output / (key + "-observations.json"), row)
            print(
                f"DONE {key}: update512 median={median:.6f}s; learning ratio={row['final_over_initial_total']:.6g}; gate={passed}",
                flush=True,
            )
        results["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        results["all_learning_gates_pass"] = all(
            row["learning_gate_pass"] for row in results["models"].values()
        )
        save(args.output / "summary.json", results)
        if not results["all_learning_gates_pass"]:
            raise RuntimeError(
                "bounded learning gate failed; no automatic extra updates or recipe changes"
            )
        save(
            args.output / "COMMITTED.json",
            {p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
        )
    except Exception:
        save(
            args.output / "FAILED.json",
            {"traceback": traceback.format_exc(), "full_training_started": False},
        )
        raise


if __name__ == "__main__":
    main()
