"""GPU compatibility/parity/timing only. Hard ceiling99 disposable GPU updates."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from competitors.gnode_pignn_gpu.bundle import sha
from competitors.gnode_pignn_gpu.runtime import configure


def save(path: Path, data: dict) -> None:
    """No-overwrite JSON evidence."""
    with path.open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)


def main() -> None:
    """Require GPU and TRAIN-only bundle; reject failed gates, never train epochs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-disposable-gpu-probes", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        environment = configure()  # must fail on CPU-only hosts, before model updates
        import numpy as np
        import tensorflow as tf

        from competitors.gnode_pignn_gpu.checks import (
            compare_adam_slots,
            compare_update,
            relative_l2,
        )
        from competitors.gnode_pignn_tf.models import compiled_rollout
        from competitors.gnode_pignn_training.checkpoint import restore_checkpoint, save_checkpoint
        from competitors.gnode_pignn_training.data import PairSampler, load_train_pairs
        from competitors.gnode_pignn_training.engine import StepEngine
        from competitors.gnode_pignn_training.recipes import RECIPES

        root = Path(__file__).resolve().parents[2]
        manifest = json.loads((root / "bundle_manifest.json").read_text())
        for name, record in manifest["files"].items():
            if sha(root / name) != record["sha256"]:
                raise ValueError(f"bundled input changed: {name}")
        evidence = json.loads((root / "cpu_preflight_evidence.json").read_text())
        plan = {
            "environment": environment,
            "bundle_manifest_sha256": sha(root / "bundle_manifest.json"),
            "recipes": {k: v.record() for k, v in RECIPES.items()},
            "parity_batches": [7, 192, 512],
            "forward_rtol": 2e-4,
            "forward_atol": 2e-5,
            "gradient_relative_l2_limit": 5e-4,
            "adam_momentum_relative_l2_limit": 5e-4,
            "adam_velocity_relative_l2_limit": 1e-3,
            "adam_update_relative_l2_limit": 0.01,
            "warmup_updates": 2,
            "host_feed_timed_updates": 10,
            "resident_timed_updates": 10,
            "same_gpu_exact_resume_required": True,
            "max_gpu_updates": 99,
            "max_cpu_updates": 9,
            "train_standin_rollouts": [80, 1999],
            "full_training_authorized": False,
            "contains_validation_or_comparison_records": False,
        }
        save(args.output / "FROZEN_GPU_PROBE_PLAN.json", plan)
        x, y, ids = load_train_pairs(root / "data/output/daniela-rk4/train.h5")
        rng = np.random.default_rng(2026)
        systems = rng.choice(320, size=4, replace=False)
        small = np.concatenate(
            [system * 1999 + rng.choice(1999, size=4, replace=False) for system in systems]
        )
        benchmark = rng.choice(len(x), size=512, replace=False)
        identity = {
            "environment": environment,
            "bundle": plan["bundle_manifest_sha256"],
            "selected_flat_indices": small.tolist(),
            "scope": "GPU_probe_not_CPU_resume_or_full_training",
        }
        results = {"plan": plan, "selected_training_ids": ids[systems].tolist(), "models": {}}

        def gradients(engine: object, bx: np.ndarray, by: np.ndarray, device: str) -> tuple:
            with tf.device(device):
                tx, ty = tf.constant(bx), tf.constant(by)
                with tf.GradientTape() as tape:
                    losses = engine.loss_parts(tx, ty)
                gs = tape.gradient(losses["total"], engine.model.trainable_variables)
                if any(g is None for g in gs):
                    raise ValueError("disconnected GPU parity gradient")
                output = engine.model(tx, training=False).numpy()
            return output, {k: float(v) for k, v in losses.items()}, [g.numpy() for g in gs]

        def gpu_engine(recipe: object) -> object:
            with tf.device("/GPU:0"):
                engine = StepEngine(recipe)
            devices = [v.value.device for v in engine.model.trainable_variables]
            if not devices or not all("GPU:0" in device.upper() for device in devices):
                raise RuntimeError(f"model variables not on required GPU: {devices}")
            return engine

        def gpu_step(engine: object, bx: object, by: object) -> dict:
            with tf.device("/GPU:0"):
                row = engine.step(bx, by)
            _ = float(row["total"])  # synchronize, do not time asynchronous dispatch only
            if "GPU:0" not in row["total"].device.upper():
                raise RuntimeError("training loss not on GPU; CPU fallback forbidden")
            return row

        for key, recipe in RECIPES.items():
            print(f"START GPU {key}", flush=True)
            tf.config.experimental.reset_memory_stats("GPU:0")
            parity = []
            for size in (7, 192, 512):
                with tf.device("/CPU:0"):
                    cpu = StepEngine(recipe)
                gpu = gpu_engine(recipe)
                initial = cpu.model.get_weights()
                gpu.model.set_weights(initial)  # coupled test initialization, not CPU resume
                bx, by = x[benchmark[:size]], y[benchmark[:size]]
                cp, cl, cg = gradients(cpu, bx, by, "/CPU:0")
                gp, gl, gg = gradients(gpu, bx, by, "/GPU:0")
                np.testing.assert_allclose(cp, gp, rtol=2e-4, atol=2e-5)
                for name in cl:
                    np.testing.assert_allclose(cl[name], gl[name], rtol=2e-4, atol=1e-10)
                gradient_difference = relative_l2(cg, gg)
                if gradient_difference > 5e-4:
                    raise ValueError(
                        f"CPU/GPU gradient relative L2 {gradient_difference} exceeds5e-4"
                    )
                with tf.device("/CPU:0"):
                    cpu.step(bx, by)
                gpu_step(gpu, bx, by)
                update = compare_update(cpu.model.get_weights(), gpu.model.get_weights(), initial)
                moments = compare_adam_slots(
                    cpu.optimizer.variables,
                    gpu.optimizer.variables,
                    len(cpu.model.trainable_variables),
                )
                parity.append(
                    {
                        "batch": size,
                        "forward_maxabs": float(np.max(np.abs(cp - gp))),
                        "gradient_relative_l2": gradient_difference,
                        **moments,
                        **update,
                    }
                )
            # Fresh GPU seed42 model; never time a CPU-restored model.
            engine = gpu_engine(recipe)
            bx, by = x[benchmark], y[benchmark]
            gpu_step(engine, bx, by)
            gpu_step(engine, bx, by)
            host_times = []
            for _ in range(10):
                started = time.perf_counter()
                gpu_step(engine, bx, by)
                host_times.append(time.perf_counter() - started)
            with tf.device("/GPU:0"):
                resident_x, resident_y = tf.constant(bx), tf.constant(by)
            resident_times = []
            for _ in range(10):
                started = time.perf_counter()
                gpu_step(engine, resident_x, resident_y)
                resident_times.append(time.perf_counter() - started)
            assert int(engine.optimizer.iterations) == 22
            with tf.device("/GPU:0"):
                concrete = engine.step.get_concrete_function()
            matrix_devices = {
                op.name: op.device
                for op in concrete.graph.get_operations()
                if op.type in ("MatMul", "BatchMatMul", "BatchMatMulV2", "Einsum")
            }
            if not matrix_devices or not all("GPU:0" in d.upper() for d in matrix_devices.values()):
                raise RuntimeError(f"unverified GPU matrix placement: {matrix_devices}")
            # Same-GPU exact restore/replay across partial batch and epoch boundary.
            resume_engine = gpu_engine(recipe)
            sampler = PairSampler(13, 5, recipe.kind == "gnode")
            for _ in range(2):
                idx = small[sampler.next_indices()]
                gpu_step(resume_engine, x[idx], y[idx])
            with tf.device("/GPU:0"):
                save_checkpoint(args.output / key, resume_engine, sampler, identity, {"updates": 2})
            replay = []
            for _ in range(3):
                local = sampler.next_indices()
                replay.append(local)
                gpu_step(resume_engine, x[small[local]], y[small[local]])
            restored = gpu_engine(recipe)
            sampler2 = PairSampler(13, 5, recipe.kind == "gnode")
            with tf.device("/GPU:0"):
                restore_checkpoint(args.output / key, restored, sampler2, identity)
            for local in replay:
                np.testing.assert_array_equal(sampler2.next_indices(), local)
                gpu_step(restored, x[small[local]], y[small[local]])
            for a, b in zip(resume_engine.arrays(), restored.arrays(), strict=True):
                np.testing.assert_array_equal(a, b)
            with tf.device("/GPU:0"):
                execute = compiled_rollout(engine.inference_view(), recipe.kind)
                execute(tf.constant(x[:1]), tf.constant(1)).numpy()
                initial80 = x[np.arange(80) * 1999]
                started = time.perf_counter()
                finite_rollouts = 0
                for start in range(0, 80, 64):
                    pred = execute(
                        tf.constant(initial80[start : start + 64]), tf.constant(1999)
                    ).numpy()
                    finite_rollouts += int(np.isfinite(pred).all(axis=(1, 2, 3)).sum())
                rollout_seconds = time.perf_counter() - started
            memory = tf.config.experimental.get_memory_info("GPU:0")
            median = float(np.median(host_times))
            row = {
                "cpu_gpu_parity": parity,
                "same_gpu_resume_exact": True,
                "host_feed_update_seconds": host_times,
                "resident_update_seconds": resident_times,
                "host_feed_median_seconds": median,
                "gpu_matrix_devices": matrix_devices,
                "gpu_allocator_bytes": memory,
                "train80_standin_rollout_seconds": rollout_seconds,
                "train80_standin_finite": finite_rollouts,
                "projected100epochs_plus25rollouts_hours": (125000 * median + 25 * rollout_seconds)
                / 3600,
                "projection_caveat": "short warm host-feed timing; excludes sampler, scoring, backup/IO and long-run thermal/runtime variation",
                "cpu_timing_reference_seconds": evidence["cpu_recipe_observations"][key][
                    "median_batch512_update_seconds"
                ],
                "gpu_updates": 33,
                "cpu_reference_updates": 3,
            }
            results["models"][key] = row
            save(args.output / (key + "-observations.json"), row)
            print(f"DONE GPU {key}: host-feed update512 median={median:.6f}s", flush=True)
        save(args.output / "summary.json", results)
        save(
            args.output / "COMMITTED.json",
            {p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
        )
        print(
            "PASS: bounded GPU preflight only; no full training authorised or started.", flush=True
        )
    except Exception:
        save(
            args.output / "FAILED.json",
            {"traceback": traceback.format_exc(), "full_training_started": False},
        )
        raise


if __name__ == "__main__":
    main()
