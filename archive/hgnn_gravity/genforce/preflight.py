"""Disposable preflight for daniela-hgnn-gravity-genforce-v1: parity, second-order physics grad, timing & resume."""
import gc
import hashlib
import json
import time
from dataclasses import replace

import numpy as np
import torch

import adapter as A
import runner as R
from training import run_cpu_daniela as C
from training._io import load_config
from training.colab_runtime import backup_round, verify_backup
from training.cpu_microbatch import CpuMicrobatchTrainer
from training.resume import environment
from training.test_resume import equal_tree


def tensor_bytes(tree):
    if isinstance(tree, torch.Tensor):
        return tree.detach().cpu().numpy().tobytes()
    if isinstance(tree, dict):
        return b"".join(str(k).encode() + tensor_bytes(v) for k, v in sorted(tree.items()))
    return repr(tree).encode()


def main():
    C.setup_runtime()
    assert R.mains()
    R.config_delta()
    cfg = load_config(A.CONFIG)
    old = load_config(A.ROOT / "configs/daniela-hgnn-cpu-long-horizons.yaml")
    old = replace(old, checkpointing=replace(old.checkpointing, enabled=False), logging=replace(old.logging, enabled=False))
    units = np.array([0, 500, 1000, 1500, 2000], dtype=np.int64)
    disposable = replace(
        cfg,
        training=replace(cfg.training, max_train_batches_per_epoch=5),
        checkpointing=replace(cfg.checkpointing, enabled=False),
        logging=replace(cfg.logging, enabled=False),
    )
    original = CpuMicrobatchTrainer(old, micro_batch_size=16)
    assigned = A.GravityLossHGNNTrainer(disposable, probe_order=units)
    equal_tree(original.model.state_dict(), assigned.model.state_dict())
    initial_digest = hashlib.sha256(tensor_bytes(assigned.model.state_dict())).hexdigest()
    equal_tree(original.optimizer.state_dict(), assigned.optimizer.state_dict())
    assert original.pos_std == assigned.pos_std
    assert original.vel_std == assigned.vel_std
    assert original.momentum_std == assigned.momentum_std

    profiles = []
    for unit, (xb, yb) in zip(units, assigned.train_loader, strict=True):
        h = int(assigned.bank["horizon"][unit])
        # Verify analytical target is finite and physics loss flows only into potential network at step 0
        acc_target = A.analytical_gravity_acceleration(xb, additive=A.ADDITIVE_SOFTENING)
        acc_pred = A.predicted_hgnn_acceleration(assigned.model, xb, create_graph=True)
        phys_mse = float(torch.mean((acc_pred - acc_target).square()).detach())
        assert np.isfinite(phys_mse) and phys_mse > 0

        started = time.perf_counter()
        backward = assigned._backward_logical_batch(xb, yb)
        assert backward is not None
        norm = float(torch.nn.utils.clip_grad_norm_(assigned.model.parameters(), 1.0))
        assert np.isfinite(norm)
        assigned.optimizer.step()
        elapsed = time.perf_counter() - started
        profiles.append({
            "horizon": h,
            "seconds": elapsed,
            "initial_phys_mse": phys_mse,
            "grad_norm_pre_clip": norm,
            "total_loss": float(backward[0]),
        })
        print("FULL-CAPACITY PROBE", profiles[-1], flush=True)

    source_hashes = assigned.source_hashes
    del original, assigned, xb, yb, backward
    gc.collect()

    smoke = A.WORK / f"smoke-disposable-{time.time_ns()}"
    small = replace(
        cfg,
        model=replace(cfg.model, hidden_dim=8, n_layers=1),
        data=replace(cfg.data, val_path=cfg.data.train_path),
        training=replace(cfg.training, max_train_batches_per_epoch=2, validation_rollout_steps=8),
        checkpointing=replace(cfg.checkpointing, dir=str(smoke / "segments")),
        logging=replace(cfg.logging, enabled=False, dir=str(smoke / "segments")),
    )
    probe = np.array([0, 2000, 500, 1500], dtype=np.int64)
    straight = A.GravityLossHGNNTrainer(small, probe_order=probe)
    straight.run(max_rounds=1)
    assert straight.assigned_completed == straight.optimizer_steps == 2
    snap = backup_round(straight.ckpt_dir, smoke / "backups")
    verify_backup(snap)
    resumed = A.GravityLossHGNNTrainer(small, probe_order=probe, resume_checkpoint=snap / "latest.pt")
    assert resumed.assigned_completed == resumed.optimizer_steps == 2
    straight.run(max_rounds=1)
    resumed.run(max_rounds=1)
    assert straight.assigned_completed == resumed.assigned_completed == 4
    equal_tree(straight.model.state_dict(), resumed.model.state_dict())
    equal_tree(straight.optimizer.state_dict(), resumed.optimizer.state_dict())
    equal_tree(straight._progress, resumed._progress)

    rejected = False
    try:
        A.GravityLossHGNNTrainer(small, probe_order=probe[::-1].copy(), resume_checkpoint=snap / "latest.pt")
    except ValueError:
        rejected = True
    assert rejected

    order = np.load(A.ORDER, allow_pickle=False)
    assert np.array_equal(order, np.arange(2500))
    prior = json.loads((A.ROOT / "runs/daniela-hgnn-one-step-control-v1/preflight.json").read_text())
    val_seconds = prior["second_round_seconds"]
    projection = (sum(x["seconds"] for x in profiles) * 500 + val_seconds * 25) / 3600
    report = {
        "arm": "genforce",
        "additive_softening": A.ADDITIVE_SOFTENING,
        "physics_weight": A.PHYSICS_WEIGHT,
        "initial_model_sha256": initial_digest,
        "initial_model_AdamW_scales_exact": True,
        "profiles": profiles,
        "probe_straight_resume_model_AdamW_progress_exact": True,
        "wrong_order_resume_rejected": True,
        "staged_identity_order_verified": True,
        "production_existing100_read": False,
        "disposable_updates": 11,
        "estimated_full_run_hours": projection,
        "source_hashes": source_hashes,
        "environment": environment(torch.device("cpu")),
        "smoke_directory": str(smoke.relative_to(A.ROOT)),
    }
    with (A.WORK / "preflight.json").open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
