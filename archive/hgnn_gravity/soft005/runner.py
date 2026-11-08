"""Bounded production runner for daniela-hgnn-gravity-soft005-v1."""
import argparse
import fcntl
import json
import math
import os
import signal
import time
from dataclasses import asdict
from pathlib import Path

import torch
import adapter as A
from training import run_cpu_daniela as C
from training._io import load_checkpoint, load_config
from training.colab_runtime import backup_round, verify_backup
from training.fixed_validation import file_sha256
from training.resume import environment

ROOT = A.ROOT
WORK = A.WORK
STOP = {"reason": None}


def mains():
    return any(p.read_text().strip() == "1" for p in Path("/sys/class/power_supply").glob("*/online"))


def check_power():
    if not mains():
        STOP["reason"] = "paused_battery"


def config_delta():
    before = asdict(load_config(ROOT / "configs/daniela-hgnn-cpu-long-horizons.yaml"))
    after = asdict(load_config(A.CONFIG))
    changed = {}

    def walk(a, b, prefix=""):
        assert set(a) == set(b)
        for k in a:
            key = prefix + k
            if isinstance(a[k], dict):
                walk(a[k], b[k], key + ".")
            elif a[k] != b[k]:
                changed[key] = [a[k], b[k]]

    walk(before, after)
    assert set(changed) == {"checkpointing.dir", "logging.dir"}, changed
    return changed


def verify_ready():
    ready = json.loads((WORK / "READY.json").read_text())
    for rel, h in ready["files"].items():
        if file_sha256(ROOT / rel) != h:
            raise ValueError("Readiness identity mismatch: " + rel)
    assert environment(torch.device("cpu")) == ready["environment"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--authorise-full-run", action="store_true")
    parser.add_argument("--max-wall-hours", type=float, default=6.0)
    args = parser.parse_args()
    if not args.check and not args.authorise_full_run:
        parser.error("Explicit authorisation required")
    if not math.isfinite(args.max_wall_hours) or not 0 < args.max_wall_hours <= 6:
        parser.error("Wall limit must be positive and <=6h")
    os.chdir(ROOT)
    C.setup_runtime()
    config_delta()
    verify_ready()
    if not mains():
        raise SystemExit("Mains required")
    if args.check:
        print("CHECK OK: daniela-hgnn-gravity-soft005-v1; additive=0.05; physics_weight=0.1; 2500 staged updates")
        return
    lock = (WORK / "run.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (WORK / "RESULT.json").exists():
        print("Already complete; no updates repeated")
        return
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda *_: STOP.__setitem__("reason", "paused_signal"))
    verified = [(verify_backup(p.parent)["epoch"], p.parent) for p in (WORK / "backups").glob("*/*/COMMITTED.json")]
    latest = max(verified, key=lambda x: x[0])[1] if verified else None
    if latest:
        old = json.loads((WORK / "status.json").read_text())
        if old["state"].startswith("stopped_nonfinite"):
            raise RuntimeError("Numerical failure must not automatically resume")
    cfg = load_config(A.CONFIG)
    plan = {
        "config": asdict(cfg),
        "environment": environment(torch.device("cpu")),
        "operational_config_delta": config_delta(),
        "scientific_intervention": "Staged HGNN (seed 42) + 0.1 * MSE(a_hat, a_grav) with Daniela notebook softening s=0.05",
        "additive_softening": A.ADDITIVE_SOFTENING,
        "physics_weight": A.PHYSICS_WEIGHT,
        "preregistration_sha256": file_sha256(WORK / "PREREGISTRATION.json"),
        "max_wall_hours": args.max_wall_hours,
        "stop_policy": "Signal/battery/wall pauses complete current 100-update round+validation, then commit; nonfinite update or validation is terminal",
    }
    if (WORK / "plan.json").exists():
        assert json.loads((WORK / "plan.json").read_text()) == json.loads(json.dumps(plan))
    else:
        with (WORK / "plan.json").open("x") as f:
            json.dump(plan, f, indent=2)
    trainer = A.GravityLossHGNNTrainer(cfg, resume_checkpoint=latest / "latest.pt" if latest else None)
    trainer.power_or_signal_check = check_power
    assert not trainer.probe and trainer.units_per_round == 100
    started = time.monotonic()

    def publish(state, snapshot=None):
        score = trainer._progress.get("best_selected_score", float("inf"))
        C.write_status(
            WORK / "status.json",
            {
                "state": state,
                "pid": os.getpid(),
                "rounds_done": len(trainer._progress.get("train_history", [])),
                "total_rounds": 25,
                "optimizer_steps": trainer.optimizer_steps,
                "attempted_batches": trainer.attempted_train_batches,
                "assigned_batches_completed": trainer.assigned_completed,
                "best_round": trainer._progress.get("best_epoch"),
                "best_VAL80_MAE": score if math.isfinite(score) else None,
                "snapshot": str(snapshot) if snapshot else None,
                "elapsed_segment_seconds": time.monotonic() - started,
                "stop_requested": STOP["reason"],
            },
        )

    publish("running")
    print("RESUMED " + str(latest) if latest else "FRESH daniela-hgnn-gravity-soft005-v1; seed42; additive=0.05", flush=True)
    while len(trainer._progress.get("train_history", [])) < 25:
        check_power()
        if time.monotonic() - started > args.max_wall_hours * 3600:
            STOP["reason"] = "paused_wall_limit"
        if STOP["reason"]:
            publish(STOP["reason"], latest)
            return
        trainer.run(max_rounds=1)
        latest = backup_round(trainer.ckpt_dir, WORK / "backups")
        verify_backup(latest)
        if trainer.optimizer_steps != trainer.attempted_train_batches:
            publish("stopped_nonfinite_update", latest)
            raise RuntimeError("Skipped update; terminal stop")
        if not trainer.validation_history[-1]["model"]["eligible"]:
            publish("stopped_nonfinite_validation", latest)
            raise RuntimeError("Nonfinite validation; terminal stop")
        publish("running", latest)
        print("COMMITTED", latest, "updates", trainer.optimizer_steps, "best", trainer._progress["best_selected_score"], flush=True)
    assert trainer.optimizer_steps == trainer.assigned_completed == 2500
    best = load_checkpoint(trainer.ckpt_dir / "best.pt", torch.device("cpu"))
    with (WORK / "RESULT.json").open("x") as f:
        json.dump(
            {
                "updates": 2500,
                "best_round": best.epoch,
                "best_VAL80_MAE": best.selected_score,
                "final_snapshot": str(latest),
                "additive_softening": A.ADDITIVE_SOFTENING,
                "physics_weight": A.PHYSICS_WEIGHT,
                "selection": "minimum eligible full1999-future VAL80 position MAE; earlier exact ties",
                "existing100_read": False,
                "preregistration_sha256": file_sha256(WORK / "PREREGISTRATION.json"),
            },
            f,
            indent=2,
        )
    publish("complete", latest)
    print("COMPLETE", best.epoch, best.selected_score, flush=True)


if __name__ == "__main__":
    main()
