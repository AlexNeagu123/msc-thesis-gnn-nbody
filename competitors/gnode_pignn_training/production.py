"""Bounded production runner for one named shared-split competitor recipe.

Scope: fresh from-scratch training on the exact TRAIN320 systems, full1999-step
future VAL80 position-MAE selection, explicit staged checkpoints and exact resume.
It reads ONLY train.h5 and val.h5; existing-100 records are never opened here.

Call ``runtime()`` (or use the CLI) before importing TensorFlow-dependent modules.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
import resource
import shutil
import signal
import tempfile
import time
import traceback
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "daniela-shared-production-v1"
SELECTION_EVERY = 4
VALIDATION_BATCH = 40
STOP_EXIT = {"complete": 0, "paused_signal": 0, "paused_battery": 75, "paused_wall_limit": 75}


def sha(path: Path) -> str:
    """Hash exact bytes without loading very large files at once."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def runtime() -> dict:
    """Apply the frozen one-thread CPU policy before any TensorFlow work."""
    from competitors.gnode_pignn_tf.runtime import configure

    return configure()


def power_ok() -> bool:
    """Require mains power; absent power-supply information is treated as unsafe."""
    online = [
        (p / "online").read_text().strip() == "1"
        for p in Path("/sys/class/power_supply").glob("*")
        if (p / "online").is_file()
    ]
    return any(online)


def atomic_json(path: Path, value: dict) -> None:
    """Atomically publish small mutable status; never used for checkpoints."""
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_validation(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read VAL80 only: mass-first float32 initial states and float64 shared reference."""
    with h5py.File(path) as handle:
        if handle.attrs.get("split") != "val" or float(handle.attrs["dt"]) != 0.001:
            raise ValueError("validation selection requires the VAL split and dt=.001")
        ids = handle["simulation_ids"][:]
        if len(ids) != 80 or len(np.unique(ids)) != 80:
            raise ValueError("expected exact80 whole validation systems")
        reference = handle["trajectories"][:][np.argsort(ids)].astype(np.float64)
    if reference.shape != (80, 2000, 3, 5) or not np.isfinite(reference).all():
        raise ValueError("unexpected validation data")
    initial = reference[:, 0][..., [4, 0, 1, 2, 3]].astype(np.float32)
    return initial, reference, np.sort(ids)


def validation_score(
    rollout: object, initial: np.ndarray, reference: np.ndarray, batch: int = VALIDATION_BATCH
) -> dict:
    """Mean per-system future position MAE over all steps; any nonfinite system is ineligible."""
    import tensorflow as tf

    steps = reference.shape[1] - 1
    errors, finite = [], []
    for start in range(0, len(initial), batch):
        stop = start + batch
        predicted = np.asarray(rollout(tf.constant(initial[start:stop]), tf.constant(steps)))
        if predicted.shape != (len(initial[start:stop]), steps, 3, 5):
            raise ValueError("unexpected rollout shape")
        finite.extend(np.isfinite(predicted).all(axis=(1, 2, 3)).tolist())
        delta = predicted[..., 1:3].astype(np.float64) - reference[start:stop, 1:, :, :2]
        with np.errstate(over="ignore", invalid="ignore"):
            errors.extend(np.abs(delta).mean(axis=(1, 2, 3)).tolist())
    eligible = bool(all(finite)) and bool(np.isfinite(errors).all())
    return {
        "eligible": eligible,
        "score": float(np.mean(errors)) if eligible else None,
        "finite_systems": int(sum(finite)),
        "n_systems": len(finite),
    }


def verify_bundle(directory: Path, updates: int | None = None) -> dict:
    """Verify an existing committed checkpoint's hashes; optionally its update counter."""
    hashes = json.loads((directory / "COMMITTED.json").read_text())
    for name, digest in hashes.items():
        if Path(name).name != name or sha(directory / name) != digest:
            raise ValueError(f"checkpoint bundle hash mismatch: {directory}")
    metadata = json.loads((directory / "metadata.json").read_text())
    if updates is not None and metadata["progress"].get("updates") != updates:
        raise ValueError(f"existing checkpoint has different progress: {directory}")
    return metadata


def save_if_absent(
    directory: Path, engine: object, sampler: object, identity: dict, p: dict
) -> None:
    """Idempotent after crashes: reuse only a verified checkpoint with identical progress."""
    from competitors.gnode_pignn_training.checkpoint import save_checkpoint

    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        verify_bundle(directory, p["updates"])
        return
    save_checkpoint(directory, engine, sampler, identity, json.loads(json.dumps(p)))


def latest_checkpoint(run_dir: Path) -> Path | None:
    """Select the highest-update committed checkpoint; pending directories are ignored."""
    found = []
    for path in (run_dir / "checkpoints").glob("u*"):
        if (path / "COMMITTED.json").is_file() and ".pending-" not in path.name:
            found.append((int(path.name[1:]), path))
    return max(found)[1] if found else None


def selection_epochs(epochs: int, every: int = SELECTION_EVERY) -> list[int]:
    """Frozen validation schedule; always includes the final epoch."""
    return sorted({*range(every, epochs + 1, every), epochs})


def fresh_progress() -> dict:
    """Initial JSON-safe scientific progress; no NaN values."""
    return {
        "updates": 0,
        "epochs_done": 0,
        "epoch_updates": 0,
        "epoch_sums": {"data": 0.0, "physics": 0.0, "total": 0.0},
        "epoch_history": [],
        "validations": [],
        "best": None,
        "final": None,
    }


def run_training(
    recipe: object,
    run_dir: Path,
    x: np.ndarray,
    y: np.ndarray,
    initial: np.ndarray,
    reference: np.ndarray,
    identity: dict,
    *,
    selection_every: int = SELECTION_EVERY,
    max_wall_seconds: float | None = None,
    stop_requested: object = lambda: False,
    mains_power: object = power_ok,
    score_fn: object | None = None,
    log: object = print,
) -> dict:
    """Train/resume until complete or a controlled boundary; return a state record."""
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir / "run.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError(
            "another process owns this run directory; no duplicate launched"
        ) from error
    try:
        return _run_locked(
            recipe,
            run_dir,
            x,
            y,
            initial,
            reference,
            identity,
            selection_every=selection_every,
            max_wall_seconds=max_wall_seconds,
            stop_requested=stop_requested,
            mains_power=mains_power,
            score_fn=score_fn,
            log=log,
        )
    finally:
        lock.close()


def _run_locked(
    recipe: object,
    run_dir: Path,
    x: np.ndarray,
    y: np.ndarray,
    initial: np.ndarray,
    reference: np.ndarray,
    identity: dict,
    *,
    selection_every: int,
    max_wall_seconds: float | None,
    stop_requested: object,
    mains_power: object,
    score_fn: object | None,
    log: object,
) -> dict:
    """Run body while the exclusive run-directory lock is held."""
    import tensorflow as tf

    from competitors.gnode_pignn_tf.models import compiled_rollout
    from competitors.gnode_pignn_training.checkpoint import restore_checkpoint
    from competitors.gnode_pignn_training.data import PairSampler
    from competitors.gnode_pignn_training.engine import StepEngine

    plan = {
        "protocol": PROTOCOL,
        "recipe": recipe.record(),
        "identity": identity,
        "selection_every_epochs": selection_every,
        "selection_epochs": selection_epochs(recipe.epochs, selection_every),
        "validation_batch": VALIDATION_BATCH,
        "validation_future_steps": reference.shape[1] - 1,
        "metric": "mean per-system position MAE over all future steps",
    }
    plan_path = run_dir / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != json.loads(json.dumps(plan)):
            raise ValueError("saved run plan differs; refusing to overwrite or mix recipes")
    else:
        with plan_path.open("x") as handle:
            json.dump(plan, handle, indent=2, allow_nan=False)
    if (run_dir / "RESULT.json").exists():
        log("Run already complete; nothing repeated.")
        return {"state": "complete", "already_complete": True}
    if shutil.disk_usage(run_dir).free < 2 * 1024**3:
        raise RuntimeError("less than2GiB free disk")

    engine = StepEngine(recipe)
    sampler = PairSampler(len(x), recipe.batch_size, recipe.kind == "gnode", recipe.seed)
    rollout = compiled_rollout(engine.inference_view(), recipe.kind)
    score = score_fn or (lambda: validation_score(rollout, initial, reference))
    previous = latest_checkpoint(run_dir)
    progress = fresh_progress()
    if previous is not None:
        progress = restore_checkpoint(previous, engine, sampler, identity)
        log(f"RESUMED verified checkpoint {previous.name}: updates={progress['updates']}")
        if progress["best"] is not None:
            verify_bundle(run_dir / progress["best"]["checkpoint"])
    else:
        log("FRESH from-scratch run; no public or disposable weights used.")
    status_path = run_dir / "status.json"
    started = time.monotonic()
    segment_start_updates = progress["updates"]
    schedule = set(plan["selection_epochs"])

    def publish(state: str, **extra: object) -> None:
        elapsed = time.monotonic() - started
        done = progress["updates"] - segment_start_updates
        atomic_json(
            status_path,
            {
                "state": state,
                "pid": os.getpid(),
                "utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "updates": progress["updates"],
                "total_updates": recipe.epochs * -(-len(x) // recipe.batch_size),
                "epochs_done": progress["epochs_done"],
                "best": progress["best"],
                "segment_seconds": elapsed,
                "segment_updates": done,
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                **extra,
            },
        )

    def checkpoint_here() -> Path:
        path = run_dir / "checkpoints" / f"u{progress['updates']:07d}"
        save_if_absent(path, engine, sampler, identity, progress)
        return path

    def stop_reason() -> str | None:
        if stop_requested():
            return "paused_signal"
        if max_wall_seconds is not None and time.monotonic() - started > max_wall_seconds:
            return "paused_wall_limit"
        if progress["updates"] % 50 == 0 and not mains_power():
            return "paused_battery"
        return None

    publish("running")
    try:
        while progress["epochs_done"] < recipe.epochs:
            reason = stop_reason()
            if reason:
                path = checkpoint_here()
                publish(reason, checkpoint=str(path))
                log(f"STOP {reason}: committed {path.name}")
                return {"state": reason, "checkpoint": str(path)}
            indices = sampler.next_indices()
            row = engine.step(x[indices], y[indices])
            progress["updates"] = int(engine.optimizer.iterations)
            progress["epoch_updates"] += 1
            for key in progress["epoch_sums"]:
                progress["epoch_sums"][key] += float(row[key])
            if sampler.cursor != sampler.size:
                if progress["updates"] % 250 == 0:
                    publish("running")
                continue
            epoch = progress["epochs_done"] = progress["epochs_done"] + 1
            means = {k: v / progress["epoch_updates"] for k, v in progress["epoch_sums"].items()}
            progress["epoch_history"].append(
                {"epoch": epoch, "updates": progress["updates"], **means}
            )
            progress["epoch_sums"] = {"data": 0.0, "physics": 0.0, "total": 0.0}
            progress["epoch_updates"] = 0
            note = f"epoch {epoch}/{recipe.epochs} updates={progress['updates']} train_total={means['total']:.8g}"
            if epoch in schedule:
                begin = time.monotonic()
                result = score()
                result.update(
                    epoch=epoch, updates=progress["updates"], seconds=time.monotonic() - begin
                )
                progress["validations"].append(result)
                note += f" VAL80 MAE={result['score']} eligible={result['eligible']}"
                best = progress["best"]
                if result["eligible"] and (best is None or result["score"] < best["score"]):
                    name = f"selected/best-epoch-{epoch:04d}"
                    progress["best"] = {
                        "epoch": epoch,
                        "updates": progress["updates"],
                        "score": result["score"],
                        "checkpoint": name,
                    }
                    save_if_absent(run_dir / name, engine, sampler, identity, progress)
                    note += " NEW BEST"
                if not result["eligible"]:
                    path = checkpoint_here()
                    publish("stopped_nonfinite_validation", checkpoint=str(path))
                    raise RuntimeError(
                        "nonfinite validation prediction; run stopped after preserving state"
                    )
            if epoch == recipe.epochs:
                name = f"selected/final-epoch-{epoch:04d}"
                progress["final"] = {
                    "epoch": epoch,
                    "updates": progress["updates"],
                    "checkpoint": name,
                }
                save_if_absent(run_dir / name, engine, sampler, identity, progress)
            if epoch in schedule or epoch == recipe.epochs:
                checkpoint_here()
            log(note)
            publish("running")
    except tf.errors.InvalidArgumentError as error:
        publish("stopped_nonfinite_update", error=str(error)[:500])
        raise
    if progress["best"] is None:
        raise RuntimeError("completed with no eligible validation checkpoint")
    final = {
        "protocol": PROTOCOL,
        "recipe_id": recipe.model_id,
        "best": progress["best"],
        "final": progress["final"],
        "validations": progress["validations"],
        "epoch_history": progress["epoch_history"],
        "updates": progress["updates"],
        "identity_sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest(),
        "comparison_records_read": False,
        "bundle_hashes": {
            str(p.relative_to(run_dir)): sha(p)
            for p in sorted((run_dir / "selected").glob("*/COMMITTED.json"))
        },
    }
    with (run_dir / "RESULT.json").open("x") as handle:
        json.dump(final, handle, indent=2, allow_nan=False)
    publish("complete")
    log(f"COMPLETE: best epoch {progress['best']['epoch']} VAL80 MAE {progress['best']['score']}")
    return {"state": "complete", "best": progress["best"]}


def build_identity(environment: dict, recipe: object, train: Path, val: Path) -> dict:
    """Bind recipe/source/data/runtime; train/val must equal the frozen shared split."""
    frozen = json.loads(
        (ROOT / "runs/daniela-public-replay-v1/FROZEN_BEFORE_COMPARISON.json").read_text()
    )["input_sha256"]
    outputs = json.loads((ROOT / "data/output/daniela-rk4/manifest.json").read_text())["outputs"]
    hashes = {"train": outputs["train"]["sha256"], "val": outputs["val"]["sha256"]}
    # The import manifest is the authority for VAL; TRAIN must also match the earlier freeze.
    if frozen[str(train.relative_to(ROOT))] != hashes["train"]:
        raise ValueError("TRAIN hash differs from the frozen comparison record")
    for name, path in (("train", train), ("val", val)):
        if sha(path) != hashes[name]:
            raise ValueError(f"{path.name} differs from the import manifest")
    sources = {
        str(p.relative_to(ROOT)): sha(p)
        for folder in ("gnode_pignn_tf", "gnode_pignn_training")
        for p in sorted((ROOT / "competitors" / folder).glob("*.py"))
        if not p.name.startswith("test_")
    }
    return {
        "environment": environment,
        "train_sha256": hashes["train"],
        "val_sha256": hashes["val"],
        "sources": sources,
        "recipe_id": recipe.model_id,
        "scope": "production_shared320_80_from_scratch",
        "selection": "VAL80 only; comparison records unread",
    }


def main() -> None:
    """Explicit bounded CLI; no implicit full-run defaults without authorisation flag."""
    from competitors.gnode_pignn_training.recipes import RECIPES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=sorted(RECIPES), required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--max-wall-hours", type=float, default=12.0)
    parser.add_argument("--check", action="store_true", help="verify identities, then exit")
    parser.add_argument("--authorise-full-run", action="store_true")
    args = parser.parse_args()
    if not args.check and not args.authorise_full_run:
        parser.error("training requires --authorise-full-run; use --check otherwise")
    if args.max_wall_hours <= 0:
        parser.error("positive wall limit required")
    os.chdir(ROOT)
    if not power_ok():
        raise SystemExit("Connect the charger first.")
    environment = runtime()
    recipe = RECIPES[args.recipe]
    train = ROOT / "data/output/daniela-rk4/train.h5"
    val = ROOT / "data/output/daniela-rk4/val.h5"
    identity = build_identity(environment, recipe, train, val)
    ready = ROOT / "runs/daniela-shared-training-v1/READY.json"
    if not ready.is_file():
        raise SystemExit("READY.json missing; tests have not been recorded.")
    for name, digest in json.loads(ready.read_text())["files"].items():
        if sha(ROOT / name) != digest:
            raise SystemExit(f"tested file changed: {name}; revalidate before launch")
    run_dir = args.run_dir or ROOT / "runs/daniela-shared-training-v1" / args.recipe
    if args.check:
        print(f"CHECK OK for {args.recipe}; no run directory created and no training started.")
        return
    from competitors.gnode_pignn_training.data import load_train_pairs

    x, y, train_ids = load_train_pairs(train)
    initial, reference, val_ids = load_validation(val)
    if set(train_ids.tolist()) & set(val_ids.tolist()):
        raise ValueError("train and validation systems overlap")
    stop = {"flag": False}
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.__setitem__("flag", True))
    try:
        result = run_training(
            recipe,
            run_dir,
            x,
            y,
            initial,
            reference,
            identity,
            max_wall_seconds=args.max_wall_hours * 3600,
            stop_requested=lambda: stop["flag"],
        )
    except BaseException:
        with contextlib.suppress(OSError):
            (run_dir / "FAILED.txt").open("a").write(traceback.format_exc() + "\n")
        raise
    raise SystemExit(STOP_EXIT.get(result["state"], 1))


if __name__ == "__main__":
    main()
