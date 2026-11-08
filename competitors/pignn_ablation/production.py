"""Isolated Block A production entry point; historical registry remains unchanged."""

import argparse
import contextlib
import hashlib
import json
import math
import os
import signal
import traceback
from pathlib import Path

from competitors.pignn_ablation.recipes import CELLS
from competitors.gnode_pignn_training import production as P  # noqa: N812

READY = P.ROOT / "runs/daniela-pi-ablation-production-v1/READY.json"
RUNS = P.ROOT / "runs/daniela-pi-ablation-production-v1"


def cell_recipe(cell: str) -> object:
    """Restrict this entry point to the two newly prepared cells."""
    if cell not in ("A10", "A01"):
        raise ValueError("Only A10 and A01 are production candidates")
    return CELLS[cell]


def identity(environment: dict, cell: str) -> dict:
    """Bind original dependencies plus every isolated non-test module."""
    recipe = cell_recipe(cell)
    data = P.ROOT / "data/output/daniela-rk4"
    result = P.build_identity(environment, recipe, data / "train.h5", data / "val.h5")
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if not path.name.startswith("test_"):
            result["sources"][str(path.relative_to(P.ROOT))] = P.sha(path)
    result["ablation_cell"] = cell
    result["adapter_protocol"] = "daniela-pi-factorial-production-v1"
    import numpy as np

    from competitors.gnode_pignn_training.engine import StepEngine

    engine = StepEngine(recipe)
    digest = hashlib.sha256()
    for array in engine.arrays():
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    result["initial_model_adam_arrays_sha256"] = digest.hexdigest()
    result["epoch_pair_indices_sha256"] = hashlib.sha256(
        np.arange(639680, dtype=np.int64).tobytes()
    ).hexdigest()
    result["pairing_scope"] = (
        "same-runtime deterministic reconstruction; no historical initial-array artifact"
    )
    return result


def verify_ready(path: Path = READY) -> None:
    """Refuse launch if any tested file or evidence has changed."""
    ready = json.loads(path.read_text())
    if ready["protocol"] != "daniela-pi-factorial-readiness-v1":
        raise ValueError("Unsupported readiness record")
    for rel, digest in ready["files"].items():
        if P.sha(P.ROOT / rel) != digest:
            raise ValueError(f"Readiness hash mismatch: {rel}")


def assert_resumable(run_dir: Path) -> None:
    """Numerical failures need a new scientific decision, not automatic resumption."""
    status = run_dir / "status.json"
    if status.exists():
        state = json.loads(status.read_text())["state"]
        if state.startswith("stopped_nonfinite"):
            raise ValueError("Numerical failure is terminal for this recipe run; do not resume")


def main() -> None:
    """Check or explicitly authorise one bounded production candidate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=["A10", "A01"], required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--authorise-full-run", action="store_true")
    parser.add_argument("--max-wall-hours", type=float, default=12.0)
    args = parser.parse_args()
    if not args.check and not args.authorise_full_run:
        parser.error("Use --check or explicit --authorise-full-run")
    if not math.isfinite(args.max_wall_hours) or not 0 < args.max_wall_hours <= 12:
        parser.error("Wall budget must be finite, positive and at most12h")
    os.chdir(P.ROOT)
    if not P.power_ok():
        raise SystemExit("Connect mains before checking/starting")
    environment = P.runtime()
    recipe = cell_recipe(args.cell)
    bound = identity(environment, args.cell)
    verify_ready()
    tested_environment = json.loads(READY.read_text())["environment"]
    if environment != tested_environment:
        raise ValueError("Runtime differs from tested readiness environment")
    run_dir = RUNS / recipe.model_id
    assert_resumable(run_dir)
    if args.check:
        print(f"CHECK OK {args.cell}; no run directory created or optimiser updates")
        return
    from competitors.gnode_pignn_training.data import load_train_pairs

    x, y, train_ids = load_train_pairs(P.ROOT / "data/output/daniela-rk4/train.h5")
    initial, reference, val_ids = P.load_validation(P.ROOT / "data/output/daniela-rk4/val.h5")
    if set(train_ids.tolist()) & set(val_ids.tolist()):
        raise ValueError("TRAIN/VAL overlap")
    stop = {"requested": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__("requested", True))
    try:
        result = P.run_training(
            recipe,
            run_dir,
            x,
            y,
            initial,
            reference,
            bound,
            max_wall_seconds=args.max_wall_hours * 3600,
            stop_requested=lambda: stop["requested"],
        )
    except BaseException:
        with contextlib.suppress(OSError), (run_dir / "FAILED.txt").open("a") as handle:
            handle.write(traceback.format_exc() + "\n")
        raise
    raise SystemExit(P.STOP_EXIT.get(result["state"], 1))


if __name__ == "__main__":
    main()
