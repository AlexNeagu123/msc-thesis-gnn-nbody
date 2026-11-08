"""Prepare shared Daniela-style systems or score saved model rollouts as JSON/CSV.

This separate protocol does not change the original evaluator or require Torch.
Future model adapters export NPZ with predicted=(run,step,body,5) and run_ids.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np

from data.import_daniela import sha256
from evaluation._io import _json_safe
from evaluation.rk4_metrics import build_daniela_report, invariants

DT = 0.001
EPSILON = 0.05
DEFAULT_SOURCE = Path(__file__).resolve().parents[3] / "daniela/nbody-net"


def source_provenance(repo: Path) -> dict:
    """Record exact notebook content as well as the repository commit."""
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "repository_commit": commit,
        "notebook_sha256": {
            name: sha256(repo / "notebooks" / name)
            for name in ("1_dataset_generation.ipynb", "2_training.ipynb", "3_testing.ipynb")
        },
    }


def generate_initial_states(seed: int, n_runs: int = 10) -> np.ndarray:
    """Match generate_bodies draw order, using an isolated seeded Python RNG."""
    if n_runs < 1:
        raise ValueError("n_runs must be positive")
    rng = random.Random(seed)
    unit = math.sqrt(149597870700**3 / (6.6743e-11 * 1e30)) / 149597870700
    initial = np.empty((n_runs, 3, 5), dtype=np.float64)
    for run in range(n_runs):
        momentum = np.zeros(2)
        for body in range(3):
            mass = rng.uniform(0.5, 2)
            pos = [rng.uniform(-1, 1), rng.uniform(-1, 1)]
            if body < 2:
                vel = np.array([rng.uniform(-2e4, 2e4) * unit, rng.uniform(-2e4, 2e4) * unit])
                momentum += mass * vel
            else:
                vel = -momentum / mass
            initial[run, body] = [*pos, *vel, mass]
    return initial


def rk4_reference(initial: np.ndarray, n_steps: int = 2000) -> np.ndarray:
    """Integrate the notebook's softened G=1 RK4, including initial state at index 0."""
    initial = np.asarray(initial, dtype=np.float64)
    if (
        initial.ndim != 3
        or initial.shape[1:] != (3, 5)
        or len(initial) < 1
        or not np.isfinite(initial).all()
        or np.any(initial[..., 4] <= 0)
        or n_steps < 1
    ):
        raise ValueError("need finite initial (runs,3,5), positive masses, and positive n_steps")
    mass = initial[..., 4]

    def acceleration(pos: np.ndarray) -> np.ndarray:
        delta = pos[:, None, :, :] - pos[:, :, None, :]
        denominator = (np.square(delta).sum(axis=-1) + EPSILON**2) ** 1.5
        return (mass[:, None, :, None] * delta / denominator[..., None]).sum(axis=2)

    output = np.empty((len(initial), n_steps + 1, 3, 5), dtype=np.float64)
    output[:, 0] = initial
    pos, vel = initial[..., :2].copy(), initial[..., 2:4].copy()
    for step in range(1, n_steps + 1):
        k1q, k1v = vel, acceleration(pos)
        k2q, k2v = vel + DT * 0.5 * k1v, acceleration(pos + DT * 0.5 * k1q)
        k3q, k3v = vel + DT * 0.5 * k2v, acceleration(pos + DT * 0.5 * k2q)
        k4q, k4v = vel + DT * k3v, acceleration(pos + DT * k3q)
        pos = pos + DT / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
        vel = vel + DT / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        output[:, step, :, :2] = pos
        output[:, step, :, 2:4] = vel
        output[:, step, :, 4] = mass
    if not np.isfinite(output).all():
        raise ValueError("nonfinite RK4 reference; do not publish an invalid benchmark")
    return output


def _staging(output: Path) -> Path:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))


def prepare_reference(
    output_dir: Path,
    *,
    seed: int,
    n_runs: int = 10,
    n_steps: int = 2000,
    source_repo: Path = DEFAULT_SOURCE,
    purpose: str = "development",
) -> dict:
    """Freeze the shared initial conditions, references and exact-source manifest."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    if purpose not in ("development", "final"):
        raise ValueError("purpose must be development or final")
    provenance = source_provenance(source_repo)
    initial = generate_initial_states(seed, n_runs)
    trajectories = rk4_reference(initial, n_steps)
    report = {
        "schema_version": 1,
        "protocol": "daniela-repository-compat-v1",
        "purpose": purpose,
        "seed": seed,
        "n_runs": n_runs,
        "n_steps": n_steps,
        "dt": DT,
        "epsilon": EPSILON,
        "G": 1,
        "initial_state_features": ["x", "y", "vx", "vy", "m"],
        "reference_includes_initial": True,
        "source": provenance,
        "generator_sha256": sha256(Path(__file__)),
        "numpy_version": np.__version__,
        "note": "Fresh random systems, not CSV test trajectories. Seeded batch preservation is an explicit reproducibility addition. Do not tune on a batch labelled final.",
    }
    staging = _staging(output_dir)
    try:
        with h5py.File(staging / "reference.h5", "w") as handle:
            handle["initial_states"] = initial
            handle["run_ids"] = np.arange(n_runs, dtype=np.int64)
            handle.create_dataset(
                "trajectories", data=trajectories, compression="lzf", shuffle=True
            )
            handle["times"] = np.arange(n_steps + 1) * DT
            handle["energies"] = invariants(trajectories, epsilon=EPSILON)["energy"]
            handle.attrs["dt"] = DT
            handle.attrs["epsilon"] = EPSILON
        report["reference_sha256"] = sha256(staging / "reference.h5")
        (staging / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        os.rename(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report


def _read_reference(directory: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / "reference.h5"
    if (
        manifest["protocol"] != "daniela-repository-compat-v1"
        or sha256(path) != manifest["reference_sha256"]
    ):
        raise ValueError("reference protocol or checksum mismatch")
    if manifest["dt"] != DT or manifest["epsilon"] != EPSILON or manifest["G"] != 1:
        raise ValueError("reference physics do not match the repository-defined protocol")
    with h5py.File(path, "r") as handle:
        trajectories, ids = handle["trajectories"][:], handle["run_ids"][:]
        expected_shape = (manifest["n_runs"], manifest["n_steps"] + 1, 3, 5)
        if trajectories.shape != expected_shape or not np.array_equal(
            ids, np.arange(manifest["n_runs"])
        ):
            raise ValueError("reference shape/run IDs disagree with manifest")
        if not np.array_equal(handle["initial_states"][:], trajectories[:, 0]):
            raise ValueError("reference initial states disagree with trajectory frame zero")
        if not np.array_equal(handle["times"][:], np.arange(manifest["n_steps"] + 1) * DT):
            raise ValueError("reference time grid disagrees with protocol")
    return manifest, trajectories, ids


def score_predictions(
    reference_dir: Path,
    output_dir: Path,
    *,
    model_name: str,
    predictions_path: Path | None = None,
    baseline: str | None = None,
    metadata: dict | None = None,
    include_series: bool = False,
    runtime_scope: str | None = None,
) -> dict:
    """Score explicit future-state NPZs, or a cheap baseline, into a distinct schema.

    NPZ keys: predicted (runs,steps,bodies,5), run_ids, optional runtime_seconds.
    Predictions must not contain the initial state. IDs/order must match exactly.
    Existing EvaluationReport parsers are intentionally not used on this schema.
    """
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    if (predictions_path is None) == (baseline is None):
        raise ValueError("provide exactly one predictions_path or baseline")
    manifest, reference, ids = _read_reference(reference_dir)
    runtime = None
    prediction_provenance = {}
    if predictions_path is not None:
        prediction_provenance = {"file": str(predictions_path), "sha256": sha256(predictions_path)}
        with np.load(predictions_path, allow_pickle=False) as bundle:
            if not np.array_equal(bundle["run_ids"], ids):
                raise ValueError("prediction run IDs/order do not match the frozen reference")
            predicted = bundle["predicted"]
            if "runtime_seconds" in bundle:
                runtime = bundle["runtime_seconds"]
    else:
        if baseline not in ("constant_velocity", "persistence"):
            raise ValueError("unknown baseline")
        predicted = np.repeat(reference[:, :1], reference.shape[1] - 1, axis=1)
        if baseline == "constant_velocity":
            times = np.arange(1, reference.shape[1]) * DT
            predicted[..., :2] += reference[:, :1, :, 2:4] * times[None, :, None, None]
        prediction_provenance = {"baseline": baseline, "inference_precision": "float64"}
    info = {
        **(metadata or {}),
        "model_name": model_name,
        "reference_manifest": manifest,
        "reference_manifest_sha256": sha256(reference_dir / "manifest.json"),
        "run_ids": ids.tolist(),
        "prediction_source": prediction_provenance,
        "scorer_sha256": sha256(Path(__file__)),
        "metrics_code_sha256": sha256(Path(__file__).with_name("rk4_metrics.py")),
    }
    report = build_daniela_report(
        reference,
        predicted,
        epsilon=manifest["epsilon"],
        metadata=info,
        include_series=include_series,
        runtime_seconds=runtime,
        runtime_scope=runtime_scope,
    )
    staging = _staging(output_dir)
    try:
        safe = _json_safe(report)
        (staging / "daniela_metrics.json").write_text(
            json.dumps(safe, indent=2, allow_nan=False) + "\n"
        )
        row = {
            "model_name": model_name,
            "protocol": report["protocol"],
            "purpose": manifest["purpose"],
            "n_runs": len(ids),
            "n_nonfinite_runs": report["failures"]["n_nonfinite_runs"],
        }
        for window, entry in report["trajectory"].items():
            row[f"{window}_steps_scored"] = entry["n_steps_scored"]
            for metric in ("mae", "mse"):
                for stat in ("mean", "std", "finite_runs"):
                    row[f"position_{window}_{metric}_{stat}"] = entry[metric][stat]
        for invariant, entry in report["conservation_legacy"].items():
            for reduction in ("temporal_mean", "temporal_max"):
                for stat in ("mean", "std"):
                    row[f"legacy_{invariant}_{reduction}_{stat}"] = entry[reduction][stat]
        for reduction in ("temporal_mean", "temporal_max"):
            for stat in ("mean", "std"):
                row[f"physical_softened_energy_{reduction}_{stat}"] = report[
                    "physical_softened_energy"
                ][reduction][stat]
        if report["runtime"] is not None:
            for stat in ("mean", "std"):
                row[f"runtime_seconds_{stat}"] = report["runtime"]["seconds"][stat]
            row["runtime_scope"] = runtime_scope
        with (staging / "daniela_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(_json_safe(row))
        if baseline is not None:
            np.savez_compressed(staging / "predictions.npz", predicted=predicted, run_ids=ids)
        os.rename(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report


def main() -> None:
    """Prepare a development/final batch or score future predictions without Torch."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--seed", type=int, required=True)
    prepare.add_argument("--runs", type=int, default=10)
    prepare.add_argument("--steps", type=int, default=2000)
    prepare.add_argument("--source-repo", type=Path, default=DEFAULT_SOURCE)
    prepare.add_argument("--purpose", choices=("development", "final"), default="development")
    score = commands.add_parser("score")
    score.add_argument("--reference-dir", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--model-name", required=True)
    source = score.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path)
    source.add_argument("--baseline", choices=("constant_velocity", "persistence"))
    score.add_argument(
        "--metadata", type=Path, help="JSON with checkpoint/config identity and inference details"
    )
    score.add_argument("--runtime-scope", help="required when NPZ supplies runtime_seconds")
    score.add_argument("--include-series", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_reference(
            args.output_dir,
            seed=args.seed,
            n_runs=args.runs,
            n_steps=args.steps,
            source_repo=args.source_repo,
            purpose=args.purpose,
        )
    else:
        score_predictions(
            args.reference_dir,
            args.output_dir,
            model_name=args.model_name,
            predictions_path=args.predictions,
            baseline=args.baseline,
            metadata=json.loads(args.metadata.read_text()) if args.metadata else None,
            runtime_scope=args.runtime_scope,
            include_series=args.include_series,
        )
    print(f"Wrote: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
