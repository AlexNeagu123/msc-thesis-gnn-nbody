"""Convert Daniela's grouped N-body CSV into reproducible, unnormalized HDF5 splits.

The original 80/20 split matches sklearn's train_test_split on df.id_sim.unique(),
g
with random_state=42. Validation is drawn ONLY from that training partition.
No sklearn, pandas, PyTorch, simulator, or training dependencies are required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np


def sha256(path: Path) -> str:
    """Return a file's SHA-256 without loading the entire file into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate contiguous simulation blocks; sort frames and reorder body features.

    Simulation order follows first appearance, matching pandas.unique in the
    reference notebook. Float64 preserves CSV precision; no normalization occurs.
    """
    ids: list[int] = []
    seen: set[int] = set()
    trajectories: list[np.ndarray] = []
    grids: list[np.ndarray] = []
    block: list[list[float]] = []
    current_id: int | None = None

    def finish_block() -> None:
        if not block:
            return
        values = np.asarray(block, dtype=np.float64)
        values = values[np.argsort(values[:, 0], kind="stable")]
        times = values[:, 0]
        if len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError(f"simulation {current_id}: need >=2 distinct increasing timestamps")
        if not np.allclose(np.diff(times), np.diff(times)[0], rtol=1e-7, atol=1e-10):
            raise ValueError(f"simulation {current_id}: nonuniform timestep")
        if grids and (
            times.shape != grids[0].shape or not np.allclose(times, grids[0], rtol=0, atol=1e-10)
        ):
            raise ValueError(f"simulation {current_id}: inconsistent frame count/time grid")
        raw = values[:, 1:].reshape(len(times), n_bodies, 5)
        masses = raw[..., 0]
        if np.any(masses <= 0) or not np.allclose(masses, masses[0], rtol=0, atol=1e-12):
            raise ValueError(f"simulation {current_id}: masses must be positive and constant")
        trajectories.append(raw[..., [1, 2, 3, 4, 0]])
        grids.append(times)
        block.clear()

    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        n_bodies = (len(header) - 2) // 5
        expected = ["id_sim", "time"] + [
            f"{key}{i}" for i in range(1, n_bodies + 1) for key in ("m", "x", "y", "vx", "vy")
        ]
        if n_bodies < 2 or header != expected:
            raise ValueError("unexpected CSV header; expected id_sim,time,m1,x1,y1,vx1,vy1,...")
        for line, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"line {line}: incorrect column count")
            try:
                simulation_id = int(row[0])
                numeric = [float(value) for value in row[1:]]
            except ValueError as error:
                raise ValueError(f"line {line}: invalid simulation ID or numeric value") from error
            if not all(math.isfinite(value) for value in numeric):
                raise ValueError(f"line {line}: non-finite value")
            if simulation_id != current_id:
                finish_block()
                if simulation_id in seen:
                    raise ValueError(f"simulation {simulation_id}: interleaved simulation blocks")
                seen.add(simulation_id)
                ids.append(simulation_id)
                current_id = simulation_id
            block.append(numeric)
        finish_block()
    if not trajectories:
        raise ValueError("CSV contains no trajectories")
    return np.asarray(ids, dtype=np.int64), np.stack(grids), np.stack(trajectories)


def split_indices(
    n_simulations: int,
    *,
    test_fraction: float = 0.2,
    val_fraction: float = 0.2,
    split_seed: int = 42,
    val_seed: int = 42,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Match sklearn ShuffleSplit, then reserve validation from training only.

    val_fraction is a fraction of the ORIGINAL TRAINING pool, not of all data.
    RandomState/MT19937 intentionally matches sklearn, not default_rng.
    """
    if not 0 < test_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("test_fraction and val_fraction must lie strictly between 0 and 1")
    n_test = math.ceil(n_simulations * test_fraction)
    n_pool = n_simulations - n_test
    n_val = math.ceil(n_pool * val_fraction)
    if n_test < 1 or n_val < 1 or n_pool - n_val < 1:
        raise ValueError("split must leave at least one simulation in train, validation, and test")
    order = np.random.RandomState(split_seed).permutation(n_simulations)
    test, pool = order[:n_test], order[n_test:]
    inner = np.random.RandomState(val_seed).permutation(n_pool)
    return {"train": pool[inner[n_val:]], "val": pool[inner[:n_val]], "test": test}, pool


def softened_energy(states: np.ndarray, *, G: float, epsilon: float) -> np.ndarray:
    """Compute physical energy with the generator's Plummer-softened potential."""
    mass = states[..., 4]
    kinetic = 0.5 * (mass * np.square(states[..., 2:4]).sum(axis=-1)).sum(axis=-1)
    potential = np.zeros(states.shape[:2], dtype=np.float64)
    for i in range(states.shape[2]):
        for j in range(i + 1, states.shape[2]):
            delta = states[:, :, i, :2] - states[:, :, j, :2]
            distance = np.sqrt(np.square(delta).sum(axis=-1) + epsilon**2)
            potential -= G * mass[:, :, i] * mass[:, :, j] / distance
    return kinetic + potential


def convert(
    csv_path: Path,
    output_dir: Path,
    *,
    test_fraction: float = 0.2,
    val_fraction: float = 0.2,
    split_seed: int = 42,
    val_seed: int = 42,
    G: float = 1.0,
    epsilon: float = 0.05,
) -> dict:
    """Write a complete conversion atomically; never overwrite an existing output.

    The legacy metadata group assumes a single mass, so imported provenance is
    stored separately rather than fabricating incompatible metadata values.
    """
    csv_path, output_dir = csv_path.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    if not math.isfinite(G) or G <= 0 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("G and epsilon must be finite and positive")
    source_hash = sha256(csv_path)
    ids, times, states = load_csv(csv_path)
    splits, pool = split_indices(
        len(ids),
        test_fraction=test_fraction,
        val_fraction=val_fraction,
        split_seed=split_seed,
        val_seed=val_seed,
    )
    if sha256(csv_path) != source_hash:
        raise ValueError("source CSV changed during import")
    dt = float(times[0, 1] - times[0, 0])
    manifest = {
        "schema_version": 1,
        "source": {
            "filename": csv_path.name,
            "sha256": source_hash,
            "bytes": csv_path.stat().st_size,
        },
        "layout": {
            "axes": ["simulation", "frame", "body", "feature"],
            "features": ["x", "y", "vx", "vy", "m"],
            "dtype": "float64",
            "n_bodies": states.shape[2],
            "n_frames": states.shape[1],
            "normalized": False,
        },
        "physics": {
            "G": G,
            "epsilon": epsilon,
            "potential": "-G*m_i*m_j/sqrt(r_ij^2+epsilon^2)",
            "origin": "explicit importer parameters; not encoded in source CSV",
        },
        "time": {
            "dt": dt,
            "first_recorded": float(times[0, 0]),
            "last_recorded": float(times[0, -1]),
            "note": "Source labels preserved. Reference generator saves after each integration step, labeling the first saved state 0; no pre-step initial state is reconstructed.",
        },
        "split": {
            "algorithm": "sklearn-compatible ShuffleSplit using numpy.random.RandomState.permutation",
            "split_seed": split_seed,
            "test_fraction": test_fraction,
            "validation_seed": val_seed,
            "validation_fraction_of_original_train": val_fraction,
            "original_train_ids": ids[pool].tolist(),
            "original_test_ids": ids[splits["test"]].tolist(),
        },
        "outputs": {},
        "cautions": [
            "Reference held-out simulations were used as validation in Daniela's notebook; these are not guaranteed historically untouched test data.",
            "Normalization must be fitted on the final training split only.",
            "No encounter buckets are manufactured; do not use bucket_macro_rollout_score on these files.",
            "Stored softened energies do not change existing unsoftened evaluation functions.",
            "HGNN unequal-mass dynamics must be adapted/tested before training.",
        ],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        for name, indices in splits.items():
            target = staging / f"{name}.h5"
            subset = states[indices]
            energies = softened_energy(subset, G=G, epsilon=epsilon)
            with h5py.File(target, "w") as handle:
                handle.create_dataset("trajectories", data=subset, compression="lzf", shuffle=True)
                handle.create_dataset("energies", data=energies, compression="lzf", shuffle=True)
                handle.create_dataset("simulation_ids", data=ids[indices])
                handle.create_dataset("times", data=times[indices])
                handle.attrs["source_sha256"] = source_hash
                handle.attrs["dt"] = dt
                handle.attrs["G"] = G
                handle.attrs["epsilon"] = epsilon
                handle.attrs["mass_mode"] = (
                    "per-particle constant, variable across particles/simulations"
                )
                handle.attrs["energy_definition"] = manifest["physics"]["potential"]
                handle.attrs["split"] = name
            manifest["outputs"][name] = {
                "file": target.name,
                "sha256": sha256(target),
                "shape": list(subset.shape),
                "simulation_ids": ids[indices].tolist(),
            }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        os.rename(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main() -> None:
    """Run the adapter without importing training or simulation dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="fraction of original training simulations reserved for validation",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-seed", type=int, default=42)
    parser.add_argument("--G", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    args = parser.parse_args()
    manifest = convert(
        args.csv,
        args.output_dir,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        val_seed=args.val_seed,
        G=args.G,
        epsilon=args.epsilon,
    )
    print(f"Source SHA-256: {manifest['source']['sha256']}")
    for name, output in manifest["outputs"].items():
        print(f"{name}: {output['shape']} -> {args.output_dir / output['file']}")
    print(f"Manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
