"""I/O for the data pipeline: HDF5 trajectories and YAML data-gen configs.

Owns the on-disk schema (dataset names, metadata group) so producers
(data/generate.py) and consumers (data/dataset.py, evaluation) never
touch raw h5py keys.

References:
    - Trajectory bundle: data/_types.py (Trajectories, TrajectoryMetadata, EncounterBin)
    - Data-gen config:  data/_types.py (DataGenConfig)
"""

import json
from dataclasses import fields
from pathlib import Path

import h5py
import numpy as np
import yaml

from data._types import DataGenConfig, EncounterBin, Trajectories, TrajectoryMetadata

_TRAJECTORIES_KEY = "trajectories"
_ENERGIES_KEY = "energies"
_METADATA_GROUP = "metadata"
_ENCOUNTER_BIN_ID_KEY = "encounter_bin_id"
_ENCOUNTER_BIN_NAME_KEY = "encounter_bin_name"
_MIN_PAIRWISE_DISTANCE_KEY = "min_pairwise_distance"
_ENCOUNTER_BINS_ATTR = "encounter_bins_json"


def load_data_config(path: str | Path) -> DataGenConfig:
    """Load a YAML data-gen config into a typed DataGenConfig."""
    with Path(path).open() as f:
        raw = yaml.safe_load(f)
    return DataGenConfig.from_dict(raw)


def read_states(path: Path) -> np.ndarray:
    """Load only the trajectory states (hot path for dataset/evaluator).

    Skips the energies and metadata datasets — use read_trajectories if you
    need the full bundle.
    """
    with h5py.File(path, "r") as f:
        return f[_TRAJECTORIES_KEY][:]


def read_trajectories(path: Path) -> Trajectories:
    """Load trajectories, energies, and optional metadata + stratification.

    Stratification fields are atomic: a file with all four pieces produces
    a Trajectories with all four populated; a file with none produces a
    Trajectories with all four None (legacy / uniform); a partial file
    raises ValueError listing which pieces are missing.
    """
    with h5py.File(path, "r") as f:
        states = f[_TRAJECTORIES_KEY][:]
        energies = f[_ENERGIES_KEY][:]
        metadata = _read_metadata(f) if _METADATA_GROUP in f else None
        strat = _read_stratification(f)
    trajectories = Trajectories(
        states=states,
        energies=energies,
        metadata=metadata,
        encounter_bin_id=strat["encounter_bin_id"],
        encounter_bin_name=strat["encounter_bin_name"],
        min_pairwise_distance=strat["min_pairwise_distance"],
        encounter_bins=strat["encounter_bins"],
    )
    # symmetric guard: corrupted or foreign files surface a clear ValueError here
    # instead of leaking bad data into downstream consumers.
    _validate_stratification(trajectories)
    return trajectories


def write_trajectories(path: Path, trajectories: Trajectories) -> None:
    """Save trajectories, energies, and optional metadata to HDF5.

    When stratification fields are present they are persisted as
    (`encounter_bin_id`, `encounter_bin_name`, `min_pairwise_distance`)
    per-trajectory datasets plus an `encounter_bins_json` attribute on
    the file root. Stratification is atomic: either all four fields are
    populated and survive validation, or none are written.
    """
    _validate_stratification(trajectories)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset(_TRAJECTORIES_KEY, data=trajectories.states)
        f.create_dataset(_ENERGIES_KEY, data=trajectories.energies)
        if trajectories.metadata is not None:
            _write_metadata(f, trajectories.metadata)
        if trajectories.encounter_bin_id is not None:
            _write_stratification(f, trajectories)


def _read_metadata(f: h5py.File) -> TrajectoryMetadata:
    """Build TrajectoryMetadata from h5py attrs, coercing numpy scalars."""
    attrs = f[_METADATA_GROUP].attrs
    return TrajectoryMetadata(
        n_trajectories=int(attrs["n_trajectories"]),
        n_particles=int(attrs["n_particles"]),
        n_steps=int(attrs["n_steps"]),
        t_end=float(attrs["t_end"]),
        dt=float(attrs["dt"]),
        G=float(attrs["G"]),
        mass=float(attrs["mass"]),
        min_distance=float(attrs["min_distance"]),
        pos_scale=float(attrs["pos_scale"]),
        vel_scale=float(attrs["vel_scale"]),
        seed=int(attrs["seed"]),
        rejection_rate=float(attrs["rejection_rate"]),
    )


def _write_metadata(f: h5py.File, metadata: TrajectoryMetadata) -> None:
    """Write all TrajectoryMetadata fields into the metadata group as attrs."""
    group = f.create_group(_METADATA_GROUP)
    for field in fields(metadata):
        group.attrs[field.name] = getattr(metadata, field.name)


def _validate_stratification(t: Trajectories) -> None:
    """Enforce the all-or-none stratification contract before write.

    The contract is intentionally strict: a partial set of stratification
    fields is treated as a producer bug rather than silently downgraded
    to legacy. `n_trajectories` is taken from `states.shape[0]`.
    """
    parts = {
        "encounter_bin_id": t.encounter_bin_id,
        "encounter_bin_name": t.encounter_bin_name,
        "min_pairwise_distance": t.min_pairwise_distance,
        "encounter_bins": t.encounter_bins,
    }
    set_parts = [name for name, value in parts.items() if value is not None]

    if not set_parts:
        return  # legacy / uniform: nothing to write
    if len(set_parts) != len(parts):
        missing = [name for name, value in parts.items() if value is None]
        msg = f"stratification fields are atomic; got {set_parts} but missing {missing}"
        raise ValueError(msg)

    n_traj = t.states.shape[0]
    if t.encounter_bin_id.shape != (n_traj,):
        msg = (
            f"encounter_bin_id shape {t.encounter_bin_id.shape} does not match "
            f"n_trajectories={n_traj}"
        )
        raise ValueError(msg)
    if not np.issubdtype(t.encounter_bin_id.dtype, np.integer):
        msg = f"encounter_bin_id must be integer-typed; got {t.encounter_bin_id.dtype}"
        raise ValueError(msg)

    if t.encounter_bin_name.shape != (n_traj,):
        msg = (
            f"encounter_bin_name shape {t.encounter_bin_name.shape} does not "
            f"match n_trajectories={n_traj}"
        )
        raise ValueError(msg)

    if t.min_pairwise_distance.shape != (n_traj,):
        msg = (
            f"min_pairwise_distance shape {t.min_pairwise_distance.shape} does "
            f"not match n_trajectories={n_traj}"
        )
        raise ValueError(msg)
    if not np.issubdtype(t.min_pairwise_distance.dtype, np.floating):
        msg = f"min_pairwise_distance must be float-typed; got {t.min_pairwise_distance.dtype}"
        raise ValueError(msg)
    if not np.isfinite(t.min_pairwise_distance).all():
        msg = "min_pairwise_distance must be finite for every trajectory"
        raise ValueError(msg)

    if not t.encounter_bins:
        msg = "encounter_bins must be non-empty when stratification fields are set"
        raise ValueError(msg)

    _validate_stratification_consistency(t)


def _validate_stratification_consistency(t: Trajectories) -> None:
    """Cross-check id <-> name <-> distance against the bin definitions.

    Fired only after the structural checks pass, so all four arrays /
    tuples are guaranteed populated and length-aligned. Vectorized so it
    stays cheap for large datasets.
    """
    bins = t.encounter_bins
    n_bins = len(bins)
    ids = t.encounter_bin_id

    if (ids < 0).any() or (ids >= n_bins).any():
        bad = int(np.argmax((ids < 0) | (ids >= n_bins)))
        msg = f"encounter_bin_id[{bad}]={int(ids[bad])} is out of range [0, {n_bins})"
        raise ValueError(msg)

    expected_names = np.array([bins[bid].name for bid in ids])
    name_mismatch = t.encounter_bin_name != expected_names
    if name_mismatch.any():
        bad = int(np.argmax(name_mismatch))
        actual = str(t.encounter_bin_name[bad])
        expected = str(expected_names[bad])
        msg = (
            f"encounter_bin_name[{bad}]={actual!r} does not match bin "
            f"{int(ids[bad])} name {expected!r}"
        )
        raise ValueError(msg)

    los = np.array([bins[bid].lo for bid in ids], dtype=np.float64)
    his = np.array([bins[bid].hi for bid in ids], dtype=np.float64)
    d = t.min_pairwise_distance
    in_bin = (los <= d) & (d < his)
    if not in_bin.all():
        bad = int(np.argmax(~in_bin))
        msg = (
            f"min_pairwise_distance[{bad}]={float(d[bad])} is outside bin "
            f"{bins[int(ids[bad])].name!r} [lo={los[bad]}, hi={his[bad]})"
        )
        raise ValueError(msg)


def _encode_bins(bins: tuple[EncounterBin, ...]) -> str:
    """Encode bins as strict JSON.

    Only the top-of-range `hi == +inf` is serialised as the string
    sentinel `"inf"`; valid configs never have non-finite `lo`, so it is
    encoded as a plain number. `allow_nan=False` is a defensive backstop
    that surfaces any upstream bug (NaN, -inf in lo, etc.) as a clear
    `ValueError` from the encoder rather than letting it persist.
    """
    payload = [
        {
            "name": b.name,
            "lo": b.lo,
            "hi": "inf" if b.hi == float("inf") else b.hi,
        }
        for b in bins
    ]
    return json.dumps(payload, allow_nan=False)


def _decode_bins(s: str) -> tuple[EncounterBin, ...]:
    """Inverse of `_encode_bins`; restores `"inf"` sentinels to `float("inf")`."""

    def _to_float(v: object) -> float:
        return float("inf") if v == "inf" else float(v)  # type: ignore[arg-type]

    payload = json.loads(s)
    return tuple(
        EncounterBin(name=b["name"], lo=_to_float(b["lo"]), hi=_to_float(b["hi"])) for b in payload
    )


def _write_stratification(f: h5py.File, t: Trajectories) -> None:
    """Persist the four stratification fields. Caller has validated atomicity."""
    f.create_dataset(_ENCOUNTER_BIN_ID_KEY, data=t.encounter_bin_id)
    # h5py vlen string write needs object-dtype input; numpy unicode dtype
    # (`<U`) is not accepted directly.
    f.create_dataset(
        _ENCOUNTER_BIN_NAME_KEY,
        data=np.asarray(t.encounter_bin_name, dtype=object),
        dtype=h5py.string_dtype(encoding="utf-8"),
    )
    f.create_dataset(_MIN_PAIRWISE_DISTANCE_KEY, data=t.min_pairwise_distance)
    f.attrs[_ENCOUNTER_BINS_ATTR] = _encode_bins(t.encounter_bins)


def _read_stratification(f: h5py.File) -> dict[str, object]:
    """Read the four stratification fields, enforcing all-or-none on disk."""
    pieces = {
        "encounter_bin_id": _ENCOUNTER_BIN_ID_KEY in f,
        "encounter_bin_name": _ENCOUNTER_BIN_NAME_KEY in f,
        "min_pairwise_distance": _MIN_PAIRWISE_DISTANCE_KEY in f,
        "encounter_bins": _ENCOUNTER_BINS_ATTR in f.attrs,
    }
    if not any(pieces.values()):
        return {name: None for name in pieces}
    if not all(pieces.values()):
        missing = [name for name, present in pieces.items() if not present]
        msg = f"file has partial stratification metadata; missing {missing}"
        raise ValueError(msg)

    return {
        "encounter_bin_id": f[_ENCOUNTER_BIN_ID_KEY][:],
        "encounter_bin_name": f[_ENCOUNTER_BIN_NAME_KEY].asstr()[:],
        "min_pairwise_distance": f[_MIN_PAIRWISE_DISTANCE_KEY][:],
        "encounter_bins": _decode_bins(f.attrs[_ENCOUNTER_BINS_ATTR]),
    }
