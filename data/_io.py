"""HDF5 serialization for trajectory data.

Owns the on-disk schema (dataset names, metadata group) so producers
(data/generate.py) and consumers (data/dataset.py, evaluation) never
touch raw h5py keys.

References:
    - Trajectory bundle: data/_types.py (Trajectories, TrajectoryMetadata)
"""

from dataclasses import fields
from pathlib import Path

import h5py

from data._types import Trajectories, TrajectoryMetadata

_TRAJECTORIES_KEY = "trajectories"
_ENERGIES_KEY = "energies"
_METADATA_GROUP = "metadata"


def read_trajectories(path: Path) -> Trajectories:
    """Load trajectories, energies, and optional metadata from an HDF5 file."""
    with h5py.File(path, "r") as f:
        states = f[_TRAJECTORIES_KEY][:]
        energies = f[_ENERGIES_KEY][:]
        metadata = _read_metadata(f) if _METADATA_GROUP in f else None
    return Trajectories(states=states, energies=energies, metadata=metadata)


def write_trajectories(path: Path, trajectories: Trajectories) -> None:
    """Save trajectories, energies, and optional metadata to HDF5."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset(_TRAJECTORIES_KEY, data=trajectories.states)
        f.create_dataset(_ENERGIES_KEY, data=trajectories.energies)
        if trajectories.metadata is not None:
            _write_metadata(f, trajectories.metadata)


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
