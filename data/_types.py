"""Typed contracts for the data pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationParams:
    """Physical and numerical parameters for a single trajectory."""

    n_particles: int
    t_end: float
    dt: float
    G: float
    mass: float
    min_distance: float
    max_position: float
    pos_scale: float
    vel_scale: float


@dataclass
class SplitConfig:
    """Configuration for a single data split (train, val, or test)."""

    name: str
    n_trajectories: int
    path: str
    seed: int


@dataclass
class DataGenConfig:
    """Top-level data generation configuration."""

    simulation: SimulationParams
    splits: list[SplitConfig]

    @staticmethod
    def from_dict(d: dict) -> DataGenConfig:
        """Build a DataGenConfig from a parsed YAML dict."""
        simulation = SimulationParams(
            n_particles=d["n_particles"],
            t_end=d["t_end"],
            dt=d["dt"],
            G=d["G"],
            mass=d["mass"],
            min_distance=d["min_distance"],
            max_position=d.get("max_position", float("inf")),
            pos_scale=d["pos_scale"],
            vel_scale=d["vel_scale"],
        )

        seed = d["seed"]
        splits = [
            SplitConfig("train", d["n_train"], d["train_path"], seed),
            SplitConfig("val", d["n_val"], d["val_path"], seed + 1000),
            SplitConfig("test", d["n_test"], d["test_path"], seed + 2000),
        ]

        return DataGenConfig(simulation=simulation, splits=splits)


@dataclass
class TrajectoryMetadata:
    """Provenance attributes attached to each trajectory HDF5 file."""

    n_trajectories: int
    n_particles: int
    n_steps: int
    t_end: float
    dt: float
    G: float
    mass: float
    min_distance: float
    pos_scale: float
    vel_scale: float
    seed: int
    rejection_rate: float


@dataclass
class Trajectories:
    """Typed bundle for the contents of one trajectory HDF5 file.

    `metadata` is optional because test fixtures and legacy files may
    omit the metadata group; production files always include it.
    """

    states: np.ndarray  # (n_trajectories, n_steps, n_particles, 5)
    energies: np.ndarray  # (n_trajectories, n_steps)
    metadata: TrajectoryMetadata | None = None
