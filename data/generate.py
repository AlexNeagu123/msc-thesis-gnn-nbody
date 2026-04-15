"""Trajectory generation using REBOUND + IAS15.

Generates chaotic 3-body gravitational trajectories and saves them to HDF5.
Trajectories with close encounters are filtered out during generation.

References:
    - REBOUND docs (simulation API, IAS15 integrator): https://rebound.hanno-rein.de/
    - IAS15 adaptive timestep paper: https://arxiv.org/html/2401.02849v1
    - EGNN N-body data generation (Gaussian init, burn-in approach): https://arxiv.org/pdf/2102.09844
    - EGNN reference implementation: https://github.com/vgsatorras/egnn
    - HGNN (Bishnoi et al. 2023) — repulsive gravity setup: https://arxiv.org/abs/2307.05299
    - Architecture specs (data specification): ../../../edu/architecture-specs.md
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import rebound
import yaml

from data._types import DataGenConfig, SimulationParams
from utils import get_logger

logger = get_logger(__name__)


class Generator:
    """Generates chaotic N-body trajectories and saves them to HDF5.

    Usage::

        cfg = DataGenConfig.from_dict(raw)
        Generator(cfg).run()
    """

    def __init__(self, cfg: DataGenConfig) -> None:
        """Initialize the generator with a typed config.

        Args:
            cfg: data generation configuration.
        """
        self.cfg = cfg
        self.params = cfg.simulation

    def run(self) -> None:
        """Generate all splits defined in the config."""
        for split in self.cfg.splits:
            logger.info(
                "generating %s split (%d trajectories)...",
                split.name,
                split.n_trajectories,
            )
            self._generate_split(
                n_trajectories=split.n_trajectories,
                output_path=split.path,
                seed=split.seed,
            )

    def _generate_split(
        self,
        n_trajectories: int,
        output_path: str,
        seed: int,
    ) -> None:
        """Generate one dataset split and save to HDF5.

        Args:
            n_trajectories: number of valid trajectories to generate.
            output_path: path to write the HDF5 file.
            seed: random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)
        n_steps = int(self.params.t_end / self.params.dt)

        all_states = np.zeros((n_trajectories, n_steps, self.params.n_particles, 5))
        all_energies = np.zeros((n_trajectories, n_steps))

        collected = 0
        attempted = 0

        while collected < n_trajectories:
            attempted += 1
            result = self._simulate_trajectory(rng)

            if result is None:
                continue

            states, energies = result
            all_states[collected] = states
            all_energies[collected] = energies
            collected += 1

            logger.info(
                "trajectory %d/%d (attempted: %d)",
                collected,
                n_trajectories,
                attempted,
            )

        self._save_hdf5(
            all_states,
            all_energies,
            n_trajectories,
            output_path,
            seed,
            attempted,
        )

    def _simulate_trajectory(
        self,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Simulate one trajectory using REBOUND + IAS15.

        Args:
            rng: numpy random generator.

        Returns:
            Tuple of (states, energies) or None if a close encounter is detected.
        """
        params = self.params

        sim = rebound.Simulation()
        sim.G = params.G
        sim.integrator = "ias15"

        positions, velocities = self._sample_initial_conditions(rng)

        # add particles to simulation (2D: z=0, vz=0)
        for i in range(params.n_particles):
            sim.add(
                m=params.mass,
                x=positions[i, 0],
                y=positions[i, 1],
                z=0,
                vx=velocities[i, 0],
                vy=velocities[i, 1],
                vz=0,
            )

        return self._integrate(sim)

    def _sample_initial_conditions(
        self,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample Gaussian initial positions and velocities with COM subtracted.

        Args:
            rng: numpy random generator.

        Returns:
            Tuple of (positions, velocities), each shape (n_particles, 2).
        """
        params = self.params

        positions = rng.normal(0, params.pos_scale, size=(params.n_particles, 2))
        velocities = rng.normal(0, params.vel_scale, size=(params.n_particles, 2))

        # zero out center of mass (isolated system, no drift)
        positions -= positions.mean(axis=0)
        velocities -= velocities.mean(axis=0)

        return positions, velocities

    def _integrate(
        self,
        sim: rebound.Simulation,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Integrate the simulation and record snapshots.

        Args:
            sim: initialized REBOUND simulation.

        Returns:
            Tuple of (states, energies) or None if a close encounter is detected.
        """
        params = self.params
        n_steps = int(params.t_end / params.dt)
        states = np.zeros((n_steps, params.n_particles, 5))
        energies = np.zeros(n_steps)

        for step in range(n_steps):
            sim.integrate(sim.t + params.dt)

            for i in range(params.n_particles):
                p = sim.particles[i]
                states[step, i] = [p.x, p.y, p.vx, p.vy, p.m]

            energies[step] = sim.energy()

            if self._has_close_encounter(states[step]):
                return None

            if self._has_ejection(states[step]):
                return None

        return states, energies

    def _has_ejection(self, snapshot: np.ndarray) -> bool:
        """Check if any particle has been ejected beyond max_position.

        Args:
            snapshot: state at one timestep, shape (n_particles, 5).

        Returns:
            True if any particle's position exceeds the threshold.
        """
        max_pos = self.params.max_position
        if max_pos == float("inf"):
            return False
        positions = snapshot[:, :2]
        return bool(np.abs(positions).max() > max_pos)

    def _has_close_encounter(self, snapshot: np.ndarray) -> bool:
        """Check if any pair of particles is closer than min_distance.

        Args:
            snapshot: state at one timestep, shape (n_particles, 4).

        Returns:
            True if a close encounter is detected.
        """
        for i in range(self.params.n_particles):
            for j in range(i + 1, self.params.n_particles):
                dx = snapshot[i, 0] - snapshot[j, 0]
                dy = snapshot[i, 1] - snapshot[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < self.params.min_distance:
                    return True
        return False

    def _save_hdf5(
        self,
        states: np.ndarray,
        energies: np.ndarray,
        n_trajectories: int,
        output_path: str,
        seed: int,
        attempted: int,
    ) -> None:
        """Save trajectories and metadata to HDF5.

        Args:
            states: trajectory data, shape (n_traj, n_steps, n_particles, 4).
            energies: energy data, shape (n_traj, n_steps).
            n_trajectories: number of trajectories.
            output_path: path to write the HDF5 file.
            seed: random seed used for this split.
            attempted: total number of trajectory attempts.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        params = self.params
        with h5py.File(output, "w") as f:
            f.create_dataset("trajectories", data=states)
            f.create_dataset("energies", data=energies)

            meta = f.create_group("metadata")
            meta.attrs["n_trajectories"] = n_trajectories
            meta.attrs["n_particles"] = params.n_particles
            meta.attrs["n_steps"] = states.shape[1]
            meta.attrs["t_end"] = params.t_end
            meta.attrs["dt"] = params.dt
            meta.attrs["G"] = params.G
            meta.attrs["mass"] = params.mass
            meta.attrs["min_distance"] = params.min_distance
            meta.attrs["pos_scale"] = params.pos_scale
            meta.attrs["vel_scale"] = params.vel_scale
            meta.attrs["seed"] = seed
            meta.attrs["rejection_rate"] = 1 - n_trajectories / attempted

        logger.info("saved %d trajectories to %s", n_trajectories, output_path)
        logger.info("rejection rate: %.1f%%", (1 - n_trajectories / attempted) * 100)


def generate_trajectory(
    params: SimulationParams,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Convenience function for simulating a single trajectory.

    Args:
        params: simulation parameters.
        rng: numpy random generator for reproducibility.

    Returns:
        Tuple of (states, energies) or None if a close encounter is detected.
    """
    if rng is None:
        rng = np.random.default_rng()

    cfg = DataGenConfig(simulation=params, splits=[])
    return Generator(cfg)._simulate_trajectory(rng)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate N-body trajectory datasets.")
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    args = parser.parse_args()

    with Path(args.config).open() as f:
        raw = yaml.safe_load(f)

    config = DataGenConfig.from_dict(raw)
    Generator(config).run()
