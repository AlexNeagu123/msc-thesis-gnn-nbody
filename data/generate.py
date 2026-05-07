"""Trajectory generation using REBOUND and IAS15.

References:
    - REBOUND docs (simulation API, IAS15 integrator): https://rebound.hanno-rein.de/
    - IAS15 adaptive timestep paper: https://arxiv.org/html/2401.02849v1
    - EGNN N-body data generation (Gaussian init, burn-in approach): https://arxiv.org/pdf/2102.09844
    - EGNN reference implementation: https://github.com/vgsatorras/egnn
    - HGNN (Bishnoi et al. 2023), repulsive gravity setup: https://arxiv.org/abs/2307.05299
    - Architecture specs (data specification): ../../../edu/architecture-specs.md
    - Stratified d_min binning: data/encounters.py
"""

import argparse
from pathlib import Path

import numpy as np
import rebound

from data._io import load_data_config, write_trajectories
from data._types import (
    DataGenConfig,
    SimulationParams,
    SplitConfig,
    Trajectories,
    TrajectoryMetadata,
)
from data.encounters import (
    assign_encounter_bin,
    min_pairwise_distance_over_time,
    target_counts_from_distribution,
)
from utils import get_logger

logger = get_logger(__name__)


class Generator:
    """Generate chaotic N-body trajectories and save them to HDF5."""

    def __init__(self, cfg: DataGenConfig) -> None:
        """Store the typed generation config."""
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
            self._generate_split(split)

    def _generate_split(self, split: SplitConfig) -> None:
        """Dispatch to uniform (legacy) or stratified generation."""
        if self.cfg.stratified is None:
            self._generate_uniform_split(split)
        else:
            self._generate_stratified_split(split)

    def _generate_uniform_split(self, split: SplitConfig) -> None:
        """Generate one dataset split with the legacy first-N-accepted policy."""
        rng = np.random.default_rng(split.seed)
        n_steps = int(self.params.t_end / self.params.dt)
        n_trajectories = split.n_trajectories

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
            split.path,
            split.seed,
            attempted,
        )

    def _generate_stratified_split(self, split: SplitConfig) -> None:
        """Generate one split with per-bin acceptance quotas.

        Repeatedly simulates a candidate trajectory, computes its
        true minimum pairwise distance over time, classifies it into
        the configured encounter bins, and keeps it iff that bin's
        quota for this split is not yet full. The accepted trajectories
        are shuffled with the same split RNG before saving so training
        is mixed and val/test are not grouped by bin.

        The candidate-attempt cap (`StratifiedConfig.max_attempts` if set,
        otherwise `max(10000, n_trajectories * 2000)`) prevents silently
        infinite loops when an extreme bin is rare under the configured
        Gaussian initial conditions.
        """
        if self.cfg.stratified is None:  # narrowing for type checkers; caller dispatched
            msg = "_generate_stratified_split called without a stratified config"
            raise RuntimeError(msg)
        strat = self.cfg.stratified

        distributions = {
            "train": strat.train_distribution,
            "val": strat.val_distribution,
            "test": strat.test_distribution,
        }
        if split.name not in distributions:
            msg = (
                f"stratified mode requires split name in {sorted(distributions)}; "
                f"got {split.name!r}"
            )
            raise ValueError(msg)

        targets = target_counts_from_distribution(split.n_trajectories, distributions[split.name])
        bin_name_to_id = {b.name: i for i, b in enumerate(strat.bins)}

        logger.info(
            "stratified %s | per-bin targets: %s | total=%d",
            split.name,
            targets,
            split.n_trajectories,
        )

        rng = np.random.default_rng(split.seed)
        accepted_states: list[np.ndarray] = []
        accepted_energies: list[np.ndarray] = []
        accepted_bin_ids: list[int] = []
        accepted_bin_names: list[str] = []
        accepted_distances: list[float] = []

        accepted_per_bin = {b.name: 0 for b in strat.bins}
        over_quota_per_bin = {b.name: 0 for b in strat.bins}
        simulator_rejections = 0
        attempted = 0

        # safer default than n*500: rare bins (e.g. smooth >= 0.2 under the
        # current Gaussian setup is roughly 5-6%) need many candidate draws
        # to fill quotas; explicit StratifiedConfig.max_attempts overrides.
        max_attempts = strat.max_attempts or max(10000, split.n_trajectories * 2000)
        progress_step = max(1, split.n_trajectories // 20)

        while sum(accepted_per_bin.values()) < split.n_trajectories:
            if attempted >= max_attempts:
                remaining = {
                    name: targets[name] - accepted_per_bin[name]
                    for name in targets
                    if accepted_per_bin[name] < targets[name]
                }
                msg = (
                    f"stratified generation hit max_attempts={max_attempts} for split "
                    f"{split.name!r} with quotas unmet; "
                    f"accepted={accepted_per_bin}, remaining={remaining}"
                )
                raise RuntimeError(msg)

            attempted += 1
            result = self._simulate_trajectory(rng)
            if result is None:
                simulator_rejections += 1
                continue

            states, energies = result
            d_min = min_pairwise_distance_over_time(states)
            bin_name = assign_encounter_bin(d_min, strat.bins)

            if accepted_per_bin[bin_name] >= targets[bin_name]:
                over_quota_per_bin[bin_name] += 1
                continue

            accepted_states.append(states)
            accepted_energies.append(energies)
            accepted_bin_ids.append(bin_name_to_id[bin_name])
            accepted_bin_names.append(bin_name)
            accepted_distances.append(d_min)
            accepted_per_bin[bin_name] += 1

            total_accepted = sum(accepted_per_bin.values())
            if total_accepted % progress_step == 0 or total_accepted == split.n_trajectories:
                logger.info(
                    "%s: %d/%d (attempted=%d, sim_rejected=%d, over_quota=%d) | per-bin=%s",
                    split.name,
                    total_accepted,
                    split.n_trajectories,
                    attempted,
                    simulator_rejections,
                    sum(over_quota_per_bin.values()),
                    accepted_per_bin,
                )

        # shuffle so training is mixed and val/test rows are not grouped by bin
        permutation = rng.permutation(split.n_trajectories)
        if accepted_states:
            states_arr = np.stack(accepted_states, axis=0)[permutation]
            energies_arr = np.stack(accepted_energies, axis=0)[permutation]
            bin_ids_arr = np.array(accepted_bin_ids, dtype=np.int64)[permutation]
            bin_names_arr = np.array(accepted_bin_names)[permutation]
            distances_arr = np.array(accepted_distances, dtype=np.float64)[permutation]
        else:
            # zero-trajectory split: emit empty arrays rather than crashing on np.stack
            n_steps = int(self.params.t_end / self.params.dt)
            states_arr = np.zeros((0, n_steps, self.params.n_particles, 5), dtype=np.float64)
            energies_arr = np.zeros((0, n_steps), dtype=np.float64)
            bin_ids_arr = np.array([], dtype=np.int64)
            bin_names_arr = np.array([], dtype=object)
            distances_arr = np.array([], dtype=np.float64)

        # candidate-discard rate: combines simulator rejections AND over-quota
        # discards, so it is broader than the uniform path's rejection_rate.
        rejection_rate = 1 - split.n_trajectories / attempted if attempted > 0 else 0.0
        metadata = TrajectoryMetadata(
            n_trajectories=split.n_trajectories,
            n_particles=self.params.n_particles,
            n_steps=states_arr.shape[1],
            t_end=self.params.t_end,
            dt=self.params.dt,
            G=self.params.G,
            mass=self.params.mass,
            min_distance=self.params.min_distance,
            pos_scale=self.params.pos_scale,
            vel_scale=self.params.vel_scale,
            seed=split.seed,
            rejection_rate=rejection_rate,
        )
        write_trajectories(
            Path(split.path),
            Trajectories(
                states=states_arr,
                energies=energies_arr,
                metadata=metadata,
                encounter_bin_id=bin_ids_arr,
                encounter_bin_name=bin_names_arr,
                min_pairwise_distance=distances_arr,
                encounter_bins=strat.bins,
            ),
        )

        logger.info(
            "saved %d stratified trajectories to %s | per-bin=%s | "
            "sim_rejected=%d, over_quota=%s, attempted=%d, rejection_rate=%.1f%%",
            split.n_trajectories,
            split.path,
            accepted_per_bin,
            simulator_rejections,
            over_quota_per_bin,
            attempted,
            rejection_rate * 100,
        )

    def _simulate_trajectory(
        self,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Simulate one trajectory, rejecting close encounters and ejections."""
        params = self.params

        sim = rebound.Simulation()
        sim.G = params.G
        sim.integrator = "ias15"

        positions, velocities = self._sample_initial_conditions(rng)

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
        """Sample Gaussian initial conditions with center-of-mass drift removed."""
        params = self.params

        positions = rng.normal(0, params.pos_scale, size=(params.n_particles, 2))
        velocities = rng.normal(0, params.vel_scale, size=(params.n_particles, 2))

        positions -= positions.mean(axis=0)
        velocities -= velocities.mean(axis=0)

        return positions, velocities

    def _integrate(
        self,
        sim: rebound.Simulation,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Integrate the simulation and record snapshots."""
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
        """Check if any particle exceeds the position threshold."""
        max_pos = self.params.max_position
        if max_pos == float("inf"):
            return False
        positions = snapshot[:, :2]
        return bool(np.abs(positions).max() > max_pos)

    def _has_close_encounter(self, snapshot: np.ndarray) -> bool:
        """Check whether any particle pair is too close."""
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
        """Save trajectories and metadata to HDF5."""
        params = self.params
        rejection_rate = 1 - n_trajectories / attempted

        metadata = TrajectoryMetadata(
            n_trajectories=n_trajectories,
            n_particles=params.n_particles,
            n_steps=states.shape[1],
            t_end=params.t_end,
            dt=params.dt,
            G=params.G,
            mass=params.mass,
            min_distance=params.min_distance,
            pos_scale=params.pos_scale,
            vel_scale=params.vel_scale,
            seed=seed,
            rejection_rate=rejection_rate,
        )

        write_trajectories(
            Path(output_path),
            Trajectories(states=states, energies=energies, metadata=metadata),
        )

        logger.info("saved %d trajectories to %s", n_trajectories, output_path)
        logger.info("rejection rate: %.1f%%", rejection_rate * 100)


def generate_trajectory(
    params: SimulationParams,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Simulate one trajectory with a temporary Generator."""
    if rng is None:
        rng = np.random.default_rng()

    cfg = DataGenConfig(simulation=params, splits=[])
    return Generator(cfg)._simulate_trajectory(rng)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate N-body trajectory datasets.")
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    args = parser.parse_args()

    config = load_data_config(args.config)
    Generator(config).run()
