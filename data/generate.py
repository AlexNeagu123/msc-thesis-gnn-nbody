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
import os

import h5py
import numpy as np
import rebound
import yaml

from utils import get_logger

logger = get_logger(__name__)


def generate_trajectory(
    n_particles: int = 3,
    t_end: float = 10.0,
    dt: float = 0.05,
    G: float = 1.0,
    mass: float = 1.0,
    min_distance: float = 0.001,
    pos_scale: float = 1.0,
    vel_scale: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Simulate one 3-body trajectory using REBOUND + IAS15.

    Args:
        n_particles: number of particles in the simulation.
        t_end: total simulation time.
        dt: output snapshot interval.
        G: gravitational constant.
        mass: mass of each particle (equal masses).
        min_distance: minimum pairwise distance before rejection.
        pos_scale: standard deviation of initial position Gaussian.
        vel_scale: standard deviation of initial velocity Gaussian.
        rng: numpy random generator for reproducibility.

    Returns:
        Tuple of (states, energies) where states has shape
        (n_steps, n_particles, 4) with columns [x, y, vx, vy] and
        energies has shape (n_steps,), or None if a close encounter
        is detected.
    """
    if rng is None:
        rng = np.random.default_rng()

    sim = rebound.Simulation()
    sim.G = G
    sim.integrator = "ias15"

    # sample initial conditions from Gaussian, then subtract center of mass
    positions = rng.normal(0, pos_scale, size=(n_particles, 2))
    velocities = rng.normal(0, vel_scale, size=(n_particles, 2))

    # zero out center of mass position and velocity (isolated system, no drift)
    positions -= positions.mean(axis=0)
    velocities -= velocities.mean(axis=0)

    # add particles to simulation (2D: z=0, vz=0)
    for i in range(n_particles):
        sim.add(
            m=mass,
            x=positions[i, 0],
            y=positions[i, 1],
            z=0,
            vx=velocities[i, 0],
            vy=velocities[i, 1],
            vz=0,
        )

    # integrate and record snapshots
    n_steps = int(t_end / dt)
    states = np.zeros((n_steps, n_particles, 4))
    energies = np.zeros(n_steps)

    for step in range(n_steps):
        sim.integrate(sim.t + dt)

        for i in range(n_particles):
            p = sim.particles[i]
            states[step, i] = [p.x, p.y, p.vx, p.vy]

        energies[step] = sim.energy()

        # check all pairwise distances for close encounters
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                dx = states[step, i, 0] - states[step, j, 0]
                dy = states[step, i, 1] - states[step, j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < min_distance:
                    return None

    return states, energies


def generate_dataset(
    n_trajectories: int,
    n_particles: int = 3,
    t_end: float = 10.0,
    dt: float = 0.05,
    G: float = 1.0,
    mass: float = 1.0,
    min_distance: float = 0.001,
    pos_scale: float = 1.0,
    vel_scale: float = 0.5,
    output_path: str = "data/train.h5",
    seed: int = 0,
) -> None:
    """Generate valid trajectories and save to HDF5.

    Rejected trajectories (close encounters) are resampled until the
    requested count is reached.

    Args:
        n_trajectories: number of valid trajectories to generate.
        n_particles: number of particles per simulation.
        t_end: total simulation time per trajectory.
        dt: output snapshot interval.
        G: gravitational constant.
        mass: mass of each particle.
        min_distance: minimum pairwise distance before rejection.
        pos_scale: standard deviation of initial position Gaussian.
        vel_scale: standard deviation of initial velocity Gaussian.
        output_path: path to write the HDF5 file.
        seed: random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(t_end / dt)

    all_states = np.zeros((n_trajectories, n_steps, n_particles, 4))
    all_energies = np.zeros((n_trajectories, n_steps))

    collected = 0
    attempted = 0

    while collected < n_trajectories:
        attempted += 1
        result = generate_trajectory(
            n_particles=n_particles,
            t_end=t_end,
            dt=dt,
            G=G,
            mass=mass,
            min_distance=min_distance,
            pos_scale=pos_scale,
            vel_scale=vel_scale,
            rng=rng,
        )

        if result is None:
            continue

        states, energies = result
        all_states[collected] = states
        all_energies[collected] = energies
        collected += 1

        logger.info(f"trajectory {collected}/{n_trajectories} (attempted: {attempted})")

    # save to HDF5
    with h5py.File(output_path, "w") as f:
        f.create_dataset("trajectories", data=all_states)
        f.create_dataset("energies", data=all_energies)

        meta = f.create_group("metadata")
        meta.attrs["n_trajectories"] = n_trajectories
        meta.attrs["n_particles"] = n_particles
        meta.attrs["n_steps"] = n_steps
        meta.attrs["t_end"] = t_end
        meta.attrs["dt"] = dt
        meta.attrs["G"] = G
        meta.attrs["mass"] = mass
        meta.attrs["min_distance"] = min_distance
        meta.attrs["pos_scale"] = pos_scale
        meta.attrs["vel_scale"] = vel_scale
        meta.attrs["seed"] = seed
        meta.attrs["rejection_rate"] = 1 - n_trajectories / attempted

    logger.info(f"saved {n_trajectories} trajectories to {output_path}")
    logger.info(f"rejection rate: {1 - n_trajectories / attempted:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate N-body trajectory datasets.")
    parser.add_argument("--config", type=str, default="data/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = os.path.dirname(cfg["train_path"])
    os.makedirs(output_dir, exist_ok=True)

    # shared params for all splits
    common = dict(
        n_particles=cfg["n_particles"],
        t_end=cfg["t_end"],
        dt=cfg["dt"],
        G=cfg["G"],
        mass=cfg["mass"],
        min_distance=cfg["min_distance"],
        pos_scale=cfg["pos_scale"],
        vel_scale=cfg["vel_scale"],
    )

    # use different seeds per split so trajectories don't overlap
    seed = cfg["seed"]
    splits = [
        ("train", cfg["n_train"], cfg["train_path"], seed),
        ("val", cfg["n_val"], cfg["val_path"], seed + 1000),
        ("test", cfg["n_test"], cfg["test_path"], seed + 2000),
    ]

    for name, count, path, split_seed in splits:
        logger.info(f"generating {name} split ({count} trajectories)...")
        generate_dataset(
            n_trajectories=count,
            output_path=path,
            seed=split_seed,
            **common,
        )
