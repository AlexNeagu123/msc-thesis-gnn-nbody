"""Generate one uniform zero-shot generalization test set for a chosen body count.

A trimmed companion to data/generate.py: it inherits every parameter from the base
config and overrides only n_particles, so N is the sole variable. Stratification is
off because the 3-body encounter bins do not transfer across N.
"""

import argparse
from dataclasses import replace

from data._io import load_data_config
from data._types import DataGenConfig, SimulationParams, SplitConfig
from data.generate import Generator
from utils import get_logger

logger = get_logger(__name__)

DEFAULT_BASE_CONFIG = "configs/data.yaml"
DEFAULT_N_TRAJECTORIES = 200
# per-N seed = base + n_particles, kept clear of the config's train/val/test seeds
DEFAULT_SEED_BASE = 7000


def default_output_path(n_particles: int) -> str:
    """Return the conventional output path for an N-body generalization set."""
    return f"data/output/generalization_{n_particles}body.h5"


def build_eval_config(
    base_simulation: SimulationParams,
    n_particles: int,
    n_trajectories: int,
    out_path: str,
    seed: int,
) -> DataGenConfig:
    """Single-split uniform config: inherit base_simulation, override n_particles."""
    if n_particles < 2:
        msg = f"n_particles must be >= 2 for a pairwise system; got {n_particles}"
        raise ValueError(msg)
    if n_trajectories < 1:
        msg = f"n_trajectories must be >= 1; got {n_trajectories}"
        raise ValueError(msg)

    simulation = replace(base_simulation, n_particles=n_particles)
    split = SplitConfig(name="test", n_trajectories=n_trajectories, path=out_path, seed=seed)
    return DataGenConfig(simulation=simulation, splits=[split], stratified=None)


def generate_eval_set(
    n_particles: int,
    *,
    n_trajectories: int = DEFAULT_N_TRAJECTORIES,
    out_path: str | None = None,
    seed: int | None = None,
    base_config_path: str = DEFAULT_BASE_CONFIG,
) -> str:
    """Generate one uniform N-body test set and return its output path.

    Defaults: out_path from default_output_path, seed from DEFAULT_SEED_BASE + n_particles.
    """
    resolved_out = out_path or default_output_path(n_particles)
    resolved_seed = seed if seed is not None else DEFAULT_SEED_BASE + n_particles

    base_cfg = load_data_config(base_config_path)
    cfg = build_eval_config(
        base_simulation=base_cfg.simulation,
        n_particles=n_particles,
        n_trajectories=n_trajectories,
        out_path=resolved_out,
        seed=resolved_seed,
    )

    logger.info(
        "generating %d-body generalization test set: %d trajectories -> %s (seed=%d)",
        n_particles,
        n_trajectories,
        resolved_out,
        resolved_seed,
    )
    Generator(cfg).run()
    return resolved_out


def main() -> None:
    """Generate an N-body generalization test set from CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a single uniform N-body zero-shot generalization test set."
    )
    parser.add_argument("--n-particles", type=int, required=True)
    parser.add_argument("--n-trajectories", type=int, default=DEFAULT_N_TRAJECTORIES)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--base-config", type=str, default=DEFAULT_BASE_CONFIG)
    args = parser.parse_args()

    generate_eval_set(
        args.n_particles,
        n_trajectories=args.n_trajectories,
        out_path=args.out,
        seed=args.seed,
        base_config_path=args.base_config,
    )


if __name__ == "__main__":
    main()
