"""Pure classifier helpers for encounter-severity stratification.

A trajectory's "encounter severity" is summarised by its minimum pairwise
distance over the full time horizon. This module owns:

    - the canonical bin definition (`DEFAULT_ENCOUNTER_BINS`);
    - the d_min computation;
    - mapping a d_min value to a bin name;
    - rounding a real-valued distribution to integer per-bin counts.

These helpers are deliberately free of REBOUND, h5py, and config types so
the stratified generator (Block 4) and downstream evaluators can import
them without pulling in the simulation stack.

References:
    - data/_types.py : EncounterBin
    - Largest-remainder method: https://en.wikipedia.org/wiki/Largest_remainder_method
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from data._types import EncounterBin

DEFAULT_ENCOUNTER_BINS: tuple[EncounterBin, ...] = (
    EncounterBin("extreme", 0.0, 0.01),
    EncounterBin("very_close", 0.01, 0.02),
    EncounterBin("close", 0.02, 0.05),
    EncounterBin("moderate", 0.05, 0.10),
    EncounterBin("mild", 0.10, 0.20),
    EncounterBin("smooth", 0.20, float("inf")),
)


def min_pairwise_distance_over_time(states: np.ndarray) -> float:
    """Return the minimum (i<j) Euclidean distance over all timesteps.

    Only the first two coordinates of each particle are used so the
    helper works for both 2-D state vectors `[x, y]` and full state
    vectors `[x, y, vx, vy, m, ...]`.

    Args:
        states: array of shape (n_steps, n_particles, dim) with dim >= 2
            and n_particles >= 2.

    Raises:
        ValueError if the shape is wrong or the particle count is < 2.
    """
    if states.ndim != 3:
        msg = f"states must be 3D (n_steps, n_particles, >=2); got shape {states.shape}"
        raise ValueError(msg)

    n_steps, n_particles, dim = states.shape
    if n_steps < 1:
        msg = f"states must have at least 1 timestep; got n_steps={n_steps}"
        raise ValueError(msg)
    if n_particles < 2:
        msg = f"need at least 2 particles to compute pairwise distance; got {n_particles}"
        raise ValueError(msg)
    if dim < 2:
        msg = f"states must have at least 2 spatial coords; got dim={dim}"
        raise ValueError(msg)

    # broadcast pairwise diffs over time, then take the upper triangle
    pos = states[..., :2]
    diffs = pos[:, :, None, :] - pos[:, None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    iu = np.triu_indices(n_particles, k=1)
    pairwise = dists[:, iu[0], iu[1]]
    return float(pairwise.min())


def assign_encounter_bin(distance: float, bins: Sequence[EncounterBin]) -> str:
    """Return the name of the bin whose [lo, hi) interval contains `distance`.

    Half-open semantics: a value sitting on a boundary belongs to the
    upper bin. Non-finite distances (NaN, +/-inf) are rejected because
    they cannot be unambiguously placed under [lo, hi). A distance that
    falls outside every supplied bin also raises, which catches misuse
    such as forgetting a top sentinel `hi=inf` bin.
    """
    if not math.isfinite(distance):
        msg = f"distance must be finite; got {distance!r}"
        raise ValueError(msg)
    for b in bins:
        if b.lo <= distance < b.hi:
            return b.name
    msg = f"distance {distance!r} does not fall in any bin {[b.name for b in bins]}"
    raise ValueError(msg)


def target_counts_from_distribution(total: int, distribution: dict[str, float]) -> dict[str, int]:
    """Round a real-valued distribution to integer counts that sum to `total`.

    Uses the largest-remainder method: each bin first gets `floor(total *
    weight / sum(weights))`, then the remaining surplus is distributed
    one-by-one to bins with the largest fractional parts. Fractional ties
    are broken by the insertion order of `distribution`, so identical
    inputs always produce identical outputs.

    Raises ValueError on negative `total`, an empty distribution,
    negative weights, or a zero (or negative) total weight.
    """
    if total < 0:
        msg = f"total must be >= 0; got {total}"
        raise ValueError(msg)
    if not distribution:
        msg = "distribution must be non-empty"
        raise ValueError(msg)
    if any(not math.isfinite(w) for w in distribution.values()):
        msg = f"weights must be finite; got {distribution}"
        raise ValueError(msg)
    if any(w < 0 for w in distribution.values()):
        msg = f"weights must be non-negative; got {distribution}"
        raise ValueError(msg)
    weight_sum = sum(distribution.values())
    if weight_sum <= 0:
        msg = f"distribution weights must sum to > 0; got {distribution}"
        raise ValueError(msg)

    raw = {name: total * w / weight_sum for name, w in distribution.items()}
    counts = {name: int(v) for name, v in raw.items()}
    remainder = total - sum(counts.values())

    # stable sort on negative fractional part: largest first; ties keep
    # insertion order, so repeat calls are deterministic.
    fractional = [(name, raw[name] - counts[name]) for name in distribution]
    fractional.sort(key=lambda kv: -kv[1])

    for name, _ in fractional[:remainder]:
        counts[name] += 1

    return counts
