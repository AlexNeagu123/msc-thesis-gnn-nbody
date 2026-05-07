"""Bin-mask helpers for stratified evaluation slicing.

Pure helpers used by evaluation/evaluate.py to slice per-trajectory
metric arrays (rollout MSE, energy drift) and flattened per-transition
arrays (single-step MSE, min pairwise distance) by encounter-severity
bin. The two helpers split the work cleanly:

    - `trajectory_masks` produces one boolean mask per bin id; callers
      apply each mask to any per-trajectory array.
    - `expand_trajectory_mask_to_transitions` lifts a per-trajectory
      mask onto the flattened (n_traj * n_transitions) layout used by
      NBodyDataset, so single-step metrics can reuse the same masks.

References:
    - Bin definitions: data/_types.py (EncounterBin, StratifiedConfig)
    - Report schema: evaluation/_types.py (EncounterBinsReport)
    - Transition flatten contract: data/dataset.py (NBodyDataset)
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def trajectory_masks(
    bin_id: npt.NDArray[np.integer],
    n_bins: int,
) -> list[npt.NDArray[np.bool_]]:
    """Return one boolean mask per bin id, in canonical order.

    Each mask has shape `(n_trajectories,)` and selects rows where
    `bin_id == i` for `i in range(n_bins)`. Empty bins yield all-False
    masks; this is intentional so downstream code can treat bins
    uniformly without special-casing absence.
    """
    if n_bins < 0:
        msg = f"n_bins must be >= 0, got {n_bins}"
        raise ValueError(msg)
    if bin_id.ndim != 1:
        msg = f"bin_id must be 1-D, got shape {bin_id.shape}"
        raise ValueError(msg)
    return [bin_id == i for i in range(n_bins)]


def expand_trajectory_mask_to_transitions(
    mask: npt.NDArray[np.bool_],
    n_transitions: int,
) -> npt.NDArray[np.bool_]:
    """Repeat each per-trajectory mask entry `n_transitions` times.

    NBodyDataset flattens `(n_traj, n_transitions)` into
    `(n_traj * n_transitions,)` in row-major dataset order, so a per-
    trajectory mask must be expanded by repetition (not tiling) before
    use against any flattened transition array.
    """
    if n_transitions < 0:
        msg = f"n_transitions must be >= 0, got {n_transitions}"
        raise ValueError(msg)
    if mask.ndim != 1:
        msg = f"mask must be 1-D, got shape {mask.shape}"
        raise ValueError(msg)
    return np.repeat(mask, n_transitions)
