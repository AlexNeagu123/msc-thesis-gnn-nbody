"""Bin-mask helpers for slicing stratified evaluation arrays by encounter bin."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def trajectory_masks(
    bin_id: npt.NDArray[np.integer],
    n_bins: int,
) -> list[npt.NDArray[np.bool_]]:
    """One boolean (n_trajectories,) mask per bin id; empty bins yield all-False masks."""
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
    """Repeat each mask entry n_transitions times to match NBodyDataset's flattened layout.

    Repetition (not tiling), since the flatten is row-major (n_traj, n_transitions).
    """
    if n_transitions < 0:
        msg = f"n_transitions must be >= 0, got {n_transitions}"
        raise ValueError(msg)
    if mask.ndim != 1:
        msg = f"mask must be 1-D, got shape {mask.shape}"
        raise ValueError(msg)
    return np.repeat(mask, n_transitions)
