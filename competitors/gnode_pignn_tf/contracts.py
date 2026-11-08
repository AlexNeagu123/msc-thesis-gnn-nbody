"""Framework-free layout/export contract, testable in the original Torch venv."""

from __future__ import annotations

import numpy as np


def to_mass_first(state: np.ndarray) -> np.ndarray:
    """Convert physical [x,y,vx,vy,m] to float32 [m,x,y,vx,vy], without scaling."""
    state = np.asarray(state)
    if state.ndim < 2 or state.shape[-2:] != (3, 5):
        raise ValueError("expected three-body states with five features")
    if not np.isfinite(state).all() or np.any(state[..., 4] <= 0):
        raise ValueError("initial states must be finite with positive masses")
    return state[..., [4, 0, 1, 2, 3]].astype(np.float32)


def to_shared_layout(predicted: np.ndarray, initial: np.ndarray) -> np.ndarray:
    """Convert rollout to shared float32 layout, retaining nonfinite state failures.

    Mass is cast once to float32, as in the frozen HGNN evaluator, not changed back
    to original float64. Future-only shape is (batch,steps,3,5).
    """
    predicted, initial = np.asarray(predicted), np.asarray(initial)
    if (
        predicted.ndim != 4
        or predicted.shape[2:] != (3, 5)
        or initial.shape != (len(predicted), 3, 5)
    ):
        raise ValueError("rollout/initial shape mismatch")
    if predicted.dtype != np.float32:
        raise ValueError("float32 model output required")
    mass = np.broadcast_to(initial[:, None, :, 4].astype(np.float32), predicted[..., 0].shape)
    if not np.array_equal(predicted[..., 0], mass):
        raise ValueError("mass changed during rollout")
    return predicted[..., [1, 2, 3, 4, 0]].copy()
