"""Declared cross-device tolerances; never silently relaxed after a failure."""

import numpy as np


def compare_adam_slots(cpu: list[object], gpu: list[object], expected_slots: int) -> dict:
    """Check counters exactly and moments separately so counters cannot hide error."""
    if len(cpu) != len(gpu) or len(cpu) < 4:
        raise ValueError("unexpected Adam state count")
    for left, right in zip(cpu[:2], gpu[:2], strict=True):
        np.testing.assert_array_equal(left.numpy(), right.numpy())
    result = {}
    for suffix, limit in (("momentum", 5e-4), ("velocity", 1e-3)):
        left = [v.numpy() for v in cpu if v.name.endswith("_" + suffix)]
        right = [v.numpy() for v in gpu if v.name.endswith("_" + suffix)]
        if len(left) != expected_slots or len(right) != expected_slots:
            raise ValueError("unexpected Adam slot hierarchy")
        difference = relative_l2(left, right)
        if difference > limit:
            raise ValueError(f"CPU/GPU Adam {suffix} parity gate failed")
        result["adam_" + suffix + "_relative_l2"] = difference
    return result


def relative_l2(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    """Compare concatenated arrays in float64, with an explicit near-zero floor."""
    left = np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in a])
    right = np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in b])
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("parity arrays must have equal size and be finite")
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(left), 1e-20))


def compare_update(cpu: list[np.ndarray], gpu: list[np.ndarray], initial: list[np.ndarray]) -> dict:
    """Compare the actual Adam update, not just much larger absolute parameters."""
    left = [a - b for a, b in zip(cpu, initial, strict=True)]
    right = [a - b for a, b in zip(gpu, initial, strict=True)]
    relative = relative_l2(left, right)
    if relative > 0.01:
        raise ValueError(f"CPU/GPU Adam update relative L2 {relative} exceeds0.01")
    return {"update_relative_l2": relative, "update_relative_l2_limit": 0.01}
