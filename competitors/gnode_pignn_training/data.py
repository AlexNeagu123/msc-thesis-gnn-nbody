"""Whole-system TRAIN pair preparation, no TensorFlow or fitted normalisation."""

from pathlib import Path

import h5py
import numpy as np


def load_train_pairs(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialise original sorted one-step pair order, never crossing systems."""
    with h5py.File(path) as handle:
        if handle.attrs.get("split") != "train" or float(handle.attrs["dt"]) != 0.001:
            raise ValueError("preflight requires TRAIN split and dt=.001")
        ids = handle["simulation_ids"][:]
        if len(ids) != 320 or len(np.unique(ids)) != 320:
            raise ValueError("expected exact320 whole training systems")
        raw = handle["trajectories"][:][np.argsort(ids)]
    if raw.shape != (320, 2000, 3, 5) or not np.isfinite(raw).all():
        raise ValueError("unexpected training data")
    if not np.array_equal(raw[..., 4], np.broadcast_to(raw[:, :1, :, 4], raw[..., 4].shape)):
        raise ValueError("mass must be constant within systems")
    x = raw[:, :-1, :, [4, 0, 1, 2, 3]].reshape(-1, 3, 5).astype(np.float32)
    y = raw[:, 1:, :, :4].reshape(-1, 3, 4).astype(np.float32)
    return x, y, np.sort(ids)


class PairSampler:
    """Explicit restorable epoch permutation/cursor; remainder is never discarded."""

    def __init__(self, size: int, batch_size: int, shuffle: bool, seed: int = 42) -> None:
        """Start a reproducible epoch stream with an explicit data-order policy."""
        if size < 1 or batch_size < 1:
            raise ValueError("positive sampler sizes required")
        self.size, self.batch_size, self.shuffle = size, batch_size, shuffle
        self.rng = np.random.default_rng(seed)
        self.epoch, self.cursor = 0, 0
        self.order = self.rng.permutation(size) if shuffle else np.arange(size)

    def next_indices(self) -> np.ndarray:
        """Yield the next batch, advancing epoch only after its final remainder."""
        if self.cursor == self.size:
            self.epoch += 1
            self.cursor = 0
            self.order = self.rng.permutation(self.size) if self.shuffle else np.arange(self.size)
        stop = min(self.cursor + self.batch_size, self.size)
        indices = self.order[self.cursor : stop].copy()
        self.cursor = stop
        return indices

    def state(self) -> dict:
        """Capture full data execution state, not just an RNG seed."""
        return {
            "size": self.size,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "order": self.order.tolist(),
            "rng": self.rng.bit_generator.state,
        }

    def restore(self, state: dict) -> None:
        """Reject mismatched data execution identities or invalid permutations."""
        for name in ("size", "batch_size", "shuffle"):
            if state[name] != getattr(self, name):
                raise ValueError("sampler identity mismatch")
        order = np.asarray(state["order"], dtype=np.int64)
        if not np.array_equal(np.sort(order), np.arange(self.size)):
            raise ValueError("invalid sampler permutation")
        if not 0 <= state["cursor"] <= self.size or state["epoch"] < 0:
            raise ValueError("invalid sampler progress")
        self.order = order
        self.epoch, self.cursor = state["epoch"], state["cursor"]
        self.rng.bit_generator.state = state["rng"]
