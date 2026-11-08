"""Matched-tail curriculum control: entire assigned-unit bank preserved exactly."""
# ruff: noqa: ANN001, ANN201, ANN204, D102, D103, D107

import hashlib

import numpy as np


def matched_tail_order(bank, seed=20261030):
    horizons = np.asarray(bank["horizon"])
    if horizons.shape != (2500,) or not np.array_equal(
        horizons, np.repeat([1, 5, 20, 50, 100], 500)
    ):
        raise ValueError("expected exact historical staged bank")
    order = np.concatenate(
        [np.random.default_rng(seed).permutation(2000), np.arange(2000, 2500)]
    ).astype(np.int64)
    assert np.array_equal(np.sort(order), np.arange(2500))
    assert np.array_equal(order[-500:], np.arange(2000, 2500))
    return order


def identity(order):
    return {
        "protocol": "hgnn-matched-tail-v1",
        "order_sha256": hashlib.sha256(np.asarray(order, dtype="<i8").tobytes()).hexdigest(),
        "prefix_permutation_seed": 20261030,
        "fixed_final_units": [2000, 2499],
        "fixed_final_horizon": 100,
        "updates": 2500,
    }


class UnitCursor:
    """Strict serializable cursor for an explicit order; never randomize on restore."""

    def __init__(self, order):
        self.order = np.asarray(order, dtype=np.int64).copy()
        self.position = 0
        if not np.array_equal(np.sort(self.order), np.arange(2500)):
            raise ValueError("invalid order")
        self.order_hash = hashlib.sha256(self.order.astype("<i8").tobytes()).hexdigest()

    def next_unit(self):
        if self.position >= len(self.order):
            raise StopIteration
        unit = int(self.order[self.position])
        self.position += 1
        return unit

    def state(self):
        return {"position": self.position, "order_sha256": self.order_hash}

    def restore(self, state):
        if (
            state["order_sha256"] != self.order_hash
            or not isinstance(state["position"], int)
            or not 0 <= state["position"] <= 2500
        ):
            raise ValueError("wrong schedule identity/cursor")
        self.position = state["position"]
