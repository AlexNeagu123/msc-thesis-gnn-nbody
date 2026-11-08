"""Framework-independent competitor export checks in the unchanged research venv."""

import numpy as np
import pytest

from competitors.gnode_pignn_tf.contracts import to_mass_first, to_shared_layout
from evaluation.rk4_metrics import build_daniela_report


def test_layout_roundtrip_future_only_and_common_scorer() -> None:
    """Check ordering, dtype, alignment and exact constant float32 masses."""
    initial = np.random.default_rng(42).normal(size=(2, 3, 5))
    initial[..., 4] = [[0.51, 1.1, 1.9], [0.8, 1.7, 0.6]]
    network = to_mass_first(initial)
    np.testing.assert_array_equal(network[..., 0], initial[..., 4].astype(np.float32))
    predictions = np.repeat(network[:, None], 3, axis=1)
    predictions[..., 1] += np.arange(1, 4)[None, :, None]
    shared = to_shared_layout(predictions, initial)
    assert shared.dtype == np.float32
    np.testing.assert_array_equal(shared[..., 0], predictions[..., 1])
    reference = np.concatenate([initial[:, None], shared.astype(np.float64)], axis=1)
    reference[..., 4] = initial[:, None, :, 4]  # reference masses remain original float64
    report = build_daniela_report(reference, shared)
    for window in report["trajectory"].values():
        if window["n_steps_scored"] > 0:
            assert window["mae"]["mean"] == 0


def test_nonfinite_state_retained_but_mass_change_rejected() -> None:
    """Keep trajectory failures instead of filtering or replacing them."""
    initial = np.ones((1, 3, 5))
    predictions = to_mass_first(initial)[:, None].copy()
    predictions[0, 0, 0, 1] = np.nan
    shared = to_shared_layout(predictions, initial)
    assert np.isnan(shared[0, 0, 0, 0])
    predictions[0, 0, 0, 0] = 2
    with pytest.raises(ValueError, match="mass changed"):
        to_shared_layout(predictions, initial)


@pytest.mark.parametrize("bad", [np.zeros((3, 4)), np.zeros((2, 2, 5)), np.zeros((2, 3, 5))])
def test_invalid_initial_state_rejected(bad: np.ndarray) -> None:
    """Reject wrong layout and nonpositive masses."""
    with pytest.raises(ValueError, match=r"expected|initial states"):
        to_mass_first(bad)
