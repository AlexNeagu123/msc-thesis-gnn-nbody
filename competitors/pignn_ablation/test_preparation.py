"""Tests of single-change identities, pairing, equations and temporal sensitivity."""

# Configure deterministic TF before its first import; compact test signatures.
# ruff: noqa: E402, ANN001, ANN201, D103, PT006
import importlib.util

import pytest

if importlib.util.find_spec("tensorflow") is None:
    pytest.skip("tensorflow not installed", allow_module_level=True)

from competitors.gnode_pignn_training.production import runtime

runtime()

import numpy as np
import tensorflow as tf

from competitors.gnode_pignn_training.checkpoint import restore_checkpoint, save_checkpoint
from competitors.gnode_pignn_training.data import PairSampler
from competitors.gnode_pignn_training.engine import StepEngine, acceleration_target, projected_state
from competitors.gnode_pignn_training.recipes import RECIPES
from competitors.pignn_ablation.recipes import CELLS, scientific_delta


@pytest.mark.parametrize(
    "cell,fields",
    [
        ("A00", set()),
        ("A10", {"force_additive"}),
        ("A01", {"train_position"}),
        ("A11", {"force_additive", "train_position"}),
    ],
)
def test_scientific_delta(cell, fields):
    assert set(scientific_delta(CELLS["A00"], CELLS[cell])) == fields
    assert CELLS[cell].record()["long_run_authorized"] is False


def test_original_registry_untouched():
    assert len(RECIPES) == 3
    assert CELLS["A10"].model_id not in RECIPES
    assert CELLS["A01"].model_id not in RECIPES


def test_exact_initial_arrays_and_streams():
    baseline = StepEngine(CELLS["A00"]).arrays()
    for recipe in CELLS.values():
        engine = StepEngine(recipe)
        for a, b in zip(baseline, engine.arrays(), strict=True):
            np.testing.assert_array_equal(a, b)
    streams = [PairSampler(639680, 512, False, r.seed) for r in CELLS.values()]
    for _ in range(1251):  # full epoch including remainder and next epoch boundary
        batches = [s.next_indices() for s in streams]
        for b in batches[1:]:
            np.testing.assert_array_equal(batches[0], b)


@pytest.mark.parametrize("cell", ["A10", "A01"])
def test_new_recipe_exact_resume(tmp_path, cell):
    recipe = CELLS[cell]
    x = np.random.default_rng(7).normal(size=(7, 3, 5)).astype(np.float32)
    x[..., 0] = 1.0
    y = x[..., 1:].copy()
    straight = StepEngine(recipe)
    straight.step(x, y)
    sampler = PairSampler(7, 7, False, 42)
    sampler.next_indices()
    save_checkpoint(
        tmp_path / "saved", straight, sampler, {"scope": "disposable-test"}, {"updates": 1}
    )
    straight.step(x, y)
    expected = straight.arrays()
    resumed = StepEngine(recipe)
    stream = PairSampler(7, 7, False, 42)
    progress = restore_checkpoint(tmp_path / "saved", resumed, stream, {"scope": "disposable-test"})
    assert progress["updates"] == 1
    np.testing.assert_array_equal(stream.next_indices(), sampler.next_indices())
    resumed.step(x, y)
    for a, b in zip(expected, resumed.arrays(), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("cell", list(CELLS))
def test_projection_and_position_gradient(cell):
    recipe = CELLS[cell]
    x = tf.constant(
        [[[1.0, 0.0, 0.0, 1.0, 2.0], [2.0, 0.1, 0.2, -1.0, 0.5], [3.0, -0.2, 0.3, 0.3, -0.4]]]
    )
    a = tf.Variable(tf.ones((1, 3, 2)))
    with tf.GradientTape() as tape:
        state = projected_state(x, a, recipe)
        position_sum = tf.reduce_sum(state[..., :2])
    g = tape.gradient(position_sum, a, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    expected = 0.0 if recipe.train_position == "explicit" else recipe.dt**2
    np.testing.assert_allclose(g.numpy(), expected, rtol=2e-7, atol=0.0)
    v = x[..., 3:5].numpy() + a.numpy() * recipe.dt
    q = x[..., 1:3].numpy() + (x[..., 3:5].numpy() if expected == 0 else v) * recipe.dt
    np.testing.assert_array_equal(state.numpy(), np.concatenate([q, v], axis=-1))
    # Independent NumPy acceleration formula, including mass indexing and sign.
    xn = x.numpy().astype(np.float64)
    delta = xn[:, None, :, 1:3] - xn[:, :, None, 1:3]
    terms = (
        xn[:, None, :, :1]
        * delta
        / (np.sum(delta**2, axis=-1, keepdims=True) + recipe.force_additive) ** 1.5
    )
    terms *= (1 - np.eye(3))[None, :, :, None]
    np.testing.assert_allclose(
        acceleration_target(x, recipe.force_additive).numpy(),
        terms.sum(axis=2),
        rtol=2e-6,
        atol=2e-6,
    )
