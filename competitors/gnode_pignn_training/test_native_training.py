"""Native training gates; disposable TRAIN-only updates, no public-weight changes."""

# ruff: noqa: ANN001, ANN201, D103
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")
# In a combined native suite the inference test module configures TF first.
import tensorflow as tf

from competitors.gnode_pignn_tf.runtime import configure

if not tf.config.threading.get_intra_op_parallelism_threads():
    configure()

from competitors.gnode_pignn_training.checkpoint import restore_checkpoint, save_checkpoint
from competitors.gnode_pignn_training.data import PairSampler, load_train_pairs
from competitors.gnode_pignn_training.engine import StepEngine, acceleration_target, projected_state
from competitors.gnode_pignn_training.recipes import RECIPES

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "runs/daniela-model-audit-v1"
pytestmark = pytest.mark.skipif(not AUDIT.exists(), reason="local source/data oracle required")


def definitions(cell, names, namespace):
    notebook = ROOT.parents[1] / "daniela/nbody-net/notebooks/2_training.ipynb"
    manifest = json.loads((AUDIT / "source-evidence/manifest.json").read_text())
    assert (
        hashlib.sha256(notebook.read_bytes()).hexdigest()
        == manifest["inputs"]["nbody-net/notebooks/2_training.ipynb"]
    )
    tree = ast.parse("".join(json.loads(notebook.read_text())["cells"][cell]["source"]))
    for name in names:
        candidates = [
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name
        ]
        assert len(candidates) == 1
        exec(
            compile(
                ast.Module(body=candidates, type_ignores=[]),
                f"original_training:cell{cell}",
                "exec",
            ),
            namespace,
        )


def original_training_model(kind, x, y):
    namespace = {"tf": tf, "np": np, "dt": 0.001}
    definitions(
        11 if kind == "gnode" else 12,
        ["Encoder", "GNN", "ODE"] if kind == "gnode" else ["GNN", "compute_acceleration"],
        namespace,
    )
    definitions(18, ["define_model"], namespace)
    model = namespace["define_model"](4 if kind == "gnode" else 5, x, y, 10)
    namespace["model"] = model
    if kind == "pignn":
        namespace["optimizer"] = tf.keras.optimizers.Adam(learning_rate=5e-4)
        definitions(12, ["train_step"], namespace)
    return model, namespace


@pytest.fixture(scope="module")
def pairs():
    return load_train_pairs(ROOT / "data/output/daniela-rk4/train.h5")


@pytest.mark.parametrize("key", ["gnode_repository_shared_v1", "pignn_repository_shared_v1"])
@pytest.mark.parametrize("size", [7, 192, 512])
def test_native_original_one_update_parity(key, size, pairs):
    x, y, _ = pairs
    engine = StepEngine(RECIPES[key])
    oracle, namespace = original_training_model(engine.recipe.kind, x[:size], y[:size])
    oracle.set_weights(engine.model.get_weights())
    if engine.recipe.kind == "gnode":
        old_loss = float(oracle.train_on_batch(x[:size], y[:size])[0])
        optimizer = oracle.optimizer
    else:
        old_loss = float(namespace["train_step"](x[:size], y[:size])[0])
        optimizer = namespace["optimizer"]
    losses = engine.step(x[:size], y[:size])
    np.testing.assert_allclose(float(losses["total"]), old_loss, rtol=2e-6, atol=1e-10)
    for ours, old in zip(engine.model.weights, oracle.weights, strict=True):
        np.testing.assert_allclose(ours.numpy(), old.numpy(), rtol=2e-6, atol=2e-8)
    for ours, old in zip(engine.optimizer.variables, optimizer.variables, strict=True):
        np.testing.assert_allclose(ours.numpy(), old.numpy(), rtol=2e-6, atol=1e-10)
    assert int(engine.optimizer.iterations) == 1


@pytest.mark.parametrize("key", list(RECIPES))
def test_exact_resume_across_partial_batch_and_epoch_boundary(key, pairs, tmp_path):
    x, y, _ = pairs
    recipe = RECIPES[key]
    engine = StepEngine(recipe)
    sampler = PairSampler(13, 5, recipe.kind == "gnode")
    for _ in range(2):
        idx = sampler.next_indices()
        engine.step(x[idx], y[idx])
    saved = tmp_path / "checkpoint"
    identity = {"protocol": "unit-test-cpu-float32", "source": "same", "train": "same"}
    with pytest.raises(ValueError, match="counter"):
        save_checkpoint(tmp_path / "bad-progress", engine, sampler, identity, {"updates": 99})
    save_checkpoint(saved, engine, sampler, identity, {"updates": 2, "best_validation": None})
    expected_indices = []
    for _ in range(3):
        idx = sampler.next_indices()
        expected_indices.append(idx)
        engine.step(x[idx], y[idx])
    expected = engine.arrays()
    restored = StepEngine(recipe)
    sampler2 = PairSampler(13, 5, recipe.kind == "gnode", seed=99)
    progress = restore_checkpoint(saved, restored, sampler2, identity)
    assert progress["updates"] == 2
    for idx in expected_indices:
        actual = sampler2.next_indices()
        np.testing.assert_array_equal(actual, idx)
        restored.step(x[actual], y[actual])
    for a, b in zip(expected, restored.arrays(), strict=True):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="identity"):
        restore_checkpoint(saved, restored, sampler2, {"source": "changed"})
    with pytest.raises(FileExistsError):
        save_checkpoint(saved, restored, sampler2, identity, {})
    metadata = saved / "metadata.json"
    metadata.write_text(metadata.read_text() + " ")
    with pytest.raises(ValueError, match="hash"):
        restore_checkpoint(saved, restored, sampler2, identity)


@pytest.mark.parametrize("key", list(RECIPES))
def test_nonfinite_rejection_leaves_model_and_adam_unchanged(key, pairs):
    x, y, _ = pairs
    engine = StepEngine(RECIPES[key])
    before = engine.arrays()
    bad = y[:7].copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(tf.errors.InvalidArgumentError, match="nonfinite"):
        engine.step(x[:7], bad)
    for a, b in zip(before, engine.arrays(), strict=True):
        np.testing.assert_array_equal(a, b)


def test_literal_position_gradient_absent_corrected_present():
    x = tf.constant(np.random.default_rng(8).normal(size=(2, 3, 5)), tf.float32)
    a = tf.Variable(tf.ones((2, 3, 2)))
    for key, expect_gradient in [
        ("pignn_repository_shared_v1", False),
        ("pignn_corrected_softening_semiimplicit_v1", True),
    ]:
        with tf.GradientTape() as tape:
            position_loss = tf.reduce_mean(tf.square(projected_state(x, a, RECIPES[key])[..., :2]))
        g = tape.gradient(position_loss, a)
        norm = 0.0 if g is None else float(tf.linalg.global_norm([g]))
        assert (norm > 0) == expect_gradient


def test_literal_force_target_exact_and_corrected_matches_generator(pairs):
    x, _, _ = pairs
    state = x[:7]
    namespace = {"tf": tf, "np": np}
    definitions(12, ["compute_acceleration"], namespace)
    np.testing.assert_array_equal(
        acceleration_target(tf.constant(state), 0.05), namespace["compute_acceleration"](state)
    )
    expected = np.zeros((7, 3, 2))
    for i in range(3):
        for j in range(3):
            if i != j:
                delta = state[:, j, 1:3].astype(np.float64) - state[:, i, 1:3]
                expected[:, i] += (
                    state[:, j, :1]
                    * delta
                    / (np.square(delta).sum(axis=-1, keepdims=True) + 0.05**2) ** 1.5
                )
    np.testing.assert_allclose(
        acceleration_target(tf.constant(state), 0.05**2), expected, rtol=2e-6, atol=2e-6
    )


def test_gnode_training_and_inference_views_share_variables(pairs):
    x, _, _ = pairs
    engine = StepEngine(RECIPES["gnode_repository_shared_v1"])
    inference = engine.inference_view()
    assert {id(v) for v in inference.weights} == {id(v) for v in engine.model.weights}
    np.testing.assert_allclose(engine.model(x[:7]), inference(x[:7]), rtol=2e-6, atol=2e-7)
