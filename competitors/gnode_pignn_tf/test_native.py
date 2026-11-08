"""Native inference gates on TRAIN data only; optional external artifacts required."""

# ruff: noqa: E402, ANN001, ANN002, ANN201, D103
import ast
import hashlib
import json
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")
from competitors.gnode_pignn_tf.runtime import configure

RUNTIME = configure()
import h5py
import tensorflow as tf

from competitors.gnode_pignn_tf.contracts import to_mass_first, to_shared_layout
from competitors.gnode_pignn_tf.models import (
    ODE,
    build_model,
    compiled_rollout,
    load_public_weights,
    next_state,
    rollout,
)

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "runs/daniela-model-audit-v1"
pytestmark = pytest.mark.skipif(
    not AUDIT.exists(), reason="local audited public artifacts required"
)


def original_definition(cell, name, index=0):
    notebook = ROOT.parents[1] / "daniela/nbody-net/notebooks/3_testing.ipynb"
    manifest = json.loads((AUDIT / "source-evidence/manifest.json").read_text())
    assert (
        hashlib.sha256(notebook.read_bytes()).hexdigest()
        == manifest["inputs"]["nbody-net/notebooks/3_testing.ipynb"]
    )
    code = "".join(json.loads(notebook.read_text())["cells"][cell]["source"])
    candidates = [
        n
        for n in ast.walk(ast.parse(code))
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name
    ]
    node = candidates[index]
    namespace = {"tf": tf, "np": np}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), f"original_testing:cell{cell}", "exec"),
        namespace,
    )
    return namespace[name], namespace


def original_model(kind):
    graph, _ = original_definition(17, "GNN", 1 if kind == "gnode" else 2)
    inputs = tf.keras.Input(shape=(None, 5))
    if kind == "gnode":
        encoder, _ = original_definition(17, "Encoder")
        ode, _ = original_definition(17, "ODE")
        nodes = encoder(hidden_dim=128, output_dim=128)(inputs)
        output = ode(0.001)(inputs, graph(128)(nodes))
    else:
        output = graph(128)(inputs)
    model = tf.keras.Model(inputs, output)
    model.load_weights(checkpoint(kind), skip_mismatch=False)
    return model


def checkpoint(kind):
    return (
        AUDIT
        / "checkpoints-public"
        / ("model_GNODE_nbody.keras" if kind == "gnode" else "model_PIGNN_nbody.keras")
    )


@pytest.fixture(scope="module")
def train_state():
    with h5py.File(ROOT / "data/output/daniela-rk4/train.h5") as f:
        assert f.attrs["split"] == "train"
        return np.stack(
            [f["trajectories"][i, t] for i, t in [(0, 0), (1, 500), (2, 1000), (3, 1500)]]
        )


@pytest.fixture(scope="module", params=["gnode", "pignn"])
def model_pair(request):
    kind = request.param
    model = build_model(kind)
    evidence = load_public_weights(model, kind, checkpoint(kind))
    return kind, model, original_model(kind), evidence


def test_strict_weight_inventory_and_native_original_forward(model_pair, train_state):
    kind, model, oracle, evidence = model_pair
    x = to_mass_first(train_state)
    assert evidence["parameters"] == (83714 if kind == "gnode" else 34690)
    assert evidence["arrays_checked"] == (14 if kind == "gnode" else 8)
    assert model(x).dtype == tf.float32
    np.testing.assert_array_equal(model(x), oracle(x))
    np.testing.assert_allclose(model(x), oracle.predict(x, verbose=0), rtol=1e-5, atol=1e-6)


def test_batching_body_order_and_finite_gradient(model_pair, train_state):
    _, model, _, _ = model_pair
    x = to_mass_first(train_state)
    batched = model(x).numpy()
    single = np.concatenate([model(x[i : i + 1]).numpy() for i in range(len(x))])
    np.testing.assert_allclose(batched, single, rtol=1e-5, atol=1e-6)
    order = [2, 0, 1]
    np.testing.assert_allclose(model(x[:, order]), batched[:, order], rtol=1e-5, atol=1e-6)
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(model(x)))
    gradients = tape.gradient(loss, model.trainable_variables)
    assert all(g is not None and np.isfinite(g.numpy()).all() for g in gradients)
    assert float(tf.linalg.global_norm(gradients)) > 0


def test_train_short_rollout_native_notebook_wrapper(model_pair, train_state):
    kind, model, oracle, _ = model_pair
    x = to_mass_first(train_state[:1])
    steps = 20
    wrapper, namespace = original_definition(
        21 if kind == "gnode" else 22, "run_gnn_gnode" if kind == "gnode" else "run_pignn"
    )
    observed = []

    def record(bodies, *_) -> None:
        observed.append([[b.mass, *b.position, *b.velocity] for b in bodies])

    namespace.update(
        {
            "MODELS": {4: "GNODE", 5: "PI-GNN"},
            "N": 3,
            "steps": steps,
            "dt": 0.001,
            "RK4Body": lambda **kw: types.SimpleNamespace(**kw),
            "calculate_validation_step": record,
        }
    )
    wrapper(x, oracle, *[[] for _ in range(14)])
    expected = np.asarray(observed, dtype=np.float32)[None]
    actual = rollout(model, x, kind, steps).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert np.isfinite(actual).all()
    shared = to_shared_layout(actual, train_state[:1])
    np.testing.assert_array_equal(
        shared[..., 4],
        np.broadcast_to(train_state[:1, None, :, 4].astype(np.float32), shared[..., 4].shape),
    )


def test_graph_loop_train_parity_and_save_reload(model_pair, train_state, tmp_path):
    kind, model, _, _ = model_pair
    x = to_mass_first(train_state)
    execute = compiled_rollout(model, kind)
    actual = execute(x, tf.constant(32)).numpy()
    expected = rollout(model, x, kind, 32).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    saved = tmp_path / "replay.weights.h5"
    model.save_weights(saved)
    reloaded = build_model(kind)
    reloaded.load_weights(saved)
    np.testing.assert_array_equal(model(x), reloaded(x))
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("dt", [0.001, 0.025])
def test_native_ode_and_pignn_testing_equations(dt):
    rng = np.random.default_rng(123)
    x = rng.normal(size=(2, 3, 5)).astype(np.float32)
    a = rng.normal(size=(2, 3, 2)).astype(np.float32)
    expected = np.concatenate(
        [x[..., 1:3] + x[..., 3:5] * dt + 0.5 * a * dt**2, x[..., 3:5] + a * dt], axis=-1
    )
    np.testing.assert_array_equal(ODE(dt)(x, a), expected)

    def pignn(state, training) -> tf.Tensor:
        return tf.convert_to_tensor(a)

    actual = next_state(pignn, tf.constant(x), "pignn", dt).numpy()
    np.testing.assert_array_equal(actual[..., 1:3], x[..., 1:3] + actual[..., 3:5] * dt)


def test_corrupt_checkpoint_rejected(tmp_path):
    bad = tmp_path / "bad.keras"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="SHA256"):
        load_public_weights(build_model("gnode"), "gnode", bad)
