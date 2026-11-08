"""Explicit inference graphs for Daniela's repository models, not training variants.

Requires the separate TensorFlow environment. Never deserialize custom model code.
GNODE uses the quantitative-testing expanded kinematic equation; PI-GNN uses the
quantitative-testing semi-implicit position update. No dataset normalization.
"""

# Keras layer attributes are intentionally identical to the public weight hierarchy.
# ruff: noqa: D101, D102, D107
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

PUBLIC_HASHES = {
    "gnode": "224038e9e3a3440100277d90ad58756c37fa383c6d2c1f8d8dd3691e26019029",
    "pignn": "429d3eb5d1f42f1c5cdc66beb2a1c45120bc4f0e3420abe53156a50f9c1aa931",
}
PARAMETERS = {"gnode": 83714, "pignn": 34690}
DT = 0.001


class Encoder(tf.keras.layers.Layer):
    def __init__(self, dim: int = 128, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(dim, activation="relu")
        self.out = tf.keras.layers.Dense(dim)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-3)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.norm(self.out(self.dense(x)))


class GNN(tf.keras.layers.Layer):
    def __init__(self, dim: int = 128, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.node = tf.keras.Sequential(
            [tf.keras.layers.Dense(dim, activation="swish"), tf.keras.layers.Dense(2)]
        )
        self.edge = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dim, activation="swish"),
                tf.keras.layers.Dense(dim, activation="swish"),
            ]
        )

    def call(self, nodes: tf.Tensor) -> tf.Tensor:
        n = tf.shape(nodes)[1]
        receiver = tf.repeat(tf.expand_dims(nodes, 2), n, axis=2)
        sender = tf.repeat(tf.expand_dims(nodes, 1), n, axis=1)
        messages = self.edge(tf.concat([receiver, sender], axis=-1))
        mask = tf.reshape(tf.eye(n, dtype=nodes.dtype), (1, n, n, 1))
        messages = messages * (tf.cast(1.0, nodes.dtype) - mask)
        return self.node(tf.reduce_sum(messages, axis=2))


class ODE(tf.keras.layers.Layer):
    def __init__(self, dt: float = DT, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.dt = dt

    def call(self, state: tf.Tensor, acceleration: tf.Tensor) -> tf.Tensor:
        q, v = state[..., 1:3], state[..., 3:5]
        return tf.concat(
            [
                q + v * self.dt + 0.5 * acceleration * (self.dt**2),
                v + acceleration * self.dt,
            ],
            axis=-1,
        )


def build_model(kind: str, dt: float = DT) -> tf.keras.Model:
    """Build only reviewed graph definitions; input layout is [m,x,y,vx,vy]."""
    if kind not in PUBLIC_HASHES or not np.isfinite(dt) or dt <= 0:
        raise ValueError("require known model kind and positive finite dt")
    inputs = tf.keras.Input(shape=(None, 5), dtype="float32", name="state_mass_first")
    nodes = Encoder(name="encoder")(inputs) if kind == "gnode" else inputs
    accelerations = GNN(name="gnn")(nodes)
    outputs = ODE(dt, name="ode")(inputs, accelerations) if kind == "gnode" else accelerations
    model = tf.keras.Model(inputs, outputs, name=f"daniela_{kind}_repository_v1")
    if model.count_params() != PARAMETERS[kind]:
        raise ValueError("architecture parameter-count mismatch")
    return model


def weight_bindings(model: tf.keras.Model, kind: str) -> list[tuple[str, object]]:
    """Enumerate every expected model variable and corresponding HDF5 dataset."""
    pairs = []
    if kind == "gnode":
        encoder = model.get_layer("encoder")
        for name in ("dense", "out"):
            layer = getattr(encoder, name)
            pairs += [
                (f"layers/encoder/{name}/vars/0", layer.kernel),
                (f"layers/encoder/{name}/vars/1", layer.bias),
            ]
        pairs += [
            ("layers/encoder/norm/vars/0", encoder.norm.gamma),
            ("layers/encoder/norm/vars/1", encoder.norm.beta),
        ]
    graph = model.get_layer("gnn")
    for name in ("edge", "node"):
        for i, layer in enumerate(getattr(graph, name).layers):
            group = "dense" if i == 0 else f"dense_{i}"
            prefix = f"layers/gnn/{name}/layers/{group}/vars"
            pairs += [(prefix + "/0", layer.kernel), (prefix + "/1", layer.bias)]
    if len(pairs) != len(model.weights) or {id(v) for _, v in pairs} != {
        id(v) for v in model.weights
    }:
        raise ValueError("incomplete variable binding")
    return pairs


def load_public_weights(model: tf.keras.Model, kind: str, checkpoint: Path) -> dict:
    """Strictly load pinned weights, independently checking every model array."""
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != PUBLIC_HASHES[kind]:
        raise ValueError("public checkpoint SHA256 mismatch")
    model.load_weights(checkpoint, skip_mismatch=False)
    bindings = weight_bindings(model, kind)
    with (
        zipfile.ZipFile(checkpoint) as archive,
        h5py.File(io.BytesIO(archive.read("model.weights.h5")), "r") as handle,
    ):
        datasets = set()
        handle.visititems(
            lambda name, obj: (
                datasets.add(name)
                if isinstance(obj, h5py.Dataset) and not name.startswith("optimizer/")
                else None
            )
        )
        if datasets != {name for name, _ in bindings}:
            raise ValueError("unexpected/missing checkpoint model arrays")
        for name, variable in bindings:
            saved = handle[name][...]
            loaded = np.asarray(variable)
            if (
                saved.dtype != np.float32
                or not np.isfinite(saved).all()
                or not np.array_equal(saved, loaded)
            ):
                raise ValueError(f"weight mismatch: {name}")
    return {
        "sha256": digest,
        "parameters": model.count_params(),
        "arrays_checked": len(bindings),
        "optimizer_restored": False,
        "load_mode": "reviewed graph + strict weights only",
    }


def next_state(model: tf.keras.Model, state: tf.Tensor, kind: str, dt: float = DT) -> tf.Tensor:
    """Return mass-first float32 state under the literal quantitative-test update."""
    output = model(state, training=False)
    if kind == "gnode":
        if model.get_layer("ode").dt != dt:
            raise ValueError("GNODE graph and requested timestep disagree")
        qv = output
    elif kind == "pignn":
        velocity = state[..., 3:5] + output * dt
        position = state[..., 1:3] + velocity * dt
        qv = tf.concat([position, velocity], axis=-1)
    else:
        raise ValueError("unknown model kind")
    return tf.concat([state[..., :1], qv], axis=-1)


def rollout(model: tf.keras.Model, initial: tf.Tensor, kind: str, steps: int) -> tf.Tensor:
    """Eager literal autoregression; output (batch,future,body,mass-first features)."""
    if steps < 1:
        raise ValueError("positive rollout steps required")
    state = tf.convert_to_tensor(initial, dtype=tf.float32)
    snapshots = []
    for _ in range(steps):
        state = next_state(model, state, kind)
        snapshots.append(state)
    return tf.stack(snapshots, axis=1)


def compiled_rollout(model: tf.keras.Model, kind: str) -> object:
    """Build non-XLA graph loop; explicit TRAIN-only parity required before use."""

    @tf.function(
        input_signature=[tf.TensorSpec([None, 3, 5], tf.float32), tf.TensorSpec([], tf.int32)],
        jit_compile=False,
    )
    def execute(initial: tf.Tensor, steps: tf.Tensor) -> tf.Tensor:
        results = tf.TensorArray(tf.float32, size=steps, element_shape=[None, 3, 5])

        def body(i: tf.Tensor, state: tf.Tensor, array: tf.TensorArray) -> tuple:
            state = next_state(model, state, kind)
            return i + 1, state, array.write(i, state)

        _, _, results = tf.while_loop(
            lambda i, _s, _a: i < steps,
            body,
            (tf.constant(0), initial, results),
            parallel_iterations=1,
        )
        return tf.transpose(results.stack(), [1, 0, 2, 3])

    return execute
