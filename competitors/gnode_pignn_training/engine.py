"""Native one-step training kernel; no epochs/full-run entry point.

Builds from scratch. GNODE keeps the training averaged-velocity expression.
Literal PI-GNN keeps its additive softening and explicit position projection.
Corrections are selected only through an independently named immutable recipe.
"""

# ruff: noqa: D101, D102, D107
from __future__ import annotations

import numpy as np
import tensorflow as tf

from competitors.gnode_pignn_tf.models import GNN, ODE, PARAMETERS, Encoder
from competitors.gnode_pignn_training.recipes import Recipe


class TrainingODE(tf.keras.layers.Layer):
    def __init__(self, dt: float, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.dt = dt

    def call(self, state: tf.Tensor, acceleration: tf.Tensor) -> tf.Tensor:
        q, v = state[..., 1:3], state[..., 3:5]
        predicted_velocity = v + acceleration * self.dt
        average_velocity = 0.5 * (v + predicted_velocity)
        return tf.concat([q + average_velocity * self.dt, v + acceleration * self.dt], axis=-1)


def acceleration_target(x: tf.Tensor, additive: float) -> tf.Tensor:
    """Reviewed all-pairs analytical target; additive is epsilon squared or legacy.05."""
    n = tf.shape(x)[1]
    mass, position = x[..., :1], x[..., 1:3]
    delta = tf.tile(tf.expand_dims(position, 1), [1, n, 1, 1]) - tf.tile(
        tf.expand_dims(position, 2), [1, 1, n, 1]
    )
    radius2 = tf.reduce_sum(tf.square(delta), axis=-1, keepdims=True) + additive
    mass_j = tf.tile(tf.expand_dims(mass, 1), [1, n, 1, 1])
    terms = mass_j * delta / (radius2 * tf.sqrt(radius2))
    mask = tf.expand_dims(tf.expand_dims(1.0 - tf.eye(n, dtype=x.dtype), 0), -1)
    return tf.reduce_sum(terms * mask, axis=2)


def projected_state(x: tf.Tensor, acceleration: tf.Tensor, recipe: Recipe) -> tf.Tensor:
    """Explicit literal or consistently semi-implicit corrected PI-GNN projection."""
    velocity = x[..., 3:5] + acceleration * recipe.dt
    if recipe.train_position == "explicit":
        position = x[..., 1:3] + x[..., 3:5] * recipe.dt
    elif recipe.train_position == "semiimplicit":
        position = x[..., 1:3] + velocity * recipe.dt
    else:
        raise ValueError("invalid PI-GNN position recipe")
    return tf.concat([position, velocity], axis=-1)


class StepEngine:
    """Model, Adam and compiled deterministic step; no sampler or training loop."""

    def __init__(self, recipe: Recipe) -> None:
        self.recipe = recipe
        tf.keras.utils.set_random_seed(recipe.seed)
        inputs = tf.keras.Input(shape=(None, 5), dtype="float32", name="state_mass_first")
        nodes = Encoder(name="encoder")(inputs) if recipe.kind == "gnode" else inputs
        acceleration = GNN(name="gnn")(nodes)
        outputs = (
            TrainingODE(recipe.dt, name="ode")(inputs, acceleration)
            if recipe.kind == "gnode"
            else acceleration
        )
        self.model = tf.keras.Model(inputs, outputs)
        if self.model.count_params() != PARAMETERS[recipe.kind]:
            raise ValueError("training architecture mismatch")
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=recipe.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=recipe.adam_epsilon,
            amsgrad=False,
        )
        self.optimizer.build(self.model.trainable_variables)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.step = tf.function(
            self._step,
            input_signature=[
                tf.TensorSpec([None, 3, 5], tf.float32),
                tf.TensorSpec([None, 3, 4], tf.float32),
            ],
            jit_compile=False,
        )

    def loss_parts(self, x: tf.Tensor, y: tf.Tensor) -> dict:
        """Expose data/physics terms separately without optimiser side effects."""
        output = self.model(x, training=True)
        if self.recipe.kind == "gnode":
            data = self.mse(y, output)
            physics = tf.constant(0.0, tf.float32)
        else:
            prediction = projected_state(x, output, self.recipe)
            data = tf.reduce_mean(tf.square(prediction - y))
            physics = tf.reduce_mean(
                tf.square(output - acceleration_target(x, self.recipe.force_additive))
            )
        return {
            "data": data,
            "physics": physics,
            "total": data + self.recipe.physics_weight * physics,
        }

    def _step(self, x: tf.Tensor, y: tf.Tensor) -> dict:
        with tf.GradientTape() as tape:
            losses = self.loss_parts(x, y)
        gradients = tape.gradient(losses["total"], self.model.trainable_variables)
        if any(g is None for g in gradients):
            raise ValueError("disconnected training gradient")
        checked = [
            tf.debugging.check_numerics(g, "nonfinite gradient; step rejected") for g in gradients
        ]
        total = tf.debugging.check_numerics(losses["total"], "nonfinite loss; step rejected")
        with tf.control_dependencies([total, *checked]):
            self.optimizer.apply_gradients(
                zip(checked, self.model.trainable_variables, strict=True)
            )
        return {**losses, "gradient_norm": tf.linalg.global_norm(checked)}

    def inference_view(self) -> tf.keras.Model:
        """Share exact variables but use the declared quantitative-test equation."""
        if self.recipe.kind == "pignn":
            return self.model
        inputs = tf.keras.Input(shape=(None, 5), dtype="float32")
        nodes = self.model.get_layer("encoder")(inputs)
        acceleration = self.model.get_layer("gnn")(nodes)
        return tf.keras.Model(inputs, ODE(self.recipe.dt, name="ode")(inputs, acceleration))

    def arrays(self) -> list[np.ndarray]:
        """Snapshot model and all Adam variables for exact-continuation tests."""
        return [v.numpy().copy() for v in [*self.model.variables, *self.optimizer.variables]]
