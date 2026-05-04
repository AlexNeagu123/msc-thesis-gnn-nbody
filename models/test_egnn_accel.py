"""Tests for models/egnn_accel.py."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from models.egnn_accel import (
    AccelerationHead,
    EGNNAccel,
    EGNNAccelLayer,
    _clip_vector_norm,
)
from training._types import (
    CheckpointConfig,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    SchedulerConfig,
    TrainConfig,
    TrainingParams,
    TrainResult,
)
from training.train import build_model, train


def _make_state(batch: int = 4, n_particles: int = 3, mass_value: float = 1.0) -> torch.Tensor:
    """Create a deterministic state tensor [x, y, vx, vy, mass]."""
    torch.manual_seed(0)
    pos_vel = torch.randn(batch, n_particles, 4)
    mass = torch.full((batch, n_particles, 1), mass_value)
    return torch.cat([pos_vel, mass], dim=-1)


def test_output_shape() -> None:
    """Forward pass returns (batch, n_particles, 5)."""
    model = EGNNAccel()
    state = _make_state(batch=4)
    out = model(state)

    assert out.shape == (4, 3, 5)


def test_single_sample() -> None:
    """Forward pass works with batch size 1."""
    model = EGNNAccel()
    state = _make_state(batch=1)
    out = model(state)

    assert out.shape == (1, 3, 5)


def test_parameter_count() -> None:
    """At hidden=128, n_layers=6 the model should sit between 600K and 750K params."""
    model = EGNNAccel(hidden_dim=128, n_layers=6)
    n_params = sum(p.numel() for p in model.parameters())

    assert 600_000 < n_params < 750_000


def test_deterministic() -> None:
    """Same input produces identical output in eval mode."""
    model = EGNNAccel()
    model.eval()
    state = _make_state()

    out1 = model(state)
    out2 = model(state)

    assert torch.allclose(out1, out2)


def test_mass_passthrough() -> None:
    """Mass column is passed through exactly."""
    model = EGNNAccel()
    state = _make_state()
    out = model(state)

    assert torch.allclose(out[..., 4:], state[..., 4:])


def test_constant_velocity_at_init() -> None:
    """Tiny-init head means a fresh model integrates as the constant-velocity baseline.

    With a_norm ≈ 0:
        v_next ≈ v
        x_next ≈ x + dt * v
    """
    dt = 0.05
    model = EGNNAccel(dt=dt)
    model.eval()
    state = _make_state()
    out = model(state)

    pos_in = state[..., :2]
    vel_in = state[..., 2:4]

    # velocity should be very close to input velocity
    assert torch.allclose(out[..., 2:4], vel_in, atol=1e-3)
    # position should be very close to x + dt * v
    expected_pos = pos_in + dt * vel_in
    assert torch.allclose(out[..., :2], expected_pos, atol=1e-4)


def test_translation_equivariance_of_position() -> None:
    """Shifting input positions by d shifts output positions by the same d.

    Velocities and accelerations should be unchanged under translation.
    """
    model = EGNNAccel()
    model.eval()
    state = _make_state()

    d = torch.tensor([3.0, -2.0])
    shifted = state.clone()
    shifted[..., :2] += d

    out_orig = model(state)
    out_shifted = model(shifted)

    pos_diff = out_shifted[..., :2] - out_orig[..., :2]
    assert torch.allclose(pos_diff, d.expand_as(pos_diff), atol=1e-4)
    assert torch.allclose(out_shifted[..., 2:4], out_orig[..., 2:4], atol=1e-4)


def test_rotation_equivariance() -> None:
    """Rotating input pos and vel rotates the output by the same matrix."""
    model = EGNNAccel()
    model.eval()
    state = _make_state()

    theta = torch.tensor(torch.pi / 2)
    rot = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)],
        ]
    )

    rotated = state.clone()
    rotated[..., :2] = state[..., :2] @ rot.T
    rotated[..., 2:4] = state[..., 2:4] @ rot.T

    out_orig = model(state)
    out_rotated = model(rotated)

    expected_pos = out_orig[..., :2] @ rot.T
    expected_vel = out_orig[..., 2:4] @ rot.T

    assert torch.allclose(out_rotated[..., :2], expected_pos, atol=1e-4)
    assert torch.allclose(out_rotated[..., 2:4], expected_vel, atol=1e-4)


def test_permutation_equivariance() -> None:
    """Permuting input particles permutes the output equivalently."""
    model = EGNNAccel()
    model.eval()
    state = _make_state()

    perm = [2, 1, 0]
    permuted = state[:, perm, :]

    out_orig = model(state)
    out_permuted = model(permuted)

    assert torch.allclose(out_permuted, out_orig[:, perm, :], atol=1e-4)


def test_velocity_affects_position_via_dt_drift() -> None:
    """Shifting velocity by delta shifts output position by exactly dt * delta.

    This holds at any training state, not just at init: the acceleration head
    depends only on positions and mass, so changing only `v` leaves `a`
    unchanged. Both `v_next = v + dt * a` and `x_next = x + dt * v_next` then
    pick up the velocity shift linearly.
    """
    dt = 0.05
    model = EGNNAccel(dt=dt)
    model.eval()

    state1 = _make_state()
    state2 = state1.clone()
    delta_v = torch.tensor([0.7, -0.3])
    state2[..., 2:4] += delta_v

    out1 = model(state1)
    out2 = model(state2)

    pos_diff = out2[..., :2] - out1[..., :2]
    expected_pos_diff = (dt * delta_v).expand_as(pos_diff)
    assert torch.allclose(pos_diff, expected_pos_diff, atol=1e-4)


def test_gradient_flow() -> None:
    """Backward pass leaves gradients on every trained parameter."""
    model = EGNNAccel()
    state = _make_state()

    out = model(state)
    loss = out.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert param.grad.abs().sum() > 0, f"zero gradient for {name}"


def test_accel_head_tiny_init() -> None:
    """The acceleration head's final scalar layer is tiny-initialised."""
    model = EGNNAccel(hidden_dim=64, n_layers=2)
    final = model.accel_head.mlp_w[-1]

    assert final.bias is None
    assert final.weight.abs().max().item() < 0.01


def test_layer_only_updates_h() -> None:
    """EGNNAccelLayer takes h, x and returns only an updated h."""
    layer = EGNNAccelLayer(hidden_dim=64)
    h = torch.randn(4, 3, 64)
    x = torch.randn(4, 3, 2)

    h_new = layer(h, x)
    assert h_new.shape == (4, 3, 64)


def test_clip_vector_norm_passes_small_vectors_through() -> None:
    """Vectors with norm <= max_norm are unchanged."""
    v = torch.tensor([[3.0, 4.0], [0.5, 0.5]])  # norms 5.0 and ~0.71
    out = _clip_vector_norm(v, max_norm=100.0)
    assert torch.allclose(out, v)


def test_clip_vector_norm_scales_large_vectors_to_max() -> None:
    """Vectors above max_norm are scaled to exactly max_norm, direction preserved."""
    v = torch.tensor([[300.0, 400.0]])  # norm = 500
    out = _clip_vector_norm(v, max_norm=100.0)

    expected_norm = torch.tensor(100.0)
    assert torch.allclose(out.norm(dim=-1), expected_norm, atol=1e-5)
    # direction preserved (collinear with input)
    assert torch.allclose(out / out.norm(), v / v.norm(), atol=1e-5)


def test_clip_vector_norm_is_rotation_equivariant() -> None:
    """Clipping commutes with rotation, even in the activated (large-norm) regime."""
    torch.manual_seed(0)
    v = torch.randn(4, 3, 3, 2) * 200.0  # large enough to trigger clipping

    theta = torch.tensor(torch.pi / 3)
    rot = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)],
        ]
    )

    # clip then rotate
    clip_first = _clip_vector_norm(v, max_norm=50.0) @ rot.T
    # rotate then clip
    rotate_first = _clip_vector_norm(v @ rot.T, max_norm=50.0)

    assert torch.allclose(clip_first, rotate_first, atol=1e-5)


def test_accel_head_shape() -> None:
    """AccelerationHead returns one 2-D acceleration vector per particle."""
    head = AccelerationHead(hidden_dim=64)
    h = torch.randn(4, 3, 64)
    x = torch.randn(4, 3, 2)

    a = head(h, x)
    assert a.shape == (4, 3, 2)


def test_build_model_constructs_egnn_accel() -> None:
    """The trainer factory builds EGNNAccel when cfg.model.name == 'egnn_accel'."""
    cfg = TrainConfig(
        model=ModelConfig(name="egnn_accel", hidden_dim=32, n_layers=2),
        data=DataConfig(train_path="x", val_path="y", dt=0.05),
        training=TrainingParams(epochs=1, batch_size=8, lr=1e-3, weight_decay=0.0),
        scheduler=SchedulerConfig(enabled=False),
        checkpointing=CheckpointConfig(enabled=False),
        logging=LoggingConfig(enabled=False),
    )
    model = build_model(cfg, pos_std=1.0, vel_std=1.0)

    assert isinstance(model, EGNNAccel)
    # dt buffer should match cfg.data.dt
    assert float(model.dt) == pytest.approx(0.05)


def test_build_model_rejects_unknown_name() -> None:
    """The factory still raises on an unknown model name."""
    cfg = TrainConfig(
        model=ModelConfig(name="not_a_model", hidden_dim=32, n_layers=2),
        data=DataConfig(train_path="x", val_path="y", dt=0.05),
        training=TrainingParams(epochs=1, batch_size=8, lr=1e-3, weight_decay=0.0),
        scheduler=SchedulerConfig(enabled=False),
        checkpointing=CheckpointConfig(enabled=False),
        logging=LoggingConfig(enabled=False),
    )
    with pytest.raises(ValueError, match="Unknown model"):
        build_model(cfg, pos_std=1.0, vel_std=1.0)


def test_egnn_accel_trains_end_to_end(tmp_path: Path) -> None:
    """Small synthetic run completes through the Trainer."""
    rng = np.random.default_rng(42)
    for name in ("train.h5", "val.h5"):
        trajectories = rng.normal(size=(5, 10, 3, 5))
        trajectories[:, :, :, 4] = 1.0
        energies = rng.normal(size=(5, 10))
        with h5py.File(tmp_path / name, "w") as f:
            f.create_dataset("trajectories", data=trajectories)
            f.create_dataset("energies", data=energies)

    cfg = TrainConfig(
        model=ModelConfig(name="egnn_accel", hidden_dim=16, n_layers=2),
        data=DataConfig(
            train_path=str(tmp_path / "train.h5"),
            val_path=str(tmp_path / "val.h5"),
            dt=0.05,
        ),
        training=TrainingParams(
            epochs=3,
            batch_size=8,
            lr=1e-3,
            weight_decay=0.0,
            loss="mse",
            seed=42,
            device="cpu",
        ),
        scheduler=SchedulerConfig(enabled=False),
        checkpointing=CheckpointConfig(enabled=False),
        logging=LoggingConfig(enabled=False),
    )

    result = train(cfg)

    assert isinstance(result, TrainResult)
    assert result.final_train_loss < float("inf")
    assert result.train_history[0] > result.train_history[-1]
