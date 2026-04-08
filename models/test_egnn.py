"""Tests for models/egnn.py."""

from pathlib import Path

import h5py
import numpy as np
import torch

from models.egnn import EGNN, EGCLLayer
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
from training.train import train


def _make_state(batch: int = 4, n_particles: int = 3) -> torch.Tensor:
    """Create a random state tensor [x, y, vx, vy, mass].

    Args:
        batch: batch size.
        n_particles: number of particles.

    Returns:
        State tensor of shape (batch, n_particles, 5).
    """
    torch.manual_seed(0)
    state = torch.randn(batch, n_particles, 4)
    mass = torch.ones(batch, n_particles, 1)
    return torch.cat([state, mass], dim=-1)


def test_output_shape() -> None:
    """Forward pass returns (batch, n_particles, 5)."""
    model = EGNN()
    state = _make_state(batch=4)
    out = model(state)

    assert out.shape == (4, 3, 5)


def test_single_sample() -> None:
    """Forward pass works with batch size 1."""
    model = EGNN()
    state = _make_state(batch=1)
    out = model(state)

    assert out.shape == (1, 3, 5)


def test_parameter_count() -> None:
    """Total parameters should be approximately 121K."""
    model = EGNN(hidden_dim=64, n_layers=4)
    n_params = sum(p.numel() for p in model.parameters())

    assert 100_000 < n_params < 130_000


def test_deterministic() -> None:
    """Same input produces identical output in eval mode."""
    model = EGNN()
    model.eval()
    state = _make_state()

    out1 = model(state)
    out2 = model(state)

    assert torch.allclose(out1, out2)


def test_translation_equivariance() -> None:
    """Shifting input positions by d shifts output positions by the same d."""
    model = EGNN()
    model.eval()
    state = _make_state()

    # translate positions by a random vector
    d = torch.tensor([3.0, -2.0])
    shifted = state.clone()
    shifted[..., :2] += d

    out_orig = model(state)
    out_shifted = model(shifted)

    # positions should shift by d
    pos_diff = out_shifted[..., :2] - out_orig[..., :2]
    assert torch.allclose(pos_diff, d.expand_as(pos_diff), atol=1e-4)

    # velocities should be unchanged
    assert torch.allclose(out_shifted[..., 2:4], out_orig[..., 2:4], atol=1e-4)


def test_rotation_equivariance() -> None:
    """Rotating input positions/velocities rotates the output equivalently."""
    model = EGNN()
    model.eval()
    state = _make_state()

    # 90-degree rotation matrix
    theta = torch.tensor(torch.pi / 2)
    rot = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)],
        ]
    )

    rotated = state.clone()
    rotated[..., :2] = state[..., :2] @ rot.T  # rotate positions
    rotated[..., 2:4] = state[..., 2:4] @ rot.T  # rotate velocities

    out_orig = model(state)
    out_rotated = model(rotated)

    # output positions and velocities should be rotated equivalently
    expected_pos = out_orig[..., :2] @ rot.T
    expected_vel = out_orig[..., 2:4] @ rot.T

    assert torch.allclose(out_rotated[..., :2], expected_pos, atol=1e-4)
    assert torch.allclose(out_rotated[..., 2:4], expected_vel, atol=1e-4)


def test_permutation_equivariance() -> None:
    """Permuting input particles permutes the output equivalently."""
    model = EGNN()
    model.eval()
    state = _make_state()

    # swap particle 0 and particle 2
    perm = [2, 1, 0]
    permuted = state[:, perm, :]

    out_orig = model(state)
    out_permuted = model(permuted)

    # output should be permuted the same way
    assert torch.allclose(out_permuted, out_orig[:, perm, :], atol=1e-4)


def test_mass_passthrough() -> None:
    """Mass column is passed through unchanged."""
    model = EGNN()
    state = _make_state()
    out = model(state)

    assert torch.allclose(state[..., 4:], out[..., 4:])


def test_gradient_flow() -> None:
    """All used parameters receive gradients after backward pass."""
    model = EGNN()
    state = _make_state()

    out = model(state)
    loss = out.sum()
    loss.backward()

    # last layer's mlp_h is unused (h is discarded after final layer)
    last = len(model.layers) - 1
    for name, param in model.named_parameters():
        if name.startswith(f"layers.{last}.mlp_h"):
            assert param.grad is None, f"unexpected gradient for {name}"
            continue
        assert param.grad is not None, f"no gradient for {name}"
        assert param.grad.abs().sum() > 0, f"zero gradient for {name}"


def test_egcl_layer_shapes() -> None:
    """EGCLLayer returns correct shapes for h, x, v."""
    layer = EGCLLayer(hidden_dim=64)
    h = torch.randn(4, 3, 64)
    x = torch.randn(4, 3, 2)
    v = torch.randn(4, 3, 2)

    h_new, x_new, v_new = layer(h, x, v)

    assert h_new.shape == (4, 3, 64)
    assert x_new.shape == (4, 3, 2)
    assert v_new.shape == (4, 3, 2)


def test_velocity_affects_position() -> None:
    """Different input velocities produce different output positions."""
    model = EGNN()
    model.eval()

    state1 = _make_state()
    state2 = state1.clone()
    state2[..., 2:4] += 1.0  # perturb velocities

    out1 = model(state1)
    out2 = model(state2)

    assert not torch.allclose(out1[..., :2], out2[..., :2])


def test_integration_with_trainer(tmp_path: Path) -> None:
    """EGNN trains end-to-end through the Trainer pipeline."""
    rng = np.random.default_rng(42)

    # create small HDF5 files with 5-feature state
    for name in ("train.h5", "val.h5"):
        trajectories = rng.normal(size=(5, 10, 3, 5))
        # set mass column to 1.0
        trajectories[:, :, :, 4] = 1.0
        energies = rng.normal(size=(5, 10))
        with h5py.File(tmp_path / name, "w") as f:
            f.create_dataset("trajectories", data=trajectories)
            f.create_dataset("energies", data=energies)

    cfg = TrainConfig(
        model=ModelConfig(name="egnn", hidden_dim=32, n_layers=2),
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
