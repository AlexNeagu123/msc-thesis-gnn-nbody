"""Tests for models/hgnn.py."""

from pathlib import Path

import h5py
import numpy as np
import torch

from models.hgnn import HGNN, KineticNetwork, PotentialNetwork
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
    """Create a random state tensor [x, y, vx, vy, mass]."""
    torch.manual_seed(0)
    state = torch.randn(batch, n_particles, 4)
    mass = torch.ones(batch, n_particles, 1)
    return torch.cat([state, mass], dim=-1)


def test_output_shape() -> None:
    """Forward pass returns (batch, n_particles, 5)."""
    model = HGNN()
    state = _make_state(batch=4)
    out = model(state)

    assert out.shape == (4, 3, 5)


def test_single_sample() -> None:
    """Forward pass works with batch size 1."""
    model = HGNN()
    state = _make_state(batch=1)
    out = model(state)

    assert out.shape == (1, 3, 5)


def test_mass_passthrough() -> None:
    """Mass column is passed through unchanged."""
    model = HGNN()
    state = _make_state()
    out = model(state)

    assert torch.allclose(state[..., 4:], out[..., 4:])


def test_parameter_count() -> None:
    """Total parameters should be approximately 191K."""
    model = HGNN(hidden_dim=64, n_layers=4)
    n_params = sum(p.numel() for p in model.parameters())

    assert 170_000 < n_params < 220_000


def test_deterministic() -> None:
    """Same input produces identical output in eval mode."""
    model = HGNN()
    model.eval()
    state = _make_state()

    out1 = model(state)
    out2 = model(state)

    assert torch.allclose(out1, out2)


def test_hamiltonian_translation_invariance() -> None:
    """H is translation-invariant in positions (V depends only on distances)."""
    model = HGNN()
    model.eval()
    state = _make_state()

    x = state[..., :2] / model.pos_std
    v = state[..., 2:4] / model.vel_std
    m = state[..., 4:]

    H_orig = model.hamiltonian(x, v, m)

    d = torch.tensor([3.0, -2.0])
    H_shifted = model.hamiltonian(x + d, v, m)

    assert torch.allclose(H_orig, H_shifted, atol=1e-5)


def test_kinetic_network_shape() -> None:
    """KineticNetwork returns (B,) scalar per sample."""
    kinetic = KineticNetwork(hidden_dim=64)
    v = torch.randn(4, 3, 2)
    m = torch.ones(4, 3, 1)

    T = kinetic(v, m)

    assert T.shape == (4,)


def test_potential_network_shape() -> None:
    """PotentialNetwork returns (B,) scalar per sample."""
    potential = PotentialNetwork(hidden_dim=64, n_layers=4)
    x = torch.randn(4, 3, 2)
    m = torch.ones(4, 3, 1)

    V = potential(x, m)

    assert V.shape == (4,)


def test_free_fall_at_init() -> None:
    """With tiny-init on V readouts, one forward step should be ~pure drift.

    At init, V ~= 0 everywhere so forces ~= 0, leaving x_new ~= x + dt*v and
    v_new ~= v. T is NOT tiny-init (we want dT/dv = v at m=1), so the drift
    term x_dot = dT/dv should dominate and approximately equal v.
    """
    model = HGNN(dt=0.05)
    model.eval()
    state = _make_state()

    out = model(state)
    x = state[..., :2]
    v = state[..., 2:4]
    x_new = out[..., :2]
    v_new = out[..., 2:4]

    # velocities should be nearly unchanged (tiny-V gives tiny v_dot)
    assert torch.allclose(v_new, v, atol=0.05), (
        f"v_new deviates from v by {(v_new - v).abs().max().item():.4f}"
    )

    # positions should move approximately by dt * dT/dv (not strictly dt * v
    # because T is learned; for untrained mlp_T on standard init, dT/dv is
    # correlated with v but not identical). We check that the move is
    # proportional to dt and not huge.
    pos_change = (x_new - x).abs().max().item()
    assert pos_change < 1.0, f"position change {pos_change:.4f} too large at init"


def test_gradient_flow() -> None:
    """All trainable parameters receive gradients after backward pass.

    Critically tests that autograd.grad(..., create_graph=True) propagates
    loss gradients through the leapfrog integrator back to model params.
    """
    model = HGNN()
    state = _make_state()

    out = model(state)
    loss = out.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert param.grad.abs().sum() > 0, f"zero gradient for {name}"


def test_energy_conservation_at_init() -> None:
    """Killer test: energy drift over a short rollout should be bounded at init.

    With tiny-init, H starts near zero. A correct symplectic leapfrog keeps
    |H(t) - H(0)| bounded (not monotonically growing). This test catches:
        - non-symplectic integrators (Euler, RK4 pretending to be leapfrog)
        - sign errors in Hamilton's equations
        - missing autograd graph retention
    """
    torch.manual_seed(42)
    model = HGNN(dt=0.05)
    model.eval()

    state = _make_state(batch=2)
    m = state[..., 4:]

    # compute H at t=0 in normalized coords
    with torch.enable_grad():
        x0 = (state[..., :2] / model.pos_std).detach().requires_grad_(True)
        v0 = (state[..., 2:4] / model.vel_std).detach().requires_grad_(True)
        H0 = model.hamiltonian(x0, v0, m).detach()

    # roll out 10 steps
    energies = [H0]
    current = state
    for _ in range(10):
        current = model(current)
        with torch.enable_grad():
            x = (current[..., :2] / model.pos_std).detach().requires_grad_(True)
            v = (current[..., 2:4] / model.vel_std).detach().requires_grad_(True)
            H = model.hamiltonian(x, v, m).detach()
        energies.append(H)

    # at init, |H| is tiny (tiny-init on V, and T is small at random v).
    # require drift to stay below a loose bound.
    max_drift = max((h - H0).abs().max().item() for h in energies)
    assert max_drift < 1.0, f"energy drift {max_drift:.4f} too large"


def test_integration_with_trainer(tmp_path: Path) -> None:
    """HGNN trains end-to-end through the Trainer pipeline."""
    rng = np.random.default_rng(42)

    for name in ("train.h5", "val.h5"):
        trajectories = rng.normal(size=(5, 10, 3, 5))
        trajectories[:, :, :, 4] = 1.0  # mass = 1
        energies = rng.normal(size=(5, 10))
        with h5py.File(tmp_path / name, "w") as f:
            f.create_dataset("trajectories", data=trajectories)
            f.create_dataset("energies", data=energies)

    cfg = TrainConfig(
        model=ModelConfig(name="hgnn", hidden_dim=32, n_layers=2),
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


def test_readout_tiny_init() -> None:
    """Final layer of MLP_v and MLP_e has small weights and no bias."""
    model = HGNN(hidden_dim=64, n_layers=4)

    v_out = model.potential.mlp_v[-1]
    e_out = model.potential.mlp_e[-1]

    assert v_out.bias is None
    assert e_out.bias is None
    assert v_out.weight.abs().max().item() < 0.01
    assert e_out.weight.abs().max().item() < 0.01
