"""Unit tests for daniela-hgnn-gravity-genforce-v1 gravity-loss adapter and analytical target."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gravity_adapter_genforce", HERE / "adapter.py")
A = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = A
spec.loader.exec_module(A)

from models.hgnn_canonical import CanonicalHGNN


def test_analytical_gravity_two_body_and_momentum_conservation():
    # Two bodies at (-1, 0) and (+1, 0) with masses m1=2.0, m2=3.0
    state = torch.tensor([[[ -1.0, 0.0, 0.0, 0.0, 2.0 ],
                           [  1.0, 0.0, 0.0, 0.0, 3.0 ],
                           [  0.0, 2.0, 0.0, 0.0, 0.0001 ]]], dtype=torch.float64)
    acc = A.analytical_gravity_acceleration(state, additive=A.ADDITIVE_SOFTENING)
    assert acc.shape == (1, 3, 2)
    assert torch.isfinite(acc).all()
    # Total gravitational force sum_i m_i a_i must vanish identically by Newton's 3rd law!
    total_force = torch.sum(state[..., 4:] * acc, dim=1)
    assert torch.allclose(total_force, torch.zeros_like(total_force), atol=1e-12)


def test_predicted_hgnn_acceleration_conservatively_trains_potential_only():
    torch.manual_seed(42)
    model = CanonicalHGNN(hidden_dim=16, n_layers=2, dt=0.001, pos_std=0.7216, momentum_std=1.3989)
    state = torch.randn(4, 3, 5)
    state[..., 4] = torch.rand(4, 3) + 0.5
    pred_acc = A.predicted_hgnn_acceleration(model, state, create_graph=True)
    target_acc = A.analytical_gravity_acceleration(state, additive=A.ADDITIVE_SOFTENING)
    loss = torch.mean((pred_acc - target_acc).square())
    loss.backward()
    # Potential parameters must receive finite non-zero gradients; kinetic parameters must have Nonegrad
    pot_grads = [p.grad for p in model.potential.parameters()]
    kin_grads = [p.grad for p in model.kinetic.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in pot_grads)
    assert any(g.abs().sum() > 0 for g in pot_grads)
    assert all(g is None for g in kin_grads)


def test_normalized_physics_gradient_is_raw_gradient_divided_by_scale():
    import json
    scale = json.loads((HERE / 'acceleration_scale.json').read_text())
    assert scale['independent_numpy_target_verified']
    assert A.ACCELERATION_MEAN_SQUARE == scale['mean_square'] > 0
    torch.manual_seed(42)
    model = CanonicalHGNN(hidden_dim=8, n_layers=1, dt=.001, pos_std=.72, momentum_std=1.4)
    state = torch.randn(4,3,5); state[...,4] = 1
    raw = (A.predicted_hgnn_acceleration(model,state,create_graph=True)-A.analytical_gravity_acceleration(state)).square().mean()
    params = list(model.potential.parameters())
    raw_grads = torch.autograd.grad(raw,params,retain_graph=True)
    scaled_grads = torch.autograd.grad(raw/A.ACCELERATION_MEAN_SQUARE,params)
    for a,b in zip(raw_grads,scaled_grads,strict=True):
        torch.testing.assert_close(b,a/A.ACCELERATION_MEAN_SQUARE,atol=1e-10,rtol=2e-5)


def test_staged_order_is_identity():
    order = np.load(HERE / "staged-order.npy", allow_pickle=False)
    assert np.array_equal(order, np.arange(2500))
    with np.load(A.BANK, allow_pickle=False) as z:
        assert np.array_equal(
            z["horizon"][order],
            np.repeat([1, 5, 20, 50, 100], 500),
        )
