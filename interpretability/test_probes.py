"""Tests for interpretability/probes.py using tiny synthetic models (no PySR)."""

from pathlib import Path

import h5py
import numpy as np
import torch

from interpretability import probes
from models.egnn import EGNN
from models.hgnn import HGNN


def _write_h5(path: Path, n_traj: int = 3, n_steps: int = 6, n: int = 3) -> None:
    """Write a tiny 3-body trajectory file."""
    rng = np.random.default_rng(0)
    traj = rng.normal(size=(n_traj, n_steps, n, 5)).astype(np.float32)
    traj[..., 4] = 1.0
    energies = rng.normal(size=(n_traj, n_steps)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("trajectories", data=traj)
        f.create_dataset("energies", data=energies)


def _hgnn() -> HGNN:
    """Tiny untrained HGNN for shape tests."""
    torch.manual_seed(0)
    return HGNN(hidden_dim=8, n_layers=2).eval()


def _egnn() -> EGNN:
    """Tiny untrained EGNN for shape tests."""
    torch.manual_seed(0)
    return EGNN(hidden_dim=8, n_layers=2).eval()


def test_potential_2body_shape() -> None:
    """potential_2body returns one finite value per separation."""
    v = probes.potential_2body(_hgnn(), np.linspace(0.2, 2.0, 12))
    assert v.shape == (12,)
    assert np.isfinite(v).all()


def test_physical_force_probes_finite() -> None:
    """Both physical-force probes return finite per-separation accelerations."""
    model = _hgnn()
    r = np.linspace(0.3, 2.0, 10)
    a_fwd = probes.physical_force_forward(model, r, dt=0.05)
    a_grad = probes.physical_force_gradient(model, r, pos_std=1.0, vel_std=1.0)
    assert a_fwd.shape == (10,)
    assert a_grad.shape == (10,)
    assert np.isfinite(a_fwd).all()
    assert np.isfinite(a_grad).all()


def test_kinetic_grid_shape() -> None:
    """kinetic_grid returns a square grid of finite energies."""
    vx, _vy, t = probes.kinetic_grid(_hgnn(), smax=1.5, n=8)
    assert vx.shape == (8, 8)
    assert t.shape == (8, 8)
    assert np.isfinite(t).all()


def test_egnn_probes_finite() -> None:
    """EGNN effective-force and layer-0 weight probes return finite values."""
    model = _egnn()
    r = np.linspace(0.3, 2.0, 10)
    accel, _vx0 = probes.egnn_effective_force(model, r, dt=0.05)
    w = probes.egnn_layer0_weight(model, r, pos_std=1.0)
    assert accel.shape == (10,)
    assert w.shape == (10,)
    assert np.isfinite(accel).all()
    assert np.isfinite(w).all()


def test_data_probes(tmp_path: Path) -> None:
    """Data-backed probes load a tiny file and return aligned arrays."""
    path = tmp_path / "data.h5"
    _write_h5(path)

    lo, hi = probes.distance_support(str(path))
    assert 0 <= lo < hi

    feats, v = probes.total_potential_3body(
        _hgnn(), str(path), pos_std=1.0, support_norm=(0.0, 1e9)
    )
    assert feats.shape[1] == 3
    assert feats.shape[0] == v.shape[0]

    edge = probes.per_edge_potential(_hgnn(), str(path), pos_std=1.0)
    assert edge.r_ij.shape == edge.v_ij.shape
    assert np.isfinite(edge.v_ij).all()
