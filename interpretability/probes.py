"""Build (input -> target) datasets by probing a loaded model.

The HGNN potential/kinetic probes call the public `decompose`/forward methods; the EGNN
layer-0 probe composes its public submodules. All distances/energies for the HGNN potential
are in normalized coordinates (the model normalizes by pos_std/vel_std internally).

References:
    - Cranmer et al. 2020 (probing GNN edge functions): https://arxiv.org/abs/2006.11287
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from data.dataset import NBodyDataset

BATCH = 4096


@dataclass(frozen=True)
class PerEdgeProbe:
    """Per-edge readout and per-state node energy gathered over real 3-body states."""

    r_ij: np.ndarray
    v_ij: np.ndarray
    other_min: np.ndarray
    v_node: np.ndarray
    sum_inv: np.ndarray


def distance_support(data_path: str, low: float = 0.02, high: float = 0.98) -> tuple[float, float]:
    """Physical pairwise-distance percentile range from a dataset."""
    ds = NBodyDataset(data_path)
    pos = ds.inputs[..., :2]
    n = pos.shape[1]
    dx = pos[:, :, None, :] - pos[:, None, :, :]
    d = torch.sqrt((dx**2).sum(-1) + 1e-12)
    iu = torch.triu_indices(n, n, offset=1)
    dd = d[:, iu[0], iu[1]].reshape(-1).numpy()
    return float(np.quantile(dd, low)), float(np.quantile(dd, high))


def normalized_pairwise_distances(data_path: str, pos_std: float) -> np.ndarray:
    """All pairwise distances in normalized coordinates (physical distance / pos_std)."""
    ds = NBodyDataset(data_path)
    pos = ds.inputs[..., :2] / pos_std
    n = pos.shape[1]
    dx = pos[:, :, None, :] - pos[:, None, :, :]
    d = torch.sqrt((dx**2).sum(-1) + 1e-12)
    iu = torch.triu_indices(n, n, offset=1)
    return d[:, iu[0], iu[1]].reshape(-1).numpy()


def speed_support(data_path: str, vel_std: float, high: float = 0.98) -> float:
    """Normalized-speed percentile from dataset velocities."""
    ds = NBodyDataset(data_path)
    v = (ds.inputs[..., 2:4] / vel_std).reshape(-1, 2).numpy()
    return float(np.quantile(np.linalg.norm(v, axis=1), high))


def potential_2body(model: nn.Module, r_norm: np.ndarray) -> np.ndarray:
    """Total learned potential on 2-body configs at normalized separations r."""
    r = torch.tensor(r_norm, dtype=torch.float32).reshape(-1, 1)
    b = r.shape[0]
    x = torch.zeros(b, 2, 2)
    x[:, 1, 0] = r[:, 0]
    m = torch.ones(b, 2, 1)
    with torch.no_grad():
        return model.potential(x, m).numpy()


def total_potential_3body(
    model: nn.Module,
    data_path: str,
    pos_std: float,
    support_norm: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-state (pairwise distances, V_total) for states with all distances in support."""
    lo, hi = support_norm
    ds = NBodyDataset(data_path)
    x = ds.inputs[..., :2] / pos_std
    m = ds.inputs[..., 4:5]
    n = ds.inputs.shape[1]
    iu = torch.triu_indices(n, n, offset=1)
    d = torch.sqrt(((x[:, :, None, :] - x[:, None, :, :]) ** 2).sum(-1) + 1e-12)
    dpairs = d[:, iu[0], iu[1]]
    ok = ((dpairs >= lo) & (dpairs <= hi)).all(1)
    return dpairs[ok].numpy(), _batched_potential(model, x[ok], m[ok])


def _batched_potential(model: nn.Module, x: torch.Tensor, m: torch.Tensor) -> np.ndarray:
    """Evaluate the total potential in batches to bound memory."""
    out = []
    with torch.no_grad():
        for s in range(0, x.shape[0], BATCH):
            out.append(model.potential(x[s : s + BATCH], m[s : s + BATCH]).numpy())
    return np.concatenate(out)


def per_edge_potential(model: nn.Module, data_path: str, pos_std: float) -> PerEdgeProbe:
    """Gather per-edge V_ij vs r_ij and per-state V_node on real states, via decompose."""
    ds = NBodyDataset(data_path)
    inp = ds.inputs
    n = inp.shape[1]
    iu = torch.triu_indices(n, n, offset=1)
    edges = list(zip(iu[0].tolist(), iu[1].tolist(), strict=True))
    third = [next(k for k in range(n) if k not in (a, b)) for a, b in edges]

    r_ij, omin, v_ij, v_node, sum_inv = [], [], [], [], []
    with torch.no_grad():
        for s in range(0, inp.shape[0], BATCH):
            batch = inp[s : s + BATCH]
            x = batch[..., :2] / pos_std
            m = batch[..., 4:5]
            parts = model.potential.decompose(x, m)
            d = parts.d_ij
            for (a, b), k in zip(edges, third, strict=True):
                r_ij.append(d[:, a, b])
                omin.append(torch.minimum(d[:, a, k], d[:, b, k]))
                v_ij.append(parts.v_ij[:, a, b])
            v_node.append(parts.v_node)
            sum_inv.append((1.0 / d[:, iu[0], iu[1]]).sum(-1))

    return PerEdgeProbe(
        r_ij=torch.cat(r_ij).numpy(),
        v_ij=torch.cat(v_ij).numpy(),
        other_min=torch.cat(omin).numpy(),
        v_node=torch.cat(v_node).numpy(),
        sum_inv=torch.cat(sum_inv).numpy(),
    )


def physical_force_forward(model: nn.Module, r_phys: np.ndarray, dt: float) -> np.ndarray:
    """Physical acceleration on body 0 via a v=0 forward step: a = |v_out| / dt."""
    r = torch.tensor(r_phys, dtype=torch.float32).reshape(-1, 1)
    b = r.shape[0]
    state = torch.zeros(b, 2, 5)
    state[:, 1, 0] = r[:, 0]
    state[:, :, 4] = 1.0
    with torch.no_grad():
        out = model(state)
    return (torch.linalg.norm(out[:, 0, 2:4], dim=-1) / dt).numpy()


def physical_force_gradient(
    model: nn.Module, r_phys: np.ndarray, pos_std: float, vel_std: float
) -> np.ndarray:
    """Physical acceleration from the potential gradient: a = vel_std * |dV/dx_norm|."""
    r = torch.tensor(r_phys, dtype=torch.float32).reshape(-1, 1)
    b = r.shape[0]
    x = torch.zeros(b, 2, 2)
    x[:, 1, 0] = r[:, 0] / pos_std
    x.requires_grad_(True)
    m = torch.ones(b, 2, 1)
    v = model.potential(x, m).sum()
    (dvdx,) = torch.autograd.grad(v, x)
    return (vel_std * torch.linalg.norm(dvdx[:, 0], dim=-1)).detach().numpy()


def kinetic_grid(
    model: nn.Module, smax: float, n: int = 80
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-particle kinetic energy T over a (vx, vy) grid, normalized velocity units."""
    g = np.linspace(-smax, smax, n)
    vx_g, vy_g = np.meshgrid(g, g)
    v = torch.tensor(np.stack([vx_g.ravel(), vy_g.ravel()], -1), dtype=torch.float32).reshape(
        -1, 1, 2
    )
    m = torch.ones(v.shape[0], 1, 1)
    with torch.no_grad():
        t = model.kinetic(v, m).numpy().reshape(vx_g.shape)
    return vx_g, vy_g, t


def egnn_effective_force(
    model: nn.Module, r_phys: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """EGNN effective acceleration on body 0 from a v=0 step, plus its x-component sign."""
    r = torch.tensor(r_phys, dtype=torch.float32).reshape(-1, 1)
    b = r.shape[0]
    state = torch.zeros(b, 2, 5)
    state[:, 1, 0] = r[:, 0]
    state[:, :, 4] = 1.0
    with torch.no_grad():
        out = model(state)
    v_out = out[:, 0, 2:4]
    return (torch.linalg.norm(v_out, dim=-1) / dt).numpy(), v_out[:, 0].numpy()


def egnn_layer0_weight(model: nn.Module, r_phys: np.ndarray, pos_std: float) -> np.ndarray:
    """EGNN layer-0 per-edge coordinate weight w(r); velocity-free, a clean function of r."""
    layer0 = model.layers[0]
    c = model.mlp_embed(torch.ones(1, 1, 1)).reshape(-1)
    r = torch.tensor(r_phys / pos_std, dtype=torch.float32)
    r_sq = (r**2).reshape(-1, 1)
    b = r_sq.shape[0]
    edge_input = torch.cat([c.expand(b, -1), c.expand(b, -1), r_sq], dim=-1)
    with torch.no_grad():
        m_ij = layer0.mlp_e(edge_input)
        w = layer0.mlp_x(m_ij).squeeze(-1)
    return w.numpy()
