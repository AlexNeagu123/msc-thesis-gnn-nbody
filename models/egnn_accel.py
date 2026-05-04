"""Equivariant GNN with an acceleration-based second-order dynamics update.

The model retains EGNN's equivariant message passing but constrains the state
update to symplectic Euler over a learned acceleration field:

    a = accel_head(message_passing(mass_embed(m), x_norm))
    v_next = v + dt * a
    x_next = x + dt * v_next
    m_next = m

Internally, message passing operates on `x_norm = x / pos_std` for numerical
stability. The head emits `a_norm`, a unitless normalized velocity increment
per timestep; `a_real = a_norm * (vel_std / dt)` recovers a real acceleration
so the integrator can apply `dt` explicitly. This is mathematically identical
to predicting a normalized velocity delta directly, but preserves the
interpretation that the network outputs an acceleration-like field.

Tiny-init on the head's final scalar makes `a_norm ≈ 0` at initialization, so
a fresh model approximately reproduces the constant-velocity baseline:

    v_next ≈ v
    x_next ≈ x + dt * v

The "approximately" is bounded by the tiny-init scale (`xavier_uniform_` gain
of 1e-3), and tests assert this with a numerical tolerance.

References:
    - EGNN paper (Satorras et al., 2021): https://arxiv.org/abs/2102.09844
    - Reuses the per-edge w_ij * dx mechanism from models.egnn.EGCLLayer.
    - Architecture motivation: thesis discussion 2026-05-04.
"""

import torch
from torch import Tensor, nn

_ACCEL_CLIP_NORM = 100.0


def _clip_vector_norm(vectors: Tensor, max_norm: float) -> Tensor:
    """Clip vectors to a maximum L2 norm, preserving direction.

    Component-wise clamping (`torch.clamp(v, min=-c, max=c)`) is NOT
    rotation-equivariant once the clamp activates: clipping x and y
    independently distorts diagonal vectors. Norm-based scaling is
    rotation-equivariant because the L2 norm is rotation-invariant.

    Args:
        vectors: tensor with last dimension = vector dimension (2D here).
        max_norm: maximum allowed L2 norm. Vectors below this pass through.

    Returns:
        Tensor of the same shape with each vector's norm bounded by `max_norm`.
    """
    norms = vectors.norm(dim=-1, keepdim=True)
    factor = torch.clamp_max(max_norm / norms.clamp_min(1e-12), 1.0)
    return vectors * factor


class EGNNAccelLayer(nn.Module):
    """Equivariant message-passing layer that updates only node features.

    Distances feed the edge MLP for message construction; positions are not
    updated inside the layer. The acceleration head at the model's tail
    handles the position/velocity update.
    """

    def __init__(self, hidden_dim: int) -> None:
        """Create edge-message and node-update MLPs."""
        super().__init__()
        h = hidden_dim

        self.mlp_e = nn.Sequential(
            nn.Linear(2 * h + 1, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )

        self.mlp_h = nn.Sequential(
            nn.Linear(2 * h, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )

    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        """Aggregate messages over pairwise distances and update node features."""
        n = x.size(1)

        dx = x.unsqueeze(2) - x.unsqueeze(1)
        r_sq = (dx**2).sum(dim=-1, keepdim=True)

        h_i = h.unsqueeze(2).expand(-1, -1, n, -1)
        h_j = h.unsqueeze(1).expand(-1, n, -1, -1)
        edge_input = torch.cat([h_i, h_j, r_sq], dim=-1)
        m_ij = self.mlp_e(edge_input)

        mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
        m_ij = m_ij * mask[None, :, :, None]

        m_agg = m_ij.sum(dim=2)
        return h + self.mlp_h(torch.cat([h, m_agg], dim=-1))


class AccelerationHead(nn.Module):
    """Equivariant read-off mapping node features to a per-particle acceleration.

    The head reuses EGCL's coordinate-update mechanism: per-edge scalar weights
    `w_ij` modulate the relative-position vectors `(x_i - x_j)`, producing an
    equivariant 2-D output. The final scalar layer is tiny-init so the
    aggregated acceleration is near zero at initialization.
    """

    def __init__(self, hidden_dim: int) -> None:
        """Create edge MLP and tiny-init scalar weight MLP."""
        super().__init__()
        h = hidden_dim

        self.mlp_e = nn.Sequential(
            nn.Linear(2 * h + 1, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )

        # tiny-init final scalar so accel ≈ 0 → constant-velocity at init
        accel_out = nn.Linear(h, 1, bias=False)
        nn.init.xavier_uniform_(accel_out.weight, gain=0.001)
        self.mlp_w = nn.Sequential(
            nn.Linear(h, h),
            nn.SiLU(),
            accel_out,
        )

    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        """Compute equivariant acceleration as sum_j w_ij * (x_i - x_j) / (n - 1)."""
        n = x.size(1)

        dx = x.unsqueeze(2) - x.unsqueeze(1)
        r_sq = (dx**2).sum(dim=-1, keepdim=True)

        h_i = h.unsqueeze(2).expand(-1, -1, n, -1)
        h_j = h.unsqueeze(1).expand(-1, n, -1, -1)
        edge_input = torch.cat([h_i, h_j, r_sq], dim=-1)
        m_ij = self.mlp_e(edge_input)

        mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
        m_ij = m_ij * mask[None, :, :, None]

        w_ij = self.mlp_w(m_ij)
        accel_terms = _clip_vector_norm(dx * w_ij, _ACCEL_CLIP_NORM)
        return accel_terms.sum(dim=2) / (n - 1)


class EGNNAccel(nn.Module):
    """Stacked EGNNAccel layers + acceleration head + symplectic Euler integrator."""

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 6,
        dt: float = 0.05,
        pos_std: float = 1.0,
        vel_std: float = 1.0,
    ) -> None:
        """Build mass embedding, message-passing stack, accel head, and register dt/stds."""
        super().__init__()

        self.register_buffer("pos_std", torch.tensor(pos_std))
        self.register_buffer("vel_std", torch.tensor(vel_std))
        self.register_buffer("dt", torch.tensor(dt))

        self.mlp_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.layers = nn.ModuleList([EGNNAccelLayer(hidden_dim) for _ in range(n_layers)])
        self.accel_head = AccelerationHead(hidden_dim)

    def forward(self, state: Tensor) -> Tensor:
        """Predict the next state via Newtonian acceleration + symplectic Euler."""
        pos = state[..., :2]
        vel = state[..., 2:4]
        mass = state[..., 4:]

        pos_norm = pos / self.pos_std

        h = self.mlp_embed(mass)
        for layer in self.layers:
            h = layer(h, pos_norm)

        a_norm = self.accel_head(h, pos_norm)
        a_real = a_norm * (self.vel_std / self.dt)

        vel_new = vel + self.dt * a_real
        pos_new = pos + self.dt * vel_new

        return torch.cat([pos_new, vel_new, mass], dim=-1)
