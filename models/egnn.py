"""Equivariant Graph Neural Network for N-body prediction.

References:
    - EGNN paper (Satorras et al., 2021): https://arxiv.org/abs/2102.09844
    - Architecture specs: edu/architecture-specs.md
    - EGNN reference implementation: https://github.com/vgsatorras/egnn
"""

import torch
from torch import Tensor, nn


class EGCLLayer(nn.Module):
    """Single equivariant graph convolutional layer."""

    def __init__(self, hidden_dim: int) -> None:
        """Create edge, coordinate, velocity, and node MLPs."""
        super().__init__()
        h = hidden_dim

        self.mlp_e = nn.Sequential(
            nn.Linear(2 * h + 1, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )

        # tiny output init keeps early coordinate updates near zero
        coord_out = nn.Linear(h, 1, bias=False)
        nn.init.xavier_uniform_(coord_out.weight, gain=0.001)
        self.mlp_x = nn.Sequential(
            nn.Linear(h, h),
            nn.SiLU(),
            coord_out,
        )

        self.mlp_v = nn.Linear(h, 1)

        self.mlp_h = nn.Sequential(
            nn.Linear(2 * h, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )

    def forward(
        self,
        h: Tensor,
        x: Tensor,
        v: Tensor,
        *,
        update_h: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply one EGCL layer."""
        n = x.size(1)

        dx = x.unsqueeze(2) - x.unsqueeze(1)
        r_sq = (dx**2).sum(dim=-1, keepdim=True)

        h_i = h.unsqueeze(2).expand(-1, -1, n, -1)
        h_j = h.unsqueeze(1).expand(-1, n, -1, -1)
        edge_input = torch.cat([h_i, h_j, r_sq], dim=-1)
        m_ij = self.mlp_e(edge_input)

        mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
        m_ij = m_ij * mask[None, :, :, None]

        w_ij = self.mlp_x(m_ij)
        coord_trans = torch.clamp(dx * w_ij, min=-100.0, max=100.0)

        coord_agg = coord_trans.sum(dim=2) / (n - 1)
        v_scale = self.mlp_v(h)
        v_new = v_scale * v + coord_agg
        x_new = x + v_new

        if update_h:
            m_agg = m_ij.sum(dim=2)
            h = h + self.mlp_h(torch.cat([h, m_agg], dim=-1))

        return h, x_new, v_new


class EGNN(nn.Module):
    """Stacked EGCL model with the Trainer-compatible state interface."""

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 4,
        pos_std: float = 1.0,
        vel_std: float = 1.0,
    ) -> None:
        """Create mass embedding and EGCL layers."""
        super().__init__()

        self.register_buffer("pos_std", torch.tensor(pos_std))
        self.register_buffer("vel_std", torch.tensor(vel_std))

        self.mlp_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.layers = nn.ModuleList([EGCLLayer(hidden_dim) for _ in range(n_layers)])

    def forward(self, state: Tensor) -> Tensor:
        """Predict the next state."""
        pos = state[..., :2] / self.pos_std
        vel = state[..., 2:4] / self.vel_std
        mass = state[..., 4:]

        h = self.mlp_embed(mass)

        for i, layer in enumerate(self.layers):
            is_last = i == len(self.layers) - 1
            h, pos, vel = layer(h, pos, vel, update_h=not is_last)

        pos = pos * self.pos_std
        vel = vel * self.vel_std

        return torch.cat([pos, vel, mass], dim=-1)
