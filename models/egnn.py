"""EGNN — Equivariant Graph Neural Network (Satorras et al., 2021).

Predicts next positions and velocities directly via stacked EGCL layers.
Designed for N-body gravitational systems in 2D with a fully connected graph.

References:
    - EGNN paper (Satorras et al., 2021): https://arxiv.org/abs/2102.09844
    - Architecture specs: edu/architecture-specs.md
    - EGNN reference implementation: https://github.com/vgsatorras/egnn
"""

import torch
from torch import Tensor, nn


class EGCLLayer(nn.Module):
    """Single Equivariant Graph Convolutional Layer.

    Performs one round of message passing on a fully connected graph,
    updating node embeddings, positions, and velocities equivariantly.

    Args:
        hidden_dim: dimension of node embeddings.
    """

    def __init__(self, hidden_dim: int) -> None:
        """Initialize the four MLPs that define the layer.

        Args:
            hidden_dim: dimension of node embeddings (H).
        """
        super().__init__()
        h = hidden_dim

        # edge message: MLP_e(h_i || h_j || r_ij^2) -> (B, N, N, H)
        self.mlp_e = nn.Sequential(
            nn.Linear(2 * h + 1, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )

        # coordinate weight: MLP_x(m_ij) -> scalar per edge
        # final layer uses tiny init (gain=0.001, no bias) so coordinate
        # updates start near zero — key stability trick from reference impl
        coord_out = nn.Linear(h, 1, bias=False)
        nn.init.xavier_uniform_(coord_out.weight, gain=0.001)
        self.mlp_x = nn.Sequential(
            nn.Linear(h, h),
            nn.SiLU(),
            coord_out,
        )

        # velocity scaling: MLP_v(h_i) -> scalar per node
        self.mlp_v = nn.Linear(h, 1)

        # node update: MLP_h(h_i || agg_i) -> (B, N, H)
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
        """Apply one EGCL layer.

        Args:
            h: node embeddings, shape (batch, n_particles, H).
            x: positions, shape (batch, n_particles, 2).
            v: velocities, shape (batch, n_particles, 2).
            update_h: whether to compute the node embedding update. Set to
                False on the last layer to skip unused computation.

        Returns:
            Tuple of (h_new, x_new, v_new) with same shapes as inputs.
        """
        n = x.size(1)

        # pairwise position differences: (B, N, N, 2)
        dx = x.unsqueeze(2) - x.unsqueeze(1)

        # squared distances: (B, N, N, 1)
        r_sq = (dx**2).sum(dim=-1, keepdim=True)

        # edge messages: concat h_i, h_j, r_ij^2 -> MLP_e
        h_i = h.unsqueeze(2).expand(-1, -1, n, -1)  # (B, N, N, H)
        h_j = h.unsqueeze(1).expand(-1, n, -1, -1)  # (B, N, N, H)
        edge_input = torch.cat([h_i, h_j, r_sq], dim=-1)  # (B, N, N, 2H+1)
        m_ij = self.mlp_e(edge_input)  # (B, N, N, H)

        # mask out self-loops (diagonal)
        mask = ~torch.eye(n, dtype=torch.bool, device=x.device)  # (N, N)
        m_ij = m_ij * mask[None, :, :, None]

        # coordinate weights: (B, N, N, 1)
        w_ij = self.mlp_x(m_ij)

        # emergency clamp on coordinate translations (reference impl safety net)
        coord_trans = torch.clamp(dx * w_ij, min=-100.0, max=100.0)

        # velocity update
        coord_agg = coord_trans.sum(dim=2) / (n - 1)  # (B, N, 2)
        v_scale = self.mlp_v(h)  # (B, N, 1)
        v_new = v_scale * v + coord_agg

        # position update
        x_new = x + v_new

        # node update (with residual connection) — skipped on last layer
        if update_h:
            m_agg = m_ij.sum(dim=2)  # (B, N, H)
            h = h + self.mlp_h(torch.cat([h, m_agg], dim=-1))

        return h, x_new, v_new


class EGNN(nn.Module):
    """Equivariant Graph Neural Network for N-body prediction.

    Stacks multiple EGCLLayers and wraps the interface expected by Trainer:
    forward(x) where x is (batch, n_particles, 5) with [x, y, vx, vy, mass].

    Args:
        hidden_dim: dimension of node embeddings.
        n_layers: number of EGCL layers to stack.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 4,
        pos_std: float = 1.0,
        vel_std: float = 1.0,
    ) -> None:
        """Initialize mass embedding MLP and EGCL layer stack.

        Args:
            hidden_dim: dimension of node embeddings (H).
            n_layers: number of EGCL layers.
            pos_std: standard deviation of positions for normalization.
            vel_std: standard deviation of velocities for normalization.
        """
        super().__init__()

        # normalization constants (not trainable, saved with model)
        self.register_buffer("pos_std", torch.tensor(pos_std))
        self.register_buffer("vel_std", torch.tensor(vel_std))

        # mass -> initial node embedding: MLP_embed(m_i) -> (B, N, H)
        self.mlp_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.layers = nn.ModuleList([EGCLLayer(hidden_dim) for _ in range(n_layers)])

    def forward(self, state: Tensor) -> Tensor:
        """Predict next state from current state.

        Internally normalizes positions and velocities, runs through EGCL
        layers, then denormalizes the output back to physical units.

        Args:
            state: input tensor of shape (batch, n_particles, 5) where
                features are [x, y, vx, vy, mass].

        Returns:
            Predicted next state, shape (batch, n_particles, 5).
        """
        pos = state[..., :2] / self.pos_std  # normalize
        vel = state[..., 2:4] / self.vel_std
        mass = state[..., 4:]

        h = self.mlp_embed(mass)  # (B, N, H)

        for i, layer in enumerate(self.layers):
            is_last = i == len(self.layers) - 1
            h, pos, vel = layer(h, pos, vel, update_h=not is_last)

        pos = pos * self.pos_std  # denormalize
        vel = vel * self.vel_std

        return torch.cat([pos, vel, mass], dim=-1)  # (B, N, 5)
