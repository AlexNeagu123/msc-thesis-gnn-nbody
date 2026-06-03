"""Hamiltonian Graph Neural Network with a decomposed H = T + V.

References:
    - HGNN paper (Bishnoi et al., 2023): https://arxiv.org/abs/2307.05299
    - Architecture specs: edu/architecture-specs.md
    - HOGN (Sanchez-Gonzalez et al., 2019): https://arxiv.org/abs/1909.12790
    - PySR (Cranmer, 2023): https://github.com/MilesCranmer/PySR
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PotentialParts:
    """Per-component breakdown of the potential energy for one batch."""

    total: Tensor
    v_node: Tensor
    v_edge: Tensor
    v_i: Tensor
    v_ij: Tensor
    d_ij: Tensor


class KineticNetwork(nn.Module):
    """Learned kinetic energy T(v, m) = Sum_i T_i."""

    def __init__(self, hidden_dim: int) -> None:
        """Create kinetic embedding and readout networks."""
        super().__init__()
        h = hidden_dim

        self.mlp_embed = nn.Sequential(
            nn.Linear(3, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )

        # final bias would only add an unlearnable Hamiltonian constant
        self.mlp_T = nn.Sequential(
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, 1, bias=False),
        )

    def decompose(self, v: Tensor, m: Tensor) -> Tensor:
        """Per-particle kinetic energy T_i, before the sum over particles."""
        h0 = self.mlp_embed(torch.cat([v, m], dim=-1))
        return self.mlp_T(h0).squeeze(-1)

    def forward(self, v: Tensor, m: Tensor) -> Tensor:
        """Compute total kinetic energy per sample."""
        return self.decompose(v, m).sum(dim=-1)


class PotentialMPLayer(nn.Module):
    """Message passing layer for the potential energy pathway."""

    def __init__(self, hidden_dim: int) -> None:
        """Create message, node, and edge update networks."""
        super().__init__()
        h = hidden_dim

        self.mlp_msg = nn.Sequential(
            nn.Linear(3 * h, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )

        self.mlp_node = nn.Sequential(
            nn.Linear(2 * h, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )

        self.mlp_edge = nn.Sequential(
            nn.Linear(2 * h, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )

    def forward(self, z: Tensor, e: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Apply one round of message passing."""
        n = z.size(1)

        z_i = z.unsqueeze(2).expand(-1, -1, n, -1)
        z_j = z.unsqueeze(1).expand(-1, n, -1, -1)
        msg_input = torch.cat([z_i, z_j, e], dim=-1)
        msg = self.mlp_msg(msg_input)

        msg = msg * mask[None, :, :, None]
        agg = msg.sum(dim=2)
        z_new = z + self.mlp_node(torch.cat([z, agg], dim=-1))
        e_new = e + self.mlp_edge(torch.cat([e, msg], dim=-1))

        return z_new, e_new


class PotentialNetwork(nn.Module):
    """Learned potential energy V(x, m) from pairwise distances."""

    def __init__(self, hidden_dim: int, n_layers: int) -> None:
        """Create embeddings, message passing layers, and readouts."""
        super().__init__()
        h = hidden_dim

        self.mlp_edge_embed = nn.Sequential(
            nn.Linear(1, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )

        self.mlp_node_embed = nn.Sequential(
            nn.Linear(1, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )

        self.mp_layers = nn.ModuleList([PotentialMPLayer(h) for _ in range(n_layers)])

        v_out = nn.Linear(h, 1, bias=False)
        nn.init.xavier_uniform_(v_out.weight, gain=0.001)
        self.mlp_v = nn.Sequential(
            nn.Linear(h, h),
            nn.SiLU(),
            v_out,
        )

        # this edge readout is the PySR target
        e_out = nn.Linear(h, 1, bias=False)
        nn.init.xavier_uniform_(e_out.weight, gain=0.001)
        self.mlp_e = nn.Sequential(
            nn.Linear(h, h),
            nn.SiLU(),
            e_out,
        )

    def decompose(self, x: Tensor, m: Tensor) -> PotentialParts:
        """Per-node, per-edge, and total potential energy for one batch."""
        n = x.size(1)

        dx = x.unsqueeze(2) - x.unsqueeze(1)
        d_ij = torch.sqrt((dx**2).sum(dim=-1, keepdim=True) + 1e-8)

        e = self.mlp_edge_embed(d_ij)
        z = self.mlp_node_embed(m)

        mask = ~torch.eye(n, dtype=torch.bool, device=x.device)

        for layer in self.mp_layers:
            z, e = layer(z, e, mask)

        V_i = self.mlp_v(z).squeeze(-1)
        V_ij = self.mlp_e(e).squeeze(-1)

        V_node = V_i.sum(dim=-1)

        i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=x.device)
        V_edge = V_ij[:, i_idx, j_idx].sum(dim=-1)

        return PotentialParts(
            total=V_node + V_edge,
            v_node=V_node,
            v_edge=V_edge,
            v_i=V_i,
            v_ij=V_ij,
            d_ij=d_ij.squeeze(-1),
        )

    def forward(self, x: Tensor, m: Tensor) -> Tensor:
        """Compute total potential energy per sample."""
        return self.decompose(x, m).total


class HGNN(nn.Module):
    """Hamiltonian Graph Neural Network for N-body prediction.

    Dynamics come from Hamilton's equations:
        H(x, v, m) = T(v, m) + V(x, m)
        dx/dt =  dH/dv = dT/dv
        dv/dt = -dH/dx = -dV/dx
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 4,
        dt: float = 0.05,
        pos_std: float = 1.0,
        vel_std: float = 1.0,
    ) -> None:
        """Create energy networks and non-trainable scale buffers."""
        super().__init__()
        self.kinetic = KineticNetwork(hidden_dim)
        self.potential = PotentialNetwork(hidden_dim, n_layers)

        self.register_buffer("pos_std", torch.tensor(float(pos_std)))
        self.register_buffer("vel_std", torch.tensor(float(vel_std)))
        self.register_buffer("dt", torch.tensor(float(dt)))

    def hamiltonian(self, x: Tensor, v: Tensor, m: Tensor) -> Tensor:
        """Compute H = T + V in normalized coordinates."""
        return self.kinetic(v, m) + self.potential(x, m)

    def energies(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Return (T, V) for a raw state [x, y, vx, vy, m], normalizing as forward does."""
        x = state[..., :2] / self.pos_std
        v = state[..., 2:4] / self.vel_std
        m = state[..., 4:]
        return self.kinetic(v, m), self.potential(x, m)

    def forward(self, state: Tensor) -> Tensor:
        """Predict next state from current state via one leapfrog step.

        `enable_grad` is required because validation and rollout code usually
        call the model under `torch.no_grad()`.
        """
        mass = state[..., 4:]

        with torch.enable_grad():
            # leaf tensors are required for autograd.grad
            x = (state[..., :2] / self.pos_std).detach().requires_grad_(True)
            v = (state[..., 2:4] / self.vel_std).detach().requires_grad_(True)

            v_dot_0 = self._v_dot(x, mass)
            v_half = v + (self.dt / 2) * v_dot_0

            x_dot_half = self._x_dot(v_half, mass)
            x_new = x + self.dt * x_dot_half

            v_dot_1 = self._v_dot(x_new, mass)
            v_new = v_half + (self.dt / 2) * v_dot_1

        pos_out = x_new * self.pos_std
        vel_out = v_new * self.vel_std

        return torch.cat([pos_out, vel_out, mass], dim=-1)

    def _v_dot(self, x: Tensor, m: Tensor) -> Tensor:
        """Compute dv/dt = -dV/dx at position x via autograd.

        `create_graph=True` retains second-order information so
        loss.backward() can propagate through the integrator.
        """
        V = self.potential(x, m).sum()
        dV_dx = torch.autograd.grad(V, x, create_graph=True)[0]
        return -dV_dx

    def _x_dot(self, v: Tensor, m: Tensor) -> Tensor:
        """Compute dx/dt = dT/dv at velocity v via autograd."""
        T = self.kinetic(v, m).sum()
        dT_dv = torch.autograd.grad(T, v, create_graph=True)[0]
        return dT_dv
