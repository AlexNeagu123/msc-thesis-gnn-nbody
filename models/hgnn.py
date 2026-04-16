"""HGNN - Hamiltonian Graph Neural Network (Bishnoi et al., 2023).

Predicts dynamics via a decomposed Hamiltonian H = T + V.
Kinetic energy T is learned from (v, m) per node (no message passing).
Potential energy V is learned from pairwise distances after 4 rounds of
message passing. Hamilton's equations derive (dx/dt, dv/dt), and a
symplectic leapfrog step advances the state by dt.

Designed for N-body gravitational systems in 2D with a fully connected graph.

References:
    - HGNN paper (Bishnoi et al., 2023): https://arxiv.org/abs/2307.05299
    - Architecture specs: edu/architecture-specs.md
    - HOGN (Sanchez-Gonzalez et al., 2019): https://arxiv.org/abs/1909.12790
    - PySR (Cranmer, 2023): https://github.com/MilesCranmer/PySR
"""

import torch
from torch import Tensor, nn


class KineticNetwork(nn.Module):
    """Learned kinetic energy T(v, m) = Sum_i MLP_T(h_i^0).

    Computes an initial per-node embedding h_i^0 from (velocity, mass), then
    emits a scalar kinetic energy per particle. No message passing - kinetic
    energy is a per-particle property and must not depend on other particles.

    Args:
        hidden_dim: dimension of node embeddings (H).
    """

    def __init__(self, hidden_dim: int) -> None:
        """Initialize the two MLPs.

        Args:
            hidden_dim: dimension of node embeddings (H).
        """
        super().__init__()
        h = hidden_dim

        # h_i^0 = MLP_embed(v_i || m_i): Linear(3, H) -> SiLU -> Linear(H, H) -> SiLU
        self.mlp_embed = nn.Sequential(
            nn.Linear(3, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )

        # T_i = MLP_T(h_i^0): Linear(H, H) -> SiLU -> Linear(H, 1, bias=False).
        # Bias on the final layer is a physically unlearnable constant: H is
        # only defined up to an additive constant, and dT/dv does not depend
        # on the final bias. Dropping it avoids a stuck parameter.
        self.mlp_T = nn.Sequential(
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, 1, bias=False),
        )

    def forward(self, v: Tensor, m: Tensor) -> Tensor:
        """Compute total kinetic energy T = Sum_i T_i for each sample in the batch.

        Args:
            v: velocities, shape (batch, n_particles, 2).
            m: masses, shape (batch, n_particles, 1).

        Returns:
            Scalar kinetic energy per sample, shape (batch,).
        """
        h0 = self.mlp_embed(torch.cat([v, m], dim=-1))  # (B, N, H)
        T_per_particle = self.mlp_T(h0).squeeze(-1)  # (B, N)
        return T_per_particle.sum(dim=-1)  # (B,)


class PotentialMPLayer(nn.Module):
    """Single round of message passing for the potential energy pathway.

    Updates node embeddings z_i and edge embeddings e_ij via:
        msg_ij = MLP_msg(z_i || z_j || e_ij)
        z_i   <- z_i + MLP_node(z_i || Sum_j msg_ij)    (residual)
        e_ij  <- e_ij + MLP_edge(e_ij || msg_ij)         (residual)

    Args:
        hidden_dim: dimension of node / edge embeddings (H).
    """

    def __init__(self, hidden_dim: int) -> None:
        """Initialize the three MLPs.

        Args:
            hidden_dim: dimension of embeddings (H).
        """
        super().__init__()
        h = hidden_dim

        # msg_ij = MLP_msg(z_i || z_j || e_ij): Linear(3H, H) -> SiLU -> Linear(H, H) -> SiLU
        self.mlp_msg = nn.Sequential(
            nn.Linear(3 * h, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )

        # z update: MLP_node(z_i || agg_i): Linear(2H, H) -> SiLU -> Linear(H, H) -> SiLU
        self.mlp_node = nn.Sequential(
            nn.Linear(2 * h, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )

        # e update: MLP_edge(e_ij || msg_ij): Linear(2H, H) -> SiLU -> Linear(H, H) -> SiLU
        self.mlp_edge = nn.Sequential(
            nn.Linear(2 * h, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )

    def forward(self, z: Tensor, e: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Apply one round of message passing.

        Args:
            z: node embeddings, shape (batch, n_particles, H).
            e: edge embeddings, shape (batch, n_particles, n_particles, H).
            mask: self-loop mask, shape (n_particles, n_particles) with True off-diagonal.

        Returns:
            Tuple (z_new, e_new) with residual updates applied.
        """
        n = z.size(1)

        # build msg input: concat z_i, z_j, e_ij
        z_i = z.unsqueeze(2).expand(-1, -1, n, -1)  # (B, N, N, H)
        z_j = z.unsqueeze(1).expand(-1, n, -1, -1)  # (B, N, N, H)
        msg_input = torch.cat([z_i, z_j, e], dim=-1)  # (B, N, N, 3H)
        msg = self.mlp_msg(msg_input)  # (B, N, N, H)

        # zero out self-loops
        msg = msg * mask[None, :, :, None]

        # aggregate messages into node i
        agg = msg.sum(dim=2)  # (B, N, H)

        # node residual update
        z_new = z + self.mlp_node(torch.cat([z, agg], dim=-1))

        # edge residual update
        e_new = e + self.mlp_edge(torch.cat([e, msg], dim=-1))

        return z_new, e_new


class PotentialNetwork(nn.Module):
    """Learned potential energy V(x, m) via message passing on pairwise distances.

    Architecture:
        - Edge init from distance d_ij: e_ij^0 = MLP_edge_embed(d_ij)
        - Node init from mass m_i:      z_i^0  = MLP_node_embed(m_i)
        - n_layers rounds of PotentialMPLayer (residual on z and e)
        - Readouts:
            V_i  = MLP_v(z_i^L)       -> per-node potential (tiny-init)
            V_ij = MLP_e(e_ij^L)      -> per-edge potential, PySR target (tiny-init)
        - Total V = Sum_i V_i + Sum_{i<j} V_ij

    Args:
        hidden_dim: dimension of embeddings (H).
        n_layers: number of message passing rounds.
    """

    def __init__(self, hidden_dim: int, n_layers: int) -> None:
        """Initialize embeddings, message passing stack, and readout MLPs.

        Args:
            hidden_dim: dimension of embeddings (H).
            n_layers: number of message passing rounds.
        """
        super().__init__()
        h = hidden_dim

        # edge init: MLP_edge_embed(d_ij): Linear(1, H) -> SiLU -> Linear(H, H) -> SiLU
        self.mlp_edge_embed = nn.Sequential(
            nn.Linear(1, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )

        # node init: MLP_node_embed(m_i): Linear(1, H) -> SiLU -> Linear(H, H) -> SiLU
        self.mlp_node_embed = nn.Sequential(
            nn.Linear(1, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )

        # stacked message passing
        self.mp_layers = nn.ModuleList([PotentialMPLayer(h) for _ in range(n_layers)])

        # per-node readout V_i = MLP_v(z_i^L) with tiny-init on the final layer
        v_out = nn.Linear(h, 1, bias=False)
        nn.init.xavier_uniform_(v_out.weight, gain=0.001)
        self.mlp_v = nn.Sequential(
            nn.Linear(h, h),
            nn.SiLU(),
            v_out,
        )

        # per-edge readout V_ij = MLP_e(e_ij^L) with tiny-init on the final layer.
        # this is the PySR target: after training, MLP_e(e_ij) should recover V(r) = -G m_i m_j / r
        e_out = nn.Linear(h, 1, bias=False)
        nn.init.xavier_uniform_(e_out.weight, gain=0.001)
        self.mlp_e = nn.Sequential(
            nn.Linear(h, h),
            nn.SiLU(),
            e_out,
        )

    def forward(self, x: Tensor, m: Tensor) -> Tensor:
        """Compute total potential energy V for each sample in the batch.

        Args:
            x: positions, shape (batch, n_particles, 2).
            m: masses, shape (batch, n_particles, 1).

        Returns:
            Scalar potential energy per sample, shape (batch,).
        """
        n = x.size(1)

        # pairwise distances: d_ij = ||x_i - x_j||, with epsilon for autograd stability at r=0
        dx = x.unsqueeze(2) - x.unsqueeze(1)  # (B, N, N, 2)
        d_ij = torch.sqrt((dx**2).sum(dim=-1, keepdim=True) + 1e-8)  # (B, N, N, 1)

        # initial embeddings
        e = self.mlp_edge_embed(d_ij)  # (B, N, N, H)
        z = self.mlp_node_embed(m)  # (B, N, H)

        # self-loop mask (True off-diagonal)
        mask = ~torch.eye(n, dtype=torch.bool, device=x.device)  # (N, N)

        # message passing
        for layer in self.mp_layers:
            z, e = layer(z, e, mask)

        # readouts
        V_i = self.mlp_v(z).squeeze(-1)  # (B, N)
        V_ij = self.mlp_e(e).squeeze(-1)  # (B, N, N)

        # per-node sum
        V_node = V_i.sum(dim=-1)  # (B,)

        # upper-triangle edge sum (avoid double counting V_ij and V_ji)
        i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=x.device)
        V_edge = V_ij[:, i_idx, j_idx].sum(dim=-1)  # (B,)

        return V_node + V_edge


class HGNN(nn.Module):
    """Hamiltonian Graph Neural Network for N-body prediction.

    Predicts next (positions, velocities) via Hamilton's equations:
        H(x, v, m) = T(v, m) + V(x, m)
        dx/dt =  dH/dv = dT/dv
        dv/dt = -dH/dx = -dV/dx
    integrated via a single symplectic leapfrog step (half-kick, drift, half-kick).

    Operates on normalized coordinates internally (x / pos_std, v / vel_std),
    denormalizes on output. Normalization is a pure linear rescale so the
    symplectic structure is preserved.

    Args:
        hidden_dim: dimension of node / edge embeddings.
        n_layers: number of message passing rounds for the V pathway.
        dt: integrator step size (physical time units per forward pass).
        pos_std: standard deviation of positions for normalization.
        vel_std: standard deviation of velocities for normalization.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 4,
        dt: float = 0.05,
        pos_std: float = 1.0,
        vel_std: float = 1.0,
    ) -> None:
        """Initialize kinetic and potential networks and register buffers.

        Args:
            hidden_dim: dimension of embeddings (H).
            n_layers: number of message passing rounds.
            dt: leapfrog step size.
            pos_std: position std for normalization.
            vel_std: velocity std for normalization.
        """
        super().__init__()
        self.kinetic = KineticNetwork(hidden_dim)
        self.potential = PotentialNetwork(hidden_dim, n_layers)

        # non-trainable constants, saved with state_dict
        self.register_buffer("pos_std", torch.tensor(float(pos_std)))
        self.register_buffer("vel_std", torch.tensor(float(vel_std)))
        self.register_buffer("dt", torch.tensor(float(dt)))

    def hamiltonian(self, x: Tensor, v: Tensor, m: Tensor) -> Tensor:
        """Total Hamiltonian H = T + V for each sample.

        Used for energy conservation tests and PySR extraction.

        Args:
            x: positions (normalized), shape (B, N, 2).
            v: velocities (normalized), shape (B, N, 2).
            m: masses, shape (B, N, 1).

        Returns:
            Scalar H per sample, shape (B,).
        """
        return self.kinetic(v, m) + self.potential(x, m)

    def forward(self, state: Tensor) -> Tensor:
        """Predict next state from current state via one leapfrog step.

        Uses torch.enable_grad() internally so Hamilton's equations work even
        when the caller is inside torch.no_grad() (e.g. validation epoch).

        Args:
            state: input tensor of shape (B, N, 5) with features [x, y, vx, vy, m].

        Returns:
            Predicted next state, shape (B, N, 5). Mass column unchanged.
        """
        mass = state[..., 4:]

        with torch.enable_grad():
            # normalize and enable autograd on the leaf tensors
            x = (state[..., :2] / self.pos_std).detach().requires_grad_(True)
            v = (state[..., 2:4] / self.vel_std).detach().requires_grad_(True)

            # first kick: velocity half-step using force at x
            v_dot_0 = self._v_dot(x, mass)
            v_half = v + (self.dt / 2) * v_dot_0

            # drift: position full-step using x_dot at v_half
            x_dot_half = self._x_dot(v_half, mass)
            x_new = x + self.dt * x_dot_half

            # second kick: velocity half-step using force at x_new
            v_dot_1 = self._v_dot(x_new, mass)
            v_new = v_half + (self.dt / 2) * v_dot_1

        # denormalize
        pos_out = x_new * self.pos_std
        vel_out = v_new * self.vel_std

        return torch.cat([pos_out, vel_out, mass], dim=-1)

    def _v_dot(self, x: Tensor, m: Tensor) -> Tensor:
        """Compute dv/dt = -dV/dx at position x via autograd.

        x must be in the autograd graph (either a leaf with requires_grad=True
        or derived from one). create_graph=True retains second-order info so
        loss.backward() can propagate through the integrator.

        Args:
            x: positions, shape (B, N, 2).
            m: masses, shape (B, N, 1).

        Returns:
            dv/dt, same shape as x.
        """
        V = self.potential(x, m).sum()
        dV_dx = torch.autograd.grad(V, x, create_graph=True)[0]
        return -dV_dx

    def _x_dot(self, v: Tensor, m: Tensor) -> Tensor:
        """Compute dx/dt = dT/dv at velocity v via autograd.

        Args:
            v: velocities, shape (B, N, 2).
            m: masses, shape (B, N, 1).

        Returns:
            dx/dt, same shape as v.
        """
        T = self.kinetic(v, m).sum()
        dT_dv = torch.autograd.grad(T, v, create_graph=True)[0]
        return dT_dv
