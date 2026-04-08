"""
HGNN — Hamiltonian Graph Neural Network with decomposed T + V (Bishnoi et al., 2023).

Learns H = T(v) + V(x) via separate pathways, derives dynamics via Hamilton's
equations (autodiff), and integrates with a symplectic Leapfrog step.
"""

import torch
import torch.nn as nn


class KineticNetwork(nn.Module):
    """Computes total kinetic energy T = sum_i MLP_T(v_i, m_i)."""

    def __init__(self, hidden_dim: int) -> None:
        pass

    def forward(self, v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Returns scalar T."""
        pass


class PotentialNetwork(nn.Module):
    """
    Computes total potential energy V via 4-layer GNN on pairwise distances.
    V = sum_i V_i(node) + sum_{i<j} V_ij(edge)
    """

    def __init__(self, hidden_dim: int, n_layers: int) -> None:
        pass

    def forward(
        self,
        x: torch.Tensor,     # (N, 2) positions
        t: torch.Tensor,     # (N, K) particle type embeddings
        edges: torch.Tensor, # (2, E) edge index
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (V_scalar, edge_embeddings) — edge_embeddings used for PySR extraction."""
        pass


class HGNN(nn.Module):
    """
    Full HGNN: computes H = T + V, derives Hamilton's equations via autograd,
    and steps forward using a symplectic Leapfrog integrator.

    Input state: (n_particles, 4) — columns [x, y, vx, vy]
    Output state: (n_particles, 4) — predicted next state
    """

    def __init__(self, hidden_dim: int, n_layers: int, dt: float) -> None:
        pass

    def hamiltonian(self, state: torch.Tensor) -> torch.Tensor:
        """Returns scalar H = T + V. Requires grad on state."""
        pass

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        pass

    def extract_potentials(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (r, V_ij) pairs from the edge potential MLP across all edges.
        Used as input to PySR for symbolic regression.
        """
        pass
