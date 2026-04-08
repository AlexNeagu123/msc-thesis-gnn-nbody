"""
EGNN — Equivariant Graph Neural Network (Satorras et al., 2021).

Predicts next positions and velocities directly via 4 EGCL layers.
"""

import torch
import torch.nn as nn


class EGCL(nn.Module):
    """Single Equivariant Graph Convolutional Layer."""

    def __init__(self, hidden_dim: int) -> None:
        pass

    def forward(
        self,
        h: torch.Tensor,   # (N, hidden_dim) node features
        x: torch.Tensor,   # (N, 2) positions
        v: torch.Tensor,   # (N, 2) velocities
        edges: torch.Tensor,  # (2, E) edge index
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns updated (h, x, v)."""
        pass


class EGNN(nn.Module):
    """
    Stack of n_layers EGCL layers.

    Input state: (n_particles, 4) — columns [x, y, vx, vy]
    Output state: (n_particles, 4) — predicted next state
    """

    def __init__(self, hidden_dim: int, n_layers: int) -> None:
        pass

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        pass

    def extract_weights(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (r, w) pairs from MLP_x across all edges in the batch.
        Used as input to PySR for symbolic regression.
        """
        pass
