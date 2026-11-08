"""Versioned, mass-compatible HGNN; the original models.hgnn remains unchanged.

External state: [x,y,vx,vy,m]. Internal integration: physical positions and
canonical momenta p=m*v. Network inputs are scaled, but differentiation is with
respect to PHYSICAL coordinates, including the normalization chain rule.
"""

import math

import torch
from torch import Tensor, nn

from models.hgnn import KineticNetwork, PotentialNetwork, PotentialParts

MODEL_ID = "hgnn_canonical_v1"


class SymmetricPotentialNetwork(PotentialNetwork):
    """Keep the original message passing, averaging both directed pair readouts."""

    def decompose(self, x: Tensor, m: Tensor) -> PotentialParts:
        """Return permutation-invariant energy and symmetric per-pair contributions."""
        parts = super().decompose(x, m)
        symmetric = 0.5 * (parts.v_ij + parts.v_ij.transpose(1, 2))
        i, j = torch.triu_indices(x.size(1), x.size(1), offset=1, device=x.device)
        edge = symmetric[:, i, j].sum(dim=-1)
        return PotentialParts(
            total=parts.v_node + edge,
            v_node=parts.v_node,
            v_edge=edge,
            v_i=parts.v_i,
            v_ij=symmetric,
            d_ij=parts.d_ij,
        )


class CanonicalHGNN(nn.Module):
    """Learn T(p,m)+V(x,m) and advance physical x/p by kick-drift-kick.

    Reuses the original learned kinetic MLP (same 3 inputs), now on p/p_std and
    m rather than v/v_std and m. No analytic kinetic energy or force supervision
    is introduced. A learned physical-coordinate Hamiltonian still need not equal
    the true physical energy, nor does finite-step leapfrog conserve it exactly.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 4,
        dt: float = 0.001,
        pos_std: float = 1.0,
        momentum_std: float = 1.0,
    ) -> None:
        """Create learned energy branches and persistent physical-input scales."""
        super().__init__()
        for name, value in (("dt", dt), ("pos_std", pos_std), ("momentum_std", momentum_std)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            self.register_buffer(name, torch.tensor(float(value), dtype=torch.float64))
        self.kinetic = KineticNetwork(hidden_dim)
        self.potential = SymmetricPotentialNetwork(hidden_dim, n_layers)
        self.register_load_state_dict_pre_hook(self._check_identity)

    def get_extra_state(self) -> dict:
        """Save an explicit coordinate/version identity in every state dict."""
        return {
            "model_id": MODEL_ID,
            "coordinates": "physical_x_p",
            "pair_readout": "symmetric_mean",
        }

    def set_extra_state(self, state: dict) -> None:
        """Refuse a checkpoint with different coordinate or readout semantics."""
        if state != self.get_extra_state():
            raise RuntimeError(f"incompatible canonical HGNN identity: {state!r}")

    def _check_identity(
        self,
        module: nn.Module,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Reject legacy/unversioned weights even if a caller requests strict=False."""
        if state_dict.get(prefix + "_extra_state") != self.get_extra_state():
            error_msgs.append(
                f"{prefix or MODEL_ID}: expected versioned physical-x/p checkpoint; legacy HGNN weights cannot be silently reused"
            )
        for name in ("dt", "pos_std", "momentum_std"):
            value = state_dict.get(prefix + name)
            if (
                not isinstance(value, Tensor)
                or value.numel() != 1
                or not torch.isfinite(value).all()
                or torch.any(value <= 0)
            ):
                error_msgs.append(
                    f"{prefix + name}: checkpoint scale must be a finite positive scalar"
                )

    @staticmethod
    def to_momentum(velocity: Tensor, mass: Tensor) -> Tensor:
        """Convert external velocity to canonical momentum without breaking gradients."""
        return mass * velocity

    @staticmethod
    def to_velocity(momentum: Tensor, mass: Tensor) -> Tensor:
        """Convert updated canonical momentum back to external velocity."""
        return momentum / mass

    def hamiltonian(self, x: Tensor, p: Tensor, m: Tensor) -> Tensor:
        """Evaluate H on PHYSICAL x/p, normalizing only the branch inputs."""
        return self.kinetic(p / self.momentum_std, m) + self.potential(x / self.pos_std, m)

    def energies(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Return learned T/V from the unnormalized external state."""
        mass = state[..., 4:]
        p = self.to_momentum(state[..., 2:4], mass)
        return self.kinetic(p / self.momentum_std, mass), self.potential(
            state[..., :2] / self.pos_std, mass
        )

    def _force(self, x: Tensor, m: Tensor, *, create_graph: bool) -> Tensor:
        energy = self.potential(x / self.pos_std, m).sum()
        return -torch.autograd.grad(energy, x, create_graph=create_graph)[0]

    def _velocity(self, p: Tensor, m: Tensor, *, create_graph: bool) -> Tensor:
        energy = self.kinetic(p / self.momentum_std, m).sum()
        return torch.autograd.grad(energy, p, create_graph=create_graph)[0]

    def forward(self, state: Tensor) -> Tensor:
        """Preserve full rollout graphs in training; support no_grad evaluation.

        A tensor need not be a leaf to serve as an autograd.grad input. Never
        detach a state that already belongs to the previous rollout's graph.
        inference_mode is explicitly unsupported because spatial derivatives
        still require autograd, even when no parameter gradients are requested.
        """
        if torch.is_inference_mode_enabled():
            raise RuntimeError(
                "CanonicalHGNN requires autograd; use torch.no_grad(), not inference_mode()"
            )
        if state.ndim != 3 or state.shape[1] < 2 or state.shape[2] != 5:
            raise ValueError("expected external state (batch, bodies>=2, 5)")
        track_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            mass = state[..., 4:]
            if not torch.isfinite(mass).all() or torch.any(mass <= 0):
                raise ValueError("masses must be finite and positive")
            x = state[..., :2]
            p = self.to_momentum(state[..., 2:4], mass)
            if not x.requires_grad:
                x = x.requires_grad_(True)
            if not p.requires_grad:
                p = p.requires_grad_(True)
            p_half = p + self.dt * 0.5 * self._force(x, mass, create_graph=track_graph)
            x_new = x + self.dt * self._velocity(p_half, mass, create_graph=track_graph)
            p_new = p_half + self.dt * 0.5 * self._force(x_new, mass, create_graph=track_graph)
            result = torch.cat([x_new, self.to_velocity(p_new, mass), mass], dim=-1)
        return result if track_graph else result.detach()
