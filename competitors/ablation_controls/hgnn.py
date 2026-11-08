"""Exact/learned kinetic HGNN and differentiable objective; frozen model untouched."""
# ruff: noqa: ANN001, ANN003, ANN201, ANN204, D101, D102, D103, D107

import torch
from torch import nn

from models.hgnn_canonical import CanonicalHGNN

ACCELERATION_MEAN_SQUARE = 882.8257392887597


class PhysicalKinetic(nn.Module):
    """Accept the parent's normalized p but return physical sum p^2/(2m)."""

    def __init__(self, momentum_std):
        super().__init__()
        self.register_buffer("momentum_std", torch.tensor(float(momentum_std), dtype=torch.float64))

    def decompose(self, normalized_p, mass):
        p = normalized_p * self.momentum_std
        return 0.5 * (p.square() / mass).sum(-1)

    def forward(self, normalized_p, mass):
        return self.decompose(normalized_p, mass).sum(-1)


class KineticAblationHGNN(CanonicalHGNN):
    def __init__(self, *, kinetic="exact", **kwargs):
        if kinetic not in {"learned", "exact"}:
            raise ValueError("unknown kinetic prior")
        self.kinetic_mode = kinetic
        # Always construct the complete parent first: removing T must not shift potential RNG.
        super().__init__(**kwargs)
        if kinetic == "exact":
            self.kinetic = PhysicalKinetic(float(self.momentum_std))

    def get_extra_state(self):
        return {
            **super().get_extra_state(),
            "model_id": "hgnn_kinetic_ablation_v1",
            "kinetic": self.kinetic_mode,
        }


def acceleration_target(state):
    q = state[..., :2]
    m = state[..., 4:]
    d = q.unsqueeze(1) - q.unsqueeze(2)
    radius2 = d.square().sum(-1, keepdim=True) + 0.05**2
    return (m.unsqueeze(1) * d / radius2.pow(1.5)).sum(2)


def objective(model, initial, targets, velocity_std, gravity="off"):
    """Mean full-BPTT elapsed-time trajectory loss plus initial-state-only force term.

    targets:[batch,horizon,bodies,5], mass-last. No optimizer/sampler side effects.
    """
    if gravity not in {"off", "scaled"}:
        raise ValueError("only off/scaled gravity registered")
    if targets.ndim != 4 or targets.shape[1] < 1 or velocity_std <= 0:
        raise ValueError("invalid target/scaling")
    state = initial
    trajectory = initial.new_zeros(())
    for k in range(1, targets.shape[1] + 1):
        state = model(state)
        error = state[..., :4] - targets[:, k - 1, :, :4]
        trajectory = (
            trajectory
            + 0.5
            * (
                (error[..., :2] / (k * float(model.dt) * velocity_std)).square().mean()
                + (error[..., 2:] / velocity_std).square().mean()
            )
            / targets.shape[1]
        )
    physics = trajectory.new_zeros(())
    if gravity == "scaled":
        x = initial[..., :2].detach().requires_grad_(True)
        mass = initial[..., 4:].detach()
        predicted = model._force(x, mass, create_graph=True) / mass
        physics = (
            predicted - acceleration_target(initial.detach())
        ).square().mean() / ACCELERATION_MEAN_SQUARE
    return {"trajectory": trajectory, "physics": physics, "total": trajectory + 0.1 * physics}
