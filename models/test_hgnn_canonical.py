"""Analytical mechanics, symmetry, graph continuity and checkpoint identity tests."""

import itertools

import pytest
import torch
from torch import Tensor, nn

from models.hgnn import HGNN, PotentialNetwork
from models.hgnn_canonical import MODEL_ID, CanonicalHGNN, SymmetricPotentialNetwork


class AnalyticKinetic(nn.Module):
    """TEST ONLY: undo input scaling and evaluate alpha*p^2/(2m)."""

    def __init__(self, scale: float) -> None:
        """Store the network-input scale and a testable energy multiplier."""
        super().__init__()
        self.scale = scale
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def forward(self, p: Tensor, mass: Tensor) -> Tensor:
        """Return known physical kinetic energy from normalized momentum."""
        return self.alpha * (0.5 * (p * self.scale).square() / mass).sum(dim=(1, 2))


class AnalyticPotential(nn.Module):
    """TEST ONLY: softened Newtonian potential in physical coordinates."""

    def __init__(self, scale: float, *, zero: bool = False) -> None:
        """Store the position scale and optional free-motion fixture flag."""
        super().__init__()
        self.scale = scale
        self.zero = zero

    def forward(self, q: Tensor, mass: Tensor) -> Tensor:
        """Undo input scaling before computing the analytical energy."""
        x = q * self.scale
        energy = x.sum(dim=(1, 2)) * 0
        if self.zero:
            return energy
        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                r2 = (x[:, i] - x[:, j]).square().sum(dim=-1)
                energy = energy - mass[:, i, 0] * mass[:, j, 0] / torch.sqrt(r2 + 0.05**2)
        return energy


def _state() -> Tensor:
    return torch.tensor(
        [[[0, 0, 0.2, 0.1, 0.5], [1, 0.3, -0.1, 0.3, 2], [-0.4, 0.8, 0.1, -0.2, 1.3]]],
        dtype=torch.float64,
    )


def _analytic(pos_std: float = 1, momentum_std: float = 1, *, zero: bool = False) -> CanonicalHGNN:
    model = CanonicalHGNN(8, 2, dt=0.001, pos_std=pos_std, momentum_std=momentum_std).double()
    model.kinetic = AnalyticKinetic(momentum_std)
    model.potential = AnalyticPotential(pos_std, zero=zero)
    return model


def _acceleration(x: Tensor, mass: Tensor) -> Tensor:
    delta = x[:, None, :, :] - x[:, :, None, :]
    denominator = (delta.square().sum(-1) + 0.05**2).pow(1.5)
    return (mass[:, None, :, :] * delta / denominator[..., None]).sum(dim=2)


@pytest.mark.parametrize(("pos_std", "momentum_std"), [(1, 1), (2.7, 0.4), (0.13, 5.2)])
@pytest.mark.parametrize("n_bodies", [2, 3])
def test_analytic_unequal_mass_verlet(pos_std: float, momentum_std: float, n_bodies: int) -> None:
    """Recover velocity Verlet including the 1/m acceleration factor and scale chain rule."""
    state = _state()[:, :n_bodies]
    model = _analytic(pos_std, momentum_std)
    x, v, mass = state[..., :2], state[..., 2:4], state[..., 4:]
    half = v + 0.0005 * _acceleration(x, mass)
    q_next = x + 0.001 * half
    v_next = half + 0.0005 * _acceleration(q_next, mass)
    expected = torch.cat([q_next, v_next, mass], dim=-1)
    torch.testing.assert_close(model(state), expected, rtol=1e-12, atol=1e-13)
    initial_p = (mass * v).sum(dim=1)
    actual = model(state)
    torch.testing.assert_close(
        (actual[..., 4:] * actual[..., 2:4]).sum(dim=1), initial_p, rtol=1e-12, atol=1e-13
    )


def test_free_motion_and_conversion_round_trip() -> None:
    """Different masses have the correct same kinematic velocity, not m*v drift."""
    state = _state()
    model = _analytic(3.1, 0.27, zero=True)
    expected = state.clone()
    expected[..., :2] += 0.001 * state[..., 2:4]
    torch.testing.assert_close(model(state), expected, rtol=1e-13, atol=1e-14)
    p = model.to_momentum(state[..., 2:4], state[..., 4:])
    torch.testing.assert_close(model.to_velocity(p, state[..., 4:]), state[..., 2:4])
    assert torch.equal(model(state)[..., 4:], state[..., 4:])


def test_energy_interface_uses_physical_coordinates() -> None:
    """The public energy/hamiltonian interface follows the same canonical convention."""
    state = _state()
    model = _analytic(2.7, 0.4)
    mass = state[..., 4:]
    p = mass * state[..., 2:4]
    kinetic, potential = model.energies(state)
    torch.testing.assert_close(kinetic, (0.5 * mass * state[..., 2:4].square()).sum((1, 2)))
    torch.testing.assert_close(kinetic + potential, model.hamiltonian(state[..., :2], p, mass))


def test_legacy_permutation_problem_is_reproduced() -> None:
    """Characterize the old directed-upper-triangle behaviour, without editing it."""
    torch.manual_seed(7)
    model = PotentialNetwork(8, 2).double()
    state = _state()
    permutation = [2, 0, 1]
    original = model(state[..., :2], state[..., 4:])
    reordered = model(state[:, permutation, :2], state[:, permutation, 4:])
    assert (original - reordered).abs().item() > 1e-10


def test_symmetric_readout_and_all_permutations() -> None:
    """Scalar energy is invariant; derivatives and dynamics permute with bodies."""
    torch.manual_seed(7)
    model = CanonicalHGNN(8, 2, pos_std=2.7, momentum_std=0.4).double()
    state = _state()
    initial = model(state)
    energy = sum(model.energies(state))
    parts = model.potential.decompose(state[..., :2] / model.pos_std, state[..., 4:])
    torch.testing.assert_close(parts.v_ij, parts.v_ij.transpose(1, 2), rtol=0, atol=0)
    for permutation in itertools.permutations(range(3)):
        ids = list(permutation)
        reordered = state[:, ids]
        torch.testing.assert_close(sum(model.energies(reordered)), energy, rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(model(reordered), initial[:, ids], rtol=1e-12, atol=1e-13)


def test_parameter_count_is_unchanged() -> None:
    """Pair symmetrization and canonical coordinates add no trainable parameters."""
    legacy, canonical = HGNN(8, 2), CanonicalHGNN(8, 2)
    assert sum(p.numel() for p in legacy.parameters()) == sum(
        p.numel() for p in canonical.parameters()
    )
    assert isinstance(canonical.potential, SymmetricPotentialNetwork)


def test_potential_spatial_invariance() -> None:
    """Distance-based V remains invariant under translation and planar rotation."""
    torch.manual_seed(11)
    model = CanonicalHGNN(8, 2).double()
    state = _state()
    x, mass = state[..., :2], state[..., 4:]
    rotated = torch.stack([-x[..., 1], x[..., 0]], dim=-1)
    torch.testing.assert_close(
        model.potential(x, mass), model.potential(rotated + 3, mass), rtol=1e-12, atol=1e-14
    )


def test_terminal_loss_reaches_all_earlier_steps() -> None:
    """An alpha-scaled free drift gives an exact multi-step parameter derivative."""
    state = _state().requires_grad_(True)
    model = _analytic(2.7, 0.4, zero=True)
    first = model(state)
    first.retain_grad()
    third = model(model(first))
    loss = third[..., :2].sum()
    loss.backward()
    expected = 3 * 0.001 * state[..., 2:4].detach().sum()
    torch.testing.assert_close(model.kinetic.alpha.grad, expected, rtol=1e-12, atol=1e-14)
    assert first.grad is not None
    torch.testing.assert_close(first.grad[..., :2], torch.ones_like(first[..., :2]))
    torch.testing.assert_close(state.grad[..., :2], torch.ones_like(state[..., :2]))


def test_rollout_derivative_matches_finite_difference() -> None:
    """Terminal gradients agree with perturbing the shared parameter across a rollout."""
    model = _analytic(2.1, 0.7, zero=True)
    state = _state()

    def objective() -> Tensor:
        current = state
        for _ in range(4):
            current = model(current)
        return current[..., :2].square().sum()

    derivative = torch.autograd.grad(objective(), model.kinetic.alpha)[0]
    with torch.no_grad():
        model.kinetic.alpha.fill_(1 + 1e-5)
        high = objective()
        model.kinetic.alpha.fill_(1 - 1e-5)
        low = objective()
        model.kinetic.alpha.fill_(1)
    torch.testing.assert_close(derivative, (high - low) / 2e-5, rtol=1e-7, atol=1e-10)


def test_learned_network_backward_and_no_grad() -> None:
    """All learned parameters train through a short rollout; no_grad inference works."""
    torch.manual_seed(21)
    model = CanonicalHGNN(8, 2, pos_std=2.1, momentum_std=0.7).double()
    state = _state()
    expected = model(model(state))
    expected[..., :4].square().mean().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    with torch.no_grad():
        actual = model(model(state))
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected.detach(), rtol=1e-12, atol=1e-13)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        model(state)


@pytest.mark.parametrize("strict", [True, False])
def test_legacy_checkpoint_rejected(strict: bool) -> None:
    """Identically shaped network weights cannot hide different coordinate semantics."""
    with pytest.raises(RuntimeError, match="legacy HGNN weights"):
        CanonicalHGNN(8, 2).load_state_dict(HGNN(8, 2).state_dict(), strict=strict)


def test_versioned_checkpoint_round_trip() -> None:
    """Restore learned parameters, exact scales, dt and coordinate identity."""
    model = CanonicalHGNN(8, 2, pos_std=2.7, momentum_std=0.4).double()
    restored = CanonicalHGNN(8, 2).double()
    restored.load_state_dict(model.state_dict())
    assert restored.get_extra_state()["model_id"] == MODEL_ID
    assert restored.pos_std.item() == 2.7
    assert restored.momentum_std.item() == 0.4
    torch.testing.assert_close(restored(_state()), model(_state()))


@pytest.mark.parametrize("name", ["dt", "pos_std", "momentum_std"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_scales(name: str, value: float) -> None:
    """Reject invalid coordinate scales and step sizes at construction."""
    with pytest.raises(ValueError, match="finite and positive"):
        CanonicalHGNN(**{name: value})


def test_invalid_mass() -> None:
    """Reject zero mass before converting momentum back into velocity."""
    state = _state()
    state[0, 0, 4] = 0
    with pytest.raises(ValueError, match="masses"):
        CanonicalHGNN(8, 2).double()(state)
