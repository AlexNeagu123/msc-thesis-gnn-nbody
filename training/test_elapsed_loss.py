"""Synthetic tests for the opt-in time-scaled canonical rollout objective."""

from dataclasses import replace

import pytest
import torch

from training._types import TrainConfig
from training.test_canonical_hgnn import canonical_cfg as canonical_cfg
from training.train import Trainer


@pytest.mark.parametrize("step", [1, 5, 20])
def test_elapsed_loss_formula_and_mass_exclusion(canonical_cfg: TrainConfig, step: int) -> None:
    """Match the analytical residual scaling and exclude the mass channel."""
    cfg = replace(
        canonical_cfg,
        training=replace(canonical_cfg.training, dynamic_loss_scaling="elapsed_time_v1"),
    )
    trainer = Trainer(cfg)
    predicted = torch.zeros(2, 3, 5, requires_grad=True)
    target = torch.tensor([0.02, 0.02, 0.3, 0.3, 100.0]).expand_as(predicted)
    loss = trainer._dynamic_step_loss(predicted, target, step=step)
    expected = 0.5 * (
        (0.02 / (step * cfg.data.dt * trainer.vel_std)) ** 2 + (0.3 / trainer.vel_std) ** 2
    )
    assert float(loss.detach()) == pytest.approx(expected, rel=1e-6)
    gradient = torch.autograd.grad(loss, predicted)[0]
    assert torch.equal(gradient[..., 4], torch.zeros_like(gradient[..., 4]))
    assert torch.isfinite(gradient).all()
    assert gradient[..., :4].abs().sum() > 0


def test_elapsed_rollout_matches_explicit_weighted_objective(canonical_cfg: TrainConfig) -> None:
    """Check per-future elapsed times and full kinetic parameter gradients."""
    cfg = replace(
        canonical_cfg,
        training=replace(
            canonical_cfg.training, dynamic_loss_scaling="elapsed_time_v1", multi_step_gamma=0.7
        ),
    )
    trainer = Trainer(cfg)
    trainer.current_horizon = 3
    inputs = trainer.train_loader.dataset.states[:2, 0]
    targets = torch.stack([inputs, inputs, inputs], dim=1)
    _, actual = trainer._multi_step_rollout_loss(inputs, targets)
    state = inputs
    losses = []
    for step in range(1, 4):
        state = trainer.model(state)
        error = state[..., :4] - targets[:, step - 1, ..., :4]
        losses.append(
            0.7 ** (step - 1)
            * 0.5
            * (
                (error[..., :2] / (step * cfg.data.dt * trainer.vel_std)).square().mean()
                + (error[..., 2:4] / trainer.vel_std).square().mean()
            )
        )
    expected = sum(losses) / sum(0.7**k for k in range(3))
    torch.testing.assert_close(actual, expected)
    gradients = torch.autograd.grad(
        actual, list(trainer.model.kinetic.parameters()), allow_unused=True
    )
    assert any(g is not None and g.abs().sum() > 0 for g in gradients)


def test_elapsed_loss_rejects_legacy_model_and_bad_scale(canonical_cfg: TrainConfig) -> None:
    """Reject unsupported models, losses, names and invalid training scales."""
    cfg = replace(
        canonical_cfg,
        training=replace(canonical_cfg.training, dynamic_loss_scaling="elapsed_time_v1"),
    )
    with pytest.raises(ValueError, match="canonical HGNN"):
        Trainer(replace(cfg, model=replace(cfg.model, name="hgnn")))
    trainer = Trainer(cfg)
    trainer.vel_std = 0.0
    with pytest.raises(ValueError, match="velocity scale"):
        trainer._dynamic_step_loss(torch.zeros(2, 3, 5), torch.zeros(2, 3, 5), step=1)
    with pytest.raises(ValueError, match="unknown dynamic_loss_scaling"):
        replace(cfg.training, dynamic_loss_scaling="typo")
    with pytest.raises(ValueError, match="requires mse"):
        replace(cfg.training, loss="mae")
