"""Isolated normalized-gravity HGNN; frozen shared training/models unchanged.

L_total = L_traj + 0.1 * MSE(a_hat(x0)-a_grav(x0)) / TRAIN_acceleration_mean_square.
Initial-state acceleration a_hat = -dV/dx / mass; additive softening 0.05**2.
The denominator is an uncentred TRAIN-only mean square, not a centred variance.
"""
import hashlib
from pathlib import Path
import numpy as np
import torch

from training.cpu_microbatch import CpuMicrobatchTrainer
from training.fixed_validation import file_sha256

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
BANK = ROOT / "runs/daniela-hgnn-order-reconstruction-audit-v1/historical-assigned-batches.npz"
ORDER = WORK / "staged-order.npy"
CONFIG = ROOT / "configs/daniela-hgnn-gravity-norm-v1.yaml"
ADDITIVE_SOFTENING = 0.0025000000000000005
PHYSICS_WEIGHT = 0.1
import json
ACCELERATION_MEAN_SQUARE = json.loads((WORK / "acceleration_scale.json").read_text())["mean_square"]


def analytical_gravity_acceleration(state: torch.Tensor, additive: float = ADDITIVE_SOFTENING) -> torch.Tensor:
    """Exact pairwise Newtonian gravitational acceleration with additive softening s.

    External HGNN state is mass-last: [x, y, vx, vy, m] of shape (batch, N, 5), G = 1.0.
    For body i: a_i = sum_{j != i} m_j (x_j - x_i) / (||x_j - x_i||^2 + additive)^(3/2).
    """
    if state.ndim != 3 or state.shape[1] < 2 or state.shape[2] != 5:
        raise ValueError("expected state shape (batch, bodies>=2, 5)")
    if not np.isfinite(additive) or additive <= 0:
        raise ValueError("additive softening must be finite and positive")
    pos = state[..., :2]
    mass = state[..., 4:]
    n = pos.shape[1]
    # delta[:, i, j, :] = pos[:, j, :] - pos[:, i, :]
    delta = pos.unsqueeze(1) - pos.unsqueeze(2)
    radius2 = torch.sum(delta.square(), dim=-1, keepdim=True) + additive
    inv_r3 = (radius2 * torch.sqrt(radius2)).reciprocal()
    mass_j = mass.unsqueeze(1)
    mask = (1.0 - torch.eye(n, dtype=state.dtype, device=state.device)).view(1, n, n, 1)
    return torch.sum(mass_j * delta * inv_r3 * mask, dim=2)


def predicted_hgnn_acceleration(model: torch.nn.Module, state: torch.Tensor, *, create_graph: bool) -> torch.Tensor:
    """Conservative acceleration a_hat_i = F_i / m_i = -(1/m_i) dV(x, m)/dx_i at state."""
    mass = state[..., 4:].detach()
    x = state[..., :2].detach().requires_grad_(True)
    with torch.enable_grad():
        force = model._force(x, mass, create_graph=create_graph)
    return force / mass


class AssignedLoader:
    """One prepared 64-example unit per logical optimizer update, in staged identity order."""
    def __init__(self, trainer, dataset):
        self.trainer = trainer
        self.dataset = dataset

    def __len__(self):
        return self.trainer.units_per_round

    def __iter__(self):
        trainer = self.trainer
        first = trainer.assigned_completed
        expected = len(trainer._progress.get("train_history", [])) * trainer.units_per_round
        if first != expected:
            raise RuntimeError("Schedule cursor disagrees with completed rounds")
        if first + len(self) > len(trainer.order):
            raise RuntimeError("Assigned schedule exhausted")
        torch.empty((), dtype=torch.int64).random_()
        for position in range(first, first + len(self)):
            unit = int(trainer.order[position])
            inputs, targets = trainer.assigned_inputs(unit)
            trainer.current_horizon = int(trainer.bank["horizon"][unit])
            trainer.assigned_completed = position + 1
            yield inputs, targets


class GravityLossHGNNTrainer(CpuMicrobatchTrainer):
    def __init__(self, cfg, *, micro_batch_size=16, resume_checkpoint=None, probe_order=None):
        with np.load(BANK, allow_pickle=False) as z:
            self.bank = {k: z[k] for k in ["horizon", "trajectory_index", "start_frame"]}
        self.probe = probe_order is not None
        self.order = np.asarray(probe_order, dtype=np.int64) if self.probe else np.load(ORDER, allow_pickle=False)
        self.units_per_round = cfg.training.max_train_batches_per_epoch
        if not self.units_per_round or len(self.order) % self.units_per_round:
            raise ValueError("Whole rounds required")
        if self.order.ndim != 1 or np.any(self.order < 0) or np.any(self.order >= 2500):
            raise ValueError("Invalid schedule indices")
        if cfg.training.batch_size != 64:
            raise ValueError("Recovered units require logical batch 64")
        if not self.probe and not np.array_equal(self.order, np.arange(2500)):
            raise ValueError("Staged gravity-loss arm requires identity order 0..2499")
        if cfg.training.noise_factor or cfg.training.reset_optimizer_on_stage:
            raise ValueError("Noise/reset differs from historical recipe")
        self.additive_softening = float(ADDITIVE_SOFTENING)
        self.physics_weight = float(PHYSICS_WEIGHT)
        self.assigned_completed = 0
        self.power_or_signal_check = None
        super().__init__(cfg, micro_batch_size=micro_batch_size, resume_checkpoint=resume_checkpoint)
        self.assigned_completed = self._progress.get("assigned_batches_completed", 0)
        if self.assigned_completed != self.attempted_train_batches:
            raise ValueError("Saved schedule cursor/update mismatch")
        if self.assigned_completed != len(self._progress.get("train_history", [])) * self.units_per_round:
            raise ValueError("Not a committed schedule round")
        self._assigned_loader = AssignedLoader(self, self.train_loader.dataset)
        self.train_loader = self._assigned_loader

    def _execution_identity(self, cfg):
        return {
            **super()._execution_identity(cfg),
            "order_protocol": "historical-staged-normalized-gravity-loss-v1",
            "acceleration_mean_square": ACCELERATION_MEAN_SQUARE,
            "acceleration_scale_sha256": file_sha256(WORK / "acceleration_scale.json"),
            "order_sha256": hashlib.sha256(self.order.astype("<i8").tobytes()).hexdigest(),
            "bank_sha256": file_sha256(BANK),
            "units_per_round": self.units_per_round,
            "additive_softening": self.additive_softening,
            "physics_weight": self.physics_weight,
            "disposable_probe": self.probe,
        }

    def _setup_data(self):
        result = super()._setup_data()
        for path in [Path(__file__), WORK / "runner.py", CONFIG, BANK, ORDER, WORK / "PREREGISTRATION.json", WORK / "acceleration_scale.json"]:
            self.source_hashes[str(path.relative_to(ROOT))] = file_sha256(path)
        return result

    def _stages(self):
        return [(1, len(self.order) // self.units_per_round)]

    def _setup_loaders_for_horizon(self, horizon):
        if hasattr(self, "_assigned_loader"):
            return self._assigned_loader, self.val_loader
        return super()._setup_loaders_for_horizon(horizon)

    def assigned_inputs(self, unit):
        dataset = self._assigned_loader.dataset if hasattr(self, "_assigned_loader") else self.train_loader.dataset
        states = dataset.states
        h = int(self.bank["horizon"][unit])
        trajectories = self.bank["trajectory_index"][unit]
        starts = self.bank["start_frame"][unit]
        inputs = torch.stack([states[int(i), int(t)] for i, t in zip(trajectories, starts, strict=True)])
        targets = torch.stack([states[int(i), int(t) + 1 : int(t) + 1 + h] for i, t in zip(trajectories, starts, strict=True)])
        if targets.shape[1] != h:
            raise RuntimeError("Invalid assigned window")
        return inputs, targets[:, 0] if h == 1 else targets

    def _compute_loss(self, inputs: torch.Tensor, targets: torch.Tensor):
        preds, traj_loss, diag_targets = super()._compute_loss(inputs, targets)
        pred_acc = predicted_hgnn_acceleration(self.model, inputs, create_graph=torch.is_grad_enabled())
        target_acc = analytical_gravity_acceleration(inputs.detach(), additive=self.additive_softening)
        phys_loss = torch.mean((pred_acc - target_acc).square()) / ACCELERATION_MEAN_SQUARE
        total_loss = traj_loss + self.physics_weight * phys_loss
        return preds, total_loss, diag_targets

    def _backward_logical_batch(self, inputs, targets):
        if self.power_or_signal_check:
            self.power_or_signal_check()
        return super()._backward_logical_batch(inputs, targets)

    def _checkpoint(self, epoch, **kwargs):
        if self.assigned_completed != epoch * self.units_per_round:
            raise RuntimeError("Checkpoint schedule position mismatch")
        self._progress["assigned_batches_completed"] = self.assigned_completed
        return super()._checkpoint(epoch, **kwargs)
