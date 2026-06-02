"""Baseline-normalized rollout scoring for checkpoint selection.

Per-step ratio R_s = (model_median_mse_s + eps) / (baseline_envelope_s + eps);
scalar score = mean_s log(R_s + eps), lower is better.

Curves are indexed by step s = 1..N and must exclude step 0 (trivially zero MSE);
anchor step s reads ratios[s - 1].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from data._io import read_states, read_trajectories
from evaluation._binning import trajectory_masks
from evaluation.metrics import compute_rollout_mse, run_all_rollouts, subset_rollout_mse
from evaluation.rollout_score import BaselineEnvelopeComputer
from models.baselines import (
    ConstantVelocityBaseline,
    MeanStateBaseline,
    MeanVelocityBaseline,
    PersistenceBaseline,
)
from training._types import BucketRolloutScore, RolloutScore

DEFAULT_ANCHOR_STEPS: tuple[int, ...] = (10, 20, 50, 100, 199)
DEFAULT_EPS: float = 1e-12


def compute_rollout_score(
    model_curve: npt.NDArray[np.floating],
    baseline_envelope: npt.NDArray[np.floating],
    *,
    eps: float = DEFAULT_EPS,
    anchor_steps: tuple[int, ...] = DEFAULT_ANCHOR_STEPS,
) -> RolloutScore:
    """Compute the baseline-normalized rollout score and step-resolved diagnostics.

    `model_curve` and `baseline_envelope` are matching 1-D per-step curves; anchor_steps
    are 1-indexed and silently dropped past the curve length.
    """
    if model_curve.shape != baseline_envelope.shape:
        msg = (
            f"shape mismatch: model_curve {model_curve.shape} vs "
            f"baseline_envelope {baseline_envelope.shape}"
        )
        raise ValueError(msg)
    if model_curve.ndim != 1:
        msg = f"expected 1-D curves, got shape {model_curve.shape}"
        raise ValueError(msg)
    if model_curve.size == 0:
        msg = "model_curve and baseline_envelope must contain at least one step"
        raise ValueError(msg)

    # eps in both numerator and denominator so exact ties produce R = 1.0 exactly,
    # otherwise an eps-shift in only the denominator would count ties as wins.
    ratios = (np.asarray(model_curve, dtype=np.float64) + eps) / (
        np.asarray(baseline_envelope, dtype=np.float64) + eps
    )
    score = float(np.mean(np.log(ratios + eps)))

    beats_baseline = ratios < 1.0
    fraction_beating_baseline = float(beats_baseline.mean())

    # longest prefix where every R_s < 1: index of the first failure, or full length if none
    false_indices = np.where(~beats_baseline)[0]
    dominance_horizon = len(ratios) if len(false_indices) == 0 else int(false_indices[0])

    ratios_at_step: dict[int, float] = {}
    for s in anchor_steps:
        if 1 <= s <= len(ratios):
            ratios_at_step[s] = float(ratios[s - 1])

    final_ratio = float(ratios[-1])

    return RolloutScore(
        score=score,
        ratios=ratios,
        dominance_horizon=dominance_horizon,
        fraction_beating_baseline=fraction_beating_baseline,
        final_ratio=final_ratio,
        ratios_at_step=ratios_at_step,
    )


class RolloutScoreEvaluator:
    """Score a model against a cached baseline envelope on a fixed val set.

    The envelope and val trajectories are built lazily once and reused for every call;
    the caller owns model device placement.
    """

    def __init__(
        self,
        val_path: str | Path,
        train_path: str | Path,
        dt: float,
        device: torch.device,
    ) -> None:
        """Store paths and lazy-compute caches; no I/O happens here."""
        self.val_path = Path(val_path)
        self.train_path = Path(train_path)
        self.dt = float(dt)
        self.device = device
        self._val_traj: np.ndarray | None = None
        self._envelope: np.ndarray | None = None

    @property
    def val_traj(self) -> np.ndarray:
        """Validation trajectories loaded once on first access."""
        if self._val_traj is None:
            self._val_traj = read_states(self.val_path)
        return self._val_traj

    @property
    def baseline_envelope(self) -> np.ndarray:
        """Per-step minimum median state MSE across baselines, steps 1..N (cached)."""
        if self._envelope is None:
            self._envelope = self._build_envelope()
        return self._envelope

    def score(self, model: nn.Module) -> RolloutScore:
        """Run the model on val and return its rollout score; restores train/eval state."""
        was_training = model.training
        model.eval()
        try:
            model_curve = _curve_for_model(model, self.val_traj, self.device)
        finally:
            model.train(was_training)
        return compute_rollout_score(model_curve, self.baseline_envelope)

    def _build_envelope(self) -> np.ndarray:
        """Roll out every baseline and take the per-step minimum-median curve."""
        baselines = [b.to(self.device).eval() for b in self._build_baselines()]
        curves = np.stack([_curve_for_model(b, self.val_traj, self.device) for b in baselines])
        return curves.min(axis=0)

    def _build_baselines(self) -> list[nn.Module]:
        """Construct the four deterministic baselines, fitting from train_path."""
        return [
            PersistenceBaseline(),
            ConstantVelocityBaseline(dt=self.dt),
            MeanVelocityBaseline.fit(str(self.train_path), dt=self.dt),
            MeanStateBaseline.fit(str(self.train_path)),
        ]


def _curve_for_model(
    model: nn.Module,
    val_traj: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Per-step median state MSE for steps 1..N, dropping the trivial step-0 zero."""
    predicted = run_all_rollouts(model, val_traj, device)
    rollout_mse = compute_rollout_mse(val_traj, predicted)
    return rollout_mse.state.median[1:]


class BucketRolloutScoreEvaluator:
    """Bucket-aware rollout score on a stratified val set (sibling of RolloutScoreEvaluator).

    Construction fails fast if the val file lacks stratification. Each call rolls the model
    out once, scores every non-empty bin against its cached baseline envelope, and
    macro-averages (unweighted) over them.
    """

    def __init__(
        self,
        val_path: str | Path,
        train_path: str | Path,
        dt: float,
        device: torch.device,
    ) -> None:
        """Eagerly read the val bundle and validate stratification."""
        self.val_path = Path(val_path)
        self.train_path = Path(train_path)
        self.dt = float(dt)
        self.device = device

        bundle = read_trajectories(self.val_path)
        if bundle.encounter_bin_id is None:
            msg = (
                f"checkpoint_metric=bucket_macro_rollout_score requires a stratified "
                f"val file (with encounter_bin_id, encounter_bins, etc.); "
                f"{self.val_path} has no stratification metadata"
            )
            raise ValueError(msg)
        self._val_bundle = bundle
        # narrow nullable bundle fields once so score() doesn't repeat the assert
        assert bundle.encounter_bins is not None
        self._bin_order: tuple[str, ...] = tuple(b.name for b in bundle.encounter_bins)
        self._envelope_computer: BaselineEnvelopeComputer | None = None

    @property
    def val_traj(self) -> np.ndarray:
        """Validation trajectories (cached on construction)."""
        return self._val_bundle.states

    @property
    def bin_order(self) -> tuple[str, ...]:
        """Canonical bin order from the val file's encounter_bins."""
        return self._bin_order

    @property
    def envelope_computer(self) -> BaselineEnvelopeComputer:
        """Lazy: roll out the four baselines on val once, cache for the run."""
        if self._envelope_computer is None:
            ec = BaselineEnvelopeComputer(self.train_path, self.dt, self.device)
            ec.fit(self.val_traj)
            self._envelope_computer = ec
        return self._envelope_computer

    def score(self, model: nn.Module) -> BucketRolloutScore:
        """Run the model on val once and return its bucket-aware score; restores train/eval state."""
        was_training = model.training
        model.eval()
        try:
            predicted = run_all_rollouts(model, self.val_traj, self.device)
            full_mse = compute_rollout_mse(self.val_traj, predicted)
        finally:
            model.train(was_training)

        bin_id = self._val_bundle.encounter_bin_id
        bins = self._val_bundle.encounter_bins
        assert bin_id is not None
        assert bins is not None

        masks = trajectory_masks(bin_id, len(bins))
        per_bin: dict[str, RolloutScore] = {}
        for bin_def, mask in zip(bins, masks, strict=True):
            if not mask.any():
                continue  # empty bin: skip, do not contribute to macro
            model_curve = subset_rollout_mse(full_mse, mask).state.median[1:]
            envelope = self.envelope_computer.envelope_for_mask(mask)
            per_bin[bin_def.name] = compute_rollout_score(
                model_curve, envelope, anchor_steps=DEFAULT_ANCHOR_STEPS
            )

        if not per_bin:
            msg = "no non-empty bins in val file; cannot compute bucket macro score"
            raise RuntimeError(msg)

        macro = float(np.mean([s.score for s in per_bin.values()]))
        return BucketRolloutScore(macro=macro, per_bin=per_bin, bin_order=self._bin_order)
