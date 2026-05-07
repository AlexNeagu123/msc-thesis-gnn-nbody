"""Test-set baseline envelope for stratified evaluation reports.

Parallels training/rollout_score.py:RolloutScoreEvaluator but operates on
the evaluation test set with bin slicing. The four deterministic baselines
roll out exactly once per evaluation; per-bin envelopes are then derived
by reusing the cached RolloutMSE objects via subset_rollout_mse, so the
per-bin cost is a few array slices regardless of bin count.

References:
    - Training-time analogue: training/rollout_score.py (RolloutScoreEvaluator)
    - Score math: training/rollout_score.py:compute_rollout_score
    - Baselines: models/baselines.py
    - Per-bin slicing: evaluation/metrics.py:subset_rollout_mse
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from evaluation._types import RolloutMSE
from evaluation.metrics import compute_rollout_mse, run_all_rollouts, subset_rollout_mse
from models.baselines import (
    ConstantVelocityBaseline,
    MeanStateBaseline,
    MeanVelocityBaseline,
    PersistenceBaseline,
)


class BaselineEnvelopeComputer:
    """Per-bin baseline envelope built from a single round of test rollouts.

    Usage:
        computer = BaselineEnvelopeComputer(train_path, dt, device)
        computer.fit(test_traj)
        envelope = computer.envelope_for_mask(bin_mask)  # (n_steps - 1,)

    `fit()` rolls out the four canonical deterministic baselines on the
    full test set once and caches their RolloutMSE objects. Each
    subsequent `envelope_for_mask(mask)` call slices those cached objects
    by `mask`, takes the per-step median state MSE for each baseline, and
    returns the per-step minimum across baselines (the "envelope"),
    dropping step 0 to match the curve contract of compute_rollout_score.
    """

    def __init__(
        self,
        train_path: str | Path,
        dt: float,
        device: torch.device,
    ) -> None:
        """Store paths and device; rollouts happen lazily in fit()."""
        self.train_path = Path(train_path)
        self.dt = float(dt)
        self.device = device
        self._mse_per_baseline: list[RolloutMSE] | None = None

    def fit(self, test_traj: npt.NDArray[np.floating]) -> None:
        """Roll out every baseline once on `test_traj` and cache its RolloutMSE."""
        baselines = [b.to(self.device).eval() for b in self._build_baselines()]
        mse_per_baseline = []
        for baseline in baselines:
            predicted = run_all_rollouts(baseline, test_traj, self.device)
            mse_per_baseline.append(compute_rollout_mse(test_traj, predicted))
        self._mse_per_baseline = mse_per_baseline

    def envelope_for_mask(
        self,
        mask: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.floating]:
        """Per-step minimum-median state MSE over the bin subset, dropping step 0.

        Empty masks return an all-NaN curve so callers can short-circuit
        the per-bin score computation without guarding against warnings;
        the orchestrator skips empty bins anyway (baseline_ratios=None).
        """
        if self._mse_per_baseline is None:
            msg = "fit() must be called before envelope_for_mask()"
            raise RuntimeError(msg)

        n_envelope_steps = self._mse_per_baseline[0].state.median.shape[0] - 1

        if not mask.any():
            return np.full(n_envelope_steps, np.nan)

        curves = np.stack(
            [subset_rollout_mse(mse, mask).state.median[1:] for mse in self._mse_per_baseline]
        )
        return curves.min(axis=0)

    def _build_baselines(self) -> list[nn.Module]:
        """Construct the four deterministic baselines, fitting from train_path."""
        return [
            PersistenceBaseline(),
            ConstantVelocityBaseline(dt=self.dt),
            MeanVelocityBaseline.fit(str(self.train_path), dt=self.dt),
            MeanStateBaseline.fit(str(self.train_path)),
        ]
