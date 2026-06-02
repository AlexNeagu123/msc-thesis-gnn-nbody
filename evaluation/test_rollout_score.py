"""Tests for evaluation/rollout_score.py."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from evaluation.metrics import compute_rollout_mse, run_all_rollouts, subset_rollout_mse
from evaluation.rollout_score import BaselineEnvelopeComputer
from models.baselines import (
    ConstantVelocityBaseline,
    MeanStateBaseline,
    MeanVelocityBaseline,
    PersistenceBaseline,
)


def _write_h5(path: Path, n_traj: int = 3, n_steps: int = 5) -> None:
    """Write a tiny deterministic trajectory file matching evaluation tests."""
    rng = np.random.default_rng(42)
    trajectories = rng.normal(size=(n_traj, n_steps, 3, 5)).astype(np.float32)
    trajectories[..., 4] = 1.0
    with h5py.File(path, "w") as f:
        f.create_dataset("trajectories", data=trajectories)
        f.create_dataset("energies", data=np.zeros((n_traj, n_steps), dtype=np.float32))


def _test_traj(n_traj: int = 4, n_steps: int = 5) -> np.ndarray:
    """Build a deterministic test trajectory in-memory."""
    rng = np.random.default_rng(7)
    traj = rng.normal(size=(n_traj, n_steps, 3, 5)).astype(np.float32)
    traj[..., 4] = 1.0
    return traj


def test_envelope_before_fit_raises(tmp_path: Path) -> None:
    """Querying the envelope before fit() is a programmer error."""
    train_path = tmp_path / "train.h5"
    _write_h5(train_path)

    computer = BaselineEnvelopeComputer(train_path, dt=0.05, device=torch.device("cpu"))
    with pytest.raises(RuntimeError, match="fit\\(\\) must be called"):
        computer.envelope_for_mask(np.array([True, True]))


def test_envelope_full_mask_matches_min_over_baselines(tmp_path: Path) -> None:
    """Envelope on an all-True mask equals the per-step minimum median across baselines."""
    train_path = tmp_path / "train.h5"
    _write_h5(train_path)
    test_traj = _test_traj()
    device = torch.device("cpu")

    computer = BaselineEnvelopeComputer(train_path, dt=0.05, device=device)
    computer.fit(test_traj)

    full_mask = np.ones(test_traj.shape[0], dtype=bool)
    envelope = computer.envelope_for_mask(full_mask)

    # reproduce manually: roll out each baseline, take median[1:], take per-step min
    baselines = [
        PersistenceBaseline().to(device).eval(),
        ConstantVelocityBaseline(dt=0.05).to(device).eval(),
        MeanVelocityBaseline.fit(str(train_path), dt=0.05).to(device).eval(),
        MeanStateBaseline.fit(str(train_path)).to(device).eval(),
    ]
    expected_curves = np.stack(
        [
            compute_rollout_mse(test_traj, run_all_rollouts(b, test_traj, device)).state.median[1:]
            for b in baselines
        ]
    )
    expected = expected_curves.min(axis=0)

    np.testing.assert_allclose(envelope, expected)
    assert envelope.shape == (test_traj.shape[1] - 1,)


def test_envelope_per_bin_matches_subset_min(tmp_path: Path) -> None:
    """Per-bin envelope matches subset_rollout_mse and per-step min across baselines."""
    train_path = tmp_path / "train.h5"
    _write_h5(train_path)
    test_traj = _test_traj()
    device = torch.device("cpu")

    computer = BaselineEnvelopeComputer(train_path, dt=0.05, device=device)
    computer.fit(test_traj)

    mask = np.array([True, False, True, False])
    envelope = computer.envelope_for_mask(mask)

    # reconstruct expected using the cached MSE objects
    expected = np.stack(
        [subset_rollout_mse(mse, mask).state.median[1:] for mse in computer._mse_per_baseline]
    ).min(axis=0)

    np.testing.assert_allclose(envelope, expected)


def test_fit_runs_baselines_once_per_call(tmp_path: Path) -> None:
    """fit() populates the per-baseline MSE cache exactly once; queries reuse it."""
    train_path = tmp_path / "train.h5"
    _write_h5(train_path)
    test_traj = _test_traj()
    device = torch.device("cpu")

    computer = BaselineEnvelopeComputer(train_path, dt=0.05, device=device)
    computer.fit(test_traj)
    cached_first = computer._mse_per_baseline
    assert cached_first is not None
    assert len(cached_first) == 4

    # Multiple envelope queries must not rebuild the cache (object identity).
    computer.envelope_for_mask(np.array([True, False, True, False]))
    computer.envelope_for_mask(np.array([False, True, False, True]))
    assert computer._mse_per_baseline is cached_first


def test_envelope_for_empty_mask_returns_nan(tmp_path: Path) -> None:
    """Empty mask returns an all-NaN envelope of the right length, no warnings."""
    train_path = tmp_path / "train.h5"
    _write_h5(train_path)
    test_traj = _test_traj()

    computer = BaselineEnvelopeComputer(train_path, dt=0.05, device=torch.device("cpu"))
    computer.fit(test_traj)

    envelope = computer.envelope_for_mask(np.zeros(test_traj.shape[0], dtype=bool))

    assert envelope.shape == (test_traj.shape[1] - 1,)
    assert np.isnan(envelope).all()
