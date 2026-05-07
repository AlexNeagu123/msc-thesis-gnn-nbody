"""Tests for training/rollout_score.py."""

import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from data._io import write_trajectories
from data._types import EncounterBin, Trajectories
from models.baselines import ConstantVelocityBaseline, PersistenceBaseline
from training._types import BucketRolloutScore
from training.rollout_score import (
    DEFAULT_ANCHOR_STEPS,
    BucketRolloutScoreEvaluator,
    RolloutScoreEvaluator,
    _curve_for_model,
    compute_rollout_score,
)


def test_score_zero_when_model_matches_baseline() -> None:
    """If model_curve == baseline_envelope, R_s == 1 exactly; ties are not wins."""
    n = 50
    rng = np.random.default_rng(0)
    baseline = rng.uniform(1e-3, 1.0, size=n)

    score = compute_rollout_score(baseline.copy(), baseline)

    assert abs(score.score) < 1e-6
    assert score.final_ratio == pytest.approx(1.0, abs=1e-12)
    assert score.dominance_horizon == 0
    assert score.fraction_beating_baseline == 0.0


def test_score_negative_when_model_beats_baseline_uniformly() -> None:
    """If R_s = 0.5 everywhere, score = log(0.5 + eps) ~ -0.693."""
    n = 100
    baseline = np.full(n, 2.0)
    model = np.full(n, 1.0)

    score = compute_rollout_score(model, baseline)

    assert score.score == pytest.approx(math.log(0.5), abs=1e-6)
    assert score.dominance_horizon == n
    assert score.fraction_beating_baseline == 1.0
    assert score.final_ratio == pytest.approx(0.5, abs=1e-6)


def test_score_positive_when_model_loses_uniformly() -> None:
    """If R_s = 2 everywhere, score = log(2) ~ 0.693."""
    n = 100
    baseline = np.full(n, 1.0)
    model = np.full(n, 2.0)

    score = compute_rollout_score(model, baseline)

    assert score.score == pytest.approx(math.log(2.0), abs=1e-6)
    assert score.dominance_horizon == 0
    assert score.fraction_beating_baseline == 0.0


def test_dominance_horizon_is_longest_winning_prefix() -> None:
    """Wins until s=20 (R<1), then loses; dominance == 20."""
    n = 50
    ratios_target = np.concatenate([np.full(20, 0.5), np.full(n - 20, 1.5)])
    baseline = np.ones(n)
    model = ratios_target * baseline

    score = compute_rollout_score(model, baseline)

    assert score.dominance_horizon == 20
    assert score.fraction_beating_baseline == pytest.approx(20 / n)


def test_dominance_horizon_zero_when_first_step_loses() -> None:
    """A single early loss at s=1 reduces dominance to zero, even if later wins."""
    ratios_target = np.array([1.5, 0.5, 0.5, 0.5])
    baseline = np.ones(4)
    model = ratios_target * baseline

    score = compute_rollout_score(model, baseline)

    assert score.dominance_horizon == 0
    assert score.fraction_beating_baseline == pytest.approx(3 / 4)


def test_dominance_horizon_full_length_when_always_beats() -> None:
    """No failure in beats_baseline -> dominance_horizon == len(curve)."""
    n = 30
    score = compute_rollout_score(np.full(n, 0.1), np.full(n, 1.0))

    assert score.dominance_horizon == n


def test_anchor_step_ratios_are_one_indexed_and_clamped() -> None:
    """Anchor steps map to ratios[s-1] and steps past the curve are dropped."""
    ratios_target = np.linspace(0.1, 1.9, 20)
    baseline = np.ones(20)
    model = ratios_target * baseline

    # default anchors are (10, 20, 50, 100, 199); only 10 and 20 fit.
    score = compute_rollout_score(model, baseline)

    assert set(score.ratios_at_step.keys()) == {10, 20}
    assert score.ratios_at_step[10] == pytest.approx(ratios_target[9])
    assert score.ratios_at_step[20] == pytest.approx(ratios_target[19])


def test_anchor_steps_can_be_overridden() -> None:
    """Caller-supplied anchor_steps replace the default set."""
    ratios_target = np.linspace(0.1, 1.0, 5)
    baseline = np.ones(5)
    model = ratios_target * baseline

    score = compute_rollout_score(model, baseline, anchor_steps=(1, 3, 5))

    assert set(score.ratios_at_step.keys()) == {1, 3, 5}


def test_final_ratio_is_last_step() -> None:
    """final_ratio equals R at the last index of the curve."""
    n = 50
    rng = np.random.default_rng(1)
    baseline = rng.uniform(1e-3, 1.0, size=n)
    model = rng.uniform(1e-3, 1.0, size=n)

    score = compute_rollout_score(model, baseline)

    expected_last = (model[-1] + 1e-12) / (baseline[-1] + 1e-12)
    assert score.final_ratio == pytest.approx(expected_last, rel=1e-6)


def test_shape_mismatch_raises() -> None:
    """Curves with different shapes raise before computing anything."""
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_rollout_score(np.ones(10), np.ones(11))


def test_non_1d_raises() -> None:
    """2-D curves are rejected; the function operates on a single horizon series."""
    with pytest.raises(ValueError, match="expected 1-D"):
        compute_rollout_score(np.ones((5, 5)), np.ones((5, 5)))


def test_empty_curve_raises() -> None:
    """A zero-length curve has no defined score."""
    with pytest.raises(ValueError, match="at least one step"):
        compute_rollout_score(np.array([]), np.array([]))


def test_baseline_zero_does_not_blow_up() -> None:
    """A zero baseline step stays finite thanks to eps in the denominator.

    Realistic edge: persistence baseline yields zero MSE on synthetic
    stationary trajectories. The score must still be finite.
    """
    n = 10
    baseline = np.zeros(n)
    model = np.full(n, 1e-6)

    score = compute_rollout_score(model, baseline)

    assert math.isfinite(score.score)
    assert math.isfinite(score.final_ratio)


def test_default_anchor_steps_constant_matches_doc() -> None:
    """The advertised anchor steps are the ones used by default."""
    assert DEFAULT_ANCHOR_STEPS == (10, 20, 50, 100, 199)


def _write_traj_h5(path: Path, n_traj: int = 3, n_frames: int = 6, seed: int = 0) -> None:
    """Write a small synthetic trajectory file in the standard h5 layout."""
    rng = np.random.default_rng(seed)
    trajectories = rng.normal(size=(n_traj, n_frames, 3, 5)).astype(np.float32)
    trajectories[..., 4] = 1.0
    with h5py.File(path, "w") as f:
        f.create_dataset("trajectories", data=trajectories)
        f.create_dataset("energies", data=np.zeros((n_traj, n_frames), dtype=np.float32))


def test_evaluator_baseline_envelope_length_matches_rollout_steps(tmp_path: Path) -> None:
    """Envelope covers steps 1..N where N = n_frames - 1, matching the score contract."""
    val_path = tmp_path / "val.h5"
    train_path = tmp_path / "train.h5"
    _write_traj_h5(val_path, n_frames=8)
    _write_traj_h5(train_path, n_frames=8)

    ev = RolloutScoreEvaluator(val_path, train_path, dt=0.05, device=torch.device("cpu"))

    assert ev.baseline_envelope.shape == (7,)


def test_evaluator_envelope_is_per_step_minimum_across_baselines(tmp_path: Path) -> None:
    """Envelope at each step is the smallest median MSE achieved by any baseline."""
    val_path = tmp_path / "val.h5"
    train_path = tmp_path / "train.h5"
    _write_traj_h5(val_path, n_frames=10)
    _write_traj_h5(train_path, n_frames=10)

    ev = RolloutScoreEvaluator(val_path, train_path, dt=0.05, device=torch.device("cpu"))
    envelope = ev.baseline_envelope

    # the envelope must be <= each baseline's individual curve at every step
    for baseline in ev._build_baselines():
        baseline = baseline.to(torch.device("cpu")).eval()
        curve = _curve_for_model(baseline, ev.val_traj, torch.device("cpu"))
        assert (envelope <= curve + 1e-9).all(), "envelope exceeds an individual baseline curve"


def test_evaluator_caches_envelope(tmp_path: Path) -> None:
    """Repeated access returns the same array object; baselines roll out once."""
    val_path = tmp_path / "val.h5"
    train_path = tmp_path / "train.h5"
    _write_traj_h5(val_path)
    _write_traj_h5(train_path)

    ev = RolloutScoreEvaluator(val_path, train_path, dt=0.05, device=torch.device("cpu"))

    first = ev.baseline_envelope
    second = ev.baseline_envelope
    assert first is second


def test_evaluator_score_against_persistence_baseline_yields_unit_ratio(tmp_path: Path) -> None:
    """A model that is the persistence baseline should match it step-for-step.

    Persistence is one of the envelope members, so it can never be strictly
    better than the envelope. Its curve equals the envelope at every step
    where persistence is the per-step argmin, giving R = 1 there. Most steps
    other baselines beat persistence, so we just check that the persistence
    score is finite and non-negative on average (it does not beat the envelope).
    """
    val_path = tmp_path / "val.h5"
    train_path = tmp_path / "train.h5"
    _write_traj_h5(val_path, n_frames=8)
    _write_traj_h5(train_path, n_frames=8)

    ev = RolloutScoreEvaluator(val_path, train_path, dt=0.05, device=torch.device("cpu"))

    score = ev.score(PersistenceBaseline().eval())

    assert math.isfinite(score.score)
    # persistence cannot strictly beat the envelope (it's a member); score >= 0
    assert score.score >= -1e-9
    # but it ties at every step where it is the minimum baseline
    assert score.dominance_horizon == 0


@pytest.mark.parametrize("start_training", [True, False])
def test_evaluator_preserves_caller_model_training_state(
    tmp_path: Path, start_training: bool
) -> None:
    """Scoring restores model.training to whatever it was before the call.

    Today's models have no dropout/batchnorm so eval vs train is observationally
    identical, but the contract still matters: rollout validation must use eval
    mode internally for determinism, and the caller's state must survive intact.
    """
    val_path = tmp_path / "val.h5"
    train_path = tmp_path / "train.h5"
    _write_traj_h5(val_path)
    _write_traj_h5(train_path)

    ev = RolloutScoreEvaluator(val_path, train_path, dt=0.05, device=torch.device("cpu"))
    model = ConstantVelocityBaseline(dt=0.05)
    model.train(start_training)

    ev.score(model)

    assert model.training is start_training


def test_curve_contract_is_one_indexed_steps_one_through_n() -> None:
    """Curves cover rollout steps 1..N; anchor s reads ratios[s-1].

    Locks in the contract that callers must slice off step 0 before calling
    the scorer, and prevents future regressions where someone passes the full
    0..N curve and silently shifts every anchor by one.
    """
    n = 199
    # encode the rollout step into the value at each index so the test is
    # readable: ratios[i] should expose step i+1.
    model = np.arange(1, n + 1, dtype=np.float64)
    baseline = np.ones(n, dtype=np.float64)

    score = compute_rollout_score(model, baseline)

    # default anchors all fall within the curve; each one reads ratios[s-1] = s
    assert score.ratios_at_step == {
        10: pytest.approx(10.0, rel=1e-12),
        20: pytest.approx(20.0, rel=1e-12),
        50: pytest.approx(50.0, rel=1e-12),
        100: pytest.approx(100.0, rel=1e-12),
        199: pytest.approx(199.0, rel=1e-12),
    }
    assert score.final_ratio == pytest.approx(199.0, rel=1e-12)


def _write_stratified_h5(
    path: Path,
    *,
    bin_id: list[int],
    n_frames: int = 6,
    seed: int = 0,
) -> None:
    """Write a small stratified trajectory file with two canonical bins.

    `bin_id` controls how many trajectories are assigned to each bin and
    in what order; the per-trajectory min pairwise distance is set
    deterministically inside each bin's interval.
    """
    bins = (
        EncounterBin(name="extreme", lo=0.0, hi=0.05),
        EncounterBin(name="smooth", lo=0.05, hi=float("inf")),
    )
    representative_d_min = {0: 0.02, 1: 0.5}
    rng = np.random.default_rng(seed)
    n_traj = len(bin_id)

    states = rng.normal(size=(n_traj, n_frames, 3, 5)).astype(np.float32)
    states[..., 4] = 1.0
    energies = np.zeros((n_traj, n_frames), dtype=np.float32)

    bundle = Trajectories(
        states=states,
        energies=energies,
        encounter_bin_id=np.array(bin_id, dtype=np.int64),
        encounter_bin_name=np.array([bins[b].name for b in bin_id]),
        min_pairwise_distance=np.array([representative_d_min[b] for b in bin_id], dtype=np.float64),
        encounter_bins=bins,
    )
    write_trajectories(path, bundle)


def test_bucket_evaluator_construction_rejects_non_stratified_val(tmp_path: Path) -> None:
    """Non-stratified val raises with a clear configuration message."""
    val_path = tmp_path / "val.h5"
    train_path = tmp_path / "train.h5"
    _write_traj_h5(val_path)
    _write_traj_h5(train_path)

    with pytest.raises(ValueError, match="requires a stratified val file"):
        BucketRolloutScoreEvaluator(
            val_path=val_path,
            train_path=train_path,
            dt=0.05,
            device=torch.device("cpu"),
        )


def test_bucket_evaluator_construction_succeeds_on_stratified_val(tmp_path: Path) -> None:
    """Stratified val constructs and exposes canonical bin order."""
    val_path = tmp_path / "val.h5"
    train_path = tmp_path / "train.h5"
    _write_stratified_h5(val_path, bin_id=[0, 1, 0, 1])
    _write_traj_h5(train_path)

    ev = BucketRolloutScoreEvaluator(
        val_path=val_path,
        train_path=train_path,
        dt=0.05,
        device=torch.device("cpu"),
    )

    assert ev.bin_order == ("extreme", "smooth")
    assert ev.val_traj.shape[0] == 4


def test_bucket_evaluator_score_returns_macro_and_per_bin(tmp_path: Path) -> None:
    """score() returns one RolloutScore per non-empty bin and a finite macro."""
    val_path = tmp_path / "val.h5"
    train_path = tmp_path / "train.h5"
    _write_stratified_h5(val_path, bin_id=[0, 1, 0, 1], n_frames=8)
    _write_traj_h5(train_path, n_frames=8)

    ev = BucketRolloutScoreEvaluator(
        val_path=val_path,
        train_path=train_path,
        dt=0.05,
        device=torch.device("cpu"),
    )
    bucket = ev.score(PersistenceBaseline().eval())

    assert isinstance(bucket, BucketRolloutScore)
    assert bucket.bin_order == ("extreme", "smooth")
    assert set(bucket.per_bin) == {"extreme", "smooth"}
    assert math.isfinite(bucket.macro)


def test_bucket_evaluator_macro_is_arithmetic_mean(tmp_path: Path) -> None:
    """Macro must equal the unweighted arithmetic mean of populated per-bin scores.

    Pinned explicitly so any future weighted-averaging would have to
    deliberately update this assertion. Using imbalanced bin counts
    (3 vs 1) so a sample-weighted mean would diverge from the macro.
    """
    val_path = tmp_path / "val.h5"
    train_path = tmp_path / "train.h5"
    _write_stratified_h5(val_path, bin_id=[0, 0, 0, 1], n_frames=8)
    _write_traj_h5(train_path, n_frames=8)

    ev = BucketRolloutScoreEvaluator(
        val_path=val_path,
        train_path=train_path,
        dt=0.05,
        device=torch.device("cpu"),
    )
    bucket = ev.score(PersistenceBaseline().eval())

    expected = (bucket.per_bin["extreme"].score + bucket.per_bin["smooth"].score) / 2
    assert bucket.macro == pytest.approx(expected, rel=1e-12)


def test_bucket_evaluator_skips_empty_bins(tmp_path: Path) -> None:
    """Bins with zero trajectories are absent from per_bin and the macro."""
    val_path = tmp_path / "val.h5"
    train_path = tmp_path / "train.h5"
    # all trajectories in the smooth bin: extreme is empty
    _write_stratified_h5(val_path, bin_id=[1, 1, 1, 1], n_frames=8)
    _write_traj_h5(train_path, n_frames=8)

    ev = BucketRolloutScoreEvaluator(
        val_path=val_path,
        train_path=train_path,
        dt=0.05,
        device=torch.device("cpu"),
    )
    bucket = ev.score(PersistenceBaseline().eval())

    assert set(bucket.per_bin) == {"smooth"}
    assert bucket.macro == pytest.approx(bucket.per_bin["smooth"].score)
    # bin_order still carries the canonical order, including empty extreme
    assert bucket.bin_order == ("extreme", "smooth")


def test_bucket_evaluator_caches_envelope(tmp_path: Path) -> None:
    """Envelope is built once and reused across multiple score() calls."""
    val_path = tmp_path / "val.h5"
    train_path = tmp_path / "train.h5"
    _write_stratified_h5(val_path, bin_id=[0, 1, 0, 1])
    _write_traj_h5(train_path)

    ev = BucketRolloutScoreEvaluator(
        val_path=val_path,
        train_path=train_path,
        dt=0.05,
        device=torch.device("cpu"),
    )
    ev.score(PersistenceBaseline().eval())
    cached = ev.envelope_computer
    ev.score(ConstantVelocityBaseline(dt=0.05).eval())
    assert ev.envelope_computer is cached


def test_bucket_evaluator_preserves_caller_model_training_state(tmp_path: Path) -> None:
    """score() must restore model.training even though it toggles to eval internally."""
    val_path = tmp_path / "val.h5"
    train_path = tmp_path / "train.h5"
    _write_stratified_h5(val_path, bin_id=[0, 1, 0, 1])
    _write_traj_h5(train_path)

    ev = BucketRolloutScoreEvaluator(
        val_path=val_path,
        train_path=train_path,
        dt=0.05,
        device=torch.device("cpu"),
    )
    model = ConstantVelocityBaseline(dt=0.05)
    model.train(True)

    ev.score(model)
    assert model.training is True
