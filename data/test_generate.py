"""Tests for data/generate.py."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from data._io import read_trajectories
from data._types import (
    DataGenConfig,
    EncounterBin,
    SimulationParams,
    SplitConfig,
    StratifiedConfig,
)
from data.generate import Generator, generate_trajectory


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a seeded random generator."""
    return np.random.default_rng(42)


@pytest.fixture
def default_params() -> SimulationParams:
    """Return default simulation parameters for testing."""
    return SimulationParams(
        n_particles=3,
        t_end=1.0,
        dt=0.05,
        G=1.0,
        mass=1.0,
        min_distance=0.001,
        max_position=10.0,
        pos_scale=1.0,
        vel_scale=0.5,
    )


def test_trajectory_shape(
    default_params: SimulationParams,
    rng: np.random.Generator,
) -> None:
    """Output arrays have the expected shape."""
    result = generate_trajectory(default_params, rng)
    assert result is not None

    states, energies = result
    assert states.shape == (20, 3, 5)
    assert energies.shape == (20,)


def test_trajectory_state_columns(
    default_params: SimulationParams,
    rng: np.random.Generator,
) -> None:
    """State columns are [x, y, vx, vy, m]; all values should be finite."""
    result = generate_trajectory(default_params, rng)
    assert result is not None

    states, _ = result
    assert np.all(np.isfinite(states))


def test_energy_conservation(
    default_params: SimulationParams,
    rng: np.random.Generator,
) -> None:
    """IAS15 should conserve energy to near machine precision."""
    result = generate_trajectory(default_params, rng)
    assert result is not None

    _, energies = result
    rel_error = np.abs((energies - energies[0]) / energies[0])
    assert rel_error.max() < 1e-8


def test_center_of_mass_near_zero(
    default_params: SimulationParams,
    rng: np.random.Generator,
) -> None:
    """Center of mass should stay near the origin (we subtract it at init)."""
    result = generate_trajectory(default_params, rng)
    assert result is not None

    states, _ = result
    com_positions = states[:, :, :2].mean(axis=1)
    assert np.abs(com_positions).max() < 1e-8


def test_close_encounter_filtering() -> None:
    """Trajectory with particles starting very close should be rejected."""
    strict_params = SimulationParams(
        n_particles=3,
        t_end=1.0,
        dt=0.05,
        G=1.0,
        mass=1.0,
        min_distance=1.0,
        max_position=10.0,
        pos_scale=1.0,
        vel_scale=0.5,
    )

    rejected = 0
    for seed in range(50):
        r = np.random.default_rng(seed)
        result = generate_trajectory(strict_params, r)
        if result is None:
            rejected += 1

    assert rejected > 0, "no trajectories were rejected despite strict min_distance"


def test_different_seeds_produce_different_trajectories(
    default_params: SimulationParams,
) -> None:
    """Different seeds should give different trajectories."""
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)

    result1 = generate_trajectory(default_params, rng1)
    result2 = generate_trajectory(default_params, rng2)

    assert result1 is not None or result2 is not None

    if result1 is not None and result2 is not None:
        states1, _ = result1
        states2, _ = result2
        assert not np.allclose(states1, states2)


# stratified split generation
_STRAT_BINS: tuple[EncounterBin, ...] = (
    EncounterBin("close", 0.0, 0.05),
    EncounterBin("mid", 0.05, 0.5),
    EncounterBin("smooth", 0.5, float("inf")),
)


def _candidate(d_min: float, n_steps: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Build a (states, energies) pair whose pairwise min distance equals `d_min`."""
    states = np.zeros((n_steps, 3, 5), dtype=np.float64)
    states[:, 0, :2] = (0.0, 0.0)
    states[:, 1, :2] = (d_min, 0.0)
    states[:, 2, :2] = (d_min + 1.0, 0.0)
    states[..., 4] = 1.0
    energies = np.zeros(n_steps, dtype=np.float64)
    return states, energies


def _stratified_cfg(
    n_train: int,
    n_val: int,
    n_test: int,
    train_dist: dict[str, float],
    train_path: Path,
    val_path: Path,
    test_path: Path,
) -> DataGenConfig:
    """Assemble a stratified DataGenConfig pointing at the given paths."""
    sim = SimulationParams(
        n_particles=3,
        t_end=0.2,
        dt=0.05,
        G=1.0,
        mass=1.0,
        min_distance=0.0,
        max_position=10.0,
        pos_scale=1.0,
        vel_scale=0.5,
    )
    splits = [
        SplitConfig("train", n_train, str(train_path), seed=1),
        SplitConfig("val", n_val, str(val_path), seed=2),
        SplitConfig("test", n_test, str(test_path), seed=3),
    ]
    equal = {"close": 1.0, "mid": 1.0, "smooth": 1.0}
    strat = StratifiedConfig(
        bins=_STRAT_BINS,
        train_distribution=dict(train_dist),
        val_distribution=dict(equal),
        test_distribution=dict(equal),
    )
    return DataGenConfig(simulation=sim, splits=splits, stratified=strat)


class _StubGenerator(Generator):
    """Generator subclass returning scripted candidates; None models a simulator rejection."""

    def __init__(
        self,
        cfg: DataGenConfig,
        candidates: list[tuple[np.ndarray, np.ndarray] | None],
    ) -> None:
        super().__init__(cfg)
        self._iter: Iterator[tuple[np.ndarray, np.ndarray] | None] = iter(candidates)

    def _simulate_trajectory(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Pop and return the next scripted candidate."""
        return next(self._iter)


def _close() -> tuple[np.ndarray, np.ndarray]:
    """Candidate landing in the 'close' bin (d_min=0.01)."""
    return _candidate(0.01)


def _mid() -> tuple[np.ndarray, np.ndarray]:
    """Candidate landing in the 'mid' bin (d_min=0.2)."""
    return _candidate(0.2)


def _smooth() -> tuple[np.ndarray, np.ndarray]:
    """Candidate landing in the 'smooth' bin (d_min=0.7)."""
    return _candidate(0.7)


def test_stratified_split_fills_exact_quotas(tmp_path: Path) -> None:
    """Each per-bin quota is met exactly; counts in the file match the targets."""
    cfg = _stratified_cfg(
        n_train=2,
        n_val=0,
        n_test=0,
        train_dist={"close": 1.0, "mid": 1.0, "smooth": 0.0},
        train_path=tmp_path / "train.h5",
        val_path=tmp_path / "v.h5",
        test_path=tmp_path / "t.h5",
    )
    candidates = [_close(), _mid()]
    _StubGenerator(cfg, candidates)._generate_split(cfg.splits[0])

    loaded = read_trajectories(tmp_path / "train.h5")
    counts = {b.name: 0 for b in _STRAT_BINS}
    for name in loaded.encounter_bin_name:
        counts[str(name)] += 1
    assert counts == {"close": 1, "mid": 1, "smooth": 0}
    assert loaded.encounter_bins == _STRAT_BINS


def test_stratified_discards_over_quota_candidates(tmp_path: Path) -> None:
    """A candidate hitting an already-full bin is dropped, not silently included."""
    cfg = _stratified_cfg(
        n_train=2,
        n_val=0,
        n_test=0,
        train_dist={"close": 1.0, "mid": 1.0, "smooth": 0.0},
        train_path=tmp_path / "train.h5",
        val_path=tmp_path / "v.h5",
        test_path=tmp_path / "t.h5",
    )
    candidates = [_close(), _close(), _mid()]  # second close is over-quota
    gen = _StubGenerator(cfg, candidates)
    gen._generate_split(cfg.splits[0])

    loaded = read_trajectories(tmp_path / "train.h5")
    counts = {b.name: 0 for b in _STRAT_BINS}
    for name in loaded.encounter_bin_name:
        counts[str(name)] += 1
    assert counts == {"close": 1, "mid": 1, "smooth": 0}


def test_stratified_skips_simulator_rejections(tmp_path: Path) -> None:
    """A None candidate counts as a simulator rejection but does not fill a quota."""
    cfg = _stratified_cfg(
        n_train=2,
        n_val=0,
        n_test=0,
        train_dist={"close": 1.0, "mid": 1.0, "smooth": 0.0},
        train_path=tmp_path / "train.h5",
        val_path=tmp_path / "v.h5",
        test_path=tmp_path / "t.h5",
    )
    candidates = [None, _close(), None, _mid()]
    gen = _StubGenerator(cfg, candidates)
    gen._generate_split(cfg.splits[0])

    loaded = read_trajectories(tmp_path / "train.h5")
    assert loaded.encounter_bin_id.shape == (2,)


def test_stratified_output_is_shuffled(tmp_path: Path) -> None:
    """Accepted trajectories are shuffled, not stored grouped by arrival order."""
    cfg = _stratified_cfg(
        n_train=6,
        n_val=0,
        n_test=0,
        train_dist={"close": 1.0, "mid": 0.0, "smooth": 1.0},
        train_path=tmp_path / "train.h5",
        val_path=tmp_path / "v.h5",
        test_path=tmp_path / "t.h5",
    )
    candidates = [_close()] * 3 + [_smooth()] * 3
    _StubGenerator(cfg, candidates)._generate_split(cfg.splits[0])

    loaded = read_trajectories(tmp_path / "train.h5")
    arrival_order = ["close"] * 3 + ["smooth"] * 3
    assert list(loaded.encounter_bin_name) != arrival_order


def test_stratified_raises_when_max_attempts_exhausted(tmp_path: Path) -> None:
    """Quotas that cannot be satisfied raise rather than hang once max_attempts is hit."""
    cfg = _stratified_cfg(
        n_train=2,
        n_val=0,
        n_test=0,
        train_dist={"close": 1.0, "mid": 0.0, "smooth": 1.0},
        train_path=tmp_path / "train.h5",
        val_path=tmp_path / "v.h5",
        test_path=tmp_path / "t.h5",
    )
    cfg = DataGenConfig(
        simulation=cfg.simulation,
        splits=cfg.splits,
        stratified=StratifiedConfig(
            bins=cfg.stratified.bins,
            train_distribution=cfg.stratified.train_distribution,
            val_distribution=cfg.stratified.val_distribution,
            test_distribution=cfg.stratified.test_distribution,
            max_attempts=50,
        ),
    )
    # only 'close' candidates; the 'smooth' quota is never filled
    candidates: list[tuple[np.ndarray, np.ndarray] | None] = [_close()] * 51
    gen = _StubGenerator(cfg, candidates)

    with pytest.raises(RuntimeError, match="hit max_attempts=50"):
        gen._generate_split(cfg.splits[0])


def test_stratified_max_attempts_default_is_safe_for_rare_bins(tmp_path: Path) -> None:
    """Unset max_attempts uses a default large enough to absorb many rejections."""
    cfg = _stratified_cfg(
        n_train=2,
        n_val=0,
        n_test=0,
        train_dist={"close": 1.0, "mid": 1.0, "smooth": 0.0},
        train_path=tmp_path / "train.h5",
        val_path=tmp_path / "v.h5",
        test_path=tmp_path / "t.h5",
    )
    # 1500 rejections then two valid candidates; the default cap absorbs them
    candidates: list[tuple[np.ndarray, np.ndarray] | None] = [None] * 1500 + [_close(), _mid()]
    gen = _StubGenerator(cfg, candidates)
    gen._generate_split(cfg.splits[0])

    loaded = read_trajectories(tmp_path / "train.h5")
    assert loaded.encounter_bin_id.shape == (2,)


def test_stratified_zero_trajectory_split_produces_empty_arrays(tmp_path: Path) -> None:
    """A split with n=0 writes empty per-trajectory arrays without crashing."""
    cfg = _stratified_cfg(
        n_train=0,
        n_val=0,
        n_test=0,
        train_dist={"close": 1.0, "mid": 1.0, "smooth": 1.0},
        train_path=tmp_path / "train.h5",
        val_path=tmp_path / "v.h5",
        test_path=tmp_path / "t.h5",
    )
    _StubGenerator(cfg, candidates=[])._generate_split(cfg.splits[0])

    loaded = read_trajectories(tmp_path / "train.h5")
    assert loaded.states.shape == (0, 4, 3, 5)
    assert loaded.encounter_bin_id.shape == (0,)
    assert loaded.encounter_bin_name.shape == (0,)
    assert loaded.min_pairwise_distance.shape == (0,)
    # bin descriptors round-trip even with no trajectories
    assert loaded.encounter_bins == _STRAT_BINS


def test_stratified_round_trips_through_write_and_read(tmp_path: Path) -> None:
    """The persisted file passes write-side validation and reads back consistently."""
    cfg = _stratified_cfg(
        n_train=3,
        n_val=0,
        n_test=0,
        train_dist={"close": 1.0, "mid": 1.0, "smooth": 1.0},
        train_path=tmp_path / "train.h5",
        val_path=tmp_path / "v.h5",
        test_path=tmp_path / "t.h5",
    )
    candidates = [_close(), _mid(), _smooth()]
    _StubGenerator(cfg, candidates)._generate_split(cfg.splits[0])

    # read_trajectories runs _validate_stratification and would raise on a broken invariant
    loaded = read_trajectories(tmp_path / "train.h5")

    assert loaded.encounter_bin_id.shape == (3,)
    assert loaded.encounter_bin_name.shape == (3,)
    assert loaded.min_pairwise_distance.shape == (3,)
    assert loaded.encounter_bins == _STRAT_BINS

    name_by_id = {i: b.name for i, b in enumerate(_STRAT_BINS)}
    for i in range(3):
        assert str(loaded.encounter_bin_name[i]) == name_by_id[int(loaded.encounter_bin_id[i])]


def test_uniform_path_writes_no_stratification_fields(tmp_path: Path) -> None:
    """When cfg.stratified is None the saved file has all four fields as None on read."""
    sim = SimulationParams(
        n_particles=3,
        t_end=0.2,
        dt=0.05,
        G=1.0,
        mass=1.0,
        min_distance=0.0,
        max_position=10.0,
        pos_scale=1.0,
        vel_scale=0.5,
    )
    cfg = DataGenConfig(
        simulation=sim,
        splits=[SplitConfig("train", 2, str(tmp_path / "train.h5"), seed=0)],
        stratified=None,
    )
    candidates = [_close(), _mid()]
    _StubGenerator(cfg, candidates)._generate_split(cfg.splits[0])

    loaded = read_trajectories(tmp_path / "train.h5")
    assert loaded.encounter_bin_id is None
    assert loaded.encounter_bin_name is None
    assert loaded.min_pairwise_distance is None
    assert loaded.encounter_bins is None
