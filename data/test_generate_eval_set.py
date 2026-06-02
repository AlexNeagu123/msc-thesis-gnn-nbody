"""Tests for data/generate_eval_set.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from data._io import read_trajectories
from data._types import DataGenConfig, SimulationParams, SplitConfig
from data.generate import Generator
from data.generate_eval_set import (
    DEFAULT_N_TRAJECTORIES,
    DEFAULT_SEED_BASE,
    build_eval_config,
    default_output_path,
    generate_eval_set,
)


@pytest.fixture
def base_params() -> SimulationParams:
    """Return base (3-body) simulation parameters for testing."""
    return SimulationParams(
        n_particles=3,
        t_end=10.0,
        dt=0.05,
        G=1.0,
        mass=1.0,
        min_distance=0.001,
        max_position=5.0,
        pos_scale=1.0,
        vel_scale=0.5,
    )


def test_default_output_path() -> None:
    """The output path embeds the body count and the generalization tag."""
    assert default_output_path(10) == "data/output/generalization_10body.h5"


def test_build_eval_config_overrides_only_n_particles(base_params: SimulationParams) -> None:
    """Only n_particles changes; all other sim params are inherited."""
    cfg = build_eval_config(base_params, n_particles=10, n_trajectories=50, out_path="x.h5", seed=7)

    assert cfg.simulation.n_particles == 10
    assert cfg.simulation.t_end == base_params.t_end
    assert cfg.simulation.vel_scale == base_params.vel_scale
    assert cfg.stratified is None
    assert cfg.splits == [SplitConfig("test", 50, "x.h5", 7)]


@pytest.mark.parametrize("n", [1, 0, -3])
def test_build_eval_config_rejects_too_few_particles(base_params: SimulationParams, n: int) -> None:
    """A pairwise system needs at least two bodies."""
    with pytest.raises(ValueError, match="n_particles must be >= 2"):
        build_eval_config(base_params, n_particles=n, n_trajectories=10, out_path="x.h5", seed=1)


def test_build_eval_config_rejects_nonpositive_trajectories(
    base_params: SimulationParams,
) -> None:
    """Requesting zero trajectories is a configuration error."""
    with pytest.raises(ValueError, match="n_trajectories must be >= 1"):
        build_eval_config(base_params, n_particles=4, n_trajectories=0, out_path="x.h5", seed=1)


def test_generate_eval_set_defaults_path_and_seed(
    base_params: SimulationParams,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defaults resolve to the conventional path and per-N seed; sim params inherit."""
    captured: dict[str, DataGenConfig] = {}

    def fake_run(self: Generator) -> None:
        captured["cfg"] = self.cfg

    monkeypatch.setattr(Generator, "run", fake_run)
    monkeypatch.setattr(
        "data.generate_eval_set.load_data_config",
        lambda _path: DataGenConfig(simulation=base_params, splits=[], stratified=None),
    )

    out = generate_eval_set(4)

    assert out == "data/output/generalization_4body.h5"
    cfg = captured["cfg"]
    assert cfg.simulation.n_particles == 4
    assert cfg.simulation.t_end == base_params.t_end
    assert cfg.stratified is None
    assert len(cfg.splits) == 1
    assert cfg.splits[0].name == "test"
    assert cfg.splits[0].n_trajectories == DEFAULT_N_TRAJECTORIES
    assert cfg.splits[0].seed == DEFAULT_SEED_BASE + 4


def test_generated_file_has_uniform_nbody_shape(tmp_path: Path) -> None:
    """End-to-end: generation writes a uniform N-body file with N in the state shape."""
    base = SimulationParams(
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
    out = tmp_path / "generalization_5body.h5"
    cfg = build_eval_config(base, n_particles=5, n_trajectories=2, out_path=str(out), seed=3)

    Generator(cfg).run()

    loaded = read_trajectories(out)
    assert loaded.states.shape[0] == 2
    assert loaded.states.shape[2] == 5  # n_particles flows into the state shape
    assert loaded.encounter_bin_id is None  # uniform: no stratification fields
    assert loaded.metadata is not None
    assert loaded.metadata.n_particles == 5
