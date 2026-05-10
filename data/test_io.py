"""Tests for data/_io.py.

Round-trip a Trajectories bundle through write_trajectories /
read_trajectories and confirm states, energies, and metadata survive.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from data._io import (
    load_data_config,
    read_states,
    read_trajectories,
    write_trajectories,
)
from data._types import (
    DataGenConfig,
    EncounterBin,
    StratifiedConfig,
    Trajectories,
    TrajectoryMetadata,
)


def _example_trajectories() -> Trajectories:
    """Build a small Trajectories with realistic-looking metadata."""
    rng = np.random.default_rng(42)
    states = rng.normal(size=(2, 5, 3, 5)).astype(np.float64)
    energies = rng.normal(size=(2, 5)).astype(np.float64)
    metadata = TrajectoryMetadata(
        n_trajectories=2,
        n_particles=3,
        n_steps=5,
        t_end=1.0,
        dt=0.2,
        G=1.0,
        mass=1.0,
        min_distance=0.001,
        pos_scale=1.0,
        vel_scale=0.5,
        seed=42,
        rejection_rate=0.25,
    )
    return Trajectories(states=states, energies=energies, metadata=metadata)


def test_round_trip_preserves_arrays(tmp_path: Path) -> None:
    """States and energies survive write -> read byte-equal."""
    original = _example_trajectories()
    path = tmp_path / "round.h5"

    write_trajectories(path, original)
    loaded = read_trajectories(path)

    assert np.array_equal(loaded.states, original.states)
    assert np.array_equal(loaded.energies, original.energies)


def test_round_trip_preserves_metadata(tmp_path: Path) -> None:
    """All TrajectoryMetadata fields survive write -> read."""
    original = _example_trajectories()
    path = tmp_path / "round.h5"

    write_trajectories(path, original)
    loaded = read_trajectories(path)

    assert loaded.metadata == original.metadata


def test_read_tolerates_missing_metadata(tmp_path: Path) -> None:
    """Files written without a metadata group (test fixtures) read cleanly."""
    rng = np.random.default_rng(0)
    bare = Trajectories(
        states=rng.normal(size=(1, 3, 3, 5)),
        energies=rng.normal(size=(1, 3)),
        metadata=None,
    )
    path = tmp_path / "bare.h5"

    write_trajectories(path, bare)
    loaded = read_trajectories(path)

    assert loaded.metadata is None
    assert np.array_equal(loaded.states, bare.states)


def test_write_creates_parent_directory(tmp_path: Path) -> None:
    """write_trajectories creates intermediate directories if missing."""
    nested = tmp_path / "a" / "b" / "c.h5"
    write_trajectories(nested, _example_trajectories())
    assert nested.exists()


def test_read_states_skips_energies_and_metadata(tmp_path: Path) -> None:
    """read_states returns just the trajectory array."""
    original = _example_trajectories()
    path = tmp_path / "states.h5"
    write_trajectories(path, original)

    states = read_states(path)

    assert isinstance(states, np.ndarray)
    assert np.array_equal(states, original.states)


def test_load_data_config_parses_yaml(tmp_path: Path) -> None:
    """load_data_config returns a typed DataGenConfig from YAML."""
    raw = {
        "n_particles": 3,
        "t_end": 10.0,
        "dt": 0.05,
        "G": 1.0,
        "mass": 1.0,
        "min_distance": 0.1,
        "pos_scale": 1.0,
        "vel_scale": 0.5,
        "seed": 42,
        "n_train": 100,
        "n_val": 20,
        "n_test": 20,
        "train_path": "train.h5",
        "val_path": "val.h5",
        "test_path": "test.h5",
    }
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(raw))

    cfg = load_data_config(path)

    assert isinstance(cfg, DataGenConfig)
    assert cfg.simulation.n_particles == 3
    assert len(cfg.splits) == 3
    assert cfg.splits[0].name == "train"
    assert cfg.splits[0].n_trajectories == 100


# --- StratifiedConfig parsing and validation ---


def _base_data_yaml() -> dict:
    """Minimal valid YAML body for a DataGenConfig (no stratified section)."""
    return {
        "n_particles": 3,
        "t_end": 10.0,
        "dt": 0.05,
        "G": 1.0,
        "mass": 1.0,
        "min_distance": 0.001,
        "pos_scale": 1.0,
        "vel_scale": 0.5,
        "seed": 42,
        "n_train": 100,
        "n_val": 20,
        "n_test": 20,
        "train_path": "train.h5",
        "val_path": "val.h5",
        "test_path": "test.h5",
    }


def _stratified_yaml() -> dict:
    """Reusable enabled-and-valid stratified section keyed to three sample bins."""
    return {
        "enabled": True,
        "bins": [
            {"name": "a", "lo": 0.0, "hi": 0.5},
            {"name": "b", "lo": 0.5, "hi": 1.0},
            {"name": "c", "lo": 1.0, "hi": float("inf")},
        ],
        "train_distribution": {"a": 1, "b": 2, "c": 3},
        "val_distribution": {"a": 1, "b": 1, "c": 1},
        "test_distribution": {"a": 1, "b": 1, "c": 1},
    }


_VALID_BINS: tuple[EncounterBin, ...] = (
    EncounterBin("a", 0.0, 0.5),
    EncounterBin("b", 0.5, 1.0),
    EncounterBin("c", 1.0, float("inf")),
)


def _make_stratified(**overrides: object) -> StratifiedConfig:
    """Construct a StratifiedConfig with one or more fields overridden."""
    valid_dist: dict[str, float] = {"a": 1.0, "b": 1.0, "c": 1.0}
    kwargs: dict[str, object] = {
        "bins": _VALID_BINS,
        "train_distribution": dict(valid_dist),
        "val_distribution": dict(valid_dist),
        "test_distribution": dict(valid_dist),
    }
    kwargs.update(overrides)
    return StratifiedConfig(**kwargs)  # type: ignore[arg-type]


def test_data_gen_config_without_stratified_key_is_none(tmp_path: Path) -> None:
    """YAML files without a stratified section parse as stratified=None (uniform mode)."""
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(_base_data_yaml()))

    cfg = load_data_config(path)

    assert cfg.stratified is None


def test_data_gen_config_with_stratified_disabled_is_none(tmp_path: Path) -> None:
    """An explicitly-disabled stratified section also resolves to None."""
    raw = _base_data_yaml()
    raw["stratified"] = {"enabled": False, "bins": [{"name": "x", "lo": 0.0, "hi": 1.0}]}
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(raw))

    cfg = load_data_config(path)

    assert cfg.stratified is None


def test_stratified_config_round_trips_through_yaml(tmp_path: Path) -> None:
    """An enabled stratified section parses with bins and distributions populated."""
    raw = _base_data_yaml()
    raw["stratified"] = _stratified_yaml()
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(raw))

    cfg = load_data_config(path)

    assert cfg.stratified is not None
    assert [b.name for b in cfg.stratified.bins] == ["a", "b", "c"]
    assert cfg.stratified.bins[-1].hi == float("inf")
    assert cfg.stratified.train_distribution == {"a": 1, "b": 2, "c": 3}
    assert cfg.stratified.val_distribution == {"a": 1, "b": 1, "c": 1}
    assert cfg.stratified.test_distribution == {"a": 1, "b": 1, "c": 1}


# --- bin-structure validation ---


def test_stratified_rejects_empty_bins() -> None:
    """A stratified config without any bins is meaningless."""
    with pytest.raises(ValueError, match="bins must be non-empty"):
        _make_stratified(bins=())


def test_stratified_rejects_duplicate_bin_names() -> None:
    """Bin names key the distributions, so duplicates are unambiguous bugs."""
    bins = (
        EncounterBin("dup", 0.0, 0.5),
        EncounterBin("dup", 0.5, float("inf")),
    )
    with pytest.raises(ValueError, match="bin names must be unique"):
        _make_stratified(
            bins=bins,
            train_distribution={"dup": 1.0},
            val_distribution={"dup": 1.0},
            test_distribution={"dup": 1.0},
        )


def test_stratified_rejects_non_finite_bin_lo() -> None:
    """A bin's lower bound must be finite; only the top hi may be inf."""
    bins = (
        EncounterBin("a", float("inf"), float("inf") + 1),  # nonsensical lo
        EncounterBin("b", 0.0, float("inf")),
    )
    with pytest.raises(ValueError, match="non-finite lo"):
        _make_stratified(
            bins=bins,
            train_distribution={"a": 1.0, "b": 1.0},
            val_distribution={"a": 1.0, "b": 1.0},
            test_distribution={"a": 1.0, "b": 1.0},
        )


def test_stratified_rejects_inverted_bin_bounds() -> None:
    """A bin where lo >= hi covers no values; reject."""
    bins = (
        EncounterBin("a", 0.5, 0.5),
        EncounterBin("b", 0.5, float("inf")),
    )
    with pytest.raises(ValueError, match=r"lo=0\.5 must be < hi=0\.5"):
        _make_stratified(
            bins=bins,
            train_distribution={"a": 1.0, "b": 1.0},
            val_distribution={"a": 1.0, "b": 1.0},
            test_distribution={"a": 1.0, "b": 1.0},
        )


def test_stratified_rejects_first_bin_not_starting_at_zero() -> None:
    """The partition must cover [0, inf), so the first bin must start at 0.0."""
    bins = (
        EncounterBin("a", 0.1, 0.5),
        EncounterBin("b", 0.5, float("inf")),
    )
    with pytest.raises(ValueError, match=r"first bin must start at 0\.0"):
        _make_stratified(
            bins=bins,
            train_distribution={"a": 1.0, "b": 1.0},
            val_distribution={"a": 1.0, "b": 1.0},
            test_distribution={"a": 1.0, "b": 1.0},
        )


def test_stratified_rejects_last_bin_not_ending_at_inf() -> None:
    """A finite top hi leaves a half-line uncovered; reject explicitly."""
    bins = (
        EncounterBin("a", 0.0, 0.5),
        EncounterBin("b", 0.5, 1.0),
    )
    with pytest.raises(ValueError, match=r"last bin must end at \+inf"):
        _make_stratified(
            bins=bins,
            train_distribution={"a": 1.0, "b": 1.0},
            val_distribution={"a": 1.0, "b": 1.0},
            test_distribution={"a": 1.0, "b": 1.0},
        )


def test_stratified_rejects_non_contiguous_bins() -> None:
    """Bins must tile [0, inf) without gaps or overlaps."""
    bins = (
        EncounterBin("a", 0.0, 0.5),
        EncounterBin("b", 0.6, float("inf")),  # gap between 0.5 and 0.6
    )
    with pytest.raises(ValueError, match="bins must be contiguous"):
        _make_stratified(
            bins=bins,
            train_distribution={"a": 1.0, "b": 1.0},
            val_distribution={"a": 1.0, "b": 1.0},
            test_distribution={"a": 1.0, "b": 1.0},
        )


# --- distribution validation ---


def test_stratified_rejects_distribution_missing_bin_key() -> None:
    """Every bin must appear in every distribution."""
    with pytest.raises(ValueError, match=r"missing=\['c'\]"):
        _make_stratified(train_distribution={"a": 1.0, "b": 1.0})


def test_stratified_rejects_distribution_extra_key() -> None:
    """Distributions cannot reference unknown bins."""
    with pytest.raises(ValueError, match=r"extra=\['z'\]"):
        _make_stratified(
            train_distribution={"a": 1.0, "b": 1.0, "c": 1.0, "z": 1.0},
        )


def test_stratified_rejects_negative_distribution_weight() -> None:
    """Negative weights have no meaning under largest-remainder."""
    with pytest.raises(ValueError, match=r"train_distribution\['a'\] must be >= 0"):
        _make_stratified(train_distribution={"a": -0.1, "b": 1.0, "c": 1.0})


def test_stratified_rejects_non_finite_distribution_weight() -> None:
    """NaN / +-inf weights cannot be normalised; reject before generation."""
    with pytest.raises(ValueError, match=r"val_distribution\['b'\] must be finite"):
        _make_stratified(val_distribution={"a": 1.0, "b": float("nan"), "c": 1.0})


def test_stratified_rejects_zero_total_distribution_weight() -> None:
    """An all-zero distribution puts every trajectory in nothing."""
    with pytest.raises(ValueError, match=r"test_distribution weights must sum to > 0"):
        _make_stratified(test_distribution={"a": 0.0, "b": 0.0, "c": 0.0})


# --- from_dict required-field checks ---


def test_stratified_from_dict_requires_bins_when_enabled() -> None:
    """Enabling without bins is a config-shape error caught up front."""
    with pytest.raises(ValueError, match=r"stratified\.bins is required"):
        StratifiedConfig.from_dict({"enabled": True})


def test_stratified_from_dict_requires_all_three_distributions() -> None:
    """All three split distributions are mandatory when enabled."""
    payload = _stratified_yaml()
    del payload["test_distribution"]
    with pytest.raises(ValueError, match="stratified requires test_distribution"):
        StratifiedConfig.from_dict(payload)


# --- input shape hardening ---


def test_stratified_rejects_last_bin_with_negative_infinity() -> None:
    """Top sentinel must be exactly +inf; -inf is caught by the lo<hi guard upstream.

    The last-bin `hi != +inf` check is intentionally exact (belt-and-suspenders)
    even though, in practice, any sane `lo >= 0` makes `lo < -inf` fail first.
    This test pins the rejection regardless of which guard fires.
    """
    bins = (
        EncounterBin("a", 0.0, 0.5),
        EncounterBin("b", 0.5, float("-inf")),
    )
    with pytest.raises(ValueError, match=r"hi=-inf"):
        _make_stratified(
            bins=bins,
            train_distribution={"a": 1.0, "b": 1.0},
            val_distribution={"a": 1.0, "b": 1.0},
            test_distribution={"a": 1.0, "b": 1.0},
        )


@pytest.mark.parametrize("bad_name", ["", "   ", "\t\n"])
def test_stratified_rejects_blank_bin_name(bad_name: str) -> None:
    """Empty or whitespace-only names will become useless HDF5 labels later; reject."""
    bins = (
        EncounterBin(bad_name, 0.0, 0.5),
        EncounterBin("b", 0.5, float("inf")),
    )
    with pytest.raises(ValueError, match="bin names must be non-blank"):
        _make_stratified(
            bins=bins,
            train_distribution={bad_name: 1.0, "b": 1.0},
            val_distribution={bad_name: 1.0, "b": 1.0},
            test_distribution={bad_name: 1.0, "b": 1.0},
        )


def test_stratified_rejects_non_string_bin_name() -> None:
    """A non-string name (e.g. int) cannot be persisted reliably; reject."""
    bins = (
        EncounterBin(42, 0.0, 0.5),  # type: ignore[arg-type]
        EncounterBin("b", 0.5, float("inf")),
    )
    with pytest.raises(ValueError, match="bin names must be strings"):
        _make_stratified(
            bins=bins,
            train_distribution={42: 1.0, "b": 1.0},  # type: ignore[dict-item]
            val_distribution={42: 1.0, "b": 1.0},  # type: ignore[dict-item]
            test_distribution={42: 1.0, "b": 1.0},  # type: ignore[dict-item]
        )


@pytest.mark.parametrize("bad_payload", [True, False, [], "yes", 42, None])
def test_stratified_from_dict_rejects_non_mapping(bad_payload: object) -> None:
    """A non-dict YAML payload (e.g. `stratified: true`) is a config-shape bug."""
    with pytest.raises(ValueError, match="stratified must be a mapping"):
        StratifiedConfig.from_dict(bad_payload)  # type: ignore[arg-type]


def test_stratified_from_dict_rejects_non_mapping_bin_entry() -> None:
    """Each entry under bins must be a YAML mapping with name/lo/hi."""
    payload = _stratified_yaml()
    payload["bins"][0] = "not a mapping"
    with pytest.raises(ValueError, match=r"stratified\.bins\[0\] must be a mapping"):
        StratifiedConfig.from_dict(payload)


@pytest.mark.parametrize("missing_key", ["name", "lo", "hi"])
def test_stratified_from_dict_rejects_bin_missing_required_key(missing_key: str) -> None:
    """Each bin entry must declare all three required keys explicitly."""
    payload = _stratified_yaml()
    del payload["bins"][1][missing_key]
    with pytest.raises(
        ValueError, match=rf"stratified\.bins\[1\] missing required key '{missing_key}'"
    ):
        StratifiedConfig.from_dict(payload)


def test_stratified_from_dict_rejects_non_numeric_bin_bound() -> None:
    """A bin whose lo/hi cannot be coerced to float gets a precise error message."""
    payload = _stratified_yaml()
    payload["bins"][0]["lo"] = "abc"
    with pytest.raises(ValueError, match=r"stratified\.bins\[0\] has non-numeric lo/hi"):
        StratifiedConfig.from_dict(payload)


def test_stratified_accepts_max_attempts_override() -> None:
    """An explicit max_attempts in YAML round-trips into the dataclass."""
    payload = _stratified_yaml()
    payload["max_attempts"] = 12345
    cfg = StratifiedConfig.from_dict(payload)

    assert cfg is not None
    assert cfg.max_attempts == 12345


def test_stratified_max_attempts_defaults_to_none() -> None:
    """No `max_attempts` key leaves the override unset so the generator default applies."""
    cfg = StratifiedConfig.from_dict(_stratified_yaml())

    assert cfg is not None
    assert cfg.max_attempts is None


def test_stratified_rejects_non_positive_max_attempts() -> None:
    """A zero or negative cap is meaningless; reject at construction time."""
    with pytest.raises(ValueError, match=r"max_attempts must be an integer >= 1"):
        _make_stratified(max_attempts=0)
    with pytest.raises(ValueError, match=r"max_attempts must be an integer >= 1"):
        _make_stratified(max_attempts=-5)


@pytest.mark.parametrize(
    "bad",
    [
        float("nan"),  # nan < 1 is always False -> would have slipped past a bare < 1 check
        float("inf"),  # inf would silently disable the cap
        10.5,  # non-integer floats are not valid attempt counts
        "100",  # YAML strings must be parsed to int upstream, not accepted here
        True,  # bools subclass int but should not read as a cap of 1
    ],
)
def test_stratified_rejects_non_integer_max_attempts(bad: object) -> None:
    """Type-level guard rejects non-int values that the bare < 1 check would miss."""
    with pytest.raises(ValueError, match=r"max_attempts must be an integer >= 1"):
        _make_stratified(max_attempts=bad)


# --- HDF5 stratification schema ---


def _stratified_trajectories(n_traj: int = 3) -> Trajectories:
    """Build a Trajectories with stratification fields populated for tests."""
    rng = np.random.default_rng(7)
    states = rng.normal(size=(n_traj, 4, 3, 5)).astype(np.float64)
    energies = rng.normal(size=(n_traj, 4)).astype(np.float64)
    bins = (
        EncounterBin("close", 0.0, 0.05),
        EncounterBin("mid", 0.05, 0.5),
        EncounterBin("smooth", 0.5, float("inf")),
    )
    return Trajectories(
        states=states,
        energies=energies,
        encounter_bin_id=np.array([0, 1, 2], dtype=np.int64),
        encounter_bin_name=np.array(["close", "mid", "smooth"]),
        min_pairwise_distance=np.array([0.01, 0.2, 0.7], dtype=np.float64),
        encounter_bins=bins,
    )


def test_round_trip_preserves_stratification_fields(tmp_path: Path) -> None:
    """All four stratification fields survive write -> read intact."""
    original = _stratified_trajectories()
    path = tmp_path / "strat.h5"

    write_trajectories(path, original)
    loaded = read_trajectories(path)

    np.testing.assert_array_equal(loaded.encounter_bin_id, original.encounter_bin_id)
    assert list(loaded.encounter_bin_name) == list(original.encounter_bin_name)
    np.testing.assert_array_equal(loaded.min_pairwise_distance, original.min_pairwise_distance)
    assert loaded.encounter_bins == original.encounter_bins


def test_round_trip_preserves_inf_in_bin_bounds(tmp_path: Path) -> None:
    """Top-of-range hi=+inf round-trips through JSON without becoming a string."""
    original = _stratified_trajectories()
    path = tmp_path / "strat_inf.h5"

    write_trajectories(path, original)
    loaded = read_trajectories(path)

    assert loaded.encounter_bins is not None
    assert loaded.encounter_bins[-1].hi == float("inf")


def test_non_stratified_file_yields_none_for_stratification(tmp_path: Path) -> None:
    """A non-stratified file produces None for all four stratification fields."""
    path = tmp_path / "uniform.h5"
    write_trajectories(path, _example_trajectories())

    loaded = read_trajectories(path)

    assert loaded.encounter_bin_id is None
    assert loaded.encounter_bin_name is None
    assert loaded.min_pairwise_distance is None
    assert loaded.encounter_bins is None


def test_read_states_ignores_stratification_fields(tmp_path: Path) -> None:
    """The hot path used by datasets and evaluators stays unaware of stratification."""
    path = tmp_path / "strat.h5"
    write_trajectories(path, _stratified_trajectories())

    states = read_states(path)

    assert states.shape == (3, 4, 3, 5)


def test_read_raises_on_partial_stratification(tmp_path: Path) -> None:
    """A file with some stratification components but not all is rejected on read."""
    path = tmp_path / "partial.h5"
    write_trajectories(path, _stratified_trajectories())
    # corrupt the file by deleting one of the datasets after a clean write
    with h5py.File(path, "a") as f:
        del f["min_pairwise_distance"]

    with pytest.raises(
        ValueError, match=r"partial stratification metadata; missing \['min_pairwise_distance'\]"
    ):
        read_trajectories(path)


def test_write_rejects_partial_stratification(tmp_path: Path) -> None:
    """A Trajectories with some but not all stratification fields fails to write."""
    t = _stratified_trajectories()
    t = Trajectories(
        states=t.states,
        energies=t.energies,
        encounter_bin_id=t.encounter_bin_id,
        encounter_bin_name=t.encounter_bin_name,
        min_pairwise_distance=None,  # gap
        encounter_bins=t.encounter_bins,
    )

    with pytest.raises(ValueError, match=r"stratification fields are atomic"):
        write_trajectories(tmp_path / "x.h5", t)


def test_write_rejects_mismatched_array_length(tmp_path: Path) -> None:
    """Per-trajectory arrays must have shape (n_trajectories,)."""
    t = _stratified_trajectories()
    bad = Trajectories(
        states=t.states,
        energies=t.energies,
        encounter_bin_id=np.array([0, 1], dtype=np.int64),  # wrong length
        encounter_bin_name=t.encounter_bin_name,
        min_pairwise_distance=t.min_pairwise_distance,
        encounter_bins=t.encounter_bins,
    )

    with pytest.raises(ValueError, match=r"encounter_bin_id shape .* does not match"):
        write_trajectories(tmp_path / "x.h5", bad)


def test_write_rejects_non_integer_bin_id(tmp_path: Path) -> None:
    """encounter_bin_id must be integer-typed; floats / strings are silent bugs."""
    t = _stratified_trajectories()
    bad = Trajectories(
        states=t.states,
        energies=t.energies,
        encounter_bin_id=np.array([0.0, 1.0, 2.0], dtype=np.float64),
        encounter_bin_name=t.encounter_bin_name,
        min_pairwise_distance=t.min_pairwise_distance,
        encounter_bins=t.encounter_bins,
    )

    with pytest.raises(ValueError, match="encounter_bin_id must be integer-typed"):
        write_trajectories(tmp_path / "x.h5", bad)


def test_write_rejects_non_finite_distance(tmp_path: Path) -> None:
    """A NaN / +-inf entry in min_pairwise_distance is rejected before write."""
    t = _stratified_trajectories()
    bad = Trajectories(
        states=t.states,
        energies=t.energies,
        encounter_bin_id=t.encounter_bin_id,
        encounter_bin_name=t.encounter_bin_name,
        min_pairwise_distance=np.array([0.01, float("nan"), 0.7], dtype=np.float64),
        encounter_bins=t.encounter_bins,
    )

    with pytest.raises(ValueError, match="min_pairwise_distance must be finite"):
        write_trajectories(tmp_path / "x.h5", bad)


def test_write_rejects_non_floating_distance(tmp_path: Path) -> None:
    """min_pairwise_distance must be float-typed."""
    t = _stratified_trajectories()
    bad = Trajectories(
        states=t.states,
        energies=t.energies,
        encounter_bin_id=t.encounter_bin_id,
        encounter_bin_name=t.encounter_bin_name,
        min_pairwise_distance=np.array([0, 1, 2], dtype=np.int64),
        encounter_bins=t.encounter_bins,
    )

    with pytest.raises(ValueError, match="min_pairwise_distance must be float-typed"):
        write_trajectories(tmp_path / "x.h5", bad)


def test_write_rejects_id_out_of_range(tmp_path: Path) -> None:
    """An encounter_bin_id outside [0, n_bins) is internally inconsistent."""
    t = _stratified_trajectories()
    bad = Trajectories(
        states=t.states,
        energies=t.energies,
        encounter_bin_id=np.array([0, 1, 5], dtype=np.int64),  # 5 >= 3
        encounter_bin_name=t.encounter_bin_name,
        min_pairwise_distance=t.min_pairwise_distance,
        encounter_bins=t.encounter_bins,
    )

    with pytest.raises(ValueError, match=r"encounter_bin_id\[2\]=5 is out of range \[0, 3\)"):
        write_trajectories(tmp_path / "x.h5", bad)


def test_write_rejects_negative_id(tmp_path: Path) -> None:
    """Negative ids never index a real bin; reject them on write."""
    t = _stratified_trajectories()
    bad = Trajectories(
        states=t.states,
        energies=t.energies,
        encounter_bin_id=np.array([0, -1, 2], dtype=np.int64),
        encounter_bin_name=t.encounter_bin_name,
        min_pairwise_distance=t.min_pairwise_distance,
        encounter_bins=t.encounter_bins,
    )

    with pytest.raises(ValueError, match=r"encounter_bin_id\[1\]=-1 is out of range"):
        write_trajectories(tmp_path / "x.h5", bad)


def test_write_rejects_name_inconsistent_with_id(tmp_path: Path) -> None:
    """encounter_bin_name[i] must match the name of bins[encounter_bin_id[i]]."""
    t = _stratified_trajectories()
    bad = Trajectories(
        states=t.states,
        energies=t.energies,
        encounter_bin_id=t.encounter_bin_id,
        encounter_bin_name=np.array(["close", "smooth", "smooth"]),  # index 1 wrong
        min_pairwise_distance=t.min_pairwise_distance,
        encounter_bins=t.encounter_bins,
    )

    with pytest.raises(
        ValueError, match=r"encounter_bin_name\[1\]='smooth' does not match bin 1 name 'mid'"
    ):
        write_trajectories(tmp_path / "x.h5", bad)


def test_write_rejects_distance_outside_assigned_bin(tmp_path: Path) -> None:
    """min_pairwise_distance[i] must fall in the half-open interval of its bin."""
    t = _stratified_trajectories()
    bad = Trajectories(
        states=t.states,
        energies=t.energies,
        encounter_bin_id=t.encounter_bin_id,
        encounter_bin_name=t.encounter_bin_name,
        # bin id 0 is "close" with [0.0, 0.05); 0.2 is outside that interval
        min_pairwise_distance=np.array([0.2, 0.2, 0.7], dtype=np.float64),
        encounter_bins=t.encounter_bins,
    )

    with pytest.raises(ValueError, match=r"min_pairwise_distance\[0\]=0\.2 is outside bin 'close'"):
        write_trajectories(tmp_path / "x.h5", bad)


def test_encoded_bins_attr_is_strict_json_with_inf_sentinel(tmp_path: Path) -> None:
    """The persisted JSON attr is parseable by strict (allow_nan=False) decoders.

    This pins the `+inf` -> `"inf"` sentinel encoding so files remain
    portable and don't rely on Python's `Infinity` extension.
    """
    path = tmp_path / "strat.h5"
    write_trajectories(path, _stratified_trajectories())

    with h5py.File(path, "r") as f:
        attr = f.attrs["encounter_bins_json"]
    assert "Infinity" not in attr
    payload = json.loads(attr)
    # the sentinel survives as the literal string "inf"
    assert payload[-1]["hi"] == "inf"


def test_read_validates_corrupted_name_field(tmp_path: Path) -> None:
    """A file edited to break id<->name agreement raises on read, not silently load."""
    path = tmp_path / "corrupt.h5"
    write_trajectories(path, _stratified_trajectories())

    with h5py.File(path, "a") as f:
        del f["encounter_bin_name"]
        f.create_dataset(
            "encounter_bin_name",
            data=np.asarray(["smooth", "smooth", "smooth"], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    with pytest.raises(ValueError, match=r"encounter_bin_name\[0\]='smooth' does not match"):
        read_trajectories(path)


def test_write_rejects_empty_bins_when_arrays_present(tmp_path: Path) -> None:
    """Empty bins tuple alongside populated arrays is internally inconsistent.

    `()` is `is not None` so it passes the atomicity check; the dedicated
    non-empty guard catches it.
    """
    t = _stratified_trajectories()
    bad = Trajectories(
        states=t.states,
        energies=t.energies,
        encounter_bin_id=t.encounter_bin_id,
        encounter_bin_name=t.encounter_bin_name,
        min_pairwise_distance=t.min_pairwise_distance,
        encounter_bins=(),
    )

    with pytest.raises(ValueError, match="encounter_bins must be non-empty"):
        write_trajectories(tmp_path / "x.h5", bad)
