"""Tests for importing the reciprocal benchmark without modifying model code."""

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from data._io import read_states, read_trajectories
from data.import_daniela import convert, load_csv, sha256, softened_energy, split_indices


def _fixture(path: Path, n_simulations: int = 10) -> list[list]:
    """Write a small grouped CSV with unequal masses and recognizable features."""
    header = ["id_sim", "time"] + [
        f"{key}{i}" for i in range(1, 4) for key in ("m", "x", "y", "vx", "vy")
    ]
    rows = []
    for sid in range(n_simulations):
        for frame in range(4):
            row = [sid, frame * 0.001]
            for body in range(3):
                row.extend(
                    [0.5 + body * 0.5, sid + body + frame * 0.001, -body, 0.1 * body, -0.2 * body]
                )
            rows.append(row)
    _write(path, header, rows)
    return rows


def _write(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def test_feature_order_and_precision(tmp_path: Path) -> None:
    """Preserve numeric values while moving mass from first to last feature."""
    source = tmp_path / "input.csv"
    _fixture(source)
    ids, times, states = load_csv(source)
    np.testing.assert_array_equal(ids, np.arange(10))
    assert states.shape == (10, 4, 3, 5)
    assert states.dtype == np.float64
    np.testing.assert_array_equal(states[2, 0, 1], [3, -1, 0.1, -0.2, 1.0])
    np.testing.assert_array_equal(times[0], [0, 0.001, 0.002, 0.003])


def test_splits_are_disjoint_and_reproducible() -> None:
    """Reserve 80 validation IDs from the original 400, never from test."""
    splits, pool = split_indices(500)
    assert {k: len(v) for k, v in splits.items()} == {"train": 320, "val": 80, "test": 100}
    sets = {k: set(v.tolist()) for k, v in splits.items()}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]
    assert sets["train"] | sets["val"] == set(pool.tolist())
    assert set.union(*sets.values()) == set(range(500))
    again, _ = split_indices(500)
    for key in splits:
        np.testing.assert_array_equal(splits[key], again[key])


def test_split_matches_original_sklearn_notebook() -> None:
    """Verify both split orders against sklearn when that optional checker exists."""
    sklearn = pytest.importorskip("sklearn.model_selection")
    ids = np.arange(500)[::-1]  # original first-seen ID order must be preserved
    original_train, original_test = sklearn.train_test_split(ids, test_size=0.2, random_state=42)
    train, val = sklearn.train_test_split(original_train, test_size=0.2, random_state=42)
    splits, pool = split_indices(len(ids))
    np.testing.assert_array_equal(ids[pool], original_train)
    np.testing.assert_array_equal(ids[splits["test"]], original_test)
    np.testing.assert_array_equal(ids[splits["train"]], train)
    np.testing.assert_array_equal(ids[splits["val"]], val)


@pytest.mark.parametrize("n_simulations", [0, 1, 2])
def test_too_few_simulations(n_simulations: int) -> None:
    """Reject splits with empty partitions."""
    with pytest.raises(ValueError, match="at least one"):
        split_indices(n_simulations)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, float("nan")])
def test_invalid_fractions(fraction: float) -> None:
    """Validate fractions before generating any files."""
    with pytest.raises(ValueError, match="strictly between"):
        split_indices(10, val_fraction=fraction)


def test_softened_energy_analytic() -> None:
    """Use the mass product and epsilon squared in the physical energy."""
    states = np.array([[[[0, 0, 1, 0, 2], [3, 4, 0, 2, 3]]]], dtype=float)
    expected = 7.0 - 6.0 / np.sqrt(25 + 0.05**2)
    np.testing.assert_allclose(softened_energy(states, G=1, epsilon=0.05), [[expected]])


def test_conversion_reads_through_existing_io(tmp_path: Path) -> None:
    """Existing HDF5 readers accept imports without fictitious scalar-mass metadata."""
    source = tmp_path / "input.csv"
    _fixture(source)
    before = source.read_bytes()
    manifest = convert(source, tmp_path / "converted")
    assert source.read_bytes() == before
    assert manifest["source"]["sha256"] == sha256(source)
    assert json.loads((tmp_path / "converted/manifest.json").read_text()) == manifest
    for name, output in manifest["outputs"].items():
        path = tmp_path / "converted" / output["file"]
        bundle = read_trajectories(path)
        assert bundle.metadata is None
        assert bundle.encounter_bin_id is None
        np.testing.assert_array_equal(read_states(path), bundle.states)
        np.testing.assert_array_equal(
            bundle.energies, softened_energy(bundle.states, G=1, epsilon=0.05)
        )
        assert sha256(path) == output["sha256"]
        with h5py.File(path) as handle:
            assert handle.attrs["epsilon"] == 0.05
            assert handle.attrs["split"] == name
            np.testing.assert_array_equal(handle["simulation_ids"][:], output["simulation_ids"])
            for i, sid in enumerate(output["simulation_ids"]):
                assert handle["trajectories"][i, 0, 0, 0] == sid


def test_no_overwrite(tmp_path: Path) -> None:
    """A second import cannot replace existing benchmark artifacts."""
    source = tmp_path / "input.csv"
    _fixture(source)
    output = tmp_path / "converted"
    convert(source, output)
    original = (output / "train.h5").read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        convert(source, output)
    assert (output / "train.h5").read_bytes() == original


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (3, "NaN", "non-finite"),
        (3, "", "invalid"),
        (2, 0, "positive and constant"),
        (2, 1.8, "positive and constant"),
        (1, 0, "distinct increasing"),
        (1, 0.0015, "nonuniform"),
    ],
)
def test_reject_bad_rows(tmp_path: Path, column: int, value: object, message: str) -> None:
    """Reject missing/nonfinite values, changed mass, duplicate times and gaps."""
    source = tmp_path / "input.csv"
    rows = _fixture(source)
    with source.open() as stream:
        header = next(csv.reader(stream))
    rows[1][column] = value
    _write(source, header, rows)
    with pytest.raises(ValueError, match=message):
        convert(source, tmp_path / "converted")
    assert not (tmp_path / "converted").exists()


def test_sort_frames_within_simulation(tmp_path: Path) -> None:
    """Sort timestamps without changing simulation first-appearance order."""
    source = tmp_path / "input.csv"
    rows = _fixture(source)
    with source.open() as stream:
        header = next(csv.reader(stream))
    rows[:4] = rows[:4][::-1]
    _write(source, header, rows)
    _, times, _ = load_csv(source)
    np.testing.assert_array_equal(times[0], [0, 0.001, 0.002, 0.003])


def test_reject_inconsistent_frames(tmp_path: Path) -> None:
    """Prevent silently truncating a shorter trajectory."""
    source = tmp_path / "input.csv"
    rows = _fixture(source)
    with source.open() as stream:
        header = next(csv.reader(stream))
    _write(source, header, rows[:-1])
    with pytest.raises(ValueError, match="frame count/time grid"):
        load_csv(source)


def test_reject_interleaved_blocks(tmp_path: Path) -> None:
    """Do not split one simulation into multiple trajectories."""
    source = tmp_path / "input.csv"
    rows = _fixture(source)
    with source.open() as stream:
        header = next(csv.reader(stream))
    _write(source, header, rows + rows[:4])
    with pytest.raises(ValueError, match="interleaved"):
        load_csv(source)
