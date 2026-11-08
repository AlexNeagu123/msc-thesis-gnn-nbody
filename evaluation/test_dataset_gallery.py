"""Ground-truth gallery tests without model or training dependencies."""

import csv
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from data.import_daniela import convert, sha256
from evaluation.dataset_gallery import closest_approach, create_gallery, select_examples


def _states() -> np.ndarray:
    states = np.zeros((5, 3, 5), dtype=np.float64)
    states[:, :, 0] = [0, 1, 4]
    states[:, :, 4] = [0.5, 1, 2]
    states[2, 1, 0] = 0.01
    return states


@pytest.fixture
def imported(tmp_path: Path) -> Path:
    """Create a small valid import with >=3 members in each split."""
    source = tmp_path / "source.csv"
    header = ["id_sim", "time"] + [
        f"{key}{i}" for i in range(1, 4) for key in ("m", "x", "y", "vx", "vy")
    ]
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for sid in range(20):
            states = _states()
            states[:, :, 0] *= 1 + sid * 0.1
            for frame, state in enumerate(states):
                writer.writerow([sid, frame * 0.001, *state[:, [4, 0, 1, 2, 3]].ravel()])
    output = tmp_path / "imported"
    convert(source, output)
    return output


def test_closest_approach() -> None:
    """Locate the actual frame/pair and keep the source array unchanged."""
    states = _states()
    before = states.copy()
    distance, frame, pair = closest_approach(states)
    assert distance == pytest.approx(0.01)
    assert frame == 2
    assert pair == (0, 1)
    np.testing.assert_array_equal(states, before)


@pytest.mark.parametrize("shape", [(3, 5), (5, 2, 5), (1, 3, 5), (5, 3, 4)])
def test_invalid_shape(shape: tuple[int, ...]) -> None:
    """Reject arrays that cannot represent an imported trajectory."""
    with pytest.raises(ValueError, match="shape"):
        closest_approach(np.zeros(shape))


def test_nonfinite_values() -> None:
    """Do not produce misleading plots from corrupt inputs."""
    states = _states()
    states[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        closest_approach(states)


def test_tie_breaking_by_id(tmp_path: Path) -> None:
    """Ties use original IDs, not storage order or unseeded randomness."""
    path = tmp_path / "ties.h5"
    with h5py.File(path, "w") as handle:
        handle["trajectories"] = np.stack([_states()] * 3)
        handle["simulation_ids"] = [30, 10, 20]
        handle["times"] = np.tile(np.arange(5) * 0.001, (3, 1))
    records = select_examples(path)
    assert [r["simulation_id"] for r in records] == [10, 20, 30]
    assert [r["h5_index"] for r in records] == [1, 2, 0]
    assert records == select_examples(path)
    assert records[0]["closest_pair"] == [1, 2]
    assert records[0]["encounter_time_label"] == 0.002


def test_gallery_artifacts(imported: Path, tmp_path: Path) -> None:
    """Render valid PNG/HTML/JSON with exact IDs, hashes and unchanged inputs."""
    before = {p.name: sha256(p) for p in imported.iterdir()}
    output = tmp_path / "gallery"
    report = create_gallery(imported, output, half_window=2)
    assert json.loads((output / "selection.json").read_text()) == report
    assert {p.name: sha256(p) for p in imported.iterdir()} == before
    seen = set()
    for split in ("train", "val", "test"):
        with h5py.File(imported / f"{split}.h5") as handle:
            ids = set(handle["simulation_ids"][:].tolist())
        chosen = {r["simulation_id"] for r in report["selection"][split]}
        assert len(chosen) == 3
        assert chosen <= ids
        assert seen.isdisjoint(chosen)
        seen.update(chosen)
        assert f'src="{split}.png"' in (output / "index.html").read_text()
    for name, entry in report["outputs"].items():
        assert sha256(output / name) == entry["sha256"]
        if name.endswith(".png"):
            with Image.open(output / name) as image:
                assert image.width >= 1000
                assert image.height >= 1000
                image.verify()
    assert not plt.get_fignums()


def test_input_checksum_mismatch(imported: Path, tmp_path: Path) -> None:
    """Refuse a changed dataset instead of plotting with stale provenance."""
    with (imported / "train.h5").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        create_gallery(imported, tmp_path / "gallery")
    assert not (tmp_path / "gallery").exists()


def test_no_overwrite(tmp_path: Path) -> None:
    """Never replace an existing gallery directory."""
    output = tmp_path / "gallery"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        create_gallery(tmp_path / "missing-input", output)
    assert (output / "keep.txt").read_text() == "keep"


def test_nonpositive_window(tmp_path: Path) -> None:
    """Reject an invalid zoom configuration before reading data."""
    with pytest.raises(ValueError, match="positive"):
        create_gallery(tmp_path / "missing-input", tmp_path / "gallery", half_window=0)


def test_render_failure_cleans_staging(
    imported: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed rendering leaves no partial gallery or dangling figures."""

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("render failed")

    monkeypatch.setattr("evaluation.dataset_gallery._panel", fail)
    with pytest.raises(RuntimeError, match="render failed"):
        create_gallery(imported, tmp_path / "gallery")
    assert not (tmp_path / "gallery").exists()
    assert not list(tmp_path.glob(".gallery-*"))
    assert not plt.get_fignums()
