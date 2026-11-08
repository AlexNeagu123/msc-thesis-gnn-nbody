"""Test ground-truth animation sampling, provenance and MP4 encoding."""

import json
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import writers

from data.import_daniela import sha256
from evaluation.animate_dataset import animate_gallery, frame_indices, render_video
from evaluation.dataset_gallery import create_gallery
from evaluation.test_dataset_gallery import imported as imported


def _example() -> tuple[np.ndarray, np.ndarray, dict]:
    times = np.arange(8) * 0.001
    states = np.zeros((8, 3, 5))
    states[:, :, 4] = [0.5, 1, 2]
    states[:, :, 0] = [0, 0.2, 1]
    states[:, :, 1] = times[:, None] * np.array([1, -1, 2])[None, :]
    record = {"split": "train", "simulation_id": 7, "selection": "test fixture"}
    return states, times, record


def test_sampling_endpoints() -> None:
    """Cover the whole trajectory, without duplicate or invented states."""
    indices = frame_indices(2000, 300)
    assert len(indices) == 300
    assert indices[0] == 0
    assert indices[-1] == 1999
    assert np.all(np.diff(indices) > 0)
    np.testing.assert_array_equal(frame_indices(5, 300), np.arange(5))


@pytest.mark.parametrize(("n_states", "requested"), [(1, 10), (5, 1), (0, 0)])
def test_invalid_frame_counts(n_states: int, requested: int) -> None:
    """Reject video requests that cannot show a path."""
    with pytest.raises(ValueError, match="at least two"):
        frame_indices(n_states, requested)


def test_invalid_times(tmp_path: Path) -> None:
    """Never render an animation with mismatched timestamps."""
    states, times, record = _example()
    with pytest.raises(ValueError, match="times must match"):
        render_video(states, times[:-1], record, tmp_path / "bad.mp4")


@pytest.mark.skipif(
    not writers.is_available("ffmpeg") or not shutil.which("ffprobe"),
    reason="requires ffmpeg and ffprobe",
)
def test_real_mp4(tmp_path: Path) -> None:
    """Encode and probe a tiny real H.264 video; preserve original inputs."""
    states, times, record = _example()
    before = states.copy()
    target = tmp_path / "smoke.mp4"
    result = render_video(states, times, record, target, fps=6, n_frames=6)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,nb_frames,pix_fmt",
            "-of",
            "json",
            str(target),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["codec_name"] == "h264"
    assert stream["pix_fmt"] == "yuv420p"
    assert (stream["width"], stream["height"]) == (1200, 600)
    assert int(stream["nb_frames"]) == 6
    assert result["duration_seconds"] == 1
    assert result["last_sample"] == 7
    np.testing.assert_array_equal(states, before)
    assert not plt.get_fignums()


def test_frozen_selection_and_viewer(
    imported: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reuse all nine existing selections; leave the static viewer unchanged."""
    gallery = tmp_path / "static"
    selection = create_gallery(imported, gallery)
    before = {p.name: sha256(p) for p in gallery.iterdir()}
    calls = []

    def fake_render(
        states: np.ndarray, times: np.ndarray, record: dict, path: Path, **kwargs: object
    ) -> dict:
        path.write_bytes(b"mock video")
        calls.append((record["split"], record["simulation_id"]))
        return {"frames": 5, "duration_seconds": 1}

    monkeypatch.setattr("evaluation.animate_dataset.render_video", fake_render)
    monkeypatch.setattr("evaluation.animate_dataset.writers.is_available", lambda _: True)
    output = tmp_path / "animated"
    report = animate_gallery(imported, gallery, output)
    assert len(calls) == 9
    assert calls == [
        (split, record["simulation_id"])
        for split, records in selection["selection"].items()
        for record in records
    ]
    assert {p.name: sha256(p) for p in gallery.iterdir()} == before
    assert json.loads((output / "animations.json").read_text()) == report
    assert (output / "index.html").read_text().count("<video controls") == 9
    for video in report["videos"]:
        assert sha256(output / video["file"]) == video["sha256"]


def test_no_overwrite(tmp_path: Path) -> None:
    """Do not replace previous animation runs."""
    output = tmp_path / "keep"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        animate_gallery(tmp_path, tmp_path, output)
