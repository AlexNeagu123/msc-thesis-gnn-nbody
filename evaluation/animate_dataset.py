"""Animate the frozen Daniela ground-truth gallery without training dependencies."""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import tempfile
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, writers
from matplotlib.patches import Circle

from data.import_daniela import sha256
from evaluation.dataset_gallery import COLORS, SPLITS, closest_approach


def frame_indices(n_states: int, requested: int) -> np.ndarray:
    """Subsample uniformly in recorded-frame index, including both endpoints."""
    if n_states < 2 or requested < 2:
        raise ValueError("need at least two recorded states and two requested video frames")
    return np.linspace(0, n_states - 1, min(n_states, requested), dtype=int)


def render_video(
    states: np.ndarray,
    times: np.ndarray,
    record: dict,
    path: Path,
    *,
    fps: int = 30,
    n_frames: int = 300,
    epsilon: float = 0.05,
) -> dict:
    """Render H.264 with fixed full-system and encounter-neighbourhood views.

    Marker AREA is proportional to mass, not physical body radius. Displayed
    states are subsampled; trails include all original samples up to each frame.
    No interpolation, prediction, simulation or moving-camera transform occurs.
    """
    distance, encounter, pair = closest_approach(states)
    if fps < 1 or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("fps and epsilon must be positive")
    if times.shape != (len(states),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("times must match states and be finite and increasing")
    masses = states[0, :, 4]
    if np.any(masses <= 0) or not np.allclose(states[..., 4], masses, rtol=0, atol=1e-12):
        raise ValueError("masses must be positive and constant")
    indices = frame_indices(len(states), n_frames)
    if not writers.is_available("ffmpeg"):
        raise RuntimeError("ffmpeg is required for MP4 output")
    pos = states[..., :2]
    lower, upper = pos.min(axis=(0, 1)), pos.max(axis=(0, 1))
    full_center = (lower + upper) / 2
    full_radius = max(float((upper - lower).max()) * 0.56, epsilon)
    zoom_center = pos[encounter, list(pair)].mean(axis=0)
    neighbourhood = pos[max(0, encounter - 50) : min(len(pos), encounter + 51), list(pair)]
    zoom_radius = max(2 * epsilon, float(np.abs(neighbourhood - zoom_center).max()) * 1.15)
    sizes = 85 * masses / masses.mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
    try:
        title = f"{record['split'].upper()} · ID {record['simulation_id']} · {record['selection']}"
        fig.suptitle(title, fontsize=14)
        clock = fig.text(0.5, 0.885, "", ha="center", fontsize=11, family="monospace")
        fig.text(
            0.5,
            0.04,
            "Ground truth only · marker area ∝ mass · fixed axes · trails contain all original past samples",
            ha="center",
            fontsize=9,
        )
        trails, dots = [], []
        for ax, center, radius, label in zip(
            axes,
            (full_center, zoom_center),
            (full_radius, zoom_radius),
            (
                "Full system",
                f"Fixed encounter view · pair {pair[0] + 1}/{pair[1] + 1} · t*={times[encounter]:.3f}",
            ),
            strict=True,
        ):
            ax.set(
                xlim=(center[0] - radius, center[0] + radius),
                ylim=(center[1] - radius, center[1] + radius),
                xlabel="x [AU]",
                ylabel="y [AU]",
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(label, fontsize=10)
            ax.grid(alpha=0.2)
            panel_trails = []
            for body, color in enumerate(COLORS):
                (line,) = ax.plot([], [], color=color, linewidth=1.1, alpha=0.7)
                panel_trails.append(line)
                ax.scatter(*pos[0, body], s=24, facecolors="none", edgecolors=color, alpha=0.7)
            dots.append(
                ax.scatter(
                    pos[0, :, 0],
                    pos[0, :, 1],
                    s=sizes,
                    c=COLORS,
                    edgecolors="white",
                    linewidths=0.7,
                    zorder=4,
                )
            )
            trails.append(panel_trails)
        axes[1].add_patch(
            Circle(
                pos[encounter, pair[0]],
                epsilon,
                fill=False,
                color="gray",
                linestyle=":",
                linewidth=1.1,
            )
        )
        legend = [
            plt.Line2D([], [], color=color, marker="o", label=f"Body {i + 1}: m={mass:.3f}")
            for i, (color, mass) in enumerate(zip(COLORS, masses, strict=True))
        ]
        fig.legend(
            handles=legend,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.075),
            ncol=3,
            frameon=False,
            fontsize=9,
        )
        fig.subplots_adjust(left=0.07, right=0.97, bottom=0.2, top=0.80, wspace=0.28)
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            extra_args=[
                "-crf",
                "23",
                "-preset",
                "fast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ],
        )
        with writer.saving(fig, str(path), dpi=100):
            for index in indices:
                clock.set_text(
                    f"Recorded t = {times[index]:.3f} TU  |  sample {index + 1}/{len(states)}"
                )
                for panel in range(2):
                    dots[panel].set_offsets(pos[index])
                    for body in range(3):
                        trails[panel][body].set_data(
                            pos[: index + 1, body, 0], pos[: index + 1, body, 1]
                        )
                writer.grab_frame(facecolor="white")
    finally:
        plt.close(fig)
    return {
        "frames": len(indices),
        "fps": fps,
        "duration_seconds": len(indices) / fps,
        "first_sample": int(indices[0]),
        "last_sample": int(indices[-1]),
        "minimum_separation": distance,
        "encounter_frame": encounter,
        "source_time_labels": [float(times[0]), float(times[-1])],
    }


def animate_gallery(
    data_dir: Path,
    gallery_dir: Path,
    output_dir: Path,
    *,
    fps: int = 30,
    n_frames: int = 300,
) -> dict:
    """Animate exactly the existing selections; publish a separate atomic viewer."""
    data_dir, gallery_dir, output_dir = (
        data_dir.resolve(),
        gallery_dir.resolve(),
        output_dir.resolve(),
    )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite animation directory: {output_dir}")
    if fps < 1 or n_frames < 2:
        raise ValueError("fps must be positive; n_frames must be at least two")
    if not writers.is_available("ffmpeg"):
        raise RuntimeError("ffmpeg is required for MP4 output")
    selection_path = gallery_dir / "selection.json"
    selection = json.loads(selection_path.read_text())
    imported = json.loads((data_dir / "manifest.json").read_text())
    if sha256(data_dir / "manifest.json") != selection["import_manifest_sha256"]:
        raise ValueError("import manifest does not match frozen gallery selection")
    paths = {}
    for split in SPLITS:
        entry = selection["inputs"][split]
        if Path(entry["file"]).name != entry["file"]:
            raise ValueError("expected a simple input filename")
        paths[split] = data_dir / entry["file"]
        if sha256(paths[split]) != entry["sha256"]:
            raise ValueError(f"input checksum mismatch: {split}")
    report = {
        "schema_version": 1,
        "kind": "ground-truth-only MP4 gallery",
        "selection_manifest_sha256": sha256(selection_path),
        "script_sha256": sha256(Path(__file__)),
        "inputs": selection["inputs"],
        "display_note": "Uniform subsampling without interpolation; trails retain all original past states. Playback seconds are not physical seconds. Fixed camera; marker area proportional to mass, not physical radius.",
        "videos": [],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        sections = []
        for split in SPLITS:
            cards = []
            with h5py.File(paths[split], "r") as handle:
                for selected in selection["selection"][split]:
                    record = dict(selected, split=split)
                    index = record["h5_index"]
                    if int(handle["simulation_ids"][index]) != record["simulation_id"]:
                        raise ValueError("frozen selection ID disagrees with input")
                    filename = f"{split}-id-{record['simulation_id']}.mp4"
                    details = render_video(
                        handle["trajectories"][index],
                        handle["times"][index],
                        record,
                        staging / filename,
                        fps=fps,
                        n_frames=n_frames,
                        epsilon=float(imported["physics"]["epsilon"]),
                    )
                    details.update(
                        file=filename,
                        split=split,
                        simulation_id=record["simulation_id"],
                        selection=record["selection"],
                        sha256=sha256(staging / filename),
                    )
                    report["videos"].append(details)
                    caption = html.escape(f"ID {record['simulation_id']} — {record['selection']}")
                    cards.append(
                        f'<article><h3>{caption}</h3><video controls loop playsinline preload="none" src="{filename}"></video><p><a href="{filename}">Open/download MP4</a></p></article>'
                    )
                    print(
                        f"Rendered {filename}: {details['frames']} frames, {details['duration_seconds']:.1f}s",
                        flush=True,
                    )
            sections.append(f"<h2>{split.upper()}</h2>" + "\n".join(cards))
        (staging / "index.html").write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>Daniela animated trajectories</title>'
            "<style>body{max-width:1300px;margin:2rem auto;padding:0 1rem;font:16px system-ui;background:#f7f7f7;color:#222}video{width:100%;background:white}article{margin:2rem 0}p{line-height:1.6}a{color:#00699b}</style>"
            "<h1>Daniela — animated ground-truth trajectories</h1><p>Press play on any example. "
            "Each video has a fixed full-system view and a fixed closest-encounter view; bodies may leave the zoom. "
            "Marker area is proportional to mass. Trails show the original sampled history. "
            "The counter uses recorded TU labels; playback is deliberately slowed, not physical real time. "
            "These are the same nine frozen examples as the static gallery, with no predictions or new simulation. "
            'Test examples must not be used to tune models.</p><p><a href="animations.json">Video provenance (JSON)</a></p>'
            + "\n".join(sections)
            + "</html>\n"
        )
        report["viewer_sha256"] = sha256(staging / "index.html")
        (staging / "animations.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        for split in SPLITS:
            if sha256(paths[split]) != selection["inputs"][split]["sha256"]:
                raise ValueError(f"input changed during rendering: {split}")
        os.rename(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report


def main() -> None:
    """Render the nine frozen examples and a browser-native playback page."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/output/daniela-rk4"))
    parser.add_argument(
        "--gallery-dir", type=Path, default=Path("results/evaluation/daniela-dataset-gallery")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/evaluation/daniela-dataset-gallery/animations"),
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()
    animate_gallery(
        args.data_dir, args.gallery_dir, args.output_dir, fps=args.fps, n_frames=args.frames
    )
    print(f"Open: {(args.output_dir / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
