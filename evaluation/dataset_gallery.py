"""Reproducible, ground-truth-only gallery of the imported Daniela dataset.

No training/model dependencies; uses NumPy, h5py and Matplotlib only.
Selections describe separation, not a measured ranking of chaos or difficulty.
"""

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
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from data.import_daniela import sha256

SPLITS = ("train", "val", "test")
COLORS = ("#0072B2", "#D55E00", "#009E73")
MODES = ("closest encounter", "median minimum separation", "widest minimum separation")


def closest_approach(states: np.ndarray) -> tuple[float, int, tuple[int, int]]:
    """Return the smallest sampled pair distance, frame index and zero-based pair."""
    if states.ndim != 3 or states.shape[1:] != (3, 5) or len(states) < 2:
        raise ValueError("expected one three-body trajectory with shape (frames>=2, 3, 5)")
    if not np.isfinite(states).all():
        raise ValueError("non-finite trajectory values")
    pairs = ((0, 1), (0, 2), (1, 2))
    distances = np.stack(
        [np.linalg.norm(states[:, i, :2] - states[:, j, :2], axis=-1) for i, j in pairs],
        axis=1,
    )
    frame, pair = np.unravel_index(np.argmin(distances), distances.shape)
    return float(distances[frame, pair]), int(frame), pairs[pair]


def select_examples(path: Path) -> list[dict]:
    """Choose closest, median and widest examples deterministically within ONE split.

    Sort by minimum sampled distance, breaking ties by original simulation ID.
    Three distinct examples are required. This does not change split membership.
    """
    with h5py.File(path, "r") as handle:
        states = handle["trajectories"]
        ids = handle["simulation_ids"][:]
        if len(ids) < 3 or len(set(ids.tolist())) != len(ids) or len(states) != len(ids):
            raise ValueError("gallery requires at least three distinct simulation IDs per split")
        records = []
        for index, simulation_id in enumerate(ids):
            distance, frame, pair = closest_approach(states[index])
            records.append(
                {
                    "h5_index": index,
                    "simulation_id": int(simulation_id),
                    "minimum_separation": distance,
                    "encounter_frame": frame,
                    "encounter_time_label": float(handle["times"][index, frame]),
                    "closest_pair": [i + 1 for i in pair],
                    "masses": states[index, 0, :, 4].tolist(),
                }
            )
    order = sorted(
        range(len(records)),
        key=lambda i: (records[i]["minimum_separation"], records[i]["simulation_id"]),
    )
    return [
        dict(records[order[rank]], selection=mode, separation_rank=rank)
        for mode, rank in zip(MODES, (0, len(order) // 2, len(order) - 1), strict=True)
    ]


def _panel(
    ax: plt.Axes,
    states: np.ndarray,
    times: np.ndarray,
    record: dict,
    epsilon: float,
    half_window: int,
    *,
    zoom: bool,
) -> None:
    """Draw original coordinates with equal axes and optional local encounter window."""
    pos = states[..., :2]
    frame = record["encounter_frame"]
    pair = np.asarray(record["closest_pair"]) - 1
    if zoom:
        lo, hi = max(0, frame - half_window), min(len(pos), frame + half_window + 1)
        visible = pos[lo:hi]
        center = pos[frame, pair].mean(axis=0)
        radius = max(1.5 * epsilon, float(np.abs(visible[:, pair] - center).max()) * 1.15)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.add_patch(
            Circle(
                pos[frame, pair[0]],
                epsilon,
                fill=False,
                color="#666666",
                linestyle=":",
                linewidth=1.2,
            )
        )
        ax.set_title(
            f"Encounter neighbourhood: t={times[lo]:.3f} to {times[hi - 1]:.3f}\n"
            f"Stars: pair {pair[0] + 1}/{pair[1] + 1} at t={times[frame]:.3f}; dotted radius ε={epsilon:g}",
            fontsize=9,
        )
    else:
        visible = pos
        lower = pos.min(axis=(0, 1))
        upper = pos.max(axis=(0, 1))
        center = 0.5 * (lower + upper)
        radius = max(float((upper - lower).max()) * 0.55, epsilon)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        masses = ", ".join(f"{mass:.3f}" for mass in record["masses"])
        ax.set_title(
            f"ID {record['simulation_id']} · {record['selection']}\n"
            f"m=({masses}); d_min={record['minimum_separation']:.5g}\n"
            f"Recorded t={times[0]:.3f} to {times[-1]:.3f} TU",
            fontsize=10,
        )
    for body, color in enumerate(COLORS):
        trail = visible[:, body]
        ax.plot(trail[:, 0], trail[:, 1], color=color, linewidth=1.2, alpha=0.9)
        if not zoom:
            ax.scatter(*pos[0, body], s=32, facecolors="none", edgecolors=color, zorder=4)
            ax.scatter(*pos[-1, body], s=27, marker="s", color=color, zorder=4)
            # Direction arrow along the original sampled path.
            at = len(trail) // 2
            end = min(at + max(1, len(trail) // 50), len(trail) - 1)
            ax.annotate(
                "",
                xy=trail[end],
                xytext=trail[at],
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.3},
            )
        if body in pair:
            ax.scatter(
                *pos[frame, body],
                marker="*",
                color=color,
                s=75,
                edgecolors="black",
                linewidths=0.4,
                zorder=5,
            )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=8)


def _legend(fig: plt.Figure) -> None:
    """Explain body colors and the recorded-state markers consistently."""
    handles = [Line2D([], [], color=c, label=f"Body {i + 1}") for i, c in enumerate(COLORS)]
    handles += [
        Line2D(
            [],
            [],
            color="black",
            marker="o",
            markerfacecolor="none",
            linestyle="none",
            label="first recorded",
        ),
        Line2D([], [], color="black", marker="s", linestyle="none", label="last recorded"),
        Line2D([], [], color="black", marker="*", linestyle="none", label="closest pair"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9, frameon=False)


def create_gallery(data_dir: Path, output_dir: Path, *, half_window: int = 50) -> dict:
    """Save overview, per-split zooms, HTML viewer and a checksummed selection manifest.

    Verify input hashes against the import manifest; refuse to overwrite outputs.
    Only one trajectory at a time is read while selecting examples.
    """
    data_dir, output_dir = data_dir.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite gallery: {output_dir}")
    if half_window < 1:
        raise ValueError("half_window must be positive")
    source_manifest = data_dir / "manifest.json"
    source = json.loads(source_manifest.read_text())
    selections = {}
    inputs = {}
    paths = {}
    for split in SPLITS:
        entry = source["outputs"][split]
        if Path(entry["file"]).name != entry["file"]:
            raise ValueError("expected a simple HDF5 filename in import manifest")
        path = data_dir / entry["file"]
        digest = sha256(path)
        if digest != entry["sha256"]:
            raise ValueError(f"input checksum mismatch: {split}")
        with h5py.File(path, "r") as handle:
            if handle["simulation_ids"][:].tolist() != entry["simulation_ids"]:
                raise ValueError(f"input simulation IDs disagree with import manifest: {split}")
        selections[split] = select_examples(path)
        inputs[split] = {"file": entry["file"], "sha256": digest}
        paths[split] = path
    epsilon = float(source["physics"]["epsilon"])
    manifest = {
        "schema_version": 1,
        "kind": "ground-truth dataset preview; no model predictions",
        "source_csv": source["source"],
        "import_manifest_sha256": sha256(source_manifest),
        "gallery_script_sha256": sha256(Path(__file__)),
        "inputs": inputs,
        "selection_policy": "Within each split independently: min, upper-median and max sampled minimum separation; ties broken by simulation ID. Purposive examples, not a random/representative sample or chaos classification.",
        "zoom_half_window_frames": half_window,
        "epsilon": epsilon,
        "time_note": source["time"]["note"],
        "selection": selections,
        "caution": "Held-out panels are ground-truth previews only. Do not change splits, select checkpoints or tune hyperparameters based on test examples.",
        "outputs": {},
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        with plt.rc_context({"font.family": "DejaVu Sans", "savefig.facecolor": "white"}):
            overview, grid = plt.subplots(3, 3, figsize=(15, 15))
            try:
                for row, split in enumerate(SPLITS):
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    try:
                        with h5py.File(paths[split], "r") as handle:
                            for column, record in enumerate(selections[split]):
                                index = record["h5_index"]
                                states, times = (
                                    handle["trajectories"][index],
                                    handle["times"][index],
                                )
                                _panel(
                                    grid[row, column],
                                    states,
                                    times,
                                    record,
                                    epsilon,
                                    half_window,
                                    zoom=False,
                                )
                                grid[row, column].set_ylabel(f"{split.upper()}\ny [AU]")
                                for zoom in (False, True):
                                    _panel(
                                        axes[int(zoom), column],
                                        states,
                                        times,
                                        record,
                                        epsilon,
                                        half_window,
                                        zoom=zoom,
                                    )
                        fig.suptitle(
                            f"Daniela RK4 ground truth · {split.upper()}\nFull recorded paths above; local closest-pair windows below (third body may be outside zoom)",
                            fontsize=15,
                        )
                        _legend(fig)
                        fig.tight_layout(rect=(0, 0.05, 1, 0.93), h_pad=3)
                        fig.savefig(staging / f"{split}.png", dpi=150)
                    finally:
                        plt.close(fig)
                overview.suptitle(
                    "Daniela RK4 dataset · fixed ground-truth examples by split\nColumns: closest / median / widest sampled minimum separation; not a ranking of chaos",
                    fontsize=15,
                )
                _legend(overview)
                overview.tight_layout(rect=(0, 0.04, 1, 0.95), h_pad=3)
                overview.savefig(staging / "overview.png", dpi=150)
            finally:
                plt.close(overview)
        sections = "\n".join(
            f'<h2>{split.upper()}</h2><a href="{split}.png"><img src="{split}.png" alt="{split} trajectories and encounter zooms"></a>'
            for split in SPLITS
        )
        (staging / "index.html").write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>Daniela trajectory gallery</title>'
            "<style>body{max-width:1500px;margin:2rem auto;padding:0 1rem;font:16px system-ui;background:#f7f7f7;color:#222}img{width:100%;background:white}p{line-height:1.6}a{color:#00699b}</style>"
            "<h1>Daniela dataset — ground truth only</h1>"
            "<p>Each split independently shows its closest, median, and widest minimum-separation examples. "
            "These are deliberately selected examples, not a random sample or a classification of chaos. "
            "All coordinates are original, with equal spatial axes. Masses are in simulation mass units; "
            "time is in TU using the original CSV labels. Circles/squares mark first/last recorded states; "
            "stars mark the closest sampled pair. Zooms show a local temporal window and a dotted softening-radius circle.</p>"
            f"<p><strong>{html.escape(manifest['caution'])}</strong></p>"
            '<p><a href="selection.json">Selection IDs, parameters and checksums (JSON)</a> · '
            '<a href="overview.png">Nine-example overview</a></p>' + sections + "</html>\n"
        )
        for path in sorted(staging.iterdir()):
            manifest["outputs"][path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        (staging / "selection.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        # Inputs were read-only; detect any external changes during rendering.
        for split in SPLITS:
            if sha256(paths[split]) != inputs[split]["sha256"]:
                raise ValueError(f"input changed while rendering: {split}")
        os.rename(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main() -> None:
    """Generate the gallery headlessly from the project root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/output/daniela-rk4"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/evaluation/daniela-dataset-gallery")
    )
    parser.add_argument("--half-window", type=int, default=50)
    args = parser.parse_args()
    manifest = create_gallery(args.data_dir, args.output_dir, half_window=args.half_window)
    for split, records in manifest["selection"].items():
        print(
            f"{split}: " + ", ".join(f"ID {r['simulation_id']} ({r['selection']})" for r in records)
        )
    print(f"Open: {(args.output_dir / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
