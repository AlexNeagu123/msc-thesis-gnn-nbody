"""Aggregate metrics.json across data-scaling runs into the methodology tables.

Reads a manifest YAML mapping (model, dataset_size) -> metrics.json path, then
prints Tables 1-4 from `edu/data-scaling-methodology.md` as markdown.

Usage::

    python -m evaluation.scaling_report --manifest configs/scaling_runs.yaml

Manifest format::

    egnn:
      1000: /path/to/runs/egnn/<run_id>/evaluation/metrics.json
      2000: /path/to/runs/egnn/<run_id>/evaluation/metrics.json
      5000: /path/to/runs/egnn/<run_id>/evaluation/metrics.json
    hgnn:
      1000: /path/to/runs/hgnn/<run_id>/evaluation/metrics.json
      2000: /path/to/runs/hgnn/<run_id>/evaluation/metrics.json
      5000: /path/to/runs/hgnn/<run_id>/evaluation/metrics.json

References:
    - Methodology: edu/data-scaling-methodology.md
    - metrics.json schema: evaluation/evaluate.py (_build_report)
"""

import argparse
import json
from pathlib import Path

import yaml

from utils import get_logger

logger = get_logger(__name__)

# rollout step keys used in the methodology tables
ROLLOUT_STEPS = (10, 50, 100, 199)


def _fmt(value: float | None, *, sci: bool = False) -> str:
    """Format a number for table display, with explicit handling for None."""
    if value is None:
        return "n/a"
    return f"{value:.3e}" if sci else f"{value:.4f}"


def load_manifest(path: Path) -> dict[str, dict[int, dict]]:
    """Load manifest YAML and read each metrics.json into a nested dict."""
    raw = yaml.safe_load(path.read_text())
    out: dict[str, dict[int, dict]] = {}
    for model, sizes in raw.items():
        out[model] = {}
        for size, metrics_path in sizes.items():
            with Path(metrics_path).open() as f:
                out[model][int(size)] = json.load(f)
    return out


def _collect_sizes(reports: dict[str, dict[int, dict]]) -> list[int]:
    """Sorted union of all dataset sizes referenced in the manifest."""
    sizes: set[int] = set()
    for sizes_for_model in reports.values():
        sizes.update(sizes_for_model.keys())
    return sorted(sizes)


def table_local_accuracy(reports: dict[str, dict[int, dict]]) -> str:
    """Table 1: val loss + single-step MSE per (size, model)."""
    lines = [
        "## Table 1: Local accuracy vs dataset size",
        "",
        "| Size | Model | Val loss | Single-step median MSE | Single-step p95 MSE |",
        "|---|---|---:|---:|---:|",
    ]
    for size in _collect_sizes(reports):
        for model in sorted(reports.keys()):
            r = reports[model].get(size)
            if r is None:
                continue
            val_loss = r["metadata"].get("checkpoint_val_loss")
            single = r["single_step"]["mse"]
            lines.append(
                f"| {size} | {model} | {_fmt(val_loss)} | "
                f"{_fmt(single['median'], sci=True)} | "
                f"{_fmt(single.get('p95'), sci=True)} |"
            )
    return "\n".join(lines)


def table_rollout_stability(reports: dict[str, dict[int, dict]]) -> str:
    """Table 2: rollout median MSE at fixed steps + final finite fraction."""
    headers = (
        "| Size | Model |"
        + "|".join(f" Step {s} median " for s in ROLLOUT_STEPS)
        + "| Final finite fraction |"
    )
    sep = "|---|---|" + "|".join(["---:"] * (len(ROLLOUT_STEPS) + 1)) + "|"
    lines = ["## Table 2: Long-horizon rollout stability", "", headers, sep]
    for size in _collect_sizes(reports):
        for model in sorted(reports.keys()):
            r = reports[model].get(size)
            if r is None:
                continue
            steps = r["rollout"]["steps"]
            cells = [_fmt(steps[str(s)]["median_mse"], sci=True) for s in ROLLOUT_STEPS]
            final_finite = r["rollout"]["finite_final_fraction"]
            lines.append(
                f"| {size} | {model} | " + " | ".join(cells) + f" | {_fmt(final_finite)} |"
            )
    return "\n".join(lines)


def _crossover_step(egnn_curve: list[float | None], hgnn_curve: list[float | None]) -> int | None:
    """First step where hgnn curve < egnn curve. None if no crossover found."""
    for step, (e, h) in enumerate(zip(egnn_curve, hgnn_curve, strict=True)):
        if step == 0:
            continue
        if e is None or h is None:
            continue
        if h < e:
            return step
    return None


def table_crossover(reports: dict[str, dict[int, dict]]) -> str:
    """Table 3: crossover steps (HGNN beats EGNN) and EGNN divergence."""
    lines = [
        "## Table 3: Crossover and divergence",
        "",
        "| Size | Median crossover | p95 crossover | EGNN <50% finite | HGNN final finite |",
        "|---|---:|---:|---:|---:|",
    ]
    for size in _collect_sizes(reports):
        egnn_r = reports.get("egnn", {}).get(size)
        hgnn_r = reports.get("hgnn", {}).get(size)
        if egnn_r is None or hgnn_r is None:
            continue
        median_x = _crossover_step(
            egnn_r["rollout"]["curves"]["median_mse"],
            hgnn_r["rollout"]["curves"]["median_mse"],
        )
        p95_x = _crossover_step(
            egnn_r["rollout"]["curves"]["p95_mse"],
            hgnn_r["rollout"]["curves"]["p95_mse"],
        )
        # find first step where EGNN finite_fraction drops below 0.5
        egnn_finite = egnn_r["rollout"]["curves"]["finite_fraction"]
        egnn_below_50 = next(
            (i for i, f in enumerate(egnn_finite) if f is not None and f < 0.5),
            None,
        )
        hgnn_final = hgnn_r["rollout"]["finite_final_fraction"]
        lines.append(
            f"| {size} | {median_x if median_x is not None else 'never'} | "
            f"{p95_x if p95_x is not None else 'never'} | "
            f"{egnn_below_50 if egnn_below_50 is not None else 'never'} | "
            f"{_fmt(hgnn_final)} |"
        )
    return "\n".join(lines)


def _drift_median(group: dict | None, key: str) -> float | None:
    """Return median drift from evaluator energy schema."""
    if group is None:
        return None
    stats = group.get(key)
    if not isinstance(stats, dict):
        return None
    return stats.get("median")


def table_energy(reports: dict[str, dict[int, dict]]) -> str:
    """Table 4: physical and learned-Hamiltonian energy drift."""
    lines = [
        "## Table 4: Energy behavior",
        "",
        "| Size | Model | Physical final drift | Physical max drift | Learned-H final drift |",
        "|---|---|---:|---:|---:|",
    ]
    for size in _collect_sizes(reports):
        for model in sorted(reports.keys()):
            r = reports[model].get(size)
            if r is None:
                continue
            phys = r["energy"]["physical"]
            learned = r["energy"].get("learned_hamiltonian")
            phys_final = _drift_median(phys, "final_relative_drift")
            phys_max = _drift_median(phys, "max_relative_drift")
            learned_final = _drift_median(learned, "final_relative_drift")
            lines.append(
                f"| {size} | {model} | {_fmt(phys_final, sci=True)} | "
                f"{_fmt(phys_max, sci=True)} | {_fmt(learned_final, sci=True)} |"
            )
    return "\n".join(lines)


def render_report(reports: dict[str, dict[int, dict]]) -> str:
    """Render all four tables into one markdown report."""
    return "\n\n".join(
        [
            "# Data-Scaling Report",
            table_local_accuracy(reports),
            table_rollout_stability(reports),
            table_crossover(reports),
            table_energy(reports),
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate scaling-study metrics into tables.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the rendered markdown report.",
    )
    args = parser.parse_args()

    reports = load_manifest(Path(args.manifest))
    report = render_report(reports)
    print(report)

    if args.output:
        Path(args.output).write_text(report)
        logger.info("wrote report to %s", args.output)
