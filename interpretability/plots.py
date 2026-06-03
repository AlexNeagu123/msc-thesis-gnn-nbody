"""Figures for the interpretability conclusions.

Matplotlib and seaborn are imported lazily inside functions, matching the evaluation/ plotting
modules. Each function builds a figure, saves it to the given paths (png+pdf), and returns it
so a notebook can display it inline.

References:
    - evaluation/report_figures.py (figure style)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from interpretability._types import SymbolicResult

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_STYLE_APPLIED = False


def _apply_style() -> None:
    """Apply a serif whitegrid style once (idempotent)."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    import seaborn as sns
    from matplotlib import pyplot as plt

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )
    _STYLE_APPLIED = True


def _save(fig: Figure, paths: Sequence[Path]) -> None:
    """Save a figure to each path."""
    for p in paths:
        fig.savefig(p)


def _pareto(ax: object, sym: SymbolicResult) -> None:
    """Draw a Pareto front (loss vs complexity, log scale) marking the knee."""
    ax.plot(sym.complexity, sym.loss, "o-")
    ax.axvline(sym.knee_complexity, color="k", ls=":", lw=1, label=f"knee c={sym.knee_complexity}")
    ax.set_yscale("log")
    ax.set_xlabel("complexity")
    ax.set_ylabel("loss (log)")
    ax.set_title("Pareto front")
    ax.legend(fontsize=8)


def plot_potential_2body(
    r: np.ndarray, v: np.ndarray, pred: np.ndarray, sym: SymbolicResult, paths: Sequence[Path]
) -> Figure:
    """2-body learned potential V(r) with the PySR knee fit and the Pareto front."""
    _apply_style()
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].scatter(r, v, s=12, alpha=0.5, label="HGNN V(r)")
    ax[0].plot(r, pred, "r-", lw=2, label=f"PySR knee: {sym.knee_equation}")
    ax[0].set_xlabel("r (normalized)")
    ax[0].set_ylabel("V")
    ax[0].set_title("2-body transfer probe: V(r) -> -C/r")
    ax[0].legend(fontsize=8)
    _pareto(ax[1], sym)
    fig.tight_layout()
    _save(fig, paths)
    return fig


def plot_total_potential(
    v_actual: np.ndarray,
    pred: np.ndarray,
    sum_inv: np.ndarray,
    sym: SymbolicResult,
    paths: Sequence[Path],
) -> Figure:
    """3-body total potential: parity, gravitational collapse onto sum(1/r), Pareto front."""
    _apply_style()
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    lim = [min(v_actual.min(), pred.min()), max(v_actual.max(), pred.max())]
    ax[0].scatter(v_actual, pred, s=10, alpha=0.4)
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set_xlabel("V_total (HGNN)")
    ax[0].set_ylabel("V_total (PySR)")
    ax[0].set_title("parity")
    ax[1].scatter(sum_inv, v_actual, s=10, alpha=0.4, label="HGNN")
    ax[1].scatter(sum_inv, pred, s=10, alpha=0.4, label="PySR knee")
    ax[1].set_xlabel("sum_pairs 1/r")
    ax[1].set_ylabel("V_total")
    ax[1].set_title("collapse onto sum 1/r")
    ax[1].legend(fontsize=8)
    _pareto(ax[2], sym)
    fig.tight_layout()
    _save(fig, paths)
    return fig


def plot_physical_force(
    r: np.ndarray,
    a_forward: np.ndarray,
    a_gradient: np.ndarray,
    support: tuple[float, float],
    g_eff: float,
    exponent: float,
    paths: Sequence[Path],
) -> Figure:
    """Physical 2-body force a(r): forward vs gradient agreement and the 1/r^2 exponent."""
    _apply_style()
    from matplotlib import pyplot as plt

    lo, hi = support
    fit = (r >= lo) & (r <= hi) & (a_gradient > 1e-9)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(r, a_forward, lw=2, label="a (forward, v=0)")
    ax[0].plot(r, a_gradient, "--", lw=2, label="a (potential gradient)")
    ax[0].plot(r, g_eff / r**2, ":", color="k", label=f"{g_eff:.3f}/r^2")
    ax[0].axvspan(lo, hi, color="green", alpha=0.08, label="training support")
    ax[0].set_yscale("log")  # a(r) spans orders of magnitude; log shows the 1/r^2 shape
    ax[0].set_xlabel("r (physical)")
    ax[0].set_ylabel("acceleration (log)")
    ax[0].set_title(f"physical force, G_eff={g_eff:.3f}")
    ax[0].legend(fontsize=8)
    ax[1].scatter(np.log(r[fit]), np.log(a_gradient[fit]), s=10)
    coef = np.polyfit(np.log(r[fit]), np.log(a_gradient[fit]), 1)
    ax[1].plot(
        np.log(r[fit]), np.polyval(coef, np.log(r[fit])), "k--", label=f"slope {exponent:.2f}"
    )
    ax[1].set_xlabel("log r")
    ax[1].set_ylabel("log a")
    ax[1].set_title("exponent test: slope -2 means 1/r^2")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, paths)
    return fig


def plot_node_edge(
    r_ij: np.ndarray,
    v_ij: np.ndarray,
    other_min: np.ndarray,
    sum_inv: np.ndarray,
    v_node: np.ndarray,
    alignment_r2: float,
    paths: Sequence[Path],
) -> Figure:
    """Companion finding: per-edge V_ij is clean in r_ij but the node channel carries energy."""
    _apply_style()
    from matplotlib import pyplot as plt

    rng = np.random.default_rng(0)
    es = rng.choice(len(r_ij), min(8000, len(r_ij)), replace=False)
    ss = rng.choice(len(v_node), min(8000, len(v_node)), replace=False)
    rel_std = v_node.std() / abs(v_node.mean())
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    sc = ax[0].scatter(r_ij[es], v_ij[es], c=other_min[es], s=5, alpha=0.4, cmap="viridis")
    fig.colorbar(sc, ax=ax[0], label="min(other two distances)")
    ax[0].set_xlabel("r_ij (normalized)")
    ax[0].set_ylabel("V_ij")
    ax[0].set_title(f"per-edge readout, R^2(r_ij)={alignment_r2:.3f}")
    ax[1].scatter(sum_inv[ss], v_node[ss], s=5, alpha=0.4)
    ax[1].set_xlabel("sum_pairs 1/r")
    ax[1].set_ylabel("V_node")
    ax[1].set_title(f"V_node carries the attraction (std/|mean|={rel_std:.2f})")
    fig.tight_layout()
    _save(fig, paths)
    return fig


def plot_kinetic(
    vx_grid: np.ndarray,
    vy_grid: np.ndarray,
    t_grid: np.ndarray,
    speed: np.ndarray,
    t: np.ndarray,
    smax: float,
    isotropy_r2: float,
    sym: SymbolicResult,
    paths: Sequence[Path],
) -> Figure:
    """Kinetic energy: isotropy heatmap, T vs speed, and the quadratic Pareto knee."""
    _apply_style()
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    pc = ax[0].pcolormesh(vx_grid, vy_grid, t_grid, shading="auto", cmap="viridis")
    fig.colorbar(pc, ax=ax[0], label="T")
    ax[0].add_patch(plt.Circle((0, 0), smax, fill=False, color="r", lw=1.5))
    ax[0].set_xlabel("vx")
    ax[0].set_ylabel("vy")
    ax[0].set_aspect("equal")
    ax[0].set_title("T(vx,vy): circular => isotropic")
    ax[1].scatter(speed, t, s=4, alpha=0.3)
    ax[1].set_xlabel("speed |v|")
    ax[1].set_ylabel("T")
    ax[1].set_title(f"isotropy R^2={isotropy_r2:.3f}")
    _pareto(ax[2], sym)
    fig.tight_layout()
    _save(fig, paths)
    return fig


def plot_egnn(
    r: np.ndarray,
    w: np.ndarray,
    support: tuple[float, float],
    paths: Sequence[Path],
) -> Figure:
    """EGNN contrast: the layer-0 coupling w(r) is wiggly and sign-changing, not gravitational."""
    _apply_style()
    from matplotlib import pyplot as plt

    lo, hi = support
    force = np.abs(w) * r
    fit = (r >= lo) & (r <= hi) & (force > 1e-12)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(r, w, lw=2)
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].axvspan(lo, hi, color="green", alpha=0.08, label="training support")
    ax[0].set_xlabel("r (physical)")
    ax[0].set_ylabel("layer-0 w_ij")
    ax[0].set_title("EGNN edge coupling (w<0 = attractive)")
    ax[0].legend(fontsize=8)
    ax[1].scatter(np.log(r[fit]), np.log(force[fit]), s=10)
    ax[1].set_xlabel("log r")
    ax[1].set_ylabel("log |w|*r")
    ax[1].set_title("implied force: no clean 1/r^2 slope")
    fig.tight_layout()
    _save(fig, paths)
    return fig
