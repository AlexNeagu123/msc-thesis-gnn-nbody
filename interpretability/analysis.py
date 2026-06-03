"""Orchestrator: probe the trained models, run PySR, save figures and a results JSON.

References:
    - Cranmer et al. 2020 (symbolic models from GNNs): https://arxiv.org/abs/2006.11287
    - Cranmer 2023 (PySR): https://arxiv.org/abs/2305.01582
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from evaluation._loader import load_trained_model
from interpretability import plots, probes
from interpretability._io import figure_paths, write_results
from interpretability._types import (
    EgnnContrast,
    InterpretabilityResults,
    KineticRecovery,
    NodeEdgeLocality,
    PhysicalForce,
    PotentialRecovery,
)
from interpretability.symbolic import fit_symbolic

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from torch import nn

DT = 0.05
HGNN_CONFIG = "configs/hgnn.yaml"
HGNN_CKPT = "runs/hgnn/20260510_211151/best.pt"
EGNN_CONFIG = "configs/egnn.yaml"
EGNN_CKPT = "runs/egnn/20260510_211102/best.pt"
TRAIN_PATH = "data/output/train.h5"
TEST_PATH = "data/output/test.h5"
OUTPUT_DIR = "runs/reports/interpretability"


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares y = slope*x + intercept; returns (slope, intercept, R^2)."""
    a = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(a, y, rcond=None)
    pred = slope * x + intercept
    r2 = 1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(slope), float(intercept), float(r2)


def _powerlaw_exponent(r: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    """Slope of log y vs log r over [lo, hi] (the power-law exponent)."""
    fit = (r >= lo) & (r <= hi) & (y > 1e-12)
    slope, _ = np.polyfit(np.log(r[fit]), np.log(y[fit]), 1)
    return float(slope)


def _binned_r2(x: np.ndarray, y: np.ndarray, lo: float, hi: float, n_bins: int = 50) -> float:
    """R^2 of y explained by x alone, using the per-bin mean as the best x-only predictor."""
    mask = (x >= lo) & (x <= hi)
    xb, yb = x[mask], y[mask]
    bins = np.linspace(lo, hi, n_bins + 1)
    which = np.clip(np.digitize(xb, bins), 1, n_bins)
    y_hat = np.zeros_like(yb)
    for b in range(1, n_bins + 1):
        sel = which == b
        if sel.any():
            y_hat[sel] = yb[sel].mean()
    ss_res = ((yb - y_hat) ** 2).sum()
    ss_tot = ((yb - yb.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


class InterpretabilityAnalysis:
    """Run every interpretability probe, recover symbolic laws, and save figures + results."""

    def __init__(
        self,
        hgnn: nn.Module,
        pos_std: float,
        vel_std: float,
        egnn: nn.Module,
        *,
        train_path: str = TRAIN_PATH,
        test_path: str = TEST_PATH,
        output_dir: str = OUTPUT_DIR,
        progress: bool = False,
        seed: int = 0,
    ) -> None:
        """Store loaded models and paths; pass pre-loaded models for testing (DI)."""
        self.hgnn = hgnn
        self.egnn = egnn
        self.pos_std = pos_std
        self.vel_std = vel_std
        self.train_path = train_path
        self.test_path = test_path
        self.figures_dir = Path(output_dir) / "figures"
        self.results_path = Path(output_dir) / "results.json"
        self.progress = progress
        self.seed = seed
        d_norm = probes.normalized_pairwise_distances(train_path, pos_std)
        self.support_norm = (float(np.quantile(d_norm, 0.02)), float(np.quantile(d_norm, 0.98)))
        self.support_phys = probes.distance_support(train_path)
        self._d_norm = d_norm

    @classmethod
    def from_checkpoints(
        cls,
        *,
        output_dir: str = OUTPUT_DIR,
        device: str = "cpu",
        progress: bool = False,
    ) -> InterpretabilityAnalysis:
        """Load the canonical HGNN and EGNN checkpoints."""
        dev = torch.device(device)
        hgnn = load_trained_model(Path(HGNN_CONFIG), Path(HGNN_CKPT), dev)
        egnn = load_trained_model(Path(EGNN_CONFIG), Path(EGNN_CKPT), dev)
        return cls(
            hgnn.model,
            hgnn.pos_std,
            hgnn.vel_std,
            egnn.model,
            output_dir=output_dir,
            progress=progress,
        )

    def recover_potential(self) -> tuple[PotentialRecovery, list[Figure]]:
        """Recover V(r) = -C/r (2-body transfer) and V_total = -C*sum(1/r) (3-body in-dist)."""
        rng = np.random.default_rng(self.seed)
        lo, hi = self.support_norm

        pool = self._d_norm[(self._d_norm >= lo) & (self._d_norm <= hi)]
        r2b = np.sort(rng.choice(pool, 400, replace=False))
        v2b = probes.potential_2body(self.hgnn, r2b)
        two_body, pred2 = fit_symbolic(
            r2b.reshape(-1, 1), v2b, ["r"], progress=self.progress, run_id="potential_2body"
        )
        slope2, _, _ = _linear_fit(1.0 / r2b, v2b)

        feats, vtot = probes.total_potential_3body(
            self.hgnn, self.test_path, self.pos_std, (lo, hi)
        )
        sum_inv = (1.0 / feats).sum(1)
        slope_t, _, r2_t = _linear_fit(sum_inv, vtot)
        idx = rng.choice(len(vtot), min(2000, len(vtot)), replace=False)
        total, pred_t = fit_symbolic(
            feats[idx],
            vtot[idx],
            ["r12", "r13", "r23"],
            progress=self.progress,
            run_id="potential_total",
        )

        result = PotentialRecovery(
            two_body=two_body,
            two_body_c=-slope2,
            total_linear_c=-slope_t,
            total_linear_r2=r2_t,
            total_symbolic=total,
        )
        figs = [
            plots.plot_potential_2body(
                r2b, v2b, pred2, two_body, figure_paths(self.figures_dir, "01_potential_2body")
            ),
            plots.plot_total_potential(
                vtot[idx],
                pred_t,
                sum_inv[idx],
                total,
                figure_paths(self.figures_dir, "02_potential_total"),
            ),
        ]
        return result, figs

    def physical_force(self) -> tuple[PhysicalForce, Figure]:
        """Recover the physical 2-body force law and the effective gravitational constant."""
        lo, hi = self.support_phys
        r = np.linspace(0.4 * lo, hi, 400)
        a_fwd = probes.physical_force_forward(self.hgnn, r, DT)
        a_grad = probes.physical_force_gradient(self.hgnn, r, self.pos_std, self.vel_std)
        exponent = _powerlaw_exponent(r, a_grad, lo, hi)
        clean = (r >= 0.8) & (r <= hi)
        g_fwd = float(np.median(a_fwd[clean] * r[clean] ** 2))
        g_grad = float(np.median(a_grad[clean] * r[clean] ** 2))
        result = PhysicalForce(force_exponent=exponent, g_eff_forward=g_fwd, g_eff_gradient=g_grad)
        fig = plots.plot_physical_force(
            r,
            a_fwd,
            a_grad,
            (lo, hi),
            g_grad,
            exponent,
            figure_paths(self.figures_dir, "03_physical_force"),
        )
        return result, fig

    def node_edge_locality(self) -> tuple[NodeEdgeLocality, Figure]:
        """Measure whether the per-edge readout localizes the pairwise potential."""
        lo, hi = self.support_norm
        edge = probes.per_edge_potential(self.hgnn, self.test_path, self.pos_std)
        r2 = _binned_r2(edge.r_ij, edge.v_ij, lo, hi)
        rel_std = float(edge.v_node.std() / abs(edge.v_node.mean()))
        corr = float(np.corrcoef(edge.v_node, edge.sum_inv)[0, 1])
        result = NodeEdgeLocality(
            alignment_r2=r2, vnode_relative_std=rel_std, vnode_suminv_corr=corr
        )
        fig = plots.plot_node_edge(
            edge.r_ij,
            edge.v_ij,
            edge.other_min,
            edge.sum_inv,
            edge.v_node,
            r2,
            figure_paths(self.figures_dir, "04_node_edge"),
        )
        return result, fig

    def recover_kinetic(self) -> tuple[KineticRecovery, Figure]:
        """Recover T(v): emergent isotropy and the quadratic form."""
        rng = np.random.default_rng(self.seed)
        smax = probes.speed_support(self.train_path, self.vel_std)
        vx_g, vy_g, t_grid = probes.kinetic_grid(self.hgnn, smax)
        vx, vy, t = vx_g.ravel(), vy_g.ravel(), t_grid.ravel()
        speed = np.sqrt(vx**2 + vy**2)
        insup = speed <= smax
        iso_r2 = _binned_r2(speed[insup], t[insup], 0.0, smax)
        coef, _, _ = _linear_fit(speed[insup] ** 2, t[insup])
        expected = 0.5 * self.vel_std / self.pos_std

        idx = rng.choice(np.flatnonzero(insup), min(1500, int(insup.sum())), replace=False)
        sym, _ = fit_symbolic(
            np.stack([vx[idx], vy[idx]], -1),
            t[idx],
            ["vx", "vy"],
            progress=self.progress,
            run_id="kinetic",
        )
        result = KineticRecovery(
            isotropy_r2=iso_r2,
            quadratic_coefficient=coef,
            expected_coefficient=float(expected),
            symbolic=sym,
        )
        fig = plots.plot_kinetic(
            vx_g,
            vy_g,
            t_grid,
            speed[insup],
            t[insup],
            smax,
            iso_r2,
            sym,
            figure_paths(self.figures_dir, "05_kinetic"),
        )
        return result, fig

    def egnn_contrast(self) -> tuple[EgnnContrast, Figure]:
        """Show the EGNN layer-0 coupling has no recoverable gravitational law."""
        lo, hi = self.support_phys
        r = np.linspace(0.4 * lo, hi, 400)
        w = probes.egnn_layer0_weight(self.egnn, r, self.pos_std)
        fit = (r >= lo) & (r <= hi)
        w_exp = _powerlaw_exponent(r, np.abs(w), lo, hi)
        force_exp = _powerlaw_exponent(r, np.abs(w) * r, lo, hi)
        attractive = float((w[fit] < 0).mean())
        result = EgnnContrast(
            layer0_weight_exponent=w_exp,
            layer0_force_exponent=force_exp,
            attractive_fraction=attractive,
        )
        fig = plots.plot_egnn(r, w, (lo, hi), figure_paths(self.figures_dir, "06_egnn_contrast"))
        return result, fig

    def run_all(self) -> tuple[InterpretabilityResults, dict[str, Figure]]:
        """Run every step, write results.json, and return the bundle plus figures."""
        potential, pot_figs = self.recover_potential()
        force, force_fig = self.physical_force()
        node_edge, ne_fig = self.node_edge_locality()
        kinetic, kin_fig = self.recover_kinetic()
        egnn, egnn_fig = self.egnn_contrast()
        results = InterpretabilityResults(
            potential=potential,
            physical_force=force,
            node_edge=node_edge,
            kinetic=kinetic,
            egnn=egnn,
        )
        write_results(self.results_path, results)
        figures = {
            "potential_2body": pot_figs[0],
            "potential_total": pot_figs[1],
            "physical_force": force_fig,
            "node_edge": ne_fig,
            "kinetic": kin_fig,
            "egnn_contrast": egnn_fig,
        }
        return results, figures
