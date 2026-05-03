"""Typed contracts for evaluation reports and metric-computation results.

Two kinds of types live here:
    - Persisted report schema (EvaluationReport and its tree), with
      EvaluationReport.from_dict as the single entry point.
    - Intermediate metric-computation containers (SingleStepMetrics, RolloutMSE),
      returned by evaluation/metrics.py and consumed by evaluate.py.

References:
    - JSON schema produced by evaluation/evaluate.py:_build_report
    - I/O wrapper: evaluation/_io.py
    - Pattern mirror: training/_types.py
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class SingleStepMetrics:
    """Per-sample single-step metrics before report summarization."""

    state_mse: npt.NDArray[np.floating]
    position_mse: npt.NDArray[np.floating]
    velocity_mse: npt.NDArray[np.floating]
    min_pairwise_distance: npt.NDArray[np.floating]


@dataclass
class RolloutMetricSeries:
    """Per-trajectory rollout MSE series for one state slice."""

    per_trajectory: npt.NDArray[np.floating]
    mean: npt.NDArray[np.floating]
    median: npt.NDArray[np.floating]
    std: npt.NDArray[np.floating]
    finite_fraction: npt.NDArray[np.floating]


@dataclass
class RolloutMSE:
    """Rollout MSE summaries split by dynamic state, position, and velocity."""

    state: RolloutMetricSeries
    position: RolloutMetricSeries
    velocity: RolloutMetricSeries

    @property
    def per_trajectory(self) -> npt.NDArray[np.floating]:
        """Backward-compatible access to full dynamic-state MSE."""
        return self.state.per_trajectory

    @property
    def mean(self) -> npt.NDArray[np.floating]:
        """Backward-compatible access to full dynamic-state mean MSE."""
        return self.state.mean

    @property
    def median(self) -> npt.NDArray[np.floating]:
        """Backward-compatible access to full dynamic-state median MSE."""
        return self.state.median

    @property
    def std(self) -> npt.NDArray[np.floating]:
        """Backward-compatible access to full dynamic-state std MSE."""
        return self.state.std

    @property
    def finite_fraction(self) -> npt.NDArray[np.floating]:
        """Backward-compatible access to full dynamic-state finite fraction."""
        return self.state.finite_fraction


@dataclass
class MseSummary:
    """Summary statistics: mean/median/max plus p95, p99."""

    mean: float | None
    median: float | None
    max: float | None
    p95: float | None
    p99: float | None

    def to_dict(self) -> dict[str, float | None]:
        """Serialize preserving JSON key order."""
        return {
            "mean": self.mean,
            "median": self.median,
            "max": self.max,
            "p95": self.p95,
            "p99": self.p99,
        }


@dataclass
class DistanceSummary:
    """Summary statistics: mean/median/max plus p5, p50."""

    mean: float | None
    median: float | None
    max: float | None
    p5: float | None
    p50: float | None

    def to_dict(self) -> dict[str, float | None]:
        """Serialize preserving JSON key order."""
        return {
            "mean": self.mean,
            "median": self.median,
            "max": self.max,
            "p5": self.p5,
            "p50": self.p50,
        }


@dataclass
class DriftSummary:
    """Summary statistics: mean/median/max plus p95."""

    mean: float | None
    median: float | None
    max: float | None
    p95: float | None

    def to_dict(self) -> dict[str, float | None]:
        """Serialize preserving JSON key order."""
        return {
            "mean": self.mean,
            "median": self.median,
            "max": self.max,
            "p95": self.p95,
        }


@dataclass
class EvaluationMetadata:
    """Provenance and dataset shape."""

    model_name: str
    checkpoint_path: str | None
    config_path: str | None
    test_path: str
    device: str
    checkpoint_epoch: int | None
    checkpoint_val_loss: float | None
    run_id: str | None
    git_commit: str | None
    pos_std: float
    vel_std: float
    n_trajectories: int
    n_frames: int
    n_transitions: int
    n_particles: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize preserving _build_report key order."""
        return {
            "model_name": self.model_name,
            "checkpoint_path": self.checkpoint_path,
            "config_path": self.config_path,
            "test_path": self.test_path,
            "device": self.device,
            "checkpoint_epoch": self.checkpoint_epoch,
            "checkpoint_val_loss": self.checkpoint_val_loss,
            "run_id": self.run_id,
            "git_commit": self.git_commit,
            "pos_std": self.pos_std,
            "vel_std": self.vel_std,
            "n_trajectories": self.n_trajectories,
            "n_frames": self.n_frames,
            "n_transitions": self.n_transitions,
            "n_particles": self.n_particles,
        }


@dataclass
class SingleStepReport:
    """Single-step evaluation block."""

    state_mse: MseSummary
    position_mse: MseSummary
    velocity_mse: MseSummary
    min_pairwise_distance: DistanceSummary

    @property
    def mse(self) -> MseSummary:
        """Backward-compatible alias for full dynamic-state MSE."""
        return self.state_mse

    def to_dict(self) -> dict[str, Any]:
        """Serialize preserving _build_report key order."""
        return {
            "state_mse": self.state_mse.to_dict(),
            "position_mse": self.position_mse.to_dict(),
            "velocity_mse": self.velocity_mse.to_dict(),
            "min_pairwise_distance": self.min_pairwise_distance.to_dict(),
        }


@dataclass
class RolloutMetricSummary:
    """Rollout MSE summary for one metric family at one step."""

    mean_finite: float | None
    median: float | None
    p95: float | None
    finite_fraction: float | None

    def to_dict(self) -> dict[str, float | None]:
        """Serialize preserving metric-summary key order."""
        return {
            "mean_finite": self.mean_finite,
            "median": self.median,
            "p95": self.p95,
            "finite_fraction": self.finite_fraction,
        }


@dataclass
class RolloutStepMetrics:
    """Rollout summaries at a single rollout step."""

    state_mse: RolloutMetricSummary
    position_mse: RolloutMetricSummary
    velocity_mse: RolloutMetricSummary

    @property
    def mean_finite_mse(self) -> float | None:
        """Backward-compatible alias for full dynamic-state mean MSE."""
        return self.state_mse.mean_finite

    @property
    def median_mse(self) -> float | None:
        """Backward-compatible alias for full dynamic-state median MSE."""
        return self.state_mse.median

    @property
    def p95_mse(self) -> float | None:
        """Backward-compatible alias for full dynamic-state p95 MSE."""
        return self.state_mse.p95

    @property
    def finite_fraction(self) -> float | None:
        """Backward-compatible alias for full dynamic-state finite fraction."""
        return self.state_mse.finite_fraction

    def to_dict(self) -> dict[str, Any]:
        """Serialize preserving _rollout_steps key order."""
        return {
            "state_mse": self.state_mse.to_dict(),
            "position_mse": self.position_mse.to_dict(),
            "velocity_mse": self.velocity_mse.to_dict(),
        }


@dataclass
class RolloutMetricCurves:
    """Full per-step rollout curves for one metric family."""

    mean_finite: list[float | None]
    median: list[float | None]
    p95: list[float | None]
    finite_fraction: list[float | None]

    def to_dict(self) -> dict[str, list[float | None]]:
        """Serialize preserving metric-curve key order."""
        return {
            "mean_finite": self.mean_finite,
            "median": self.median,
            "p95": self.p95,
            "finite_fraction": self.finite_fraction,
        }


@dataclass
class RolloutCurves:
    """Full per-step rollout curves for crossover analysis."""

    step: list[int]
    state_mse: RolloutMetricCurves
    position_mse: RolloutMetricCurves
    velocity_mse: RolloutMetricCurves

    @property
    def mean_finite_mse(self) -> list[float | None]:
        """Backward-compatible alias for full dynamic-state mean MSE."""
        return self.state_mse.mean_finite

    @property
    def median_mse(self) -> list[float | None]:
        """Backward-compatible alias for full dynamic-state median MSE."""
        return self.state_mse.median

    @property
    def p95_mse(self) -> list[float | None]:
        """Backward-compatible alias for full dynamic-state p95 MSE."""
        return self.state_mse.p95

    @property
    def finite_fraction(self) -> list[float | None]:
        """Backward-compatible alias for full dynamic-state finite fraction."""
        return self.state_mse.finite_fraction

    def to_dict(self) -> dict[str, Any]:
        """Serialize preserving _rollout_curves key order."""
        return {
            "step": self.step,
            "state_mse": self.state_mse.to_dict(),
            "position_mse": self.position_mse.to_dict(),
            "velocity_mse": self.velocity_mse.to_dict(),
        }


@dataclass
class DivergenceMetrics:
    """Rollout divergence summary at a single MSE threshold."""

    first_step: list[int | None]
    final_fraction_below: float | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize preserving _divergence_report key order."""
        return {
            "first_step": self.first_step,
            "final_fraction_below": self.final_fraction_below,
        }


@dataclass
class RolloutReport:
    """Long-horizon rollout block.

    `curves` is optional because metrics.json files predating that field
    omit it; from_dict() loads them as curves=None, to_dict() skips the
    key entirely on output.
    """

    steps: dict[str, RolloutStepMetrics]
    first_nonfinite_step: list[int | None]
    state_mse_thresholds: dict[str, DivergenceMetrics]
    position_mse_thresholds: dict[str, DivergenceMetrics]
    state_final_finite_fraction: float | None
    curves: RolloutCurves | None = None

    @property
    def thresholds(self) -> dict[str, DivergenceMetrics]:
        """Backward-compatible alias for full dynamic-state MSE thresholds."""
        return self.state_mse_thresholds

    @property
    def finite_final_fraction(self) -> float | None:
        """Backward-compatible alias for full dynamic-state finite fraction."""
        return self.state_final_finite_fraction

    def to_dict(self) -> dict[str, Any]:
        """Serialize preserving _build_report rollout key order."""
        # order: steps, curves (optional), first_nonfinite_step, thresholds, final fraction
        out: dict[str, Any] = {
            "steps": {k: v.to_dict() for k, v in self.steps.items()},
        }
        if self.curves is not None:
            out["curves"] = self.curves.to_dict()
        out["first_nonfinite_step"] = self.first_nonfinite_step
        out["state_mse_thresholds"] = {k: v.to_dict() for k, v in self.state_mse_thresholds.items()}
        out["position_mse_thresholds"] = {
            k: v.to_dict() for k, v in self.position_mse_thresholds.items()
        }
        out["state_final_finite_fraction"] = self.state_final_finite_fraction
        return out


@dataclass
class EnergyDriftReport:
    """Drift summary for one energy quantity (physical or learned)."""

    final_relative_drift: DriftSummary
    max_relative_drift: DriftSummary
    per_trajectory_final: list[float | None]
    per_trajectory_max: list[float | None]

    def to_dict(self) -> dict[str, Any]:
        """Serialize preserving _energy_drift_report key order."""
        return {
            "final_relative_drift": self.final_relative_drift.to_dict(),
            "max_relative_drift": self.max_relative_drift.to_dict(),
            "per_trajectory_final": self.per_trajectory_final,
            "per_trajectory_max": self.per_trajectory_max,
        }


@dataclass
class EnergyReport:
    """Energy block with optional learned-Hamiltonian drift (HGNN only)."""

    physical: EnergyDriftReport
    learned_hamiltonian: EnergyDriftReport | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize, omitting learned_hamiltonian when absent (matches EGNN reports)."""
        out: dict[str, Any] = {"physical": self.physical.to_dict()}
        if self.learned_hamiltonian is not None:
            out["learned_hamiltonian"] = self.learned_hamiltonian.to_dict()
        return out


def _energy_drift_from_dict(d: dict[str, Any]) -> EnergyDriftReport:
    """Construct an EnergyDriftReport from its JSON-shaped dict."""
    return EnergyDriftReport(
        final_relative_drift=DriftSummary(**d["final_relative_drift"]),
        max_relative_drift=DriftSummary(**d["max_relative_drift"]),
        per_trajectory_final=d["per_trajectory_final"],
        per_trajectory_max=d["per_trajectory_max"],
    )


def _single_step_from_dict(d: dict[str, Any]) -> SingleStepReport:
    """Construct SingleStepReport, accepting legacy `mse` reports."""
    state_raw = d.get("state_mse", d.get("mse"))
    if state_raw is None:
        msg = "single_step report must contain state_mse or legacy mse"
        raise KeyError(msg)

    return SingleStepReport(
        state_mse=MseSummary(**state_raw),
        position_mse=MseSummary(**d.get("position_mse", state_raw)),
        velocity_mse=MseSummary(**d.get("velocity_mse", state_raw)),
        min_pairwise_distance=DistanceSummary(**d["min_pairwise_distance"]),
    )


def _rollout_metric_summary_from_dict(d: dict[str, Any]) -> RolloutMetricSummary:
    """Construct RolloutMetricSummary from new or legacy key names."""
    return RolloutMetricSummary(
        mean_finite=d.get("mean_finite", d.get("mean_finite_mse")),
        median=d.get("median", d.get("median_mse")),
        p95=d.get("p95", d.get("p95_mse")),
        finite_fraction=d["finite_fraction"],
    )


def _rollout_step_from_dict(d: dict[str, Any]) -> RolloutStepMetrics:
    """Construct RolloutStepMetrics, accepting legacy flat step metrics."""
    if "state_mse" in d:
        return RolloutStepMetrics(
            state_mse=_rollout_metric_summary_from_dict(d["state_mse"]),
            position_mse=_rollout_metric_summary_from_dict(d["position_mse"]),
            velocity_mse=_rollout_metric_summary_from_dict(d["velocity_mse"]),
        )

    legacy = _rollout_metric_summary_from_dict(d)
    return RolloutStepMetrics(
        state_mse=legacy,
        position_mse=legacy,
        velocity_mse=legacy,
    )


def _rollout_metric_curves_from_dict(d: dict[str, Any]) -> RolloutMetricCurves:
    """Construct RolloutMetricCurves from new or legacy key names."""
    return RolloutMetricCurves(
        mean_finite=d.get("mean_finite", d.get("mean_finite_mse")),
        median=d.get("median", d.get("median_mse")),
        p95=d.get("p95", d.get("p95_mse")),
        finite_fraction=d["finite_fraction"],
    )


def _rollout_curves_from_dict(d: dict[str, Any]) -> RolloutCurves:
    """Construct RolloutCurves, accepting legacy flat curve metrics."""
    if "state_mse" in d:
        return RolloutCurves(
            step=d["step"],
            state_mse=_rollout_metric_curves_from_dict(d["state_mse"]),
            position_mse=_rollout_metric_curves_from_dict(d["position_mse"]),
            velocity_mse=_rollout_metric_curves_from_dict(d["velocity_mse"]),
        )

    legacy = _rollout_metric_curves_from_dict(d)
    return RolloutCurves(
        step=d["step"],
        state_mse=legacy,
        position_mse=legacy,
        velocity_mse=legacy,
    )


def _divergence_from_dict(d: dict[str, Any]) -> dict[str, DivergenceMetrics]:
    """Construct divergence-threshold mapping from JSON-shaped dict."""
    return {k: DivergenceMetrics(**v) for k, v in d.items()}


@dataclass
class EvaluationReport:
    """Top-level evaluation report."""

    metadata: EvaluationMetadata
    single_step: SingleStepReport
    rollout: RolloutReport
    energy: EnergyReport

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvaluationReport":
        """Build a typed report from a parsed metrics.json dict."""
        rollout = d["rollout"]
        energy = d["energy"]
        learned = energy.get("learned_hamiltonian")
        curves = rollout.get("curves")
        state_thresholds = rollout.get("state_mse_thresholds", rollout.get("thresholds", {}))
        position_thresholds = rollout.get("position_mse_thresholds", state_thresholds)
        state_final_finite_fraction = rollout.get(
            "state_final_finite_fraction",
            rollout.get("finite_final_fraction"),
        )

        return cls(
            metadata=EvaluationMetadata(**d["metadata"]),
            single_step=_single_step_from_dict(d["single_step"]),
            rollout=RolloutReport(
                steps={k: _rollout_step_from_dict(v) for k, v in rollout["steps"].items()},
                first_nonfinite_step=rollout["first_nonfinite_step"],
                state_mse_thresholds=_divergence_from_dict(state_thresholds),
                position_mse_thresholds=_divergence_from_dict(position_thresholds),
                state_final_finite_fraction=state_final_finite_fraction,
                curves=_rollout_curves_from_dict(curves) if curves is not None else None,
            ),
            energy=EnergyReport(
                physical=_energy_drift_from_dict(energy["physical"]),
                learned_hamiltonian=(
                    _energy_drift_from_dict(learned) if learned is not None else None
                ),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize preserving _build_report top-level key order."""
        return {
            "metadata": self.metadata.to_dict(),
            "single_step": self.single_step.to_dict(),
            "rollout": self.rollout.to_dict(),
            "energy": self.energy.to_dict(),
        }


@dataclass
class SummaryRow:
    """Flat CSV row built from EvaluationReport.

    Dynamic keys (rollout_step_<n>_*, rollout_final_fraction_below_*_mse_<t>,
    learned_h_*) are produced at to_csv_row() time rather than encoded as
    static fields, since they depend on which steps/thresholds the report
    contains and whether it includes a learned Hamiltonian.
    """

    report: EvaluationReport

    @classmethod
    def from_report(cls, report: EvaluationReport) -> "SummaryRow":
        """Wrap a report so we can render it as a CSV row."""
        return cls(report=report)

    def to_csv_row(self) -> dict[str, Any]:
        """Flatten the report into one CSV row, preserving _summary_row key order."""
        meta = self.report.metadata
        single = self.report.single_step
        physical = self.report.energy.physical
        learned = self.report.energy.learned_hamiltonian
        rollout = self.report.rollout

        row: dict[str, Any] = {
            "model_name": meta.model_name,
            "run_id": meta.run_id,
            "checkpoint_epoch": meta.checkpoint_epoch,
            "checkpoint_val_loss": meta.checkpoint_val_loss,
            "n_trajectories": meta.n_trajectories,
            "n_frames": meta.n_frames,
            "n_transitions": meta.n_transitions,
            "n_particles": meta.n_particles,
            "single_step_state_mse_mean": single.state_mse.mean,
            "single_step_state_mse_median": single.state_mse.median,
            "single_step_state_mse_p95": single.state_mse.p95,
            "single_step_state_mse_p99": single.state_mse.p99,
            "single_step_state_mse_max": single.state_mse.max,
            "single_step_position_mse_mean": single.position_mse.mean,
            "single_step_position_mse_median": single.position_mse.median,
            "single_step_position_mse_p95": single.position_mse.p95,
            "single_step_position_mse_p99": single.position_mse.p99,
            "single_step_position_mse_max": single.position_mse.max,
            "single_step_velocity_mse_mean": single.velocity_mse.mean,
            "single_step_velocity_mse_median": single.velocity_mse.median,
            "single_step_velocity_mse_p95": single.velocity_mse.p95,
            "single_step_velocity_mse_p99": single.velocity_mse.p99,
            "single_step_velocity_mse_max": single.velocity_mse.max,
            "physical_energy_final_drift_mean": physical.final_relative_drift.mean,
            "physical_energy_final_drift_median": physical.final_relative_drift.median,
            "physical_energy_final_drift_p95": physical.final_relative_drift.p95,
            "physical_energy_final_drift_max": physical.final_relative_drift.max,
            "physical_energy_max_drift_mean": physical.max_relative_drift.mean,
            "physical_energy_max_drift_median": physical.max_relative_drift.median,
            "physical_energy_max_drift_p95": physical.max_relative_drift.p95,
            "physical_energy_max_drift_max": physical.max_relative_drift.max,
        }

        for step, metrics in rollout.steps.items():
            for name, summary in (
                ("state_mse", metrics.state_mse),
                ("position_mse", metrics.position_mse),
                ("velocity_mse", metrics.velocity_mse),
            ):
                row[f"rollout_step_{step}_{name}_mean_finite"] = summary.mean_finite
                row[f"rollout_step_{step}_{name}_median"] = summary.median
                row[f"rollout_step_{step}_{name}_p95"] = summary.p95
                row[f"rollout_step_{step}_{name}_finite_fraction"] = summary.finite_fraction

        row["rollout_state_final_finite_fraction"] = rollout.state_final_finite_fraction

        for threshold, divergence in rollout.state_mse_thresholds.items():
            row[f"rollout_final_fraction_below_state_mse_{threshold}"] = (
                divergence.final_fraction_below
            )

        for threshold, divergence in rollout.position_mse_thresholds.items():
            row[f"rollout_final_fraction_below_position_mse_{threshold}"] = (
                divergence.final_fraction_below
            )

        if learned is not None:
            row["learned_h_final_drift_mean"] = learned.final_relative_drift.mean
            row["learned_h_final_drift_median"] = learned.final_relative_drift.median
            row["learned_h_final_drift_p95"] = learned.final_relative_drift.p95
            row["learned_h_final_drift_max"] = learned.final_relative_drift.max
            row["learned_h_max_drift_mean"] = learned.max_relative_drift.mean
            row["learned_h_max_drift_median"] = learned.max_relative_drift.median
            row["learned_h_max_drift_p95"] = learned.max_relative_drift.p95
            row["learned_h_max_drift_max"] = learned.max_relative_drift.max

        return row
