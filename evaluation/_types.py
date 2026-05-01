"""Typed contracts for evaluation reports.

Top-level entry point is `EvaluationReport.from_dict()`. Inner dataclasses
do not expose their own `from_dict` to match the pattern in
`training/_types.py`: a single top-level constructor builds the whole tree.

References:
    - JSON schema produced by evaluation/evaluate.py:_build_report
    - I/O wrapper: evaluation/_io.py
    - Pattern mirror: training/_types.py
"""

from dataclasses import dataclass
from typing import Any


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
    checkpoint_path: str
    config_path: str
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

    mse: MseSummary
    min_pairwise_distance: DistanceSummary

    def to_dict(self) -> dict[str, Any]:
        """Serialize preserving _build_report key order."""
        return {
            "mse": self.mse.to_dict(),
            "min_pairwise_distance": self.min_pairwise_distance.to_dict(),
        }


@dataclass
class RolloutStepMetrics:
    """Rollout summary at a single rollout step."""

    mean_finite_mse: float | None
    median_mse: float | None
    p95_mse: float | None
    finite_fraction: float | None

    def to_dict(self) -> dict[str, float | None]:
        """Serialize preserving _rollout_steps key order."""
        return {
            "mean_finite_mse": self.mean_finite_mse,
            "median_mse": self.median_mse,
            "p95_mse": self.p95_mse,
            "finite_fraction": self.finite_fraction,
        }


@dataclass
class RolloutCurves:
    """Full per-step rollout curves for crossover analysis."""

    step: list[int]
    mean_finite_mse: list[float | None]
    median_mse: list[float | None]
    p95_mse: list[float | None]
    finite_fraction: list[float | None]

    def to_dict(self) -> dict[str, list[Any]]:
        """Serialize preserving _rollout_curves key order."""
        return {
            "step": self.step,
            "mean_finite_mse": self.mean_finite_mse,
            "median_mse": self.median_mse,
            "p95_mse": self.p95_mse,
            "finite_fraction": self.finite_fraction,
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
    thresholds: dict[str, DivergenceMetrics]
    finite_final_fraction: float | None
    curves: RolloutCurves | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize preserving _build_report rollout key order."""
        # order: steps, curves (optional), first_nonfinite_step, thresholds, finite_final_fraction
        out: dict[str, Any] = {
            "steps": {k: v.to_dict() for k, v in self.steps.items()},
        }
        if self.curves is not None:
            out["curves"] = self.curves.to_dict()
        out["first_nonfinite_step"] = self.first_nonfinite_step
        out["thresholds"] = {k: v.to_dict() for k, v in self.thresholds.items()}
        out["finite_final_fraction"] = self.finite_final_fraction
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

        return cls(
            metadata=EvaluationMetadata(**d["metadata"]),
            single_step=SingleStepReport(
                mse=MseSummary(**d["single_step"]["mse"]),
                min_pairwise_distance=DistanceSummary(**d["single_step"]["min_pairwise_distance"]),
            ),
            rollout=RolloutReport(
                steps={k: RolloutStepMetrics(**v) for k, v in rollout["steps"].items()},
                first_nonfinite_step=rollout["first_nonfinite_step"],
                thresholds={k: DivergenceMetrics(**v) for k, v in rollout["thresholds"].items()},
                finite_final_fraction=rollout["finite_final_fraction"],
                curves=RolloutCurves(**curves) if curves is not None else None,
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

    Dynamic keys (rollout_step_<n>_*, rollout_final_fraction_below_mse_<t>,
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
        single = self.report.single_step.mse
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
            "single_step_mse_mean": single.mean,
            "single_step_mse_median": single.median,
            "single_step_mse_p95": single.p95,
            "single_step_mse_p99": single.p99,
            "single_step_mse_max": single.max,
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
            row[f"rollout_step_{step}_mean_finite_mse"] = metrics.mean_finite_mse
            row[f"rollout_step_{step}_median_mse"] = metrics.median_mse
            row[f"rollout_step_{step}_p95_mse"] = metrics.p95_mse
            row[f"rollout_step_{step}_finite_fraction"] = metrics.finite_fraction

        row["rollout_final_finite_fraction"] = rollout.finite_final_fraction

        for threshold, divergence in rollout.thresholds.items():
            row[f"rollout_final_fraction_below_mse_{threshold}"] = (
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
