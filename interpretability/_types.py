"""Result dataclasses for the interpretability analysis.

References:
    - Cranmer et al. 2020 (symbolic models from GNNs): https://arxiv.org/abs/2006.11287
    - Cranmer 2023 (PySR): https://arxiv.org/abs/2305.01582
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SymbolicResult:
    """A PySR Pareto front and the knee equation selected from it."""

    variable_names: list[str]
    complexity: list[int]
    loss: list[float]
    equation: list[str]
    knee_complexity: int
    knee_equation: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SymbolicResult:
        """Rebuild from a plain dict."""
        return cls(**d)


@dataclass(frozen=True)
class PotentialRecovery:
    """HGNN potential recovered as -C/r (2-body) and -C*sum(1/r) (3-body, in-distribution)."""

    two_body: SymbolicResult
    two_body_c: float
    total_linear_c: float
    total_linear_r2: float
    total_symbolic: SymbolicResult


@dataclass(frozen=True)
class PhysicalForce:
    """HGNN physical 2-body force law and effective gravitational constant."""

    force_exponent: float
    g_eff_forward: float
    g_eff_gradient: float


@dataclass(frozen=True)
class NodeEdgeLocality:
    """Whether the per-edge readout localizes the pairwise potential (companion finding)."""

    alignment_r2: float
    vnode_relative_std: float
    vnode_suminv_corr: float


@dataclass(frozen=True)
class KineticRecovery:
    """HGNN kinetic energy: emergent isotropy and quadratic form."""

    isotropy_r2: float
    quadratic_coefficient: float
    expected_coefficient: float
    symbolic: SymbolicResult


@dataclass(frozen=True)
class EgnnContrast:
    """EGNN layer-0 coupling: no recoverable gravitational law."""

    layer0_weight_exponent: float
    layer0_force_exponent: float
    attractive_fraction: float


@dataclass(frozen=True)
class InterpretabilityResults:
    """All interpretability conclusions for HGNN and EGNN."""

    potential: PotentialRecovery
    physical_force: PhysicalForce
    node_edge: NodeEdgeLocality
    kinetic: KineticRecovery
    egnn: EgnnContrast

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full result bundle to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InterpretabilityResults:
        """Rebuild the bundle, reconstructing nested dataclasses."""
        pot = d["potential"]
        kin = d["kinetic"]
        return cls(
            potential=PotentialRecovery(
                two_body=SymbolicResult.from_dict(pot["two_body"]),
                two_body_c=pot["two_body_c"],
                total_linear_c=pot["total_linear_c"],
                total_linear_r2=pot["total_linear_r2"],
                total_symbolic=SymbolicResult.from_dict(pot["total_symbolic"]),
            ),
            physical_force=PhysicalForce(**d["physical_force"]),
            node_edge=NodeEdgeLocality(**d["node_edge"]),
            kinetic=KineticRecovery(
                isotropy_r2=kin["isotropy_r2"],
                quadratic_coefficient=kin["quadratic_coefficient"],
                expected_coefficient=kin["expected_coefficient"],
                symbolic=SymbolicResult.from_dict(kin["symbolic"]),
            ),
            egnn=EgnnContrast(**d["egnn"]),
        )
