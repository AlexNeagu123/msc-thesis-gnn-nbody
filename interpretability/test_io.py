"""Tests for interpretability/_io.py."""

from pathlib import Path

from interpretability._io import figure_paths, read_results, write_results
from interpretability._types import (
    EgnnContrast,
    InterpretabilityResults,
    KineticRecovery,
    NodeEdgeLocality,
    PhysicalForce,
    PotentialRecovery,
    SymbolicResult,
)


def _symbolic() -> SymbolicResult:
    """A small Pareto front for fixtures."""
    return SymbolicResult(
        variable_names=["r"],
        complexity=[1, 5],
        loss=[1.0, 0.05],
        equation=["c", "-0.97/r + 6.1"],
        knee_complexity=5,
        knee_equation="-0.97/r + 6.1",
    )


def _results() -> InterpretabilityResults:
    """A fully-populated result bundle for fixtures."""
    return InterpretabilityResults(
        potential=PotentialRecovery(
            two_body=_symbolic(),
            two_body_c=0.97,
            total_linear_c=1.05,
            total_linear_r2=0.99,
            total_symbolic=_symbolic(),
        ),
        physical_force=PhysicalForce(force_exponent=-2.17, g_eff_forward=0.86, g_eff_gradient=0.86),
        node_edge=NodeEdgeLocality(
            alignment_r2=0.99, vnode_relative_std=0.18, vnode_suminv_corr=-0.58
        ),
        kinetic=KineticRecovery(
            isotropy_r2=0.998,
            quadratic_coefficient=0.417,
            expected_coefficient=0.334,
            symbolic=_symbolic(),
        ),
        egnn=EgnnContrast(
            layer0_weight_exponent=-1.37, layer0_force_exponent=-0.37, attractive_fraction=0.67
        ),
    )


def test_roundtrip(tmp_path: Path) -> None:
    """write_results then read_results returns an equal bundle."""
    results = _results()
    path = tmp_path / "results.json"
    write_results(path, results)
    assert read_results(path) == results


def test_figure_paths(tmp_path: Path) -> None:
    """figure_paths returns png+pdf and creates the directory."""
    paths = figure_paths(tmp_path / "figures", "potential")
    assert [p.suffix for p in paths] == [".png", ".pdf"]
    assert (tmp_path / "figures").is_dir()
