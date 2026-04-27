"""Tests for evaluation/scaling_report.py."""

from evaluation.scaling_report import _crossover_step, table_energy


def _report(*, final: float, max_drift: float, learned_final: float | None = None) -> dict:
    """Minimal metrics report containing energy fields used by scaling_report."""
    energy = {
        "physical": {
            "final_relative_drift": {"median": final},
            "max_relative_drift": {"median": max_drift},
        }
    }
    if learned_final is not None:
        energy["learned_hamiltonian"] = {
            "final_relative_drift": {"median": learned_final},
            "max_relative_drift": {"median": learned_final * 2},
        }
    return {"energy": energy}


def test_energy_table_reads_evaluator_energy_schema() -> None:
    """Energy report uses nested final/max relative drift median fields."""
    reports = {
        "egnn": {1000: _report(final=8.3876918254, max_drift=6.1101848446e9)},
        "hgnn": {
            1000: _report(final=2.4480650320, max_drift=6.4536938637, learned_final=0.0554721355)
        },
    }

    table = table_energy(reports)

    assert "8.388e+00" in table
    assert "6.110e+09" in table
    assert "2.448e+00" in table
    assert "5.547e-02" in table
    assert "| 1000 | hgnn | n/a" not in table


def test_crossover_step_ignores_initial_identical_state() -> None:
    """Step 0 is the known initial state, not a real rollout crossover."""
    egnn = [1.0, 0.5, 0.4, 0.3]
    hgnn = [0.9, 0.6, 0.2, 0.1]

    assert _crossover_step(egnn, hgnn) == 2
