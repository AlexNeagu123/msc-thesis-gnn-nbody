"""Tests for interpretability/symbolic.py knee selection (no PySR/Julia)."""

from interpretability.symbolic import select_knee


def test_knee_at_loss_cliff() -> None:
    """Knee is the complexity right after the sharpest log-loss drop."""
    complexity = [1, 3, 5, 7, 9]
    loss = [2.0, 1.5, 0.05, 0.04, 0.03]
    assert select_knee(complexity, loss) == 5


def test_knee_single_row() -> None:
    """A single Pareto row returns its own complexity."""
    assert select_knee([1], [1.0]) == 1


def test_knee_handles_zero_loss() -> None:
    """A zero loss does not raise and is treated as the cliff."""
    assert select_knee([1, 3, 5], [1.0, 0.9, 0.0]) == 5
