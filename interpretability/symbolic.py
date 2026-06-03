"""Deterministic PySR wrapper and Pareto-knee selection.

PySR is imported lazily inside the functions so importing this module (and the test suite)
never requires the Julia backend.

References:
    - Cranmer 2023 (PySR): https://arxiv.org/abs/2305.01582
    - PySR repo: https://github.com/MilesCranmer/PySR
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from interpretability._types import SymbolicResult

if TYPE_CHECKING:
    from pysr import PySRRegressor

BINARY_OPERATORS = ["+", "-", "*", "/"]
PYSR_OUTPUT_DIR = "runs/reports/interpretability/pysr"


def build_regressor(
    *,
    progress: bool = False,
    niterations: int = 80,
    maxsize: int = 30,
    seed: int = 0,
    run_id: str = "run",
) -> PySRRegressor:
    """Build a deterministic PySRRegressor (serial, fixed seed, + - * / only)."""
    from pysr import PySRRegressor

    return PySRRegressor(
        niterations=niterations,
        binary_operators=BINARY_OPERATORS,
        unary_operators=[],
        maxsize=maxsize,
        parallelism="serial",
        deterministic=True,
        random_state=seed,
        output_directory=PYSR_OUTPUT_DIR,
        run_id=run_id,
        progress=progress,
        verbosity=1 if progress else 0,
    )


def select_knee(complexity: Sequence[int], loss: Sequence[float]) -> int:
    """Return the complexity at the Pareto knee (end of the largest log-loss drop).

    model_selection="best" tends to over-select; this picks the point right after the
    sharpest loss cliff, which is where a recovered physical law sits.
    """
    if len(complexity) == 1:
        return int(complexity[0])
    best_i, best_drop = 1, -math.inf
    for i in range(1, len(loss)):
        prev, cur = loss[i - 1], loss[i]
        drop = math.log(prev) - math.log(cur) if prev > 0 and cur > 0 else prev - cur
        if drop > best_drop:
            best_drop, best_i = drop, i
    return int(complexity[best_i])


def fit_symbolic(
    x: np.ndarray,
    y: np.ndarray,
    variable_names: Sequence[str],
    *,
    progress: bool = False,
    niterations: int = 80,
    maxsize: int = 30,
    seed: int = 0,
    run_id: str = "run",
) -> tuple[SymbolicResult, np.ndarray]:
    """Fit PySR to (x, y); return the Pareto front and the knee equation's prediction on x."""
    x = np.asarray(x)
    reg = build_regressor(
        progress=progress, niterations=niterations, maxsize=maxsize, seed=seed, run_id=run_id
    )
    reg.fit(x, np.asarray(y), variable_names=list(variable_names))
    eq = reg.equations_
    complexity = [int(c) for c in eq["complexity"]]
    loss = [float(v) for v in eq["loss"]]
    equation = [str(s) for s in eq["equation"]]
    knee = select_knee(complexity, loss)
    knee_pos = next(i for i, c in enumerate(complexity) if c == knee)
    result = SymbolicResult(
        variable_names=list(variable_names),
        complexity=complexity,
        loss=loss,
        equation=equation,
        knee_complexity=knee,
        knee_equation=equation[knee_pos],
    )
    knee_pred = np.asarray(reg.predict(x, index=knee_pos))
    return result, knee_pred
