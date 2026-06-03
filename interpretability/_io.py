"""Read and write interpretability results and figure paths.

References:
    - evaluation/_io.py (JSON report I/O pattern)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from interpretability._types import InterpretabilityResults


def _json_safe(value: object) -> object:
    """Recursively convert values to JSON-safe forms (non-finite floats become None)."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_results(path: Path, results: InterpretabilityResults) -> None:
    """Write the result bundle as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_json_safe(results.to_dict()), f, indent=2, allow_nan=False)


def read_results(path: Path) -> InterpretabilityResults:
    """Read the result bundle from JSON."""
    with path.open() as f:
        return InterpretabilityResults.from_dict(json.load(f))


def figure_paths(figures_dir: Path, name: str) -> list[Path]:
    """Return [png, pdf] output paths for a figure, creating the directory."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    return [figures_dir / f"{name}.png", figures_dir / f"{name}.pdf"]
