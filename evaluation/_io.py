"""Serialization for evaluation reports: the metrics.json / summary.csv I/O boundary."""

import csv
import json
from pathlib import Path

import numpy as np

from evaluation._types import EvaluationReport, SummaryRow


def read_evaluation_report(path: Path) -> EvaluationReport:
    """Load and parse a metrics.json file into a typed report."""
    with path.open() as f:
        return EvaluationReport.from_dict(json.load(f))


def write_evaluation_report(path: Path, report: EvaluationReport) -> None:
    """Write a typed report as metrics.json (numpy- and NaN-safe)."""
    with path.open("w") as f:
        json.dump(_json_safe(report.to_dict()), f, indent=2, allow_nan=False)


def write_summary_csv(path: Path, report: EvaluationReport) -> None:
    """Write the flat one-row summary CSV alongside metrics.json."""
    row = SummaryRow.from_report(report).to_csv_row()
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _json_safe(value: object) -> object:
    """Convert numpy values and non-finite floats to JSON-safe values."""
    if isinstance(value, dict):
        return {key: _json_safe(v) for key, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating | float):
        coerced = float(value)
        return coerced if np.isfinite(coerced) else None
    return value
