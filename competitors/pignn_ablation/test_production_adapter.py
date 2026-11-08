"""Adapter gates; shared production engine has its own full regression tests."""

# ruff: noqa: ANN001, ANN201, D103
import json

import pytest

from competitors.pignn_ablation import production as adapter


@pytest.mark.parametrize("cell", ["A00", "A11", "unknown"])
def test_only_missing_cells(cell):
    with pytest.raises(ValueError, match="Only A10"):
        adapter.cell_recipe(cell)


@pytest.mark.parametrize("state", ["stopped_nonfinite_validation", "stopped_nonfinite_update"])
def test_numerical_failure_terminal(tmp_path, state):
    (tmp_path / "status.json").write_text(json.dumps({"state": state}))
    with pytest.raises(ValueError, match="terminal"):
        adapter.assert_resumable(tmp_path)


@pytest.mark.parametrize(
    "state", ["paused_battery", "paused_signal", "paused_wall_limit", "complete"]
)
def test_safe_pause_or_complete(tmp_path, state):
    (tmp_path / "status.json").write_text(json.dumps({"state": state}))
    adapter.assert_resumable(tmp_path)


def test_ready_corruption_rejected(tmp_path):
    payload = tmp_path / "payload"
    payload.write_text("tested")
    ready = tmp_path / "READY.json"
    ready.write_text(
        json.dumps(
            {
                "protocol": "daniela-pi-factorial-readiness-v1",
                "files": {str(payload): adapter.P.sha(payload)},
            }
        )
    )
    adapter.verify_ready(ready)
    payload.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        adapter.verify_ready(ready)
