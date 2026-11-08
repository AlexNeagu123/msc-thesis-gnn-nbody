"""Analytical and real-notebook parity tests for the separate Daniela protocol."""

import ast
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

from evaluation._io import _json_safe
from evaluation.rk4_metrics import WINDOWS, build_daniela_report, invariants
from evaluation.evaluate_rk4 import (
    DEFAULT_SOURCE,
    DT,
    generate_initial_states,
    prepare_reference,
    rk4_reference,
    score_predictions,
)


def _stationary(n_steps: int = 2000) -> np.ndarray:
    reference = np.zeros((2, n_steps + 1, 3, 5))
    reference[..., 4] = [0.5, 1, 2]
    reference[..., 0] = [0, 1, 3]
    return reference


@pytest.fixture(scope="module")
def oracle() -> dict:
    """Execute only selected functions/classes from the real notebook, never TF/Colab."""
    path = DEFAULT_SOURCE / "notebooks/3_testing.ipynb"
    if not path.exists():
        pytest.skip("Daniela checkout unavailable; analytical tests still run")
    names = {
        "Body",
        "RK4Body",
        "generate_bodies",
        "run_rk4",
        "KE",
        "PE",
        "P",
        "L",
        "energy",
        "linear_momentum",
        "angular_momentum",
        "calculate_validation_step",
        "evaluate_trajectories",
    }
    definitions = []
    for cell in json.loads(path.read_text())["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse("".join(cell["source"]))
            definitions += [
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef | ast.ClassDef) and node.name in names
            ]
    assert {node.name for node in definitions} == names
    namespace = {
        "np": np,
        "epsilon": 0.05,
        "N": 3,
        "TU_AU": math.sqrt(149597870700**3 / (6.6743e-11 * 1e30)) / 149597870700,
        "rd": random.Random(42),
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_exact_position_reduction_and_population_std() -> None:
    """Average coordinates, then time/bodies, then runs; ignore velocity for accuracy."""
    reference = _stationary()
    predicted = reference[:, 1:].copy()
    predicted[0, ..., :2] += [1, 2]
    predicted[1, ..., :2] += [3, 4]
    predicted[..., 2:4] += 100
    report = build_daniela_report(reference, predicted)
    for entry in report["trajectory"].values():
        assert entry["complete_window"]
        assert entry["mae"]["mean"] == 2.5
        assert entry["mae"]["std"] == 1
        assert entry["mse"]["mean"] == 7.5
        assert entry["mse"]["std"] == 5


def test_window_boundaries_and_no_initial_error() -> None:
    """Future step 251 belongs to medium, and 1001 belongs to long."""
    reference = _stationary()
    predicted = reference[:, 1:].copy()
    predicted[:, 250, :, :2] += 1
    predicted[:, 1000, :, :2] += 2
    windows = build_daniela_report(reference, predicted)["trajectory"]
    assert windows["0-250"]["mae"]["mean"] == 0
    assert windows["250-1000"]["mae"]["mean"] == pytest.approx(1 / 750)
    assert windows["1000-2000"]["mae"]["mean"] == pytest.approx(2 / 1000)
    with pytest.raises(ValueError, match="expected reference"):
        build_daniela_report(reference, reference)


def test_legacy_energy_is_not_silently_corrected() -> None:
    """Exact same states still get the notebook's inconsistent legacy energy offset."""
    reference = _stationary(3)
    report = build_daniela_report(reference, reference[:, 1:], include_series=True)
    assert report["conservation_legacy"]["energy"]["temporal_mean"]["mean"] > 0
    assert report["physical_softened_energy"]["temporal_max"]["mean"] == 0
    assert report["relative_percent_legacy"]["linear_momentum"]["temporal_mean"]["mean"] == 0
    assert not report["trajectory"]["0-250"]["complete_window"]
    assert report["trajectory"]["250-1000"]["n_steps_scored"] == 0
    assert _json_safe(report)["trajectory"]["250-1000"]["mae"]["mean"] is None


def test_divergence_propagates_and_strict_json() -> None:
    """Never obtain a deceptively good mean by dropping a failed run."""
    reference = _stationary()
    predicted = reference[:, 1:].copy()
    predicted[1, 1000, 0, 0] = np.nan
    report = build_daniela_report(reference, predicted)
    assert report["failures"]["nonfinite_runs"] == [
        {"run_index": 1, "first_nonfinite_future_step": 1001}
    ]
    long = report["trajectory"]["1000-2000"]["mae"]
    assert long["finite_runs"] == 1
    assert np.isnan(long["mean"])
    parsed = json.loads(json.dumps(_json_safe(report), allow_nan=False))
    assert parsed["trajectory"]["1000-2000"]["mae"]["mean"] is None


def test_runtime_requires_explicit_scope() -> None:
    """Do not pretend kernel-only timings match the notebook's entire loop."""
    reference = _stationary(2)
    with pytest.raises(ValueError, match="explicit scope"):
        build_daniela_report(reference, reference[:, 1:], runtime_seconds=np.array([1, 2]))
    report = build_daniela_report(
        reference, reference[:, 1:], runtime_seconds=np.array([1, 3]), runtime_scope="whole-loop"
    )
    assert report["runtime"]["seconds"]["mean"] == 2
    assert report["runtime"]["seconds"]["std"] == 1


def test_masses_preserved() -> None:
    """Reject invalid state layouts or models that change masses."""
    reference = _stationary(2)
    predicted = reference[:, 1:].copy()
    predicted[0, 0, 0, 4] = 3
    with pytest.raises(ValueError, match="masses"):
        build_daniela_report(reference, predicted)


def test_seeded_generator_matches_notebook(oracle: dict) -> None:
    """Match random draw order and momentum balancing, not just distributions."""
    oracle["rd"] = random.Random(42)
    actual = generate_initial_states(42, 2)
    for run in range(2):
        bodies = oracle["generate_bodies"](3)
        expected = np.array([[*b.position, *b.velocity, b.mass] for b in bodies])
        np.testing.assert_array_equal(actual[run], expected)
    np.testing.assert_allclose(invariants(actual)["linear_momentum"], 0, atol=1e-15)
    np.testing.assert_array_equal(actual, generate_initial_states(42, 2))


def test_rk4_matches_notebook_every_step(oracle: dict) -> None:
    """Replay the source's object-based integrator and compare full q/v states."""
    initial = generate_initial_states(17, 1)
    actual = rk4_reference(initial, 40)
    bodies = [oracle["RK4Body"](s[4], s[:2].copy(), s[2:4].copy()) for s in initial[0]]
    E0 = oracle["energy"](bodies, 3)
    P0, L0 = oracle["linear_momentum"](bodies), oracle["angular_momentum"](bodies)
    x, y = [[b.position[0]] for b in bodies], [[b.position[1]] for b in bodies]
    histories = [[] for _ in range(9)]
    for step in range(1, 41):
        oracle["run_rk4"](1, DT, bodies, x, y, E0, P0, L0, *histories)
        expected = np.array([[*b.position, *b.velocity, b.mass] for b in bodies])
        np.testing.assert_allclose(actual[0, step], expected, rtol=1e-12, atol=1e-13)


def test_trajectory_metrics_match_notebook_all_windows(oracle: dict) -> None:
    """Use the actual repository function as the oracle over 2,000 future steps."""
    reference = rk4_reference(generate_initial_states(53, 2))
    predicted = reference[:, 1:].copy()
    predicted[0, ..., :2] += np.linspace(0, 0.1, 2000)[:, None, None]
    predicted[1, ..., :2] -= np.linspace(0, 0.2, 2000)[:, None, None]
    expected_mae = {key: [] for key in WINDOWS}
    expected_mse = {key: [] for key in WINDOWS}
    for run in range(2):
        oracle["evaluate_trajectories"](
            WINDOWS,
            reference[run, :, :, 0].T,
            reference[run, :, :, 1].T,
            predicted[run, :, :, 0].T,
            predicted[run, :, :, 1].T,
            expected_mae,
            expected_mse,
        )
    actual = build_daniela_report(reference, predicted)
    for window in WINDOWS:
        for metric, expected in (("mae", expected_mae), ("mse", expected_mse)):
            np.testing.assert_allclose(
                actual["trajectory"][window][metric]["per_run"], expected[window], rtol=1e-14
            )
            assert actual["trajectory"][window][metric]["std"] == pytest.approx(
                np.std(expected[window])
            )


def test_invariants_and_relative_errors_match_notebook(oracle: dict) -> None:
    """Match the option=0 energy bug, vector-norm momentum, signed L and fallback."""
    reference = rk4_reference(generate_initial_states(77, 1), 4)
    predicted = reference[:, 1:].copy()
    predicted[..., :2] += 0.01
    predicted[..., 2:4] += 0.02
    initial = [oracle["RK4Body"](s[4], s[:2], s[2:4]) for s in reference[0, 0]]
    E0 = oracle["energy"](initial, 3)
    P0, L0 = oracle["linear_momentum"](initial), oracle["angular_momentum"](initial)
    histories = [[] for _ in range(9)]
    for frame in predicted[0]:
        bodies = [oracle["RK4Body"](s[4], s[:2], s[2:4]) for s in frame]
        oracle["calculate_validation_step"](bodies, 0, E0, P0, L0, *histories)
    actual = build_daniela_report(reference, predicted, include_series=True)
    for i, name in enumerate(("energy", "linear_momentum", "angular_momentum")):
        np.testing.assert_allclose(
            actual["conservation_legacy"][name]["per_run_series"][0],
            histories[3 + i],
            rtol=1e-10,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            actual["relative_percent_legacy"][name]["per_run_series"][0],
            histories[6 + i],
            rtol=1e-10,
            atol=1e-12,
        )
        assert actual["conservation_legacy"][name]["temporal_max"]["mean"] == pytest.approx(
            max(histories[3 + i])
        )


def test_reference_and_json_csv_round_trip(tmp_path: Path, oracle: dict) -> None:
    """Exercise frozen inputs, NPZ export/reload, IDs, strict JSON and no-overwrite."""
    reference_dir, baseline_dir = tmp_path / "reference", tmp_path / "baseline"
    prepare_reference(reference_dir, seed=2026, n_runs=2, n_steps=20)
    report = score_predictions(
        reference_dir, baseline_dir, model_name="constant_velocity", baseline="constant_velocity"
    )
    saved = json.loads((baseline_dir / "daniela_metrics.json").read_text())
    assert saved == _json_safe(report)
    assert (baseline_dir / "daniela_summary.csv").exists()
    copied = score_predictions(
        reference_dir,
        tmp_path / "reload",
        model_name="reloaded",
        predictions_path=baseline_dir / "predictions.npz",
    )
    assert copied["trajectory"]["0-250"] == report["trajectory"]["0-250"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        score_predictions(reference_dir, baseline_dir, model_name="x", baseline="persistence")
    with np.load(baseline_dir / "predictions.npz") as bundle:
        np.savez(tmp_path / "wrong.npz", predicted=bundle["predicted"], run_ids=[1, 0])
    with pytest.raises(ValueError, match="IDs/order"):
        score_predictions(
            reference_dir,
            tmp_path / "bad",
            model_name="bad",
            predictions_path=tmp_path / "wrong.npz",
        )
