"""Production-runner gates with tiny synthetic data; no long training."""

# ruff: noqa: ANN001, ANN003, ANN201, ANN202, D103, PT018
import fcntl
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")
import tensorflow as tf

from competitors.gnode_pignn_tf.runtime import configure

if not tf.config.threading.get_intra_op_parallelism_threads():
    configure()

from competitors.gnode_pignn_training import production
from competitors.gnode_pignn_training.checkpoint import restore_checkpoint
from competitors.gnode_pignn_training.data import PairSampler
from competitors.gnode_pignn_training.engine import StepEngine
from competitors.gnode_pignn_training.recipes import RECIPES

ROOT = Path(__file__).resolve().parents[2]
IDENTITY = {"environment": {"test": True}, "scope": "unit-test-only"}


def tiny_recipe(**changes):
    return replace(RECIPES["pignn_repository_shared_v1"], batch_size=5, epochs=3, **changes)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    x = np.concatenate(
        [
            rng.uniform(0.5, 2, (13, 3, 1)),
            rng.normal(size=(13, 3, 2)),
            0.1 * rng.normal(size=(13, 3, 2)),
        ],
        axis=-1,
    ).astype(np.float32)
    y = (x[..., 1:] + 0.001 * rng.normal(size=(13, 3, 4))).astype(np.float32)
    reference = rng.normal(size=(2, 7, 3, 5))
    reference[..., 4] = rng.uniform(0.5, 2, (2, 1, 3))
    initial = reference[:, 0][..., [4, 0, 1, 2, 3]].astype(np.float32)
    return x, y, initial, reference


def run(path, data, recipe=None, **kwargs):
    x, y, initial, reference = data
    kwargs.setdefault("selection_every", 1)
    kwargs.setdefault("log", lambda *_: None)
    return production.run_training(
        recipe or tiny_recipe(), path, x, y, initial, reference, IDENTITY, **kwargs
    )


def fake_rollout(reference, offset=0.0, bad=None):
    def execute(initial, steps):
        out = reference[:, 1:, :, [4, 0, 1, 2, 3]].astype(np.float32).copy()
        out[..., 1:3] += offset
        if bad is not None:
            out[bad, 2, 0, 1] = np.nan
        assert int(steps) == 6
        return out

    return execute


def restored_arrays(path, recipe=None):
    recipe = recipe or tiny_recipe()
    engine = StepEngine(recipe)
    sampler = PairSampler(13, recipe.batch_size, recipe.kind == "gnode", recipe.seed)
    restore_checkpoint(path, engine, sampler, IDENTITY)
    return engine.arrays()


def test_validation_score_contract(data):
    _, _, initial, reference = data
    zero = production.validation_score(fake_rollout(reference), initial, reference, 2)
    assert zero["eligible"] is True
    assert zero["score"] == pytest.approx(0, abs=1e-6)  # float32 rollout cast only
    assert (zero["finite_systems"], zero["n_systems"]) == (2, 2)
    shifted = production.validation_score(fake_rollout(reference, 0.5), initial, reference, 2)
    assert shifted["score"] == pytest.approx(0.5)
    broken = production.validation_score(fake_rollout(reference, bad=1), initial, reference, 2)
    assert broken["eligible"] is False and broken["score"] is None
    assert broken["finite_systems"] == 1
    with pytest.raises(ValueError, match="rollout shape"):
        production.validation_score(lambda *_: np.zeros((2, 5, 3, 5)), initial, reference, 2)


def test_schedule_matches_declared_recipe():
    expected = RECIPES["pignn_repository_shared_v1"].record()["selection_epochs_proposed"]
    assert production.selection_epochs(100) == expected == list(range(4, 101, 4))


def test_interrupted_resume_is_exactly_straight_run(tmp_path, data):
    straight = tmp_path / "straight"
    assert run(straight, data)["state"] == "complete"
    staged = tmp_path / "staged"
    calls = {"n": 0}

    def stop_after_four_updates():
        calls["n"] += 1
        return calls["n"] == 5  # mid second epoch, across a short remainder batch

    interrupted = run(staged, data, stop_requested=stop_after_four_updates)
    assert interrupted["state"] == "paused_signal"
    assert json.loads((staged / "status.json").read_text())["updates"] == 4
    assert run(staged, data)["state"] == "complete"
    left = json.loads((straight / "RESULT.json").read_text())
    right = json.loads((staged / "RESULT.json").read_text())
    for key in ("best", "final", "epoch_history", "updates"):
        assert left[key] == right[key]
    assert [(v["epoch"], v["score"]) for v in left["validations"]] == [
        (v["epoch"], v["score"]) for v in right["validations"]
    ]
    for name in left["bundle_hashes"]:
        a = restored_arrays(straight / Path(name).parent)
        b = restored_arrays(staged / Path(name).parent)
        for left_array, right_array in zip(a, b, strict=True):
            np.testing.assert_array_equal(left_array, right_array)
    assert (staged / "selected/final-epoch-0003/COMMITTED.json").is_file()


def test_strict_best_selection_and_separate_final(tmp_path, data):
    scores = iter([0.5, 0.4, 0.4])

    def score():
        return {"eligible": True, "score": next(scores), "finite_systems": 2, "n_systems": 2}

    assert run(tmp_path, data, score_fn=score)["state"] == "complete"
    result = json.loads((tmp_path / "RESULT.json").read_text())
    assert result["best"]["epoch"] == 2 and result["best"]["score"] == 0.4
    assert result["final"]["epoch"] == 3
    assert sorted(p.name for p in (tmp_path / "selected").iterdir()) == [
        "best-epoch-0001",
        "best-epoch-0002",
        "final-epoch-0003",
    ]
    assert result["comparison_records_read"] is False


def test_ineligible_validation_never_becomes_best(tmp_path, data):
    scores = iter([{"eligible": True, "score": 0.5}, {"eligible": False, "score": None}])

    def score():
        return {**next(scores), "finite_systems": 1, "n_systems": 2}

    with pytest.raises(RuntimeError, match="nonfinite validation"):
        run(tmp_path, data, score_fn=score)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "stopped_nonfinite_validation"
    assert status["best"]["epoch"] == 1
    assert not (tmp_path / "RESULT.json").exists()
    assert (tmp_path / "checkpoints/u0000006/COMMITTED.json").is_file()


def test_nonfinite_training_update_is_rejected_without_checkpoint(tmp_path, data):
    x, y, initial, reference = data
    bad = x.copy()
    bad[0, 0, 1] = np.nan
    with pytest.raises(tf.errors.InvalidArgumentError):
        run(tmp_path, (bad, y, initial, reference))
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "stopped_nonfinite_update"
    assert not list((tmp_path / "checkpoints").glob("u*"))
    assert not (tmp_path / "RESULT.json").exists()


def test_battery_pause_then_resume_and_complete_noop(tmp_path, data):
    assert run(tmp_path, data, mains_power=lambda: False)["state"] == "paused_battery"
    assert run(tmp_path, data)["state"] == "complete"
    assert run(tmp_path, data) == {"state": "complete", "already_complete": True}


def test_wall_limit_pauses_at_committed_state(tmp_path, data):
    assert run(tmp_path, data, max_wall_seconds=-1)["state"] == "paused_wall_limit"


def test_duplicate_plan_corruption_and_overwrite_protection(tmp_path, data):
    with (tmp_path / "run.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="no duplicate"):
            run(tmp_path, data)
    assert run(tmp_path, data, max_wall_seconds=-1)["state"] == "paused_wall_limit"
    with pytest.raises(ValueError, match="plan differs"):
        run(tmp_path, data, recipe=tiny_recipe(learning_rate=0.001))
    checkpoint = next((tmp_path / "checkpoints").glob("u*"))
    with (checkpoint / "metadata.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        run(tmp_path, data)


def test_real_validation_split_is_disjoint_and_runner_avoids_comparison_file():
    val = ROOT / "data/output/daniela-rk4/val.h5"
    train = ROOT / "data/output/daniela-rk4/train.h5"
    if not val.exists() or not train.exists():
        pytest.skip("local split required")
    initial, reference, ids = production.load_validation(val)
    assert initial.shape == (80, 3, 5) and reference.shape == (80, 2000, 3, 5)
    assert np.array_equal(ids, np.sort(ids))
    np.testing.assert_array_equal(initial[..., 0], reference[:, 0, :, 4].astype(np.float32))
    import h5py

    with h5py.File(train) as handle:
        assert not set(handle["simulation_ids"][:].tolist()) & set(ids.tolist())
    assert "test.h5" not in Path(production.__file__).read_text()
