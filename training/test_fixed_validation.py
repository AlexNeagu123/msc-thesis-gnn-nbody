"""Shared baseline protocol, batched rollout and failure handling tests."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch import Tensor, nn

from evaluation.rk4_metrics import build_daniela_report
from models.baselines import ConstantVelocityBaseline, MeanStateBaseline
from training.fixed_validation import FixedRolloutEvaluator, file_sha256


def _files(tmp_path: Path, frames: int = 9) -> tuple[Path, Path]:
    states = np.zeros((3, frames, 3, 5), dtype=np.float64)
    states[..., 4] = [0.5, 1.3, 2]
    states[..., 2:4] = [[0.1, 0.2], [-0.2, 0.3], [0.4, -0.1]]
    states[..., :2] = np.array([[0, 0], [1, 0.2], [-0.3, 0.8]])
    states[..., :2] += np.arange(frames)[None, :, None, None] * 0.001 * states[..., 2:4]
    states[..., :2] += np.arange(3)[:, None, None, None] * 2
    train_path, val_path = tmp_path / "train.h5", tmp_path / "val.h5"
    for path, multiplier in ((train_path, 10), (val_path, 1)):
        current = states.copy()
        current[..., :4] *= multiplier
        with h5py.File(path, "w") as handle:
            handle["trajectories"] = current
            handle["simulation_ids"] = [91, 24, 111]
            handle["times"] = np.broadcast_to(np.arange(frames) * 0.001, (3, frames))
    return train_path, val_path


def _evaluator(tmp_path: Path, frames: int = 9, **kwargs: object) -> FixedRolloutEvaluator:
    train, val = _files(tmp_path, frames)
    return FixedRolloutEvaluator(val, train, 0.001, torch.device("cpu"), **kwargs)


@pytest.mark.parametrize("frames", [9, 2000, 2001])
def test_matches_daniela_position_reductions(tmp_path: Path, frames: int) -> None:
    """Future-only indexing, position MAE and complete/partial window lengths match."""
    evaluator = _evaluator(tmp_path, frames, batch_size=2)
    model = ConstantVelocityBaseline(0.001)
    predicted = evaluator._predict(model)
    oracle = build_daniela_report(evaluator.reference, predicted)
    report = evaluator.score(model)
    expected = np.abs(
        predicted.astype(np.float64)[..., :2] - evaluator.reference[:, 1:, :, :2]
    ).mean()
    assert report["model"]["selection_score"] == pytest.approx(expected, abs=1e-15)
    assert report["metadata"]["simulation_ids"] == [91, 24, 111]
    assert report["metadata"]["steps"] == frames - 1
    for name, window in report["model"]["windows"].items():
        if window["available_steps"]:
            expected_window = oracle["trajectory"][name]["mae"]
            assert window["mae"]["mean"] == pytest.approx(expected_window["mean"])
            assert window["mae"]["std"] == pytest.approx(expected_window["std"])
    if frames == 2000:
        assert report["model"]["windows"]["1000-2000"]["available_steps"] == 999
        assert not report["model"]["windows"]["1000-2000"]["complete"]


class CountingVelocity(ConstantVelocityBaseline):
    """Record calls to verify batching over simulations, not a per-simulation loop."""

    def __init__(self) -> None:
        """Create the same baseline while recording input batch sizes."""
        super().__init__(0.001)
        self.batch_sizes = []

    def forward(self, state: Tensor) -> Tensor:
        """Record the batch shape then advance free motion."""
        self.batch_sizes.append(len(state))
        return super().forward(state)


@pytest.mark.parametrize("training", [True, False])
def test_batching_order_and_mode_restoration(tmp_path: Path, training: bool) -> None:
    """Batching preserves predictions/run order and restores caller train/eval mode."""
    evaluator = _evaluator(tmp_path, batch_size=2)
    model = CountingVelocity().train(training)
    actual = evaluator._predict(model)
    assert model.training is training
    assert model.batch_sizes == [2] * 8 + [1] * 8
    unbatched = FixedRolloutEvaluator(
        evaluator.val_path, evaluator.train_path, 0.001, torch.device("cpu"), batch_size=1
    )
    np.testing.assert_array_equal(actual, unbatched._predict(ConstantVelocityBaseline(0.001)))


def test_train_only_baseline_fit_and_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fit mean baselines from the requested train prefix only; freeze baseline results."""
    evaluator = _evaluator(tmp_path, n_train_trajectories=1)
    first = evaluator.score(ConstantVelocityBaseline(0.001))
    with h5py.File(evaluator.train_path) as handle:
        states = torch.from_numpy(handle["trajectories"][:1]).float()
    expected_model = MeanStateBaseline(states[..., :2].reshape(-1, 2).mean(0))
    expected, _ = evaluator._summarize(evaluator._predict(expected_model))
    assert first["baselines"]["mean_state"] == expected
    assert first["metadata"]["baseline_fit_trajectories"] == 1
    assert set(first["baselines"]) == {
        "persistence",
        "constant_velocity",
        "mean_velocity",
        "mean_state",
    }
    monkeypatch.setattr(
        "training.fixed_validation.read_states", lambda *args: pytest.fail("cache reread")
    )
    second = evaluator.score(ConstantVelocityBaseline(0.001))
    assert second["baselines"] is first["baselines"]
    assert not evaluator.reference.flags.writeable
    # CV equals its own cached winning envelope on these free-motion trajectories.
    assert abs(second["model"]["baseline_normalized_state_mse"]["score"]) < 1e-9


class NonfiniteModel(nn.Module):
    """Preserve masses but corrupt one dynamic component in a subset of trajectories."""

    def __init__(self, channel: int, value: float) -> None:
        """Choose a position/velocity channel and NaN/Inf failure value."""
        super().__init__()
        self.channel, self.value = channel, value

    def forward(self, state: Tensor) -> Tensor:
        """Fail only the first trajectory of each batch, without changing other runs."""
        result = state.clone()
        result[0, 0, self.channel] = self.value
        return result


@pytest.mark.parametrize("channel", [0, 2])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_any_divergent_run_rejects_checkpoint(tmp_path: Path, channel: int, value: float) -> None:
    """Neither a median nor finite positions may hide a failed velocity trajectory."""
    evaluator = _evaluator(tmp_path, batch_size=3)
    model = NonfiniteModel(channel, value).train()
    report = evaluator.score(model)["model"]
    assert model.training
    assert report["failed_runs"] == 1
    assert report["first_failed_step"] == [1, None, None]
    assert report["nonfinite_dynamic_values"] == 8
    assert not report["eligible"]
    assert np.isinf(report["selection_score"])
    assert np.isinf(report["baseline_normalized_state_mse"]["score"])


def test_reference_limits_and_dt(tmp_path: Path) -> None:
    """Never invent missing targets or silently evaluate the wrong physical timestep."""
    evaluator = _evaluator(tmp_path, steps=9)
    with pytest.raises(ValueError, match="exceed available"):
        _ = evaluator.reference
    evaluator.steps = 4
    assert evaluator.reference.shape[1] == 5
    evaluator = FixedRolloutEvaluator(
        evaluator.val_path, evaluator.train_path, 0.05, torch.device("cpu")
    )
    with pytest.raises(ValueError, match="time grid disagrees"):
        _ = evaluator.reference


@pytest.mark.parametrize("source", ["train", "val"])
def test_changed_source_rejected(tmp_path: Path, source: str) -> None:
    """Hashes connect cached validation/baselines to the actual training setup snapshot."""
    train, val = _files(tmp_path)
    kwargs = {f"expected_{source}_sha256": file_sha256(train if source == "train" else val)}
    evaluator = FixedRolloutEvaluator(val, train, 0.001, torch.device("cpu"), **kwargs)
    with h5py.File(train if source == "train" else val, "r+") as handle:
        handle.attrs["changed"] = "after training setup"
    with pytest.raises(RuntimeError, match="source changed since training setup"):
        evaluator.score(ConstantVelocityBaseline(0.001))


class InvalidOutputModel(nn.Module):
    """Deliberately violate either the shape or mass-preservation contract."""

    def __init__(self, problem: str) -> None:
        """Choose the contract violation to generate."""
        super().__init__()
        self.problem = problem

    def forward(self, state: Tensor) -> Tensor:
        """Return malformed dynamic state or changed masses."""
        result = state.clone()
        if self.problem == "shape":
            return result[..., :4]
        result[..., 4] *= 2
        return result


@pytest.mark.parametrize("problem", ["shape", "mass"])
def test_invalid_output_restores_mode(tmp_path: Path, problem: str) -> None:
    """Contract errors remain visible and do not leave a training model in eval mode."""
    evaluator = _evaluator(tmp_path)
    model = InvalidOutputModel(problem).train()
    with pytest.raises(ValueError, match=r"shape|masses"):
        evaluator.score(model)
    assert model.training
