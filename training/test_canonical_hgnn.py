"""Synthetic-only integration tests for the explicitly versioned canonical model."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from data.lazy_dataset import LazyTrajectoryWindowDataset
from evaluation._loader import load_trained_model
from models.hgnn import HGNN
from models.hgnn_canonical import MODEL_ID, CanonicalHGNN
from training._io import load_checkpoint
from training._types import (
    CheckpointConfig,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    SchedulerConfig,
    TrainConfig,
    TrainingParams,
)
from training.overfit_rk4 import run_overfit
from training.train import Trainer, build_model


@pytest.fixture
def canonical_cfg(tmp_path: Path) -> TrainConfig:
    """Create tiny unequal-mass synthetic train/validation data, not benchmark data."""
    rng = np.random.default_rng(41)
    for name, multiplier in (("train", 1), ("val", 100)):
        states = rng.normal(size=(3, 6, 3, 5)) * multiplier
        states[..., 4] = np.array([0.5, 1.3, 2])[None, None, :]
        with h5py.File(tmp_path / f"{name}.h5", "w") as handle:
            handle["trajectories"] = states
    return TrainConfig(
        model=ModelConfig(MODEL_ID, hidden_dim=8, n_layers=2),
        data=DataConfig(str(tmp_path / "train.h5"), str(tmp_path / "val.h5"), dt=0.001),
        training=TrainingParams(
            batch_size=3, lr=1e-3, weight_decay=0, epochs=1, multi_step_horizon=3, device="cpu"
        ),
        scheduler=SchedulerConfig(),
        checkpointing=CheckpointConfig(enabled=True, dir=str(tmp_path / "checkpoints")),
        logging=LoggingConfig(),
    )


def test_factory_requires_explicit_momentum_scale(canonical_cfg: TrainConfig) -> None:
    """Never reuse velocity normalization or an implicit unit momentum scale."""
    with pytest.raises(ValueError, match="explicit training-fitted momentum_std"):
        build_model(canonical_cfg)
    model = build_model(canonical_cfg, pos_std=2.7, vel_std=123, momentum_std=0.4)
    assert isinstance(model, CanonicalHGNN)
    assert model.momentum_std.item() == 0.4


def test_train_only_stats_and_checkpoint_reload(canonical_cfg: TrainConfig, tmp_path: Path) -> None:
    """A synthetic one-epoch run restores exact scales/identity without training data."""
    trainer = Trainer(canonical_cfg)
    inputs = trainer.train_loader.dataset.inputs
    expected = float((inputs[..., 2:4] * inputs[..., 4:]).std())
    assert trainer.momentum_std == expected
    assert trainer.model.momentum_std.item() == expected
    assert trainer.model.pos_std.item() == float(inputs[..., :2].std())
    assert trainer.model.momentum_std.item() != trainer.vel_std
    result = trainer.run()
    assert np.isfinite(result.final_train_loss)
    checkpoint_path = trainer.ckpt_dir / "best.pt"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(asdict(canonical_cfg)))
    sample = inputs[:2].clone()
    with torch.no_grad():
        expected_output = trainer.model(sample)
    # Evaluation must restore saved statistics rather than recompute from files.
    Path(canonical_cfg.data.train_path).unlink()
    loaded = load_trained_model(config_path, checkpoint_path, torch.device("cpu"))
    assert loaded.checkpoint.model_name == MODEL_ID
    assert loaded.model.momentum_std.item() == expected
    with torch.no_grad():
        torch.testing.assert_close(loaded.model(sample), expected_output)


def test_canonical_training_is_from_scratch(canonical_cfg: TrainConfig) -> None:
    """Do not initialize the reciprocal benchmark from arbitrary previous weights."""
    with pytest.raises(ValueError, match="starts from scratch"):
        Trainer(canonical_cfg, init_checkpoint="not-read.pt")


def test_legacy_factory_stays_legacy(canonical_cfg: TrainConfig) -> None:
    """Existing hgnn configs keep their original velocity-coordinate behaviour."""
    canonical_cfg.model.name = "hgnn"
    model = build_model(canonical_cfg, pos_std=2, vel_std=3)
    assert type(model) is HGNN
    assert model.vel_std.item() == 3


def test_missing_stats_and_wrong_dt_are_rejected(
    canonical_cfg: TrainConfig, tmp_path: Path
) -> None:
    """Canonical evaluation cannot silently refit scales or override time semantics."""
    trainer = Trainer(canonical_cfg)
    trainer.run()
    checkpoint_path = trainer.ckpt_dir / "best.pt"
    checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    config_path = tmp_path / "config.yaml"
    config = asdict(canonical_cfg)
    config["data"]["dt"] = 0.05
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="timestep disagrees"):
        load_trained_model(config_path, checkpoint_path, torch.device("cpu"))
    config_path.write_text(yaml.safe_dump(asdict(canonical_cfg)))
    checkpoint.pos_std = None
    torch.save(checkpoint, tmp_path / "missing-stats.pt")
    with pytest.raises(ValueError, match="refitting during evaluation is forbidden"):
        load_trained_model(config_path, tmp_path / "missing-stats.pt", torch.device("cpu"))
    checkpoint.pos_std = trainer.pos_std * 2
    torch.save(checkpoint, tmp_path / "inconsistent-stats.pt")
    with pytest.raises(ValueError, match="position scale disagrees"):
        load_trained_model(config_path, tmp_path / "inconsistent-stats.pt", torch.device("cpu"))
    checkpoint.pos_std = trainer.pos_std
    checkpoint.model["dt"] = torch.tensor(0.001, dtype=torch.float32)
    torch.save(checkpoint, tmp_path / "float32-dt.pt")
    # Float32 representation noise is not a different physical timestep.
    load_trained_model(config_path, tmp_path / "float32-dt.pt", torch.device("cpu"))


def test_lazy_curriculum_fixed_validation_and_budget(canonical_cfg: TrainConfig) -> None:
    """Changing training horizon preserves data storage, normalization and validation starts."""
    canonical_cfg.training = replace(
        canonical_cfg.training,
        curriculum_horizons=[1, 3],
        curriculum_epochs=[1, 1],
        max_train_batches_per_epoch=2,
        checkpoint_metric="fixed_position_mae",
        validation_rollout_steps=5,
        validation_batch_size=2,
    )
    trainer = Trainer(canonical_cfg)
    assert isinstance(trainer.train_loader.dataset, LazyTrajectoryWindowDataset)
    storage = trainer.train_loader.dataset.states
    pos_std = trainer.pos_std
    trainer.run()
    assert trainer.train_loader.dataset.states is storage
    assert trainer.pos_std == pos_std
    assert trainer.optimizer_steps == 4
    assert trainer.attempted_train_batches == 4
    assert len(trainer.validation_history) == 2
    assert [row["training_horizon"] for row in trainer.validation_history] == [1, 3]
    first, second = trainer.validation_history
    assert first["metadata"] == second["metadata"]
    assert first["baselines"] is second["baselines"]
    assert first["metadata"]["steps"] == 5
    assert first["source_sha256"]["models/hgnn_canonical.py"]
    records = [
        json.loads(line)
        for line in (trainer.ckpt_dir / "fixed_validation.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2
    latest = load_checkpoint(trainer.ckpt_dir / "latest.pt", torch.device("cpu"))
    assert latest.selected_metric == "fixed_position_mae"
    assert latest.selected_score == second["model"]["selection_score"]


def test_selection_and_scheduler_use_primary_mae(
    canonical_cfg: TrainConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Better fixed MAE wins even when the diagnostic state-MSE gets worse."""
    canonical_cfg.training = replace(
        canonical_cfg.training,
        epochs=2,
        max_train_batches_per_epoch=1,
        checkpoint_metric="fixed_position_mae",
    )
    canonical_cfg.scheduler.enabled = True
    trainer = Trainer(canonical_cfg)
    reports = iter(
        [
            {"model": {"selection_score": 0.2, "state4_mse": 1.0, "failed_runs": 0}},
            {"model": {"selection_score": 0.1, "state4_mse": 10.0, "failed_runs": 0}},
        ]
    )
    monkeypatch.setattr(trainer.fixed_evaluator, "score", lambda model: next(reports))
    scheduler_values = []
    monkeypatch.setattr(trainer.scheduler, "step", scheduler_values.append)
    result = trainer.run()
    assert result.best_epoch == 2
    assert scheduler_values == [0.2, 0.1]
    best = load_checkpoint(trainer.ckpt_dir / "best.pt", torch.device("cpu"))
    assert best.selected_score == 0.1
    assert best.val_loss == 10.0


def test_one_step_and_one_step_rollout_loss_match(canonical_cfg: TrainConfig) -> None:
    """The canonical one-step objective excludes mass just like multi-step training."""
    canonical_cfg.training = replace(canonical_cfg.training, multi_step_horizon=1)
    trainer = Trainer(canonical_cfg)
    inputs, targets = next(iter(trainer.train_loader))
    predicted, one_step = trainer._one_step_loss(inputs, targets)
    _, rollout = trainer._multi_step_rollout_loss(inputs, targets[:, None])
    torch.testing.assert_close(one_step, rollout)
    torch.testing.assert_close(one_step, (predicted[..., :4] - targets[..., :4]).square().mean())


def test_bounded_sampler_reproducible_and_not_prefix(canonical_cfg: TrainConfig) -> None:
    """The cap samples all training simulations, independently of model/validation RNG."""
    canonical_cfg.training = replace(canonical_cfg.training, max_train_batches_per_epoch=20)
    first = Trainer(canonical_cfg)
    torch.rand(123)
    second = Trainer(canonical_cfg)
    indices = list(first.train_loader.sampler)
    assert indices == list(second.train_loader.sampler)
    assert len(indices) == 20 * canonical_cfg.training.batch_size
    assert {idx // first.train_loader.dataset.windows_per_traj for idx in indices} == {0, 1, 2}
    assert first.run_id != second.run_id


@pytest.mark.parametrize(
    "name", ["max_train_batches_per_epoch", "validation_rollout_steps", "validation_batch_size"]
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_budget_config(canonical_cfg: TrainConfig, name: str, value: int) -> None:
    """Budgets and rollout sizes must be genuine positive integers."""
    with pytest.raises(ValueError, match=name):
        replace(canonical_cfg.training, **{name: value})


class NonfiniteTrainingModel(torch.nn.Module):
    """A deliberate failure fixture: all dynamic predictions fail, masses survive."""

    def __init__(self) -> None:
        """Register a parameter so the trainer can construct its optimizer."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Preserve the state interface while generating NaN dynamic values."""
        return torch.cat([state[..., :4] + self.weight * float("nan"), state[..., 4:]], dim=-1)


def test_skips_are_bounded_and_failed_checkpoint_never_best(canonical_cfg: TrainConfig) -> None:
    """A broken model cannot retry indefinitely or win because finite runs were filtered."""
    canonical_cfg.training = replace(
        canonical_cfg.training,
        max_train_batches_per_epoch=2,
        checkpoint_metric="fixed_position_mae",
    )
    trainer = Trainer(canonical_cfg, model=NonfiniteTrainingModel())
    result = trainer.run()
    assert trainer.attempted_train_batches == 2
    assert trainer.optimizer_steps == 0
    assert result.best_epoch == 0
    assert not (trainer.ckpt_dir / "best.pt").exists()
    assert (trainer.ckpt_dir / "latest.pt").exists()
    record = json.loads((trainer.ckpt_dir / "fixed_validation.jsonl").read_text())
    assert record["model"]["selection_score"] is None
    assert record["model"]["failed_runs"] == 3
    assert record["model"]["eligible"] is False


@pytest.mark.parametrize("horizon", [1, 2])
def test_fixed_overfit_runner_is_train_only(
    canonical_cfg: TrainConfig, tmp_path: Path, horizon: int
) -> None:
    """The manual runner uses frozen train windows and production updates, not validation."""
    Path(canonical_cfg.data.val_path).unlink()
    config_path = tmp_path / "overfit-config.yaml"
    config_path.write_text(yaml.safe_dump(asdict(canonical_cfg)))
    original = Path(canonical_cfg.data.train_path).read_bytes()
    output = tmp_path / "overfit-artifacts"
    report = run_overfit(
        config_path, output, steps=2, horizon=horizon, windows=4, trajectories=2, log_every=1
    )
    assert report["optimizer_steps"] == report["attempted_batches"] == 2
    assert report["metadata"]["validation_or_test_accessed"] is False
    assert report["metadata"]["normalization_trajectories"] == 3
    assert len(report["metadata"]["selected_simulation_ids"]) == 2
    assert len(report["history"]) == 3
    assert set(report["baselines"]) == {"constant_velocity", "persistence"}
    assert Path(canonical_cfg.data.train_path).read_bytes() == original
    assert not Path(canonical_cfg.data.val_path).exists()
    with (
        h5py.File(canonical_cfg.data.train_path) as handle,
        np.load(output / "fixed_windows.npz") as fixed,
    ):
        states = handle["trajectories"][:].astype(np.float32)
        for index, (trajectory, start) in enumerate(
            zip(fixed["trajectory_indices"], fixed["start_frames"], strict=True)
        ):
            np.testing.assert_array_equal(fixed["inputs"][index], states[trajectory, start])
            np.testing.assert_array_equal(
                fixed["targets"][index], states[trajectory, start + 1 : start + 1 + horizon]
            )
    checkpoint = load_checkpoint(output / "overfit.pt", torch.device("cpu"))
    assert checkpoint.selected_metric == "training_subset_state4_mse"
    assert np.isnan(checkpoint.val_loss)
    assert json.loads((output / "summary.json").read_text())["optimizer_steps"] == 2
    with pytest.raises(FileExistsError):
        run_overfit(config_path, output, steps=2, horizon=horizon, windows=4, trajectories=2)


def test_fixed_overfit_seed_reproducibility(canonical_cfg: TrainConfig, tmp_path: Path) -> None:
    """Same training source, seed and settings reproduce windows and final model tensors."""
    config_path = tmp_path / "overfit-config.yaml"
    config_path.write_text(yaml.safe_dump(asdict(canonical_cfg)))
    outputs = [tmp_path / "first", tmp_path / "second"]
    reports = [
        run_overfit(config_path, output, steps=2, horizon=2, windows=4, trajectories=2)
        for output in outputs
    ]
    assert reports[0]["history"] == reports[1]["history"]
    with (
        np.load(outputs[0] / "fixed_windows.npz") as first,
        np.load(outputs[1] / "fixed_windows.npz") as second,
    ):
        for key in first.files:
            np.testing.assert_array_equal(first[key], second[key])
    models = [
        load_checkpoint(output / "overfit.pt", torch.device("cpu")).model for output in outputs
    ]
    for key in models[0]:
        if key == "_extra_state":
            assert models[0][key] == models[1][key]
        else:
            torch.testing.assert_close(models[0][key], models[1][key], rtol=0, atol=0)
