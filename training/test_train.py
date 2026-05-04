"""Tests for training/train.py."""

import math
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from training._types import (
    Checkpoint,
    CheckpointConfig,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    RolloutScore,
    SchedulerConfig,
    TrainConfig,
    TrainingParams,
    TrainResult,
)
from training.train import Trainer, apply_artifact_dir, load_config, train


class DummyModel(nn.Module):
    """Minimal model that maps (batch, 3, 5) -> (batch, 3, 5)."""

    def __init__(self) -> None:
        """Initialize with a single linear layer."""
        super().__init__()
        self.net = nn.Linear(5, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, applied per-particle."""
        return self.net(x)


@pytest.fixture
def sample_h5(tmp_path: Path) -> tuple[str, str]:
    """Create small train and val HDF5 files."""
    rng = np.random.default_rng(42)

    for name in ("train.h5", "val.h5"):
        trajectories = rng.normal(size=(5, 10, 3, 5))
        energies = rng.normal(size=(5, 10))
        path = tmp_path / name
        with h5py.File(path, "w") as f:
            f.create_dataset("trajectories", data=trajectories)
            f.create_dataset("energies", data=energies)

    return str(tmp_path / "train.h5"), str(tmp_path / "val.h5")


@pytest.fixture
def make_cfg(sample_h5: tuple[str, str], tmp_path: Path) -> TrainConfig:
    """Create a minimal TrainConfig pointing to test data."""
    train_path, val_path = sample_h5
    return TrainConfig(
        model=ModelConfig(name="dummy", hidden_dim=4, n_layers=1),
        data=DataConfig(train_path=train_path, val_path=val_path, dt=0.05),
        training=TrainingParams(
            epochs=3,
            batch_size=8,
            lr=1e-3,
            weight_decay=0.0,
            loss="mse",
            seed=42,
            device="cpu",
        ),
        scheduler=SchedulerConfig(enabled=False),
        checkpointing=CheckpointConfig(enabled=True, dir=str(tmp_path / "ckpt")),
        logging=LoggingConfig(enabled=True, dir=str(tmp_path / "logs")),
    )


def test_train_returns_result(make_cfg: TrainConfig) -> None:
    """Train function returns a TrainResult with expected fields."""
    result = train(make_cfg, model=DummyModel())

    assert isinstance(result, TrainResult)
    assert result.best_epoch >= 1
    assert result.best_val_loss < float("inf")
    assert result.final_train_loss < float("inf")


def test_train_histories_length(make_cfg: TrainConfig) -> None:
    """History lists have one entry per epoch."""
    result = train(make_cfg, model=DummyModel())

    assert len(result.train_history) == make_cfg.training.epochs
    assert len(result.val_history) == make_cfg.training.epochs


def _find_run_dir(base: str) -> Path:
    """Find the single run subdirectory inside a base directory."""
    subdirs = sorted(Path(base).iterdir())
    assert len(subdirs) == 1
    return subdirs[0]


def test_checkpoint_saved(make_cfg: TrainConfig) -> None:
    """Best and latest checkpoints are written to disk."""
    train(make_cfg, model=DummyModel())

    run_dir = _find_run_dir(make_cfg.checkpointing.dir)
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "latest.pt").exists()


def test_checkpoint_contents(make_cfg: TrainConfig) -> None:
    """Checkpoint contains expected fields."""
    train(make_cfg, model=DummyModel())

    run_dir = _find_run_dir(make_cfg.checkpointing.dir)
    ckpt = torch.load(run_dir / "best.pt", weights_only=False)

    assert isinstance(ckpt, Checkpoint)
    assert isinstance(ckpt.epoch, int)
    assert isinstance(ckpt.val_loss, float)
    assert isinstance(ckpt.model, dict)
    assert isinstance(ckpt.optimizer, dict)
    assert ckpt.config is not None
    assert ckpt.model_name == make_cfg.model.name
    assert ckpt.run_id == run_dir.name
    assert isinstance(ckpt.pos_std, float)
    assert isinstance(ckpt.vel_std, float)


def test_csv_log_written(make_cfg: TrainConfig) -> None:
    """CSV metrics file is created with header and one row per epoch."""
    train(make_cfg, model=DummyModel())

    run_dir = _find_run_dir(make_cfg.logging.dir)
    csv_path = run_dir / "metrics.csv"
    assert csv_path.exists()

    lines = csv_path.read_text().strip().split("\n")
    assert lines[0] == (
        "epoch,train_loss,val_loss,lr,"
        "rollout_score,dominance_horizon,fraction_beating_baseline,final_ratio"
    )
    assert len(lines) == make_cfg.training.epochs + 1


def test_loss_decreases(make_cfg: TrainConfig) -> None:
    """Training loss should generally decrease over a few epochs."""
    result = train(make_cfg, model=DummyModel())

    # not strictly monotonic, but first should be larger than last
    assert result.train_history[0] > result.train_history[-1]


def test_mae_loss(make_cfg: TrainConfig) -> None:
    """Training works with MAE loss."""
    cfg = TrainConfig(
        model=make_cfg.model,
        data=make_cfg.data,
        training=TrainingParams(
            epochs=2,
            batch_size=8,
            lr=1e-3,
            weight_decay=0.0,
            loss="mae",
            seed=42,
            device="cpu",
        ),
        scheduler=make_cfg.scheduler,
        checkpointing=CheckpointConfig(enabled=False),
        logging=LoggingConfig(enabled=False),
    )
    result = train(cfg, model=DummyModel())

    assert isinstance(result, TrainResult)
    assert result.final_train_loss < float("inf")


def test_no_checkpointing(make_cfg: TrainConfig, tmp_path: Path) -> None:
    """Training runs fine with checkpointing disabled."""
    cfg = TrainConfig(
        model=make_cfg.model,
        data=make_cfg.data,
        training=make_cfg.training,
        scheduler=make_cfg.scheduler,
        checkpointing=CheckpointConfig(enabled=False),
        logging=LoggingConfig(enabled=False),
    )
    result = train(cfg, model=DummyModel())

    assert isinstance(result, TrainResult)


def test_scheduler_enabled(make_cfg: TrainConfig) -> None:
    """Training runs fine with scheduler enabled."""
    cfg = TrainConfig(
        model=make_cfg.model,
        data=make_cfg.data,
        training=make_cfg.training,
        scheduler=SchedulerConfig(enabled=True, patience=1, factor=0.5, min_lr=1e-6),
        checkpointing=CheckpointConfig(enabled=False),
        logging=LoggingConfig(enabled=False),
    )
    result = train(cfg, model=DummyModel())

    assert isinstance(result, TrainResult)


def test_noise_injection_modifies_pos_vel_only(make_cfg: TrainConfig) -> None:
    """Noise is applied to positions and velocities but not mass."""
    cfg = TrainConfig(
        model=make_cfg.model,
        data=make_cfg.data,
        training=TrainingParams(
            epochs=1,
            batch_size=8,
            lr=1e-3,
            weight_decay=0.0,
            loss="mse",
            noise_factor=0.05,
            seed=42,
            device="cpu",
        ),
        scheduler=SchedulerConfig(enabled=False),
        checkpointing=CheckpointConfig(enabled=False),
        logging=LoggingConfig(enabled=False),
    )
    trainer = Trainer(cfg, model=DummyModel())

    inputs, _targets = next(iter(trainer.train_loader))
    original_mass = inputs[..., 4:].clone()
    original_pos = inputs[..., :2].clone()

    noisy = trainer.apply_noise(inputs)

    # mass column unchanged
    assert torch.equal(noisy[..., 4:], original_mass)
    # position and velocity columns changed
    assert not torch.equal(noisy[..., :2], original_pos)
    assert not torch.equal(noisy[..., 2:4], inputs[..., 2:4])


def test_noise_injection_runs(make_cfg: TrainConfig) -> None:
    """Training completes with noise injection enabled."""
    cfg = TrainConfig(
        model=make_cfg.model,
        data=make_cfg.data,
        training=TrainingParams(
            epochs=2,
            batch_size=8,
            lr=1e-3,
            weight_decay=0.0,
            loss="mse",
            noise_factor=0.05,
            seed=42,
            device="cpu",
        ),
        scheduler=SchedulerConfig(enabled=False),
        checkpointing=CheckpointConfig(enabled=False),
        logging=LoggingConfig(enabled=False),
    )
    result = train(cfg, model=DummyModel())

    assert isinstance(result, TrainResult)
    assert result.final_train_loss < float("inf")


def test_load_config(tmp_path: Path) -> None:
    """Load_config parses a YAML file into a TrainConfig."""
    yaml_content = """
model:
  name: egnn
  hidden_dim: 64
  n_layers: 4
data:
  train_path: train.h5
  val_path: val.h5
  dt: 0.05
training:
  epochs: 10
  batch_size: 32
  lr: 0.001
  weight_decay: 0.00001
"""
    config_path = tmp_path / "test.yaml"
    config_path.write_text(yaml_content)

    cfg = load_config(str(config_path))

    assert isinstance(cfg, TrainConfig)
    assert cfg.model.name == "egnn"
    assert cfg.training.epochs == 10
    assert cfg.scheduler.enabled is False  # default
    assert cfg.checkpointing.enabled is False  # default


def test_multi_step_defaults_when_missing(tmp_path: Path) -> None:
    """A YAML without multi_step_* fields defaults to one-step training."""
    yaml_content = """
model:
  name: egnn
  hidden_dim: 64
  n_layers: 4
data:
  train_path: train.h5
  val_path: val.h5
  dt: 0.05
training:
  epochs: 1
  batch_size: 8
  lr: 0.001
  weight_decay: 0.0
"""
    path = tmp_path / "no_multistep.yaml"
    path.write_text(yaml_content)

    cfg = load_config(str(path))
    assert cfg.training.multi_step_horizon == 1
    assert cfg.training.multi_step_gamma == 1.0


def test_multi_step_fields_parsed_when_present(tmp_path: Path) -> None:
    """Explicit multi_step_horizon and multi_step_gamma load into TrainingParams."""
    yaml_content = """
model:
  name: egnn
  hidden_dim: 64
  n_layers: 4
data:
  train_path: train.h5
  val_path: val.h5
  dt: 0.05
training:
  epochs: 1
  batch_size: 8
  lr: 0.001
  weight_decay: 0.0
  multi_step_horizon: 5
  multi_step_gamma: 0.9
"""
    path = tmp_path / "multistep.yaml"
    path.write_text(yaml_content)

    cfg = load_config(str(path))
    assert cfg.training.multi_step_horizon == 5
    assert cfg.training.multi_step_gamma == pytest.approx(0.9)


def test_trainer_uses_nbody_dataset_for_horizon_one(make_cfg: TrainConfig) -> None:
    """horizon=1 keeps the existing one-step NBodyDataset path."""
    from data.dataset import NBodyDataset

    assert make_cfg.training.multi_step_horizon == 1
    trainer = Trainer(make_cfg, model=DummyModel())

    assert isinstance(trainer.train_loader.dataset, NBodyDataset)
    assert isinstance(trainer.val_loader.dataset, NBodyDataset)


def test_trainer_uses_window_dataset_for_horizon_above_one(make_cfg: TrainConfig) -> None:
    """horizon>1 swaps in TrajectoryWindowDataset on both train and val loaders."""
    from data.dataset import TrajectoryWindowDataset

    cfg = replace(
        make_cfg,
        training=replace(make_cfg.training, multi_step_horizon=3),
    )

    trainer = Trainer(cfg, model=DummyModel())

    train_set = trainer.train_loader.dataset
    val_set = trainer.val_loader.dataset
    assert isinstance(train_set, TrajectoryWindowDataset)
    assert isinstance(val_set, TrajectoryWindowDataset)
    assert train_set.horizon == 3
    assert val_set.horizon == 3


def test_trainer_allows_hgnn_with_multi_step_horizon(make_cfg: TrainConfig) -> None:
    """HGNN with horizon>1 constructs cleanly; curriculum mode owns the schedule."""
    cfg = replace(
        make_cfg,
        model=replace(make_cfg.model, name="hgnn"),
        training=replace(make_cfg.training, multi_step_horizon=4),
    )

    trainer = Trainer(cfg, model=DummyModel())

    assert trainer.current_horizon == 4


class _CountingModel(nn.Module):
    """DummyModel variant that counts forward calls per instance."""

    def __init__(self) -> None:
        """Initialize with a single linear layer and a counter."""
        super().__init__()
        self.net = nn.Linear(5, 5)
        self.call_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Count and forward."""
        self.call_count += 1
        return self.net(x)


class _IdentityModel(nn.Module):
    """Model that returns inputs unchanged; useful for fixed-loss assertions."""

    def __init__(self) -> None:
        """Hold a single unused parameter so the optimizer accepts the model."""
        super().__init__()
        self._unused = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return input unchanged."""
        return x


def test_multi_step_trains_end_to_end(make_cfg: TrainConfig) -> None:
    """horizon=3 multi-step path completes a full training run with finite loss."""
    cfg = replace(
        make_cfg,
        training=replace(make_cfg.training, multi_step_horizon=3),
    )

    result = train(cfg, model=DummyModel())

    assert isinstance(result, TrainResult)
    assert len(result.train_history) == cfg.training.epochs
    assert result.final_train_loss < float("inf")
    assert all(np.isfinite(x) for x in result.train_history)
    assert all(np.isfinite(x) for x in result.val_history)


def test_multi_step_calls_model_horizon_times_per_batch(make_cfg: TrainConfig) -> None:
    """Each batch in multi-step mode triggers `horizon` forward passes."""
    horizon = 3
    cfg = replace(
        make_cfg,
        training=replace(make_cfg.training, multi_step_horizon=horizon, epochs=1),
    )
    model = _CountingModel()

    trainer = Trainer(cfg, model=model)
    n_train = len(trainer.train_loader)
    n_val = len(trainer.val_loader)

    trainer.run()

    expected = horizon * (n_train + n_val)
    assert model.call_count == expected


def test_multi_step_gradients_flow_through_all_unrolls(make_cfg: TrainConfig) -> None:
    """One epoch in multi-step mode leaves non-zero gradients on every parameter."""
    cfg = replace(
        make_cfg,
        training=replace(make_cfg.training, multi_step_horizon=3, epochs=1),
    )
    model = DummyModel()

    train(cfg, model=model)

    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert param.grad.abs().sum() > 0, f"zero gradient for {name}"


def test_multi_step_loss_is_weighted_mean(make_cfg: TrainConfig) -> None:
    """Loss equals sum_k gamma^k * MSE_k / sum_k gamma^k on real batch data."""
    horizon = 3
    gamma = 0.5
    cfg = replace(
        make_cfg,
        training=replace(
            make_cfg.training,
            multi_step_horizon=horizon,
            multi_step_gamma=gamma,
            noise_factor=0.0,
            epochs=1,
        ),
    )

    trainer = Trainer(cfg, model=_IdentityModel())
    inputs, targets = next(iter(trainer.train_loader))
    inputs = inputs.to(trainer.device)
    targets = targets.to(trainer.device)

    _, loss = trainer._multi_step_rollout_loss(inputs, targets)

    loss_fn = nn.MSELoss()
    weighted_sum = sum(
        (gamma**k) * loss_fn(inputs[..., :4], targets[:, k, ..., :4]) for k in range(horizon)
    )
    weight_sum = sum(gamma**k for k in range(horizon))
    expected = weighted_sum / weight_sum
    assert torch.allclose(loss, expected, atol=1e-6)


def test_multi_step_loss_matches_one_step_scale_at_gamma_one(make_cfg: TrainConfig) -> None:
    """gamma=1 reduces the multi-step loss to the per-step MSE mean."""
    horizon = 4
    cfg = replace(
        make_cfg,
        training=replace(
            make_cfg.training,
            multi_step_horizon=horizon,
            multi_step_gamma=1.0,
            noise_factor=0.0,
            epochs=1,
        ),
    )

    trainer = Trainer(cfg, model=_IdentityModel())
    inputs, targets = next(iter(trainer.train_loader))
    inputs = inputs.to(trainer.device)
    targets = targets.to(trainer.device)

    _, loss = trainer._multi_step_rollout_loss(inputs, targets)

    loss_fn = nn.MSELoss()
    per_step = [loss_fn(inputs[..., :4], targets[:, k, ..., :4]) for k in range(horizon)]
    expected_mean = torch.stack(per_step).mean()
    assert torch.allclose(loss, expected_mean, atol=1e-6)


def test_multi_step_noise_applied_once_per_batch(make_cfg: TrainConfig) -> None:
    """apply_noise fires exactly once per batch even when horizon > 1."""
    horizon = 3
    cfg = replace(
        make_cfg,
        training=replace(
            make_cfg.training,
            multi_step_horizon=horizon,
            noise_factor=0.05,
            epochs=1,
        ),
    )

    trainer = Trainer(cfg, model=DummyModel())

    call_count = 0
    original = trainer.apply_noise

    def spy(inputs: torch.Tensor) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        return original(inputs)

    trainer.apply_noise = spy
    n_train_batches = len(trainer.train_loader)

    trainer.run()

    # noise is only applied during training, not validation
    assert call_count == n_train_batches


def test_multi_step_diagnostics_locate_window_sample(make_cfg: TrainConfig) -> None:
    """Diagnostics can locate first-step targets from TrajectoryWindowDataset."""
    cfg = replace(
        make_cfg,
        training=replace(make_cfg.training, multi_step_horizon=3),
    )
    trainer = Trainer(cfg, model=DummyModel())
    dataset = trainer.train_loader.dataset

    _inputs, targets = dataset[0]
    location = trainer.diagnostics._locate_sample(targets[0])

    assert location == "trajectory 0, step 0"


def _save_dummy_checkpoint(
    path: Path,
    *,
    state_dict: dict[str, torch.Tensor],
    model_name: str | None,
) -> None:
    """Write a Checkpoint dataclass to disk for init-checkpoint tests."""
    ckpt = Checkpoint(
        epoch=42,
        model=state_dict,
        optimizer={"state": {"momentum_buffer": torch.ones(5)}},
        val_loss=0.123,
        model_name=model_name,
        run_id="prior_run",
        pos_std=1.0,
        vel_std=1.0,
    )
    torch.save(ckpt, path)


def test_init_checkpoint_loads_model_weights(make_cfg: TrainConfig, tmp_path: Path) -> None:
    """Init checkpoint replaces fresh model weights with the saved ones."""
    saved_model = DummyModel()
    for p in saved_model.parameters():
        p.data.fill_(7.0)
    ckpt_path = tmp_path / "init.pt"
    _save_dummy_checkpoint(
        ckpt_path,
        state_dict=saved_model.state_dict(),
        model_name=make_cfg.model.name,
    )

    trainer = Trainer(make_cfg, model=DummyModel(), init_checkpoint=ckpt_path)

    for name, param in trainer.model.named_parameters():
        assert torch.equal(param.detach().cpu(), saved_model.state_dict()[name]), (
            f"mismatch on {name}"
        )


def test_init_checkpoint_optimizer_is_fresh(make_cfg: TrainConfig, tmp_path: Path) -> None:
    """Trainer never inherits the saved optimizer state when init_checkpoint is set."""
    saved_model = DummyModel()
    ckpt_path = tmp_path / "init.pt"
    _save_dummy_checkpoint(
        ckpt_path,
        state_dict=saved_model.state_dict(),
        model_name=make_cfg.model.name,
    )

    trainer = Trainer(make_cfg, model=DummyModel(), init_checkpoint=ckpt_path)

    # AdamW starts with empty per-parameter state until step() runs once
    assert len(trainer.optimizer.state) == 0


def test_init_checkpoint_rejects_model_mismatch(make_cfg: TrainConfig, tmp_path: Path) -> None:
    """A checkpoint with a different model_name raises before training starts."""
    ckpt_path = tmp_path / "init.pt"
    _save_dummy_checkpoint(
        ckpt_path,
        state_dict=DummyModel().state_dict(),
        model_name="hgnn",
    )

    with pytest.raises(ValueError, match="trained for model 'hgnn'"):
        Trainer(make_cfg, model=DummyModel(), init_checkpoint=ckpt_path)


def test_init_checkpoint_accepts_legacy_no_model_name(
    make_cfg: TrainConfig, tmp_path: Path
) -> None:
    """A legacy checkpoint with model_name=None loads without error."""
    ckpt_path = tmp_path / "init.pt"
    _save_dummy_checkpoint(
        ckpt_path,
        state_dict=DummyModel().state_dict(),
        model_name=None,
    )

    trainer = Trainer(make_cfg, model=DummyModel(), init_checkpoint=ckpt_path)
    assert trainer.model is not None


# --- curriculum schema validation ---


def _params(**overrides: object) -> TrainingParams:
    """Helper for `TrainingParams` calls that need the required fields filled."""
    base: dict[str, object] = {"batch_size": 8, "lr": 1e-3, "weight_decay": 0.0}
    base.update(overrides)
    return TrainingParams(**base)  # type: ignore[arg-type]


def test_training_params_rejects_zero_epochs_without_curriculum() -> None:
    """Single-horizon mode requires epochs > 0; otherwise the dataclass raises."""
    with pytest.raises(ValueError, match="epochs must be > 0 in single-horizon"):
        _params()


def test_training_params_accepts_curriculum_without_epochs() -> None:
    """Curriculum mode does not require epochs since the schedule supplies the count."""
    params = _params(curriculum_horizons=[1, 5], curriculum_epochs=[3, 2])
    assert params.curriculum_horizons == [1, 5]
    assert params.curriculum_epochs == [3, 2]


def test_training_params_rejects_partial_curriculum() -> None:
    """Setting only one of the two curriculum fields is ambiguous and rejected."""
    with pytest.raises(ValueError, match="must both be set or both be None"):
        _params(epochs=10, curriculum_horizons=[1, 5])

    with pytest.raises(ValueError, match="must both be set or both be None"):
        _params(epochs=10, curriculum_epochs=[3, 2])


def test_training_params_rejects_mismatched_curriculum_lengths() -> None:
    """Mismatched horizon and epoch lists describe an inconsistent schedule."""
    with pytest.raises(ValueError, match="must have the same length"):
        _params(curriculum_horizons=[1, 5, 10], curriculum_epochs=[3, 2])


def test_training_params_rejects_empty_curriculum() -> None:
    """A zero-stage schedule has nothing to run."""
    with pytest.raises(ValueError, match="must contain at least one stage"):
        _params(curriculum_horizons=[], curriculum_epochs=[])


def test_training_params_rejects_non_positive_horizons() -> None:
    """Every horizon must be at least 1."""
    with pytest.raises(ValueError, match="curriculum_horizons entry must be >= 1"):
        _params(curriculum_horizons=[1, 0, 5], curriculum_epochs=[3, 2, 1])


def test_training_params_rejects_non_positive_epochs() -> None:
    """Every stage must train for at least one epoch."""
    with pytest.raises(ValueError, match="curriculum_epochs entry must be >= 1"):
        _params(curriculum_horizons=[1, 5], curriculum_epochs=[3, 0])


# --- checkpoint_metric and rollout-score wiring ---


class _ScriptedRolloutEvaluator:
    """Stand-in for RolloutScoreEvaluator that returns scripted scores per call.

    Used to assert that the trainer's best-checkpoint logic compares
    `rollout.score` instead of `val_loss` without paying the cost of an
    actual rollout on validation trajectories.
    """

    def __init__(self, scores: list[float]) -> None:
        """Hold the queue of scores; one is popped per `score(...)` call."""
        self._scores = list(scores)

    def score(self, model: nn.Module) -> RolloutScore:
        """Return the next scripted score, ignoring the model entirely."""
        s = self._scores.pop(0)
        return RolloutScore(
            score=s,
            ratios=np.array([s]),
            dominance_horizon=0,
            fraction_beating_baseline=0.0,
            final_ratio=s,
            ratios_at_step={},
        )


def test_checkpoint_metric_default_is_val_loss(make_cfg: TrainConfig) -> None:
    """When unspecified, rollout_evaluator is not constructed."""
    trainer = Trainer(make_cfg, model=DummyModel())

    assert make_cfg.training.checkpoint_metric == "val_loss"
    assert trainer.rollout_evaluator is None


def test_checkpoint_metric_rollout_score_builds_evaluator(make_cfg: TrainConfig) -> None:
    """Setting checkpoint_metric='rollout_score' constructs a real evaluator."""
    cfg = replace(
        make_cfg,
        training=replace(make_cfg.training, checkpoint_metric="rollout_score"),
    )
    trainer = Trainer(cfg, model=DummyModel())

    assert trainer.rollout_evaluator is not None


def test_checkpoint_metric_rejects_unknown(make_cfg: TrainConfig) -> None:
    """Misspelled or unsupported values raise before any setup runs."""
    cfg = replace(
        make_cfg,
        training=replace(make_cfg.training, checkpoint_metric="rollout"),
    )
    with pytest.raises(ValueError, match="checkpoint_metric must be"):
        Trainer(cfg, model=DummyModel())


def test_csv_columns_blank_for_val_loss_metric(make_cfg: TrainConfig) -> None:
    """In the default mode, rollout columns appear in the header but rows are empty."""
    train(make_cfg, model=DummyModel())

    run_dir = _find_run_dir(make_cfg.logging.dir)
    lines = (run_dir / "metrics.csv").read_text().strip().split("\n")

    # one header + one row per epoch; trailing four columns blank in every data row
    for row in lines[1:]:
        cells = row.split(",")
        assert cells[-4:] == ["", "", "", ""]


def test_csv_columns_populated_for_rollout_score_metric(
    make_cfg: TrainConfig, tmp_path: Path
) -> None:
    """Rollout diagnostics land in the CSV when checkpoint_metric is rollout_score."""
    cfg = replace(
        make_cfg,
        training=replace(make_cfg.training, checkpoint_metric="rollout_score"),
    )
    trainer = Trainer(cfg, model=DummyModel())
    trainer.rollout_evaluator = _ScriptedRolloutEvaluator(scores=[0.7, 0.5, 0.6])
    trainer.run()

    run_dir = _find_run_dir(cfg.logging.dir)
    lines = (run_dir / "metrics.csv").read_text().strip().split("\n")

    # one header + 3 data rows; rollout_score column (index 4) is populated each row
    rollout_scores = [row.split(",")[4] for row in lines[1:]]
    assert rollout_scores == ["0.700000", "0.500000", "0.600000"]


def test_best_checkpoint_selected_by_rollout_score(make_cfg: TrainConfig) -> None:
    """Best.pt is chosen by rollout_score, not val_loss.

    Scripted scores put the win at epoch 2 regardless of how val_loss orders epochs.
    """
    cfg = replace(
        make_cfg,
        training=replace(make_cfg.training, checkpoint_metric="rollout_score"),
    )
    trainer = Trainer(cfg, model=DummyModel())
    # epoch 2 has the lowest scripted rollout_score, regardless of val_loss order
    trainer.rollout_evaluator = _ScriptedRolloutEvaluator(scores=[0.7, 0.1, 0.5])
    trainer.run()

    run_dir = _find_run_dir(cfg.checkpointing.dir)
    best_ckpt = torch.load(run_dir / "best.pt", weights_only=False)

    assert best_ckpt.epoch == 2
    assert best_ckpt.selected_metric == "rollout_score"
    assert best_ckpt.selected_score == pytest.approx(0.1)
    assert best_ckpt.rollout_score == pytest.approx(0.1)


def test_scheduler_steps_on_selected_score_in_rollout_mode(make_cfg: TrainConfig) -> None:
    """LR scheduler observes rollout_score, not val_loss, when that drives best.pt.

    Otherwise LR adaptation and best-checkpoint selection would react to
    different signals.
    """
    cfg = replace(
        make_cfg,
        training=replace(make_cfg.training, checkpoint_metric="rollout_score"),
        scheduler=replace(make_cfg.scheduler, enabled=True, patience=0),
    )
    trainer = Trainer(cfg, model=DummyModel())
    scripted = [0.7, 0.5, 0.6]
    trainer.rollout_evaluator = _ScriptedRolloutEvaluator(scores=list(scripted))

    seen: list[float] = []
    real_step = trainer.scheduler.step

    def _spy(value: float, *args: object, **kwargs: object) -> object:
        seen.append(float(value))
        return real_step(value, *args, **kwargs)

    trainer.scheduler.step = _spy  # type: ignore[method-assign]
    trainer.run()

    assert seen == scripted


def test_scheduler_steps_on_val_loss_in_default_mode(make_cfg: TrainConfig) -> None:
    """In val_loss mode the scheduler still receives val_loss (unchanged behavior)."""
    cfg = replace(
        make_cfg,
        scheduler=replace(make_cfg.scheduler, enabled=True, patience=0),
    )
    trainer = Trainer(cfg, model=DummyModel())

    seen: list[float] = []
    real_step = trainer.scheduler.step

    def _spy(value: float, *args: object, **kwargs: object) -> object:
        seen.append(float(value))
        return real_step(value, *args, **kwargs)

    trainer.scheduler.step = _spy  # type: ignore[method-assign]
    result = trainer.run()

    # the scheduler should have seen exactly the per-epoch val_loss values
    assert len(seen) == cfg.training.epochs
    assert seen == pytest.approx(result.val_history, rel=1e-9)


def test_checkpoint_val_loss_field_remains_one_step_validation_loss(
    make_cfg: TrainConfig,
) -> None:
    """Even with rollout_score selection, Checkpoint.val_loss stays as one-step val MSE.

    Locks the contract that downstream evaluation reports surfacing
    `checkpoint_val_loss` continue to mean the validation MSE, not the
    rollout score.
    """
    cfg = replace(
        make_cfg,
        training=replace(make_cfg.training, checkpoint_metric="rollout_score"),
    )
    trainer = Trainer(cfg, model=DummyModel())
    trainer.rollout_evaluator = _ScriptedRolloutEvaluator(scores=[0.7, 0.5, 0.6])
    result = trainer.run()

    run_dir = _find_run_dir(cfg.checkpointing.dir)
    best_ckpt = torch.load(run_dir / "best.pt", weights_only=False)

    # selected_score is the rollout score; val_loss is the actual one-step val MSE
    assert best_ckpt.selected_score == pytest.approx(0.5)
    assert best_ckpt.val_loss != pytest.approx(0.5)
    assert math.isfinite(best_ckpt.val_loss)
    # TrainResult.best_val_loss tracks the lowest one-step val MSE seen
    assert math.isfinite(result.best_val_loss)


# --- curriculum stage runner ---


def _curriculum_cfg(make_cfg: TrainConfig, horizons: list[int], epochs: list[int]) -> TrainConfig:
    """Build a TrainConfig with a curriculum schedule, leaving epochs/horizon unused."""
    return replace(
        make_cfg,
        training=replace(
            make_cfg.training,
            curriculum_horizons=horizons,
            curriculum_epochs=epochs,
        ),
    )


def test_curriculum_runs_through_two_stages(make_cfg: TrainConfig) -> None:
    """Two-stage schedule completes; total epochs equal sum of stage epochs."""
    cfg = _curriculum_cfg(make_cfg, horizons=[1, 3], epochs=[2, 2])

    result = train(cfg, model=DummyModel())

    assert len(result.train_history) == 4
    assert len(result.val_history) == 4
    assert all(np.isfinite(x) for x in result.train_history)


def test_curriculum_rebuilds_loader_when_horizon_changes(make_cfg: TrainConfig) -> None:
    """Loader dataset class switches from one-step to window when horizon advances."""
    from data.dataset import NBodyDataset, TrajectoryWindowDataset

    cfg = _curriculum_cfg(make_cfg, horizons=[1, 3], epochs=[1, 1])
    trainer = Trainer(cfg, model=DummyModel())

    # initial loader is built for horizon=1 (one-step dataset)
    assert isinstance(trainer.train_loader.dataset, NBodyDataset)
    assert trainer.current_horizon == 1

    trainer.run()

    # after running, the trainer is parked on the last stage's horizon and dataset class
    assert trainer.current_horizon == 3
    assert isinstance(trainer.train_loader.dataset, TrajectoryWindowDataset)
    # diagnostics tracks the active dataset, not the stale stage-0 one
    assert trainer.diagnostics.dataset is trainer.train_loader.dataset


def test_curriculum_optimizer_state_persists_across_stages(make_cfg: TrainConfig) -> None:
    """Adam state survives stage transitions; no implicit reset between horizons."""
    cfg = _curriculum_cfg(make_cfg, horizons=[1, 3], epochs=[1, 1])
    trainer = Trainer(cfg, model=DummyModel())
    trainer.run()

    # at least one parameter has accumulated optimizer state (Adam exp_avg / exp_avg_sq)
    assert len(trainer.optimizer.state) > 0
    sample_state = next(iter(trainer.optimizer.state.values()))
    assert "exp_avg" in sample_state
    assert "exp_avg_sq" in sample_state


def test_hgnn_curriculum_smoke(make_cfg: TrainConfig) -> None:
    """HGNN trains through a tiny 1->2 curriculum without the old guard tripping.

    Uses the real HGNN module (not DummyModel) so this exercises the autograd
    path the prior unconditional rejection used to block.
    """
    from models.hgnn import HGNN

    cfg = _curriculum_cfg(make_cfg, horizons=[1, 2], epochs=[1, 1])
    cfg = replace(cfg, model=replace(cfg.model, name="hgnn", hidden_dim=8, n_layers=1))

    model = HGNN(hidden_dim=8, n_layers=1, dt=cfg.data.dt)
    result = train(cfg, model=model)

    assert isinstance(result, TrainResult)
    assert len(result.train_history) == 2
    assert all(np.isfinite(x) for x in result.train_history)


def test_curriculum_best_checkpoint_indexes_globally(make_cfg: TrainConfig) -> None:
    """Best-checkpoint epoch is the global step counter across stages, not per-stage."""
    cfg = _curriculum_cfg(make_cfg, horizons=[1, 3], epochs=[2, 2])
    cfg = replace(
        cfg,
        training=replace(cfg.training, checkpoint_metric="rollout_score"),
    )
    trainer = Trainer(cfg, model=DummyModel())
    # epoch 3 (first epoch of stage 2) wins on rollout_score
    trainer.rollout_evaluator = _ScriptedRolloutEvaluator(scores=[0.7, 0.6, 0.1, 0.5])
    trainer.run()

    run_dir = _find_run_dir(cfg.checkpointing.dir)
    best_ckpt = torch.load(run_dir / "best.pt", weights_only=False)

    assert best_ckpt.epoch == 3
    assert best_ckpt.selected_score == pytest.approx(0.1)


# --- artifact dir override ---


def test_apply_artifact_dir_sets_both_dirs(make_cfg: TrainConfig) -> None:
    """Both checkpointing and logging dirs are pointed at the artifact root."""
    cfg = apply_artifact_dir(make_cfg, "runs/curriculum/egnn/n5000")

    assert cfg.checkpointing.dir == "runs/curriculum/egnn/n5000"
    assert cfg.logging.dir == "runs/curriculum/egnn/n5000"


def test_apply_artifact_dir_force_enables_both(make_cfg: TrainConfig) -> None:
    """Override implies the user wants artifacts persisted; both flags flip on."""
    cfg = replace(
        make_cfg,
        checkpointing=replace(make_cfg.checkpointing, enabled=False),
        logging=replace(make_cfg.logging, enabled=False),
    )
    cfg = apply_artifact_dir(cfg, "runs/single/egnn/n1000")

    assert cfg.checkpointing.enabled is True
    assert cfg.logging.enabled is True


def test_apply_artifact_dir_preserves_other_fields(make_cfg: TrainConfig) -> None:
    """Override only mutates the two dir fields; everything else is untouched."""
    cfg = apply_artifact_dir(make_cfg, "runs/x")

    assert cfg.model == make_cfg.model
    assert cfg.data == make_cfg.data
    assert cfg.training == make_cfg.training
    assert cfg.scheduler == make_cfg.scheduler


def test_artifact_dir_colocates_checkpoints_and_metrics(
    make_cfg: TrainConfig, tmp_path: Path
) -> None:
    """End-to-end: best.pt, latest.pt, and metrics.csv land under the same run_id."""
    artifact_dir = tmp_path / "runs" / "single" / "egnn" / "n5000"
    cfg = apply_artifact_dir(make_cfg, artifact_dir)

    train(cfg, model=DummyModel())

    run_dirs = list(artifact_dir.iterdir())
    assert len(run_dirs) == 1, f"expected exactly one run dir under {artifact_dir}"
    run_dir = run_dirs[0]
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "latest.pt").exists()
    assert (run_dir / "metrics.csv").exists()
