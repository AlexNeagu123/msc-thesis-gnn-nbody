"""Synthetic interruption/continuation tests; never trains on benchmark data."""

from dataclasses import replace
from pathlib import Path

import h5py
import pytest
import torch

from data.lazy_dataset import LazyTrajectoryWindowDataset
from training._io import load_checkpoint, save_checkpoint
from training._types import TrainConfig
from training.test_canonical_hgnn import canonical_cfg as canonical_cfg
from training.train import Trainer


def resume_config(cfg: TrainConfig, *, reset: bool = False) -> TrainConfig:
    """Use short rounds, noise, LR transitions and a scheduler to exercise all state."""
    return replace(
        cfg,
        training=replace(
            cfg.training,
            curriculum_horizons=[1, 2, 2],
            curriculum_epochs=[2, 1, 1],
            curriculum_lrs=[0.001, 0.0005, 0.0002],
            curriculum_gradient_clip_norms=[1.0, 0.5, 0.25],
            max_train_batches_per_epoch=2,
            noise_factor=0.01,
            reset_optimizer_on_stage=reset,
            dynamic_loss_scaling="elapsed_time_v1",
        ),
        scheduler=replace(cfg.scheduler, enabled=True, patience=0),
    )


def equal_tree(a: object, b: object) -> None:
    """Assert bitwise tensor equality throughout an optimizer/model tree."""
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for first, second in zip(a, b, strict=True):
            equal_tree(first, second)
    else:
        assert a == b


@pytest.mark.parametrize("rounds", [1, 2, 3])
@pytest.mark.parametrize("reset", [False, True])
def test_interrupted_run_matches_uninterrupted(
    canonical_cfg: TrainConfig, rounds: int, reset: bool
) -> None:
    """Replay matches weights, optimizer, scheduler, histories and sampling across stages."""
    cfg = resume_config(canonical_cfg, reset=reset)
    whole = Trainer(cfg)
    expected = whole.run()
    interrupted = Trainer(cfg)
    interrupted.run(max_rounds=rounds)
    path = interrupted.ckpt_dir / "latest.pt"
    resumed = Trainer(cfg, resume_checkpoint=path)
    assert resumed.run_id == interrupted.run_id
    assert resumed.ckpt_dir != interrupted.ckpt_dir
    actual = resumed.run()
    assert actual == expected
    assert resumed.optimizer_steps == whole.optimizer_steps == 8
    assert resumed.attempted_train_batches == whole.attempted_train_batches
    equal_tree(resumed.model.state_dict(), whole.model.state_dict())
    equal_tree(resumed.optimizer.state_dict(), whole.optimizer.state_dict())
    equal_tree(resumed.scheduler.state_dict(), whole.scheduler.state_dict())
    equal_tree(resumed._train_generator.get_state(), whole._train_generator.get_state())
    best_a = load_checkpoint(whole.ckpt_dir / "best.pt", torch.device("cpu"))
    best_b = load_checkpoint(resumed.ckpt_dir / "best.pt", torch.device("cpu"))
    assert best_a.selected_score == best_b.selected_score
    assert best_a.epoch == best_b.epoch
    equal_tree(best_a.model, best_b.model)


def test_legacy_repeated_run_keeps_original_behavior(canonical_cfg: TrainConfig) -> None:
    """Canonical continuation must not silently alter legacy repeated-run semantics."""
    cfg = replace(canonical_cfg, model=replace(canonical_cfg.model, name="hgnn"))
    trainer = Trainer(cfg)
    first = trainer.run()
    steps = trainer.optimizer_steps
    second = trainer.run()
    assert len(first.train_history) == len(second.train_history) == 1
    assert trainer.optimizer_steps == 2 * steps


def test_uncapped_shuffle_resume(canonical_cfg: TrainConfig) -> None:
    """Default full-pass shuffled loaders also replay their global RNG stream."""
    cfg = replace(
        canonical_cfg, training=replace(canonical_cfg.training, epochs=2, noise_factor=0.02)
    )
    whole = Trainer(cfg)
    expected = whole.run()
    first = Trainer(cfg)
    first.run(max_rounds=1)
    restored = Trainer(cfg, resume_checkpoint=first.ckpt_dir / "latest.pt")
    assert restored.run() == expected
    equal_tree(restored.model.state_dict(), whole.model.state_dict())
    equal_tree(restored.optimizer.state_dict(), whole.optimizer.state_dict())


def test_resume_does_not_refit_scales(
    canonical_cfg: TrainConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No normalization inputs property may be accessed while restoring saved scales."""
    cfg = resume_config(canonical_cfg)
    first = Trainer(cfg)
    first.run(max_rounds=1)

    def forbidden(self: LazyTrajectoryWindowDataset) -> None:
        raise AssertionError("resume attempted normalization refit")

    monkeypatch.setattr(LazyTrajectoryWindowDataset, "inputs", property(forbidden))
    resumed = Trainer(cfg, resume_checkpoint=first.ckpt_dir / "latest.pt")
    assert resumed.pos_std == first.pos_std
    assert resumed.vel_std == first.vel_std
    assert resumed.momentum_std == first.momentum_std


def test_resume_rejects_recipe_data_and_inference_weights(canonical_cfg: TrainConfig) -> None:
    """A resume is neither a changed experiment nor a weights-only warm start."""
    cfg = resume_config(canonical_cfg)
    first = Trainer(cfg)
    first.run(max_rounds=1)
    path = first.ckpt_dir / "latest.pt"
    with pytest.raises(ValueError, match="recipe"):
        Trainer(replace(cfg, training=replace(cfg.training, lr=0.123)), resume_checkpoint=path)
    with pytest.raises(ValueError, match="cannot be combined"):
        Trainer(cfg, resume_checkpoint=path, init_checkpoint=path)
    with pytest.raises(ValueError, match="full round-boundary"):
        Trainer(cfg, resume_checkpoint=first.ckpt_dir / "best.pt")
    with h5py.File(cfg.data.train_path, "r+") as handle:
        handle["trajectories"][0, 0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="data hashes"):
        Trainer(cfg, resume_checkpoint=path)


def test_resume_runtime_change_is_explicit(canonical_cfg: TrainConfig, tmp_path: Path) -> None:
    """Non-bitwise runtime portability needs an explicit override and recorded lineage."""
    cfg = resume_config(canonical_cfg)
    first = Trainer(cfg)
    first.run(max_rounds=1)
    checkpoint = load_checkpoint(first.ckpt_dir / "latest.pt", torch.device("cpu"))
    checkpoint.resume_state["environment"]["torch"] = "different runtime"
    path = tmp_path / "changed.pt"
    save_checkpoint(path, checkpoint)
    with pytest.raises(ValueError, match="runtime changed"):
        Trainer(cfg, resume_checkpoint=path)
    resumed = Trainer(cfg, resume_checkpoint=path, allow_runtime_change=True)
    assert resumed._resume_lineage[-1]["runtime_changed"] is True


def test_fixed_validation_resume_matches(canonical_cfg: TrainConfig) -> None:
    """Rebuilt fixed reference/baseline caches must not change replay or sampling."""
    cfg = resume_config(canonical_cfg)
    cfg = replace(
        cfg,
        training=replace(
            cfg.training, checkpoint_metric="fixed_position_mae", validation_rollout_steps=3
        ),
    )
    whole = Trainer(cfg)
    expected = whole.run()
    first = Trainer(cfg)
    first.run(max_rounds=2)
    restored = Trainer(cfg, resume_checkpoint=first.ckpt_dir / "latest.pt")
    assert restored.run() == expected
    equal_tree(restored.model.state_dict(), whole.model.state_dict())
    assert [r["model"] for r in restored.validation_history] == [
        r["model"] for r in whole.validation_history
    ]


def test_old_best_survives_worse_continuation(
    canonical_cfg: TrainConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The best model from before interruption is carried, not replaced by latest weights."""
    from training._types import EpochRunSummary

    original = Trainer._run_epoch

    def worsening(self: Trainer, *, training: bool, verbose: bool) -> EpochRunSummary:
        if training:
            return original(self, training=training, verbose=verbose)
        return EpochRunSummary(loss=float(self.optimizer_steps))

    monkeypatch.setattr(Trainer, "_run_epoch", worsening)
    cfg = resume_config(canonical_cfg)
    first = Trainer(cfg)
    first.run(max_rounds=1)
    best = load_checkpoint(first.ckpt_dir / "best.pt", torch.device("cpu"))
    restored = Trainer(cfg, resume_checkpoint=first.ckpt_dir / "latest.pt")
    restored.run()
    after = load_checkpoint(restored.ckpt_dir / "best.pt", torch.device("cpu"))
    assert best.epoch == after.epoch == 1
    equal_tree(best.model, after.model)


def test_atomic_checkpoint_failure_preserves_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted write must not replace a complete checkpoint with partial bytes."""
    path = tmp_path / "latest.pt"
    path.write_bytes(b"previous complete artifact")

    def fail(value: object, handle: object) -> None:
        handle.write(b"partial")
        raise OSError("simulated failed write")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="simulated"):
        save_checkpoint(path, object())
    assert path.read_bytes() == b"previous complete artifact"
    assert list(tmp_path.iterdir()) == [path]
