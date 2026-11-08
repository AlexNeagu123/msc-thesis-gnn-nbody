"""Synthetic-only parity, failure and exact CPU-adapter continuation checks."""

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
import yaml

from evaluation._loader import load_trained_model
from training._io import load_checkpoint
from training._types import TrainConfig
from training.cpu_microbatch import CpuMicrobatchTrainer
from training.test_canonical_hgnn import canonical_cfg as canonical_cfg
from training.test_resume import equal_tree, resume_config
from training.train import Trainer


@pytest.mark.parametrize("horizon", [1, 3])
@pytest.mark.parametrize("scaling", ["none", "elapsed_time_v1"])
def test_weighted_microbatch_gradient_parity(
    canonical_cfg: TrainConfig, horizon: int, scaling: str
) -> None:
    """Odd batch5 splits2+2+1; compare loss/gradient/one clipped AdamW update."""
    cfg = replace(
        canonical_cfg,
        training=replace(
            canonical_cfg.training,
            batch_size=5,
            max_train_batches_per_epoch=1,
            multi_step_horizon=horizon,
            multi_step_gamma=0.6,
            dynamic_loss_scaling=scaling,
        ),
    )
    full = Trainer(cfg)
    micro = CpuMicrobatchTrainer(cfg, micro_batch_size=2)
    inputs, targets = next(iter(full.train_loader))
    first, loss, _ = full._compute_loss(inputs, targets)
    loss.backward()
    accumulated_loss, predictions = micro._backward_logical_batch(inputs, targets)
    assert accumulated_loss == pytest.approx(float(loss.detach()), rel=2e-6, abs=1e-6)
    torch.testing.assert_close(predictions, first.detach(), rtol=2e-5, atol=1e-6)
    for a, b in zip(full.model.parameters(), micro.model.parameters(), strict=True):
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, rtol=5e-4, atol=2e-6)
    for trainer in (full, micro):
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 0.01)
        trainer.optimizer.step()
    for a, b in zip(full.model.parameters(), micro.model.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)


def test_clip_step_noise_and_diagnostics_once_per_logical_batch(
    canonical_cfg: TrainConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not silently turn effective batch5 into three optimizer/noise/clip events."""
    cfg = replace(
        canonical_cfg,
        training=replace(
            canonical_cfg.training, batch_size=5, max_train_batches_per_epoch=2, noise_factor=0.01
        ),
    )
    trainer = CpuMicrobatchTrainer(cfg, micro_batch_size=2)
    original_step, original_clip, original_noise = (
        trainer.optimizer.step,
        torch.nn.utils.clip_grad_norm_,
        trainer.apply_noise,
    )
    calls = {"step": 0, "clip": 0, "noise": 0, "diagnostic": 0}

    def step() -> None:
        calls["step"] += 1
        return original_step()

    def clip(*args: object, **kwargs: object) -> torch.Tensor:
        calls["clip"] += 1
        return original_clip(*args, **kwargs)

    def noise(inputs: torch.Tensor) -> torch.Tensor:
        calls["noise"] += 1
        assert len(inputs) == 5
        return original_noise(inputs)

    def diagnostic(
        inputs: torch.Tensor, targets: torch.Tensor, preds: torch.Tensor, *args: object
    ) -> None:
        calls["diagnostic"] += 1
        assert len(inputs) == len(targets) == len(preds) == 5
        assert not preds.requires_grad

    monkeypatch.setattr(trainer.optimizer, "step", step)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    monkeypatch.setattr(trainer, "apply_noise", noise)
    monkeypatch.setattr(trainer.diagnostics, "check_batch", diagnostic)
    summary = trainer._run_epoch(training=True, verbose=False)
    assert summary.skipped_batches == 0
    assert trainer.optimizer_steps == trainer.attempted_train_batches == 2
    assert calls == {"step": 2, "clip": 2, "noise": 2, "diagnostic": 2}


def test_bad_later_microbatch_discards_entire_logical_update(
    canonical_cfg: TrainConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finite first microbatch must never sneak in a partial optimizer update."""
    cfg = replace(
        canonical_cfg,
        training=replace(
            canonical_cfg.training,
            batch_size=5,
            max_train_batches_per_epoch=1,
            skip_nonfinite_batches=True,
        ),
    )
    trainer = CpuMicrobatchTrainer(cfg, micro_batch_size=2)
    before = copy.deepcopy(trainer.model.state_dict())
    original = trainer._compute_loss
    calls = 0

    def bad(
        inputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal calls
        calls += 1
        predictions, loss, diag = original(inputs, targets)
        return predictions, loss * float("nan") if calls == 2 else loss, diag

    monkeypatch.setattr(trainer, "_compute_loss", bad)
    summary = trainer._run_epoch(training=True, verbose=False)
    assert summary.skipped_batches == 1
    assert trainer.optimizer_steps == 0
    assert trainer.attempted_train_batches == 1
    assert all(p.grad is None for p in trainer.model.parameters())
    equal_tree(before, trainer.model.state_dict())


def test_nonfinite_accumulated_gradient_skips(
    canonical_cfg: TrainConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Always reject nonfinite gradients even if all microbatch losses are finite."""
    cfg = replace(
        canonical_cfg, training=replace(canonical_cfg.training, max_train_batches_per_epoch=1)
    )
    trainer = CpuMicrobatchTrainer(cfg, micro_batch_size=2)
    before = copy.deepcopy(trainer.model.state_dict())

    def bad(inputs: torch.Tensor, targets: torch.Tensor) -> tuple[float, torch.Tensor]:
        parameter = next(trainer.model.parameters())
        parameter.grad = torch.full_like(parameter, float("inf"))
        return 0.5, inputs.detach()

    monkeypatch.setattr(trainer, "_backward_logical_batch", bad)
    summary = trainer._run_epoch(training=True, verbose=False)
    assert summary.skipped_batches == 1
    assert trainer.optimizer_steps == 0
    assert all(p.grad is None for p in trainer.model.parameters())
    equal_tree(before, trainer.model.state_dict())


@pytest.mark.parametrize("rounds", [1, 2, 3])
@pytest.mark.parametrize("reset", [False, True])
def test_microbatch_exact_resume(canonical_cfg: TrainConfig, rounds: int, reset: bool) -> None:
    """Noise/sampler/scheduler/stages/reset histories and tensors replay bitwise."""
    cfg = resume_config(canonical_cfg, reset=reset)
    whole = CpuMicrobatchTrainer(cfg, micro_batch_size=2)
    expected = whole.run()
    first = CpuMicrobatchTrainer(cfg, micro_batch_size=2)
    first.run(max_rounds=rounds)
    restored = CpuMicrobatchTrainer(
        cfg, micro_batch_size=2, resume_checkpoint=first.ckpt_dir / "latest.pt"
    )
    assert restored.run() == expected
    assert restored.optimizer_steps == 8
    equal_tree(restored.model.state_dict(), whole.model.state_dict())
    equal_tree(restored.optimizer.state_dict(), whole.optimizer.state_dict())
    equal_tree(restored.scheduler.state_dict(), whole.scheduler.state_dict())
    equal_tree(restored._train_generator.get_state(), whole._train_generator.get_state())
    assert "training/cpu_microbatch.py" in restored.source_hashes


def test_cpu_execution_identity_rejects_changed_partition_or_ordinary_resume(
    canonical_cfg: TrainConfig,
) -> None:
    """Reject undocumented execution changes; ordinary/GPU checkpoints are not CPU resumes."""
    first = CpuMicrobatchTrainer(canonical_cfg, micro_batch_size=2)
    first.run()
    path = first.ckpt_dir / "latest.pt"
    with pytest.raises(ValueError, match="execution identity"):
        CpuMicrobatchTrainer(canonical_cfg, micro_batch_size=3, resume_checkpoint=path)
    with pytest.raises(ValueError, match="source hashes"):
        Trainer(canonical_cfg, resume_checkpoint=path)
    base = Trainer(canonical_cfg)
    base.run()
    with pytest.raises(ValueError, match="not silently converted"):
        CpuMicrobatchTrainer(canonical_cfg, resume_checkpoint=base.ckpt_dir / "latest.pt")


def test_fixed_validation_resume_and_inference(canonical_cfg: TrainConfig, tmp_path: Path) -> None:
    """Fixed reference caches replay, and selected CPU checkpoints load for inference."""
    cfg = resume_config(canonical_cfg)
    cfg = replace(
        cfg,
        training=replace(
            cfg.training, checkpoint_metric="fixed_position_mae", validation_rollout_steps=3
        ),
    )
    whole = CpuMicrobatchTrainer(cfg, micro_batch_size=2)
    expected = whole.run()
    first = CpuMicrobatchTrainer(cfg, micro_batch_size=2)
    first.run(max_rounds=2)
    restored = CpuMicrobatchTrainer(
        cfg, micro_batch_size=2, resume_checkpoint=first.ckpt_dir / "latest.pt"
    )
    assert restored.run() == expected
    equal_tree(restored.model.state_dict(), whole.model.state_dict())
    config_path = tmp_path / "inference.yaml"
    config_path.write_text(yaml.safe_dump(asdict(cfg)))
    loaded = load_trained_model(config_path, restored.ckpt_dir / "best.pt", torch.device("cpu"))
    saved = load_checkpoint(restored.ckpt_dir / "best.pt", torch.device("cpu"))
    equal_tree(loaded.model.state_dict(), saved.model)


def test_cpu_entry_pause_resume_and_completed_noop(
    canonical_cfg: TrainConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual local backup/status entry loop pauses, resumes and never repeats completion."""
    from training import run_cpu_rk4 as run_cpu_daniela

    work = tmp_path / "cpu-entry"
    monkeypatch.setattr(run_cpu_daniela, "RUN_DIR", work)
    cfg = replace(
        canonical_cfg,
        training=replace(
            canonical_cfg.training,
            curriculum_horizons=[1, 2],
            curriculum_epochs=[1, 1],
            max_train_batches_per_epoch=1,
            checkpoint_metric="fixed_position_mae",
            validation_rollout_steps=3,
        ),
        checkpointing=replace(canonical_cfg.checkpointing, dir=str(work / "segments")),
        logging=replace(canonical_cfg.logging, enabled=True, dir=str(work / "segments")),
    )
    run_cpu_daniela.train(cfg, micro_batch_size=2, rounds=1)
    assert json.loads((work / "status.json").read_text())["state"] == "paused_at_boundary"
    run_cpu_daniela.train(cfg, micro_batch_size=2, rounds=None)
    status = json.loads((work / "status.json").read_text())
    assert status["state"] == "complete"
    assert status["optimizer_steps"] == 2
    assert len(list((work / "backups").glob("*/*/COMMITTED.json"))) == 2
    run_cpu_daniela.train(cfg, micro_batch_size=2, rounds=None)
    assert len(list((work / "backups").glob("*/*/COMMITTED.json"))) == 2


def test_adapter_rejects_legacy_and_invalid_size(canonical_cfg: TrainConfig) -> None:
    """Original HGNN path and unsupported execution options remain untouched."""
    with pytest.raises(ValueError, match="canonical HGNN"):
        CpuMicrobatchTrainer(
            replace(canonical_cfg, model=replace(canonical_cfg.model, name="hgnn"))
        )
    with pytest.raises(ValueError, match="positive integer"):
        CpuMicrobatchTrainer(canonical_cfg, micro_batch_size=0)
