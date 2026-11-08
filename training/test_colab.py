"""Synthetic Colab backup, packaging, profiling and optional CUDA preflight tests."""

import copy
import json
import os
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from training._types import TrainConfig
from training.colab_bundle import build_bundle
from training.colab_runtime import backup_round, benchmark_training, restore_backup, verify_backup
from training.test_canonical_hgnn import canonical_cfg as canonical_cfg
from training.test_resume import equal_tree, resume_config
from training.train import Trainer


def test_drive_backup_commit_and_corruption(canonical_cfg: TrainConfig, tmp_path: Path) -> None:
    """Only a complete hash-verified snapshot can restore to a new local directory."""
    trainer = Trainer(resume_config(canonical_cfg))
    trainer.run(max_rounds=1)
    snapshot = backup_round(trainer.ckpt_dir, tmp_path / "drive")
    assert verify_backup(snapshot)["epoch"] == 1
    path = restore_backup(snapshot, tmp_path / "restored")
    assert path.read_bytes() == (trainer.ckpt_dir / "latest.pt").read_bytes()
    with pytest.raises(FileExistsError):
        restore_backup(snapshot, tmp_path / "restored")
    (snapshot / "latest.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        restore_backup(snapshot, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()
    incomplete = tmp_path / "partial-upload"
    incomplete.mkdir()
    with pytest.raises(FileNotFoundError):
        verify_backup(incomplete)


def test_probe_has_no_optimizer_updates(
    canonical_cfg: TrainConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Training-only timing must leave parameters unchanged and never call step."""

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("probe updated parameters")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    result = benchmark_training(
        canonical_cfg, device="cpu", horizons=(1, 2), batch_size=3, repeats=1
    )
    assert result["optimizer_steps"] == 0
    assert result["parameters_unchanged"]
    assert all(row["status"] == "ok" and row["median_seconds"] > 0 for row in result["profiles"])


def test_bundle_allowlist_and_no_overwrite(tmp_path: Path) -> None:
    """Explicit snapshot includes uncommitted code but never held-out files or credentials."""
    root = tmp_path / "project"
    allowed = [
        "models/local.py",
        "training/local.py",
        "utils.py",
        "pyproject.toml",
        "configs/hgnn-pilot-ready.yaml",
        "configs/hgnn-time-scaled.yaml",
        "data/output/daniela-rk4/train.h5",
        "data/output/daniela-rk4/val.h5",
    ]
    forbidden = ["auth.json", ".git/config", ".ssh/id_ed25519", "data/output/daniela-rk4/test.h5"]
    for name in allowed + forbidden:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    output = tmp_path / "bundle.zip"
    result = build_bundle(output, root=root)
    assert result["held_out_included"] is False
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("bundle_manifest.json"))
        assert set(manifest["files"]) == set(allowed)
        assert set(archive.namelist()) == set(allowed) | {"bundle_manifest.json"}
    with pytest.raises(FileExistsError):
        build_bundle(output, root=root)
    (root / "models/link.py").symlink_to(root / "auth.json")
    with pytest.raises(ValueError, match="regular file"):
        build_bundle(tmp_path / "unsafe.zip", root=root)


def test_notebook_code_cells_compile() -> None:
    """Validate notebook structure and Python syntax without pretending to run Colab."""
    path = Path(__file__).resolve().parents[1] / "colab/hgnn_resumable.ipynb"
    if not path.exists():
        pytest.skip("notebook not included in a minimal runtime bundle")
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = cell["source"]
            compile(source if isinstance(source, str) else "".join(source), f"cell-{index}", "exec")
            assert cell["outputs"] == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
def test_cuda_output_and_parameter_gradient_parity(canonical_cfg: TrainConfig) -> None:
    """Compare higher-order float32 autograd outputs and gradients on CPU versus GPU."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    trainer = Trainer(canonical_cfg)
    cpu = trainer.model
    gpu = copy.deepcopy(cpu).cuda()
    inputs = trainer.train_loader.dataset.states[:2, 0]
    target = trainer.train_loader.dataset.states[:2, 1]
    outputs = []
    for model, device in ((cpu, "cpu"), (gpu, "cuda")):
        state = inputs.to(device)
        for _ in range(3):
            state = model(state)
        loss = (state[..., :4] - target.to(device)[..., :4]).square().mean()
        loss.backward()
        outputs.append(state.detach().cpu())
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-4, atol=1e-6)
    for first, second in zip(cpu.parameters(), gpu.parameters(), strict=True):
        if first.grad is None:
            assert second.grad is None
        else:
            torch.testing.assert_close(first.grad, second.grad.cpu(), rtol=2e-3, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
def test_cuda_resume_replay(canonical_cfg: TrainConfig) -> None:
    """Test actual interrupted/uninterrupted CUDA continuation when GPU is available."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        cfg = resume_config(
            replace(canonical_cfg, training=replace(canonical_cfg.training, device="cuda"))
        )
        whole = Trainer(cfg)
        whole.run()
        first = Trainer(cfg)
        first.run(max_rounds=1)
        restored = Trainer(cfg, resume_checkpoint=first.ckpt_dir / "latest.pt")
        restored.run()
        equal_tree(restored.model.state_dict(), whole.model.state_dict())
        equal_tree(restored.optimizer.state_dict(), whole.optimizer.state_dict())
    finally:
        torch.use_deterministic_algorithms(previous)
