"""CPU-only preparation tests, NOT evidence that GPU execution has passed."""

# ruff: noqa: ANN001, ANN201, D103
import ast
import json
import os
import subprocess
import sys
import time
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

from competitors.gnode_pignn_gpu.bundle import FILES, build_bundle, extract_bundle, sha
from competitors.gnode_pignn_gpu.checks import compare_adam_slots, compare_update, relative_l2
from competitors.gnode_pignn_gpu.colab_support import (
    package_results,
    run_bounded,
    write_entrypoints,
)
from competitors.gnode_pignn_gpu.runtime import configure


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "project"
    for name in FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture only\n")
    proof = tmp_path / "cpu.json"
    proof.write_text(
        json.dumps(
            {
                "all_learning_gates_pass": True,
                "models": {},
                "plan": {
                    "identity": {
                        "sources": {name: sha(root / name) for name in FILES[:-1]},
                        "train_sha256": sha(root / FILES[-1]),
                        "environment": {"test_fixture": True},
                    }
                },
            }
        )
    )
    archive = tmp_path / "bundle.zip"
    info = build_bundle(root, archive, proof)
    return root, archive, info, proof


def rewrite(archive, destination, transform):
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(destination, "x") as target:
        for info in source.infolist():
            data = source.read(info.filename)
            info, data = transform(info, data)
            target.writestr(info, data)


def test_allowlist_safe_extraction_and_no_overwrite(bundle, tmp_path):
    root, archive, info, proof = bundle
    destination = tmp_path / "unpacked"
    manifest = extract_bundle(archive, destination, info["sha256"])
    assert manifest["full_training_authorized"] is False
    assert not any(
        "val.h5" in name or "test.h5" in name or "checkpoints" in name for name in manifest["files"]
    )
    for name in FILES:
        assert (destination / name).read_bytes() == (root / name).read_bytes()
    with pytest.raises(FileExistsError):
        extract_bundle(archive, destination, info["sha256"])
    with pytest.raises(FileExistsError):
        build_bundle(root, archive, proof)
    with pytest.raises(ValueError, match="SHA256"):
        extract_bundle(archive, tmp_path / "bad", "0" * 64)


@pytest.mark.parametrize("extra", ["../escape", "/tmp/escape", "data/output/daniela-rk4/test.h5"])
def test_reject_extra_or_traversal(bundle, tmp_path, extra):
    _, original, _, _ = bundle
    archive = tmp_path / "extra.zip"
    rewrite(original, archive, lambda info, data: (info, data))
    with zipfile.ZipFile(archive, "a") as target:
        target.writestr(extra, b"bad")
    with pytest.raises(ValueError, match="allowlist"):
        extract_bundle(archive, tmp_path / "out", sha(archive))
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "escape").exists()


def test_reject_duplicate(bundle, tmp_path):
    _, archive, _, _ = bundle
    with zipfile.ZipFile(archive, "a") as target, pytest.warns(UserWarning, match="Duplicate"):
        target.writestr(FILES[0], b"duplicate")
    with pytest.raises(ValueError, match="duplicated"):
        extract_bundle(archive, tmp_path / "out", sha(archive))


def test_reject_symlink_and_bad_inner_hash(bundle, tmp_path):
    _, archive, _, _ = bundle
    link = tmp_path / "link.zip"

    def transform(info, data) -> tuple:
        if info.filename == FILES[0]:
            info.create_system = 3
            info.external_attr = 0o120777 << 16
        return info, data

    rewrite(archive, link, transform)
    with pytest.raises(ValueError, match="unsafe"):
        extract_bundle(link, tmp_path / "linked", sha(link))
    changed = tmp_path / "changed.zip"
    rewrite(
        archive,
        changed,
        lambda info, data: (info, b"changed" if info.filename == FILES[0] else data),
    )
    with pytest.raises(ValueError, match="checksum"):
        extract_bundle(changed, tmp_path / "changed", sha(changed))
    assert not (tmp_path / "changed").exists()


def test_cpu_source_and_training_data_identity_pinned(bundle, tmp_path):
    root, _, _, proof = bundle
    (root / FILES[0]).write_text("changed")
    with pytest.raises(ValueError, match="CPU source"):
        build_bundle(root, tmp_path / "new.zip", proof)


def test_update_parity_uses_delta_not_large_weights():
    initial = [np.array([1000.0], dtype=np.float64)]
    left, right = [initial[0] + 1e-6], [initial[0] - 1e-6]
    np.testing.assert_allclose(left, right)  # absolute parameters look deceptively close
    with pytest.raises(ValueError, match="Adam update"):
        compare_update(left, right, initial)
    assert compare_update(left, left, initial)["update_relative_l2"] == 0
    with pytest.raises(ValueError, match="finite"):
        relative_l2([np.array([np.nan])], [np.zeros(1)])


@pytest.mark.parametrize("bad_index", [2, 3])
def test_adam_slots_not_hidden_by_large_iteration_counter(bad_index):
    def variable(name, value) -> object:
        array = np.array([value], dtype=np.float64)
        return types.SimpleNamespace(name=name, numpy=array.copy)

    original = [
        variable("iteration", 1000000),
        variable("learning_rate", 0.0005),
        variable("weight_momentum", 1e-7),
        variable("weight_velocity", 1e-14),
    ]
    assert compare_adam_slots(original, original, 1)["adam_momentum_relative_l2"] == 0
    wrong = list(original)
    wrong[bad_index] = variable(original[bad_index].name, 2 * original[bad_index].numpy()[0])
    with pytest.raises(ValueError, match="Adam"):
        compare_adam_slots(original, wrong, 1)


def test_no_gpu_rejected_before_updates(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "tensorflow",
        types.SimpleNamespace(config=types.SimpleNamespace(list_physical_devices=lambda _: [])),
    )
    for name, value in {
        "KERAS_BACKEND": "tensorflow",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TF_ENABLE_ONEDNN_OPTS": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TF_CPP_MIN_LOG_LEVEL": "2",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="CPU fallback is forbidden"):
        configure()


def test_colab_entrypoints_are_valid_pinned_python(tmp_path):
    result = write_entrypoints(tmp_path, "a" * 64)
    ast.parse(Path(result["script"]).read_text())
    notebook = json.loads(Path(result["notebook"]).read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            source = "".join(cell["source"])
            assert "a" * 64 in source
            ast.parse(source)
    with pytest.raises(FileExistsError):
        write_entrypoints(tmp_path, "a" * 64)


def test_bounded_child_group_is_terminated(tmp_path):
    marker = tmp_path / "must-not-appear"
    child = f"import time;from pathlib import Path;time.sleep(1);Path({str(marker)!r}).write_text('bad')"
    parent = f"import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',{child!r}]);time.sleep(60)"
    with pytest.raises(subprocess.TimeoutExpired):
        run_bounded(
            [sys.executable, "-c", parent],
            cwd=tmp_path,
            log=tmp_path / "timeout.log",
            environment=os.environ.copy(),
            timeout=0.2,
        )
    time.sleep(1.1)
    assert not marker.exists()


def test_result_zip_contains_only_work_directory(tmp_path):
    work = tmp_path / "results"
    work.mkdir()
    (work / "failure.log").write_text("failure evidence")
    outside = tmp_path / "do-not-package.h5"
    outside.write_text("private")
    (work / "link").symlink_to(outside)
    archive = package_results(work)
    with zipfile.ZipFile(archive) as result:
        assert result.namelist() == ["failure.log"]
