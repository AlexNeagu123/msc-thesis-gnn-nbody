"""Colab setup in a pip-less venv, bounded subprocesses, result packaging.

Only runs when the user executes the prepared Colab entry point. No system package
or GPU driver installation, no Drive mount, no global pip downgrade/replacement.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import venv
import zipfile
from pathlib import Path


def run_bounded(
    command: list[str], *, cwd: Path, log: Path, environment: dict, timeout: float
) -> None:
    """Enforce walltime and terminate the whole child group on timeout/interruption."""
    with log.open("x") as handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
    if code:
        raise RuntimeError(f"subprocess failed with exit{code}; inspect {log}")


def execute(root: Path, work: Path) -> Path:
    """Install isolated candidate runtime and launch at most the bounded preflight."""
    if sys.version_info[:2] not in ((3, 12), (3, 13)):
        raise RuntimeError("Use a Python3.12/3.13 Colab runtime; do not alter system Python.")
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("Select a GPU Colab runtime before installing dependencies.")
    if shutil.disk_usage(root).free < 12 * 1024**3:
        raise RuntimeError("Need at least12GiB free for isolated GPU wheels and results.")
    work.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(
        {
            "KERAS_BACKEND": "tensorflow",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TF_ENABLE_ONEDNN_OPTS": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "42",
        }
    )
    runtime = root.parent / (root.name + "-venv")
    if runtime.exists():
        raise FileExistsError(runtime)
    # Avoid Colab's sometimes absent ensurepip. Outer pip --python targets only
    # this new pip-less venv; it does not install TF into the notebook interpreter.
    venv.EnvBuilder(with_pip=False).create(runtime)
    python = runtime / "bin/python"
    print(
        "Installing an isolated TF2.21/Keras3.13.2 GPU environment; this may take several minutes.",
        flush=True,
    )
    run_bounded(
        [
            sys.executable,
            "-m",
            "pip",
            "--python",
            str(python),
            "install",
            "--index-url",
            "https://pypi.org/simple",
            "pip",
            "tensorflow[and-cuda]==2.21.0",
            "keras==3.13.2",
            "numpy==2.5.4",
            "h5py==3.14.0",
        ],
        cwd=root,
        log=work / "installation.log",
        environment=environment,
        timeout=1800,
    )
    run_bounded(
        [str(python), "-m", "pip", "check"],
        cwd=root,
        log=work / "pip-check.log",
        environment=environment,
        timeout=60,
    )
    with (work / "environment-freeze.txt").open("x") as handle:
        subprocess.run(
            [str(python), "-m", "pip", "freeze", "--all"],
            env=environment,
            stdout=handle,
            check=True,
            timeout=60,
        )
    site = subprocess.check_output(
        [str(python), "-c", "import site;print(site.getsitepackages()[0])"], text=True, timeout=15
    ).strip()
    libraries = sorted(str(p) for p in (Path(site) / "nvidia").glob("*/lib") if p.is_dir())
    existing = [part for part in environment.get("LD_LIBRARY_PATH", "").split(":") if part]
    environment["LD_LIBRARY_PATH"] = ":".join([*libraries, *existing])
    with (work / "launch-policy.json").open("x") as handle:
        json.dump(
            {
                "source_root": str(root),
                "python": str(python),
                "install_timeout_seconds": 1800,
                "probe_timeout_seconds": 600,
                "gpu_memory_limit_mib": 4096,
                "library_policy": "isolated venv NVIDIA wheel libraries first; system driver unchanged",
                "LD_LIBRARY_PATH": environment["LD_LIBRARY_PATH"],
                "full_training_authorized": False,
            },
            handle,
            indent=2,
        )
    print("Starting bounded GPU checks. Maximum99 GPU updates; no full training.", flush=True)
    run_bounded(
        [
            str(python),
            "-m",
            "competitors.gnode_pignn_gpu.preflight",
            "--run-disposable-gpu-probes",
            "--output",
            str(work / "probe"),
        ],
        cwd=root,
        log=work / "probe.log",
        environment=environment,
        timeout=600,
    )
    return work / "probe/summary.json"


def package_results(work: Path) -> Path:
    """Package diagnostics even after failure; never include data or the venv."""
    output = work.with_suffix(".zip")
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        if work.exists():
            for path in sorted(work.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    archive.write(path, path.relative_to(work))
    return output


def driver_source(bundle_sha256: str) -> str:
    """Generate a standalone Colab cell/script pinned to exactly one local ZIP."""
    if len(bundle_sha256) != 64 or any(c not in "0123456789abcdef" for c in bundle_sha256):
        raise ValueError("expected lowercase SHA256")
    return (
        '''# Run only in a fresh Colab GPU runtime (Python3.12/3.13).
# Upload the prepared TRAIN-only ZIP. No full training or Drive mount.
import hashlib
import json
import sys
import traceback
import uuid
import zipfile
from pathlib import Path

EXPECTED_SHA256 = "'''
        + bundle_sha256
        + """"

def main():
    from google.colab import files
    print("Upload daniela_tf_gpu_preflight_bundle.zip only.")
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise RuntimeError("Upload exactly one bundle ZIP.")
    contents = next(iter(uploaded.values()))
    if hashlib.sha256(contents).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("Bundle checksum mismatch; nothing executed.")
    tag = uuid.uuid4().hex[:12]
    archive = Path("/content") / ("daniela-gpu-" + tag + ".zip")
    archive.write_bytes(contents)
    root = Path("/content") / ("daniela-gpu-" + tag)
    # Execute only our pinned bootstrap module, after whole-archive SHA verification.
    with zipfile.ZipFile(archive) as z:
        bootstrap = z.read("competitors/gnode_pignn_gpu/bundle.py")
    namespace = {"__name__": "verified_bundle_bootstrap"}
    exec(compile(bootstrap, "verified_bundle_bootstrap", "exec"), namespace)
    namespace["extract_bundle"](archive, root, EXPECTED_SHA256)
    sys.path.insert(0, str(root))
    from competitors.gnode_pignn_gpu.colab_support import execute, package_results
    work = root.parent / (root.name + "-results")
    try:
        summary_path = execute(root, work)
        summary = json.loads(summary_path.read_text())
        for name, row in summary["models"].items():
            print(name, "seconds/update512:", row["host_feed_median_seconds"],
                  "projected compute hours:", row["projected100epochs_plus25rollouts_hours"])
        print("GPU PREFLIGHT PASSED. Full training has NOT started.")
    except Exception:
        work.mkdir(parents=True, exist_ok=True)
        (work / "COLAB_FAILURE.txt").write_text(traceback.format_exc())
        raise
    finally:
        result_zip = package_results(work)
        print("Results/diagnostics:", result_zip)
        files.download(str(result_zip))

if __name__ == "__main__":
    main()
"""
    )


def write_entrypoints(directory: Path, bundle_sha256: str) -> dict:
    """Write matching notebook and Kate-friendly Python file without overwrites."""
    source = driver_source(bundle_sha256)
    script = directory / "daniela_gnode_pignn_gpu_preflight.py"
    notebook = directory / "daniela_gnode_pignn_gpu_preflight.ipynb"
    if script.exists() or notebook.exists():
        raise FileExistsError("Colab entry points already exist")
    script.write_text(source)
    cells = [
        {
            "cell_type": "markdown",
            "id": "instructions",
            "metadata": {},
            "source": [
                "# Daniela GNODE / PI-GNN: bounded GPU preflight\n",
                "Select **Runtime → Change runtime type → GPU** in a fresh Python3.12/3.13 runtime.\n",
                "Run the cell, upload `daniela_tf_gpu_preflight_bundle.zip`, then keep the downloaded result ZIP.\n",
                "TRAIN only. At most99 disposable GPU updates +9 CPU reference updates.4GiB GPU cap.\n",
                "Isolated venv; installation may take minutes. Probe timeout10min. No full training/Drive mount.\n",
                "If a check fails, return the diagnostics ZIP; do not loosen tolerances or change recipes.\n",
            ],
        },
        {
            "cell_type": "code",
            "id": "run-gpu-preflight",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": source.splitlines(keepends=True),
        },
    ]
    with notebook.open("x") as handle:
        json.dump(
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {
                    "accelerator": "GPU",
                    "kernelspec": {"name": "python3", "display_name": "Python3"},
                    "language_info": {"name": "python"},
                },
                "cells": cells,
            },
            handle,
            indent=2,
        )
    return {"script": str(script), "notebook": str(notebook)}
