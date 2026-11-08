"""Minimal TRAIN-only GPU bundle and no-overwrite, hash-checked safe extraction."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

FILES = (
    "competitors/__init__.py",
    "competitors/gnode_pignn_tf/__init__.py",
    "competitors/gnode_pignn_tf/models.py",
    "competitors/gnode_pignn_training/__init__.py",
    "competitors/gnode_pignn_training/recipes.py",
    "competitors/gnode_pignn_training/engine.py",
    "competitors/gnode_pignn_training/data.py",
    "competitors/gnode_pignn_training/checkpoint.py",
    "competitors/gnode_pignn_gpu/__init__.py",
    "competitors/gnode_pignn_gpu/runtime.py",
    "competitors/gnode_pignn_gpu/checks.py",
    "competitors/gnode_pignn_gpu/preflight.py",
    "competitors/gnode_pignn_gpu/bundle.py",
    "competitors/gnode_pignn_gpu/colab_support.py",
    "data/output/daniela-rk4/train.h5",
)
EVIDENCE = "cpu_preflight_evidence.json"
PROTOCOL = "daniela-tf-gpu-trainonly-bundle-v1"


def sha(path: Path) -> str:
    """Stream SHA256 without unnecessarily duplicating the data array in memory."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_bundle(root: Path, output: Path, cpu_summary: Path) -> dict:
    """Capture explicit working-tree sources, TRAIN and verified CPU evidence only."""
    if output.exists():
        raise FileExistsError(output)
    previous = json.loads(cpu_summary.read_text())
    if not previous["all_learning_gates_pass"]:
        raise ValueError("CPU learning gates have not passed")
    identity = previous["plan"]["identity"]
    for name in FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"regular source file required: {name}")
        if name in identity["sources"] and sha(path) != identity["sources"][name]:
            raise ValueError(f"validated CPU source has changed: {name}")
    if sha(root / FILES[-1]) != identity["train_sha256"]:
        raise ValueError("shared TRAIN identity mismatch")
    evidence = json.dumps(
        {
            "cpu_environment": identity["environment"],
            "validated_source_hashes": identity["sources"],
            "train_sha256": identity["train_sha256"],
            "cpu_recipe_observations": previous["models"],
            "cpu_learning_gates_passed": True,
        },
        sort_keys=True,
        indent=2,
    ).encode()
    records = {
        name: {"sha256": sha(root / name), "bytes": (root / name).stat().st_size} for name in FILES
    }
    records[EVIDENCE] = {"sha256": hashlib.sha256(evidence).hexdigest(), "bytes": len(evidence)}
    git = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    manifest = {
        "protocol": PROTOCOL,
        "files": records,
        "full_training_authorized": False,
        "contents": "TRAIN only; no VAL, existing100, checkpoints, credentials or original notebooks",
        "git_head_context_only": git.stdout.strip(),
        "working_tree_hashes_authoritative": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in FILES:
                archive.write(root / name, name)
                if sha(root / name) != records[name]["sha256"]:
                    raise RuntimeError("bundle source changed during capture")
            archive.writestr(EVIDENCE, evidence)
            archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2))
        os.link(temporary, output)  # no-clobber publication
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "output": str(output),
        "sha256": sha(output),
        "bytes": output.stat().st_size,
        "files": len(records),
        "train_only": True,
        "full_training_authorized": False,
    }


def extract_bundle(archive_path: Path, destination: Path, expected_sha256: str) -> dict:
    """Reject traversal, links, duplicates, extra files, bad hashes and overwrites."""
    if sha(archive_path) != expected_sha256:
        raise ValueError("bundle SHA256 mismatch")
    if destination.exists():
        raise FileExistsError(destination)
    allowed = {*FILES, EVIDENCE, "bundle_manifest.json"}
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        names = [info.filename for info in entries]
        if len(names) != len(set(names)) or set(names) != allowed:
            raise ValueError("bundle names are duplicated, missing or outside allowlist")
        if sum(i.file_size for i in entries) > 150 * 1024**2:
            raise ValueError("bundle exceeds preflight size limit")
        for info in entries:
            path = PurePosixPath(info.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in info.filename
                or stat.S_ISLNK(info.external_attr >> 16)
            ):
                raise ValueError("unsafe bundle entry")
        manifest = json.loads(archive.read("bundle_manifest.json"))
        if manifest["protocol"] != PROTOCOL or manifest["full_training_authorized"] is not False:
            raise ValueError("unexpected bundle protocol/permission")
        if set(manifest["files"]) != allowed - {"bundle_manifest.json"}:
            raise ValueError("manifest allowlist mismatch")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".gpu-unpack-") as temp:
            payload = Path(temp) / "payload"
            payload.mkdir()
            for name in names:
                data = archive.read(name)
                if name != "bundle_manifest.json":
                    record = manifest["files"][name]
                    if (
                        len(data) != record["bytes"]
                        or hashlib.sha256(data).hexdigest() != record["sha256"]
                    ):
                        raise ValueError(f"bundle content checksum mismatch: {name}")
                path = payload / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            os.rename(payload, destination)
    return manifest
