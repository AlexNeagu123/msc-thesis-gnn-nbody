"""Package an explicit code snapshot and train/validation data, never held-out files."""

import argparse
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

from training.fixed_validation import file_sha256


def build_bundle(output: Path, *, root: Path | None = None) -> dict:
    """Publish a no-overwrite, hashed Colab bundle including uncommitted source files."""
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    if output.exists():
        raise FileExistsError(output)
    files = []
    for directory in ("models", "training", "evaluation", "data", "utils"):
        files.extend(
            path for path in (root / directory).rglob("*.py") if "__pycache__" not in path.parts
        )
    files.extend(
        root / name
        for name in (
            "configs/hgnn-pilot-ready.yaml",
            "configs/hgnn-time-scaled.yaml",
            "pyproject.toml",
            "utils.py",
            "data/output/daniela-rk4/train.h5",
            "data/output/daniela-rk4/val.h5",
        )
    )
    hashes = {}
    for path in sorted(files):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"bundle source must be a regular file: {path}")
        hashes[str(path.relative_to(root))] = file_sha256(path)
    if not hashes or any("test.h5" in name or ".git/" in name for name in hashes):
        raise ValueError("invalid bundle allowlist")
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    manifest = {
        "protocol": "canonical-colab-bundle-v1",
        "files": hashes,
        "git_commit": result.stdout.strip() if result.returncode == 0 else None,
        "note": "Hashes describe actual working-tree files, including uncommitted changes; train/val only.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, digest in hashes.items():
                archive.write(root / name, name)
                if file_sha256(root / name) != digest:
                    raise RuntimeError(f"bundle input changed: {name}")
            archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2))
        os.link(temporary, output)  # Atomic no-clobber publication on the local filesystem.
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "output": str(output.resolve()),
        "sha256": file_sha256(output),
        "bytes": output.stat().st_size,
        "files": len(hashes),
        "held_out_included": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.output), indent=2))
