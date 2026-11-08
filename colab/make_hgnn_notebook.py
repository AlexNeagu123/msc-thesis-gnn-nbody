"""Build the new canonical-HGNN notebook without touching existing Colab notebooks."""
import json
from pathlib import Path
from textwrap import dedent

cells = []


def markdown(text):
    cells.append({"cell_type": "markdown", "id": f"cell-{len(cells)}", "metadata": {}, "source": dedent(text).strip()})


def code(text):
    cells.append({"cell_type": "code", "id": f"cell-{len(cells)}", "metadata": {}, "execution_count": None, "outputs": [], "source": dedent(text).strip()})


markdown('''
# Daniela HGNN — checked GPU pilot and resumable development training

**Not final benchmark evaluation.** Only 320 training and 80 validation simulations
are bundled. The held-out 100 and final fresh shared ICs are excluded.

1. Locally build the bundle with `python -m training.colab_bundle --output ~/Downloads/daniela-colab-ready.zip`.
2. Upload that ZIP to your own Drive folder below, and this notebook to Colab.
3. Choose **Runtime → Change runtime type → GPU**. Run setup, correctness and timing cells.
4. Review timings, set `RUN_TRAINING=True`, then run training. No optimizer updates occur before that cell.

Uses float32, learned T/V, no AMP/compile. A GPU is NOT assumed faster for these small graphs.
Resume means continuation from the last **committed ROUND**, not mid-batch recovery.
An interrupted uncommitted round is replayed. Only load your OWN trusted pickle checkpoints.
''')
code('''
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from pathlib import Path
from google.colab import drive
drive.mount("/content/drive")
DRIVE_ROOT = Path("/content/drive/MyDrive/hgnn-daniela")
BUNDLE = DRIVE_ROOT / "daniela-colab-ready.zip"
EXPECTED_BUNDLE_SHA256 = ""  # Paste the local packaging command's SHA256 (recommended).
ROOT = Path("/content/hgnn-daniela")
RUN_TAG = "gpu-pilot-001"  # New tag for a NEW experiment; keep it when resuming.
BACKUP_ROOT = DRIVE_ROOT / "runs" / RUN_TAG
RESUME_SNAPSHOT = ""  # Explicit committed snapshot directory printed by a previous session.
ALLOW_RUNTIME_CHANGE = False  # Different GPU/runtime needs explicit non-bitwise permission.
RUN_TRAINING = False  # Review preflight/timing results before changing this.
ACK_GPU_NOT_FASTER = False
MAX_ROUNDS_THIS_SESSION = None  # Optional positive cap; full default plan is 15 rounds.
''')
markdown('''
## Verify the snapshot and copy data to local disk
The manifest detects changed/corrupt files; it is not an authenticity signature.
Use only the bundle you generated. Extraction rejects traversal, duplicates and symlinks.
The training HDF5 files live under `/content`, NOT on mounted Drive during training.
''')
code('''
import hashlib, json, stat, zipfile
from pathlib import PurePosixPath

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

bundle_sha = sha(BUNDLE)
print("Bundle SHA256:", bundle_sha)
if EXPECTED_BUNDLE_SHA256 and bundle_sha != EXPECTED_BUNDLE_SHA256:
    raise ValueError("Bundle SHA256 mismatch")
with zipfile.ZipFile(BUNDLE) as archive:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("Duplicate ZIP entries")
    manifest = json.loads(archive.read("bundle_manifest.json"))
    assert manifest["protocol"] == "canonical-colab-bundle-v1"
    assert set(names) == set(manifest["files"]) | {"bundle_manifest.json"}
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\\\" in name or stat.S_ISLNK(archive.getinfo(name).external_attr >> 16):
            raise ValueError("Unsafe ZIP member")
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest["files"].items():
        destination = ROOT / name
        if any(parent.is_symlink() for parent in [destination, *destination.parents]):
            raise ValueError("Local destination/parent is a symlink")
        if destination.exists():
            if sha(destination) != expected:
                raise ValueError("Existing local snapshot differs; use a fresh runtime")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(name) as source, destination.open("xb") as target:
            for chunk in iter(lambda: source.read(1024*1024), b""):
                target.write(chunk)
        if sha(destination) != expected:
            raise ValueError("Extracted file hash mismatch")
assert not (ROOT / "data/output/daniela-rk4/test.h5").exists()
os.chdir(ROOT)
import sys
sys.path.insert(0, str(ROOT))
print("Verified local code + train/val; held-out data absent.")
''')
markdown('''
## Runtime dependencies and GPU checks
Keep Colab's GPU-enabled PyTorch; do not install the workstation's CPU wheel.
This project was tested with PyTorch 2.11. If the assigned runtime is older, install
an official compatible **GPU** build of PyTorch ≥2.11 using the PyTorch installation
selector, restart the runtime, and rerun. This notebook deliberately stops rather
than guessing a CUDA-wheel/driver combination or silently falling back to CPU.
Only missing lightweight dependencies are installed; the actual environment is recorded.
''')
code('''
import importlib.util, subprocess
for module, package in [("numpy", "numpy"), ("h5py", "h5py"), ("yaml", "pyyaml"), ("torchinfo", "torchinfo==1.8.0"), ("tqdm", "tqdm"), ("pytest", "pytest")]:
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
import torch
assert sys.version_info >= (3, 12), "Python >=3.12 required"
assert tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) >= (2, 11), "Install an official compatible GPU Torch >=2.11 build, restart, rerun"
assert torch.cuda.is_available(), "No usable CUDA GPU; select a GPU runtime/check its driver"
torch.set_num_threads(1)
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
from training.resume import environment
print(json.dumps(environment(torch.device("cuda")), indent=2))
print("CUDA wheel:", torch.version.cuda, "GPU RAM GB:", torch.cuda.get_device_properties(0).total_memory / 1e9)
''')
markdown('''
## Correctness first
CPU synthetic integration tests include exact interrupted/uninterrupted replay.
The CUDA test compares outputs and parameter gradients against CPU in float32.
Passing CPU tests alone is NOT evidence that CUDA is correct or faster.
''')
code('''
import uuid, xml.etree.ElementTree as ET
junit = Path("/content") / f"hgnn-preflight-{uuid.uuid4().hex}.xml"
subprocess.run([sys.executable, "-m", "pytest", "-q", "models/test_hgnn_canonical.py", "training/test_elapsed_loss.py", "training/test_resume.py", "training/test_colab.py", f"--junitxml={junit}"], cwd=ROOT, check=True)
gpu_names = {"test_cuda_output_and_parameter_gradient_parity", "test_cuda_resume_replay"}
gpu_cases = [case for case in ET.parse(junit).iter("testcase") if case.get("name") in gpu_names]
assert len(gpu_cases) == 2 and all(case.find("skipped") is None for case in gpu_cases), "CUDA tests must actually RUN, not skip"
CUDA_PREFLIGHT_PASSED = True
''')
markdown('''
## Measure before training (ZERO optimizer updates)
Compare matched batch64 CPU/CUDA probes. These include forward/loss/backward/clipping,
not data loading, optimizer steps, fixed validation, or Drive IO; extrapolation is
NOT a full-run runtime guarantee. Longer horizons/memory are optional bounded probes.
No batch-size/LR changes are automatically applied from these measurements.
''')
code('''
from training._io import load_config
from training.colab_runtime import benchmark_training
cfg = load_config(ROOT / "configs/hgnn-pilot-ready.yaml")
profiles = {device: benchmark_training(cfg, device=device, batch_size=64, horizons=(1,5,20), repeats=3) for device in ("cpu", "cuda")}
print(json.dumps(profiles, indent=2))
import uuid
benchmark_dir = DRIVE_ROOT / "benchmarks" / uuid.uuid4().hex
benchmark_dir.mkdir(parents=True, exist_ok=False)
with (benchmark_dir / "profiles.json").open("x") as handle:
    json.dump({"bundle_sha256": bundle_sha, "profiles": profiles}, handle, indent=2)
print("Saved diagnostic profiles:", benchmark_dir)
assert all(p["status"] == "ok" for p in profiles["cuda"]["profiles"]), "GPU preflight failed/OOM; do not train"
cpu20 = profiles["cpu"]["profiles"][-1]["median_seconds"]
gpu20 = profiles["cuda"]["profiles"][-1]["median_seconds"]
GPU_FASTER = gpu20 < cpu20
print("Matched h20 forward/backward speed ratio CPU/GPU:", cpu20/gpu20)
print("Approximate 1500-update compute-only seconds:", 500*sum(p["median_seconds"] for p in profiles["cuda"]["profiles"]))
print("ADD fixed-validation, optimizer, loading and backup costs; no runtime guarantee.")
PROBE_LONGER = False
if PROBE_LONGER:
    print(json.dumps(benchmark_training(cfg, device="cuda", batch_size=16, horizons=(50,100), repeats=1), indent=2))
''')
markdown('''
## Explicit development recipe and resume
The default keeps the validated **1500 total updates**, batch64, LR2.5e-4 and horizons
1/5/20. It splits each 500-update stage into five 100-update rounds so progress can
be backed up more often. This adds validation/selection points; it is NOT the identical
checkpoint-selection schedule as the workstation pilot. Changing budgets is a NEW
recipe, not a resume. Decide any larger budget before starting a new run.

Resume uses `latest.pt`, not `best.pt`. It restores optimizer/scheduler, scales,
round/curriculum progress, best-model selection, sampler and Python/NumPy/Torch RNGs.
Same-runtime CPU replay is tested bitwise; CUDA is tested when a GPU is available,
not guaranteed bitwise across different devices/versions. Runtime changes require
explicit permission and are recorded. Source/data/recipe changes are always rejected.
''')
code('''
from dataclasses import replace, asdict
from training._types import CheckpointConfig, LoggingConfig
from training.colab_runtime import restore_backup, verify_backup
from training.train import Trainer
ROUND_BATCHES = 100
ROUNDS_PER_STAGE = [5,5,5]  # 500 updates per stage; change only BEFORE a NEW experiment.
LOCAL_ARTIFACTS = Path("/content/hgnn-artifacts")
cfg = replace(cfg,
    training=replace(cfg.training, device="cuda", max_train_batches_per_epoch=ROUND_BATCHES, curriculum_epochs=ROUNDS_PER_STAGE),
    checkpointing=CheckpointConfig(enabled=True, dir=str(LOCAL_ARTIFACTS)),
    logging=LoggingConfig(enabled=True, dir=str(LOCAL_ARTIFACTS)))
print(json.dumps(asdict(cfg), indent=2))
print("Total planned updates:", ROUND_BATCHES*sum(ROUNDS_PER_STAGE))
print("Existing committed snapshots (choose one explicitly to resume):")
for marker in sorted(BACKUP_ROOT.glob("*/*/COMMITTED.json")):
    try:
        meta = verify_backup(marker.parent)
        print(meta["epoch"], meta["optimizer_steps"], marker.parent)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print("Invalid/incomplete snapshot ignored:", marker.parent, exc)
''')
markdown('''
## Train and back up after EVERY committed round
Set `RUN_TRAINING=True` in the settings cell only after reviewing preflight/timings.
If the matched GPU probe is slower, also explicitly acknowledge that choice.
Backup failures STOP training rather than accumulating unprotected progress.
Drive snapshots are append-only, hash-verified, with `COMMITTED.json` written last.
Ignore snapshots without that marker. Drive/FUSE cannot guarantee remote durability
at the instant a client disconnects; retain multiple verified snapshots and the ZIP.

If Colab disconnects: reconnect/mount Drive, restore the SAME bundle, set the SAME
recipe/tag and explicit `RESUME_SNAPSHOT`, rerun preflight, then train. The last
uncommitted round is replayed. Ctrl+C does not promise a mid-batch checkpoint.
''')
code('''
import uuid
from training.colab_runtime import backup_round
assert RUN_TRAINING, "Review probes, set RUN_TRAINING=True, then rerun this cell"
assert CUDA_PREFLIGHT_PASSED, "Run successful CUDA correctness and resume checks first"
assert MAX_ROUNDS_THIS_SESSION is None or (isinstance(MAX_ROUNDS_THIS_SESSION, int) and MAX_ROUNDS_THIS_SESSION > 0), "Session round cap must be positive"
assert GPU_FASTER or ACK_GPU_NOT_FASTER, "GPU did not win matched probe; explicitly acknowledge or use CPU locally"
if RESUME_SNAPSHOT:
    local_checkpoint = restore_backup(Path(RESUME_SNAPSHOT), Path("/content/restore") / uuid.uuid4().hex)
    from training._io import load_checkpoint
    assert load_checkpoint(local_checkpoint, torch.device("cpu")).epoch < sum(ROUNDS_PER_STAGE), "This run is already complete; inspect/download best.pt from that snapshot instead"
    trainer = Trainer(cfg, resume_checkpoint=local_checkpoint, allow_runtime_change=ALLOW_RUNTIME_CHANGE)
else:
    assert not list(BACKUP_ROOT.glob("*/*/COMMITTED.json")), "Existing run tag: choose a NEW tag or explicitly resume"
    trainer = Trainer(cfg)
with (trainer.ckpt_dir / "session_metadata.json").open("x") as handle:
    json.dump({"bundle_sha256": bundle_sha, "environment": environment(torch.device("cuda")), "profiles": profiles, "config": asdict(cfg), "resume_snapshot": RESUME_SNAPSHOT or None}, handle, indent=2)
local_rounds = 0
while len(trainer._progress.get("train_history", [])) < sum(ROUNDS_PER_STAGE):
    result = trainer.run(max_rounds=1)
    snapshot = backup_round(trainer.ckpt_dir, BACKUP_ROOT)
    verify_backup(snapshot)
    print("COMMITTED:", snapshot, "round", len(result.train_history), "updates", trainer.optimizer_steps)
    local_rounds += 1
    if MAX_ROUNDS_THIS_SESSION is not None and local_rounds >= MAX_ROUNDS_THIS_SESSION:
        break
print("Local checkpoint directory:", trainer.ckpt_dir)
''')
code('''
from training._io import load_checkpoint
best = load_checkpoint(trainer.ckpt_dir / "best.pt", torch.device("cpu"))
print("Selected metric:", best.selected_metric, "score:", best.selected_score, "round:", best.epoch)
for record in trainer.validation_history:
    print("updates", record["optimizer_steps"], "position MAE", record["model"]["position_mae"]["mean"], "state MSE", record["model"]["state4_mse"], "failed runs", record["model"]["failed_runs"])
print("Development validation only: do not claim final/shared-IC benchmark results.")
''')

notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}, "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
path = Path(__file__).with_name("hgnn_resumable.ipynb")
with path.open("x") as handle:
    json.dump(notebook, handle, indent=2)
    handle.write("\n")
print(path)
