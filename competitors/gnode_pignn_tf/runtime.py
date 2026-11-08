"""Explicit CPU inference policy. Call before importing model implementations."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import sys
from pathlib import Path


def configure() -> dict:
    """Configure one-thread float32 CPU, deterministic ops, no XLA/TF32/oneDNN."""
    expected = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TF_ENABLE_ONEDNN_OPTS": "0",
        "CUDA_VISIBLE_DEVICES": "-1",
    }
    for name, value in expected.items():
        if name in os.environ and os.environ[name] != value:
            raise RuntimeError(f"runtime policy mismatch: {name}={os.environ[name]}")
        os.environ[name] = value
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.experimental.enable_op_determinism()
    tf.config.optimizer.set_jit(False)
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.keras.utils.set_random_seed(42)  # only discarded graph initialization; no training
    return {
        "python": sys.version,
        "executable": sys.executable,
        "executable_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("tensorflow", "keras", "numpy", "h5py")
        },
        "policy": expected,
        "intra_threads": 1,
        "inter_threads": 1,
        "dtype": "float32",
        "xla": False,
        "tf32": False,
        "amp": False,
        "deterministic_ops": True,
        "initialization_seed_only": 42,
        "historical_tensorflow_version_known": False,
    }
