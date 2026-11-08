"""Separate GPU execution identity; never modify the frozen CPU runtime policy."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys


def configure() -> dict:
    """Require a GPU before any update; enable deterministic float32, not TF32/XLA."""
    expected = {
        "KERAS_BACKEND": "tensorflow",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TF_ENABLE_ONEDNN_OPTS": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    for name, value in expected.items():
        if name in os.environ and os.environ[name] != value:
            raise RuntimeError(f"GPU preflight environment mismatch: {name}")
        os.environ[name] = value
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES=-1; CPU fallback is forbidden")
    import tensorflow as tf

    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise RuntimeError(
            "No GPU detected; CPU fallback is forbidden. Select a Colab GPU runtime."
        )
    versions = {
        name: importlib.metadata.version(name) for name in ("tensorflow", "keras", "numpy", "h5py")
    }
    if versions != {"tensorflow": "2.21.0", "keras": "3.13.2", "numpy": "2.5.4", "h5py": "3.14.0"}:
        raise RuntimeError(f"GPU preflight package mismatch: {versions}")
    tf.config.set_visible_devices(physical[:1], "GPU")
    tf.config.set_logical_device_configuration(
        physical[0], [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
    )
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.experimental.enable_op_determinism()
    tf.config.optimizer.set_jit(False)
    tf.keras.mixed_precision.set_global_policy("float32")
    # TF may place host-only shape/control operations on CPU. Model variables,
    # matrix operations and returned training loss are independently checked on GPU.
    tf.config.set_soft_device_placement(True)
    tf.keras.utils.set_random_seed(42)
    details = tf.config.experimental.get_device_details(physical[0])
    with tf.device("/GPU:0"):
        sample = tf.linalg.matmul(tf.ones((8, 8)), tf.ones((8, 8)))
    if "GPU:0" not in sample.device.upper():
        raise RuntimeError("GPU matrix smoke test was silently placed on CPU")
    try:
        smi = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10, check=False
        )
        driver_info = smi.stdout + smi.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        driver_info = str(exc)
    return {
        "protocol": "daniela-tf-gpu-preflight-v1",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "gpu": {key: str(value) for key, value in details.items()},
        "tensorflow_build": tf.sysconfig.get_build_info(),
        "nvidia_smi": driver_info,
        "environment": expected,
        "dtype": "float32",
        "tf32": False,
        "xla": False,
        "amp": False,
        "deterministic_ops": True,
        "gpu_memory_limit_mib": 4096,
        "auxiliary_host_ops_allowed": True,
        "smoke_matmul_device": sample.device,
        "cpu_gpu_bitwise_equality_claim": False,
    }
