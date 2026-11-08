# ruff: noqa: ANN001, ANN201, D103
"""Atomic development checkpoints. Explicit identity, model, Adam, cursor and RNG.

Not a production completion bundle; no automatic run/resume loop is supplied.
"""

import copy
import os
import random
import uuid
from pathlib import Path

import numpy as np
import torch


def save(path, model, optimizer, cursor, identity, progress):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".pending-" + uuid.uuid4().hex)
    value = {
        "protocol": "next-study-torch-development-v1",
        "identity": copy.deepcopy(identity),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cursor": cursor.state(),
        "progress": copy.deepcopy(progress),
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    try:
        torch.save(value, temp)
        with temp.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def restore(path, model, optimizer, cursor, identity):
    # Only locally-created trusted checkpoints. Production must add sealed hashes/bundles.
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value["protocol"] != "next-study-torch-development-v1" or value["identity"] != identity:
        raise ValueError("wrong scientific/runtime identity")
    # Validate model semantics and schedule on temporary copies before mutating live objects.
    probe = copy.deepcopy(model)
    probe.load_state_dict(value["model"], strict=True)
    probe_cursor = copy.deepcopy(cursor)
    probe_cursor.restore(value["cursor"])
    model.load_state_dict(value["model"], strict=True)
    optimizer.load_state_dict(value["optimizer"])
    cursor.restore(value["cursor"])
    torch.set_rng_state(value["rng"]["torch"])
    np.random.set_state(value["rng"]["numpy"])
    random.setstate(value["rng"]["python"])
    return value["progress"]
