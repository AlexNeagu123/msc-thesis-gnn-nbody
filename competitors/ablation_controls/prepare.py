"""Create exclusive development plan/schedule artifacts. No optimizer or data trajectories."""
# ruff: noqa: ANN001, ANN201, D103

import hashlib
import json
from pathlib import Path

import numpy as np

from .recipes import CELLS
from .schedule import identity, matched_tail_order

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "runs/daniela-ablation-next-development-v1"
BANK = ROOT / "runs/daniela-hgnn-order-reconstruction-audit-v1/historical-assigned-batches.npz"
BANK_SHA = "e16c618a59cf320c619d8249442b12b469b9bef3c0dc269782ea2bf7002e5d06"


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in ["PLAN.json", "matched-tail-order.npy", "DEVELOPMENT_MANIFEST.json"]:
        if (OUTPUT / name).exists():
            raise FileExistsError("refusing to overwrite development artifacts: " + name)
    assert sha(BANK) == BANK_SHA
    with np.load(BANK, allow_pickle=False) as bank:
        order = matched_tail_order(bank)
        for key in [
            "update",
            "horizon",
            "trajectory_index",
            "simulation_id",
            "start_frame",
            "flat_index",
        ]:
            # Preserve ALL entries once; common last500assigned batches must be byte-identical.
            a = bank[key]
            np.testing.assert_array_equal(a[order[-500:]], a[-500:])
    with (OUTPUT / "matched-tail-order.npy").open("xb") as f:
        np.save(f, order, allow_pickle=False)
    reference_runs = {
        "hgnn_learned_gravity_off": "runs/daniela-hgnn-replication-diagnostics-v1/staged_seed42",
        "hgnn_learned_gravity_scaled": "runs/daniela-norm-gnode-diagnostics-v1/normalized_hgnn_seed42",
        "gnode_native": "runs/daniela-gnode-staged-multistep-v1",
        "pi_ordered_unclipped": "runs/daniela-pi-ablation-production-v1/pignn_softening_only_ablation_v1",
        "pi_ordered_clip1": "runs/daniela-pi-a10-clip1-v1",
    }
    questions = {
        "hgnn": "Does exact physical kinetic energy help, and does it change the benefit of scaled force supervision?",
        "gnode": "Does changing only the position update improve the matched-budget staged GNODE?",
        "pi_order": "Does shuffled pair order improve A10, and is clipping conditional on data order?",
        "curriculum": "Does early ordering matter when the final500h100units are identical?",
        "central_pair": "Within a central-force architecture, does normalized force supervision improve selected rollouts?",
    }
    plan = {
        "protocol": "next-ablation-development-v1",
        "production_authorized": False,
        "optimizer_updates": 0,
        "scope": "five tracks;no production runner or queue;no new VAL/test reads",
        "cells": {n: c.record() for n, c in CELLS.items()},
        "reference_runs": reference_runs,
        "questions": questions,
        "curriculum": {
            **identity(order),
            "bank_sha256": BANK_SHA,
            "order_file": "matched-tail-order.npy",
            "file_sha256": sha(OUTPUT / "matched-tail-order.npy"),
        },
        "proposed_execution_order": [
            "finish_latest_selected_VAL_replays",
            "hgnn_exact_gravity_off",
            "hgnn_exact_gravity_scaled",
            "gnode_semiimplicit",
            "pi_shuffled_unclipped",
            "hgnn_matched_tail",
            "central_pair_gravity_off",
            "central_pair_gravity_scaled",
        ],
        "conditional": ["pi_shuffled_clip1", "seed43replication of central contrasts"],
        "required_gates": [
            "isolated production runner and loader",
            "sealed initialization/source/data/runtime identity",
            "full logical-batch real-TRAIN gradient/finite/resume checks",
            "signal/battery/wall and terminal failure tests",
            "all-horizon measured timing",
            "bounded user launch scope",
            "freeze roles before diagnostic evaluation",
        ],
        "selection": "eligible minimum full1999-future VAL80 position MAE;earlier exact ties;separate final checkpoint",
        "budget": "pending measured preflights;historical elapsed times are context, not new forecasts",
        "diagnostics": "existing100remains historically inspected post-selection diagnostic;not a tuning or untouched final test set",
    }
    with (OUTPUT / "PLAN.json").open("x") as f:
        json.dump(plan, f, indent=2)
    source = Path(__file__).parent
    manifest = {
        "development_only": True,
        "not_a_READY_seal": True,
        "sources": {
            str(p.relative_to(ROOT)): sha(p) for p in sorted(source.iterdir()) if p.is_file()
        },
        "artifacts": {p.name: sha(p) for p in sorted(OUTPUT.iterdir()) if p.is_file()},
        "bank_sha256": BANK_SHA,
        "trajectory_data_reads": False,
        "production_training_launched": False,
        "new_production_weights_initialized_from_tests": False,
    }
    with (OUTPUT / "DEVELOPMENT_MANIFEST.json").open("x") as f:
        json.dump(manifest, f, indent=2)
    print(
        "DEVELOPMENT PLAN PREPARED:13cells/5tracks,5references,8new incl1conditional. No production launch."
    )


if __name__ == "__main__":
    main()
