"""Data-scaling sweep: train one model on nested training-set sizes.

Runs the same model config at multiple n_train_trajectories values, recording
each run's checkpoint and best validation loss. Used to answer the question:
does the EGNN-vs-HGNN comparison change as more training data is added?

Usage::

    python -m training.scaling --config configs/egnn.yaml --sizes 1000,2000,5000
    python -m training.scaling --config configs/hgnn.yaml --sizes 1000,2000,5000

References:
    - Methodology: edu/data-scaling-methodology.md
"""

import argparse
from dataclasses import replace

from training._types import TrainConfig
from training.train import Trainer, load_config
from utils import get_logger

logger = get_logger(__name__)


def run_scaling(base_cfg: TrainConfig, sizes: list[int]) -> None:
    """Run training at each dataset size in `sizes`."""
    logger.info("data-scaling sweep: %d runs at sizes=%s", len(sizes), sizes)

    results = []

    for i, n_train in enumerate(sizes):
        logger.info("--- run %d/%d: n_train=%d ---", i + 1, len(sizes), n_train)

        data = replace(base_cfg.data, n_train_trajectories=n_train)
        cfg = replace(base_cfg, data=data)

        trainer = Trainer(cfg)
        result = trainer.run()

        results.append(
            {
                "run_id": trainer.run_id,
                "n_train": n_train,
                "best_val_loss": result.best_val_loss,
                "best_epoch": result.best_epoch,
                "final_train_loss": result.final_train_loss,
            }
        )

        logger.info(
            "run %d/%d done: best_val=%.6f at epoch %d",
            i + 1,
            len(sizes),
            result.best_val_loss,
            result.best_epoch,
        )

    logger.info("=" * 80)
    logger.info("DATA-SCALING RESULTS")
    logger.info("=" * 80)
    logger.info(
        "%-20s  %-10s  %-12s  %-10s  %-12s",
        "run_id",
        "n_train",
        "best_val",
        "best_epoch",
        "final_train",
    )
    logger.info("-" * 80)

    for r in results:
        logger.info(
            "%-20s  %-10d  %-12.6f  %-10d  %-12.6f",
            r["run_id"],
            r["n_train"],
            r["best_val_loss"],
            r["best_epoch"],
            r["final_train_loss"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data-scaling sweep for one model.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--sizes",
        type=str,
        default="1000,2000,5000",
        help="Comma-separated list of n_train_trajectories values.",
    )
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    config = load_config(args.config)
    run_scaling(config, sizes)
