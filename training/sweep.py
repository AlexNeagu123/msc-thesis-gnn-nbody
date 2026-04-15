"""Grid search over learning rate and noise factor.

Runs all combinations sequentially using the base EGNN config.
Prints a summary table at the end with best val loss per run.

Usage::

    python -m training.sweep
    python -m training.sweep --epochs 200 --config configs/egnn.yaml
"""

import argparse
import itertools
from dataclasses import replace

from training._types import TrainConfig
from training.train import Trainer, load_config
from utils import get_logger

logger = get_logger(__name__)

# descending order: aggressive configs run first
LRS = [2e-3, 1e-3, 5e-4]
NOISE_FACTORS = [0.1, 0.05, 0.0]


def run_sweep(base_cfg: TrainConfig, epochs: int) -> None:
    """Run grid search over lr and noise_factor.

    Args:
        base_cfg: base config to modify per run.
        epochs: number of epochs per run.
    """
    grid = list(itertools.product(LRS, NOISE_FACTORS))
    n_runs = len(grid)

    logger.info(
        "sweep: %d runs (%d lr x %d noise_factor), %d epochs each",
        n_runs, len(LRS), len(NOISE_FACTORS), epochs,
    )

    results = []

    for i, (lr, nf) in enumerate(grid):
        logger.info(
            "--- run %d/%d: lr=%.1e, noise_factor=%.2f ---",
            i + 1, n_runs, lr, nf,
        )

        training = replace(
            base_cfg.training,
            lr=lr,
            noise_factor=nf,
            epochs=epochs,
        )
        cfg = replace(base_cfg, training=training)

        trainer = Trainer(cfg)
        result = trainer.run()

        results.append({
            "run_id": trainer.run_id,
            "lr": lr,
            "noise_factor": nf,
            "best_val_loss": result.best_val_loss,
            "best_epoch": result.best_epoch,
            "final_train_loss": result.final_train_loss,
        })

        logger.info(
            "run %d/%d done: best_val=%.6f at epoch %d",
            i + 1, n_runs, result.best_val_loss, result.best_epoch,
        )

    # summary table
    logger.info("=" * 80)
    logger.info("SWEEP RESULTS")
    logger.info("=" * 80)
    logger.info(
        "%-20s  %-10s  %-12s  %-12s  %-10s  %-12s",
        "run_id", "lr", "noise_factor", "best_val", "best_epoch", "final_train",
    )
    logger.info("-" * 80)

    for r in sorted(results, key=lambda x: x["best_val_loss"]):
        logger.info(
            "%-20s  %-10.1e  %-12.2f  %-12.6f  %-10d  %-12.6f",
            r["run_id"],
            r["lr"],
            r["noise_factor"],
            r["best_val_loss"],
            r["best_epoch"],
            r["final_train_loss"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid search over lr and noise_factor.")
    parser.add_argument("--config", type=str, default="configs/egnn.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    config = load_config(args.config)
    run_sweep(config, args.epochs)
