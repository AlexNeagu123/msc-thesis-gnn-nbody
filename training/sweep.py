"""Grid search over learning rate and noise factor.

Each (lr, noise_factor) cell lands under
`<artifact_root>/<model>/lr_<lr>_nf_<nf>/<run_id>/` so every cell coexists
in the archive. Default artifact root is `runs/sweep` to match the
canonical layout. This is a 2-D hyperparameter grid and is deliberately
distinct from Colab's single-axis `runs/noise_sweep/...`.
"""

import argparse
import itertools
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from training._types import TrainConfig
from training.train import Trainer, apply_artifact_dir, load_config
from utils import get_logger

logger = get_logger(__name__)

# best candidates first, based on initial experiments
LRS = [5e-4, 1e-3, 2e-3]
NOISE_FACTORS = [0.0, 0.03, 0.05]
DEFAULT_ARTIFACT_ROOT = "runs/sweep"


def _label_lr(lr: float) -> str:
    """Path-safe label for a learning rate (e.g. 5e-4 -> '5e-04')."""
    return f"{lr:.0e}"


def _label_nf(nf: float) -> str:
    """Path-safe label for a noise factor (e.g. 0.03 -> '0p03')."""
    return f"{nf:g}".replace(".", "p")


def sweep_artifact_dir(model_name: str, lr: float, nf: float, root: str | Path) -> Path:
    """Path each per-cell run is written to under the canonical layout."""
    return Path(root) / model_name / f"lr_{_label_lr(lr)}_nf_{_label_nf(nf)}"


def run_sweep(
    base_cfg: TrainConfig,
    epochs: int,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    trainer_factory: Callable[[TrainConfig], Trainer] | None = None,
) -> None:
    """Run grid search over lr and noise_factor.

    Args:
        base_cfg: shared model/training config.
        epochs: epochs per cell, applied after the (lr, nf) override.
        artifact_root: parent directory; each cell lands at
            `<artifact_root>/<model>/lr_<lr>_nf_<nf>/`.
        trainer_factory: optional factory for tests to inject a DummyModel.
    """
    factory = trainer_factory or Trainer
    grid = list(itertools.product(LRS, NOISE_FACTORS))
    n_runs = len(grid)

    logger.info(
        "sweep: %d runs (%d lr x %d noise_factor), %d epochs each, artifact_root=%s",
        n_runs,
        len(LRS),
        len(NOISE_FACTORS),
        epochs,
        artifact_root,
    )

    results = []

    for i, (lr, nf) in enumerate(grid):
        logger.info(
            "--- run %d/%d: lr=%.1e, noise_factor=%.2f ---",
            i + 1,
            n_runs,
            lr,
            nf,
        )

        training = replace(
            base_cfg.training,
            lr=lr,
            noise_factor=nf,
            epochs=epochs,
        )
        cfg = replace(base_cfg, training=training)
        cfg = apply_artifact_dir(cfg, sweep_artifact_dir(cfg.model.name, lr, nf, artifact_root))

        trainer = factory(cfg)
        result = trainer.run()

        results.append(
            {
                "run_id": trainer.run_id,
                "lr": lr,
                "noise_factor": nf,
                "best_val_loss": result.best_val_loss,
                "best_epoch": result.best_epoch,
                "final_train_loss": result.final_train_loss,
            }
        )

        logger.info(
            "run %d/%d done: best_val=%.6f at epoch %d",
            i + 1,
            n_runs,
            result.best_val_loss,
            result.best_epoch,
        )

    logger.info("=" * 80)
    logger.info("SWEEP RESULTS")
    logger.info("=" * 80)
    logger.info(
        "%-20s  %-10s  %-12s  %-12s  %-10s  %-12s",
        "run_id",
        "lr",
        "noise_factor",
        "best_val",
        "best_epoch",
        "final_train",
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
    parser.add_argument(
        "--artifact-root",
        type=str,
        default=DEFAULT_ARTIFACT_ROOT,
        help=(
            "Parent directory under which per-cell run folders are created. "
            f"Default: {DEFAULT_ARTIFACT_ROOT}."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_sweep(config, args.epochs, artifact_root=args.artifact_root)
