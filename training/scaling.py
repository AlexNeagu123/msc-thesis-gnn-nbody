"""Data-scaling sweep: train one model on nested training-set sizes.

Runs the same model config at multiple n_train_trajectories values, recording
each run's checkpoint and best validation loss. Used to answer the question:
does the EGNN-vs-HGNN comparison change as more training data is added?

Each per-size run lands under `<artifact_root>/<model>/n<N_TRAIN>/<run_id>/`
so multiple sizes can coexist in the same archive. Default artifact root is
`runs/scaling` to match the canonical layout.

Usage::

    python -m training.scaling --config configs/egnn.yaml --sizes 1000,2000,5000
    python -m training.scaling --config configs/hgnn.yaml --sizes 1000,2000,5000

References:
    - Methodology: edu/data-scaling-methodology.md
"""

import argparse
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from training._types import TrainConfig
from training.train import Trainer, apply_artifact_dir, load_config
from utils import get_logger

logger = get_logger(__name__)

DEFAULT_ARTIFACT_ROOT = "runs/scaling"


def scaling_artifact_dir(model_name: str, n_train: int, root: str | Path) -> Path:
    """Path each per-size run is written to under the canonical layout."""
    return Path(root) / model_name / f"n{n_train}"


def run_scaling(
    base_cfg: TrainConfig,
    sizes: list[int],
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    trainer_factory: Callable[[TrainConfig], Trainer] | None = None,
) -> None:
    """Run training at each dataset size in `sizes`.

    Args:
        base_cfg: shared model/training config.
        sizes: ordered list of n_train_trajectories values to sweep.
        artifact_root: parent directory under which per-size run folders are
            created. Each run lands at `<artifact_root>/<model>/n<N_TRAIN>/`.
        trainer_factory: optional factory used to construct each per-size
            trainer. Used by tests to inject a DummyModel; production runs
            leave it None to fall through to `Trainer(cfg)`.
    """
    factory = trainer_factory or Trainer
    logger.info(
        "data-scaling sweep: %d runs at sizes=%s, artifact_root=%s",
        len(sizes),
        sizes,
        artifact_root,
    )

    results = []

    for i, n_train in enumerate(sizes):
        logger.info("--- run %d/%d: n_train=%d ---", i + 1, len(sizes), n_train)

        data = replace(base_cfg.data, n_train_trajectories=n_train)
        cfg = replace(base_cfg, data=data)
        cfg = apply_artifact_dir(cfg, scaling_artifact_dir(cfg.model.name, n_train, artifact_root))

        trainer = factory(cfg)
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
    parser.add_argument(
        "--artifact-root",
        type=str,
        default=DEFAULT_ARTIFACT_ROOT,
        help=(
            "Parent directory under which per-size run folders are created. "
            f"Default: {DEFAULT_ARTIFACT_ROOT}."
        ),
    )
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    config = load_config(args.config)
    run_scaling(config, sizes, artifact_root=args.artifact_root)
