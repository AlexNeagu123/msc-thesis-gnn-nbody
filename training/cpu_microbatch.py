"""CPU-only canonical execution adapter; legacy/Colab trainer code stays unchanged.

One optimizer update per logical batch. Each microbatch contributes its sample-weighted
mean loss; clip once after all backwards. Release graphs before the next microbatch.
CPU checkpoints fingerprint this module and preserve microbatch identity in progress.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.hgnn_canonical import MODEL_ID
from training._io import load_checkpoint
from training._types import BucketRolloutScore, EpochRunSummary, RolloutScore, TrainConfig
from training.fixed_validation import file_sha256
from training.train import Trainer

logger = logging.getLogger(__name__)
PROTOCOL = "canonical-cpu-microbatch-v1"


class CpuMicrobatchTrainer(Trainer):
    """Memory-bounded execution of the same sample-mean canonical MSE objective."""

    def __init__(
        self,
        cfg: TrainConfig,
        *,
        micro_batch_size: int = 16,
        resume_checkpoint: str | Path | None = None,
        allow_runtime_change: bool = False,
    ) -> None:
        """Start fresh or resume only matching CPU-adapter checkpoints, never GPU weights."""
        if cfg.model.name != MODEL_ID or cfg.training.device != "cpu" or cfg.training.loss != "mse":
            raise ValueError("CPU microbatch adapter requires canonical HGNN, cpu device and mse")
        if (
            isinstance(micro_batch_size, bool)
            or not isinstance(micro_batch_size, int)
            or micro_batch_size < 1
        ):
            raise ValueError("micro_batch_size must be a positive integer")
        self._micro_batch_size = micro_batch_size
        if resume_checkpoint is not None:
            saved = load_checkpoint(Path(resume_checkpoint), torch.device("cpu"))
            state = getattr(saved, "resume_state", None)
            identity = (
                state.get("progress", {}).get("cpu_execution") if isinstance(state, dict) else None
            )
            if identity != self._execution_identity(cfg):
                raise ValueError(
                    "CPU resume requires matching CPU microbatch execution identity; GPU/ordinary checkpoints are not silently converted"
                )
        super().__init__(
            cfg, resume_checkpoint=resume_checkpoint, allow_runtime_change=allow_runtime_change
        )

    @property
    def micro_batch_size(self) -> int:
        """Maximum examples per graph, not effective optimizer batch size."""
        return self._micro_batch_size

    def _execution_identity(self, cfg: TrainConfig) -> dict:
        return {
            "protocol": PROTOCOL,
            "micro_batch_size": self.micro_batch_size,
            "effective_batch_size": cfg.training.batch_size,
        }

    def _setup_data(self) -> tuple[DataLoader, DataLoader]:
        self.source_hashes["training/cpu_microbatch.py"] = file_sha256(Path(__file__))
        entry = Path(__file__).with_name("run_cpu_daniela.py")
        if entry.is_file():
            self.source_hashes["training/run_cpu_rk4.py"] = file_sha256(entry)
        return super()._setup_data()

    def _backward_logical_batch(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[float, torch.Tensor] | None:
        """Accumulate weighted gradients; no clip/step; discard whole batch on bad loss."""
        self.optimizer.zero_grad(set_to_none=True)
        count = len(inputs)
        total = 0.0
        first_predictions = []
        for start in range(0, count, self.micro_batch_size):
            stop = min(start + self.micro_batch_size, count)
            predictions, loss, diagnostic_targets = self._compute_loss(
                inputs[start:stop], targets[start:stop]
            )
            if self.cfg.training.skip_nonfinite_batches and not torch.isfinite(loss).item():
                self.optimizer.zero_grad(set_to_none=True)
                del predictions, loss, diagnostic_targets
                return None
            weighted = loss * ((stop - start) / count)
            weighted.backward()
            total += float(loss.detach()) * (stop - start)
            first_predictions.append(predictions.detach())
            # Explicitly destroy graph references before constructing another graph.
            del predictions, loss, diagnostic_targets, weighted
        return total / count, torch.cat(first_predictions)

    def _run_epoch(self, *, training: bool, verbose: bool) -> EpochRunSummary:
        if not training:
            return super()._run_epoch(training=False, verbose=verbose)
        self.model.train()
        loader = self.train_loader
        batches = tqdm(loader, desc="train (CPU microbatch)", leave=False) if verbose else loader
        total = 0.0
        samples = 0
        norms = []
        clipped = 0
        skipped = 0
        with torch.enable_grad():
            for batch_index, (inputs, targets) in enumerate(batches, start=1):
                self.attempted_train_batches += 1
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if self.cfg.training.noise_factor > 0:
                    inputs = self.apply_noise(inputs)
                backward = self._backward_logical_batch(inputs, targets)
                if backward is None:
                    logger.warning("skipped logical batch: nonfinite microbatch loss")
                    skipped += 1
                    continue
                batch_loss, predictions = backward
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.current_clip_norm
                )
                if not torch.isfinite(norm).item():
                    self.optimizer.zero_grad(set_to_none=True)
                    logger.warning("skipped logical batch: nonfinite accumulated gradients")
                    skipped += 1
                    continue
                norm_value = float(norm)
                norms.append(norm_value)
                clipped += norm_value > self.current_clip_norm
                self.optimizer.step()
                self.optimizer_steps += 1
                total += batch_loss * len(inputs)
                samples += len(inputs)
                diag_targets = targets if self.current_horizon == 1 else targets[:, 0]
                self.diagnostics.check_batch(
                    inputs, diag_targets, predictions, batch_loss, batch_index, len(loader)
                )
                if verbose:
                    batches.set_postfix(loss=f"{total / samples:.6f}")
        return EpochRunSummary(
            loss=total / samples if samples else float("nan"),
            grad_norm_mean=float(np.mean(norms)) if norms else None,
            grad_norm_max=float(np.max(norms)) if norms else None,
            grad_clip_fraction=clipped / len(norms) if norms else None,
            skipped_batches=skipped,
        )

    def _checkpoint(
        self,
        epoch: int,
        *,
        val_loss: float,
        rollout: RolloutScore | None,
        bucket: BucketRolloutScore | None,
        selected_metric: str,
        selected_score: float,
        is_best: bool,
    ) -> None:
        self._progress["cpu_execution"] = self._execution_identity(self.cfg)
        super()._checkpoint(
            epoch,
            val_loss=val_loss,
            rollout=rollout,
            bucket=bucket,
            selected_metric=selected_metric,
            selected_score=selected_score,
            is_best=is_best,
        )
