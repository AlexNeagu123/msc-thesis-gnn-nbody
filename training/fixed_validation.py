"""Fixed-horizon, batched validation for canonical HGNN checkpoint selection.

This is a DEVELOPMENT selection protocol, not the final Daniela evaluation report.
Primary: position MAE. Secondary: the legacy-form baseline-normalized state-MSE
score, explicitly using ordinary medians and rejecting any divergent trajectory.
"""

import hashlib
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from data._io import read_states
from evaluation.rk4_metrics import WINDOWS, summarize
from models.baselines import (
    ConstantVelocityBaseline,
    MeanStateBaseline,
    MeanVelocityBaseline,
    PersistenceBaseline,
)
from training.rollout_score import compute_rollout_score

METRIC = "fixed_position_mae"
PROTOCOL = "canonical-validation-v1"


def file_sha256(path: Path) -> str:
    """Hash a source without reading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FixedRolloutEvaluator:
    """Freeze validation starts, reference and baseline envelope across curriculum stages."""

    def __init__(
        self,
        val_path: str | Path,
        train_path: str | Path,
        dt: float,
        device: torch.device,
        *,
        steps: int | None = None,
        batch_size: int = 64,
        n_train_trajectories: int | None = None,
        expected_train_sha256: str | None = None,
        expected_val_sha256: str | None = None,
    ) -> None:
        """Store settings; the fixed reference/baselines are initialized on first score."""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("validation batch_size must be a positive integer")
        if steps is not None and (
            not isinstance(steps, int) or isinstance(steps, bool) or steps < 1
        ):
            raise ValueError("validation steps must be a positive integer or None")
        self.val_path, self.train_path = Path(val_path), Path(train_path)
        self.dt, self.device = dt, device
        self.steps, self.batch_size = steps, batch_size
        self.n_train_trajectories = n_train_trajectories
        self.expected_train_sha256 = expected_train_sha256
        self.expected_val_sha256 = expected_val_sha256
        self._reference: np.ndarray | None = None
        self._baseline_reports: dict | None = None
        self._envelope: np.ndarray | None = None
        self.metadata: dict = {}

    @property
    def reference(self) -> np.ndarray:
        """Load ALL validation simulations once; each starts at its saved first frame."""
        if self._reference is None:
            before = file_sha256(self.val_path)
            if self.expected_val_sha256 is not None and before != self.expected_val_sha256:
                raise RuntimeError("validation source changed since training setup")
            reference = read_states(self.val_path)
            if (
                reference.ndim != 4
                or reference.shape[0] < 1
                or reference.shape[1] < 2
                or reference.shape[2] < 2
                or reference.shape[3] != 5
                or not np.isfinite(reference).all()
            ):
                raise ValueError(
                    "expected finite validation trajectories (runs,frames,bodies>=2,5)"
                )
            available = reference.shape[1] - 1
            steps = available if self.steps is None else self.steps
            if steps > available:
                raise ValueError(
                    f"validation steps {steps} exceed available future states {available}"
                )
            reference = reference[:, : steps + 1]
            mass = reference[:, :1, :, 4]
            if np.any(mass <= 0) or not np.all(reference[..., 4] == mass):
                raise ValueError("validation masses must be positive and constant")
            with h5py.File(self.val_path, "r") as handle:
                ids = (
                    handle["simulation_ids"][:].tolist()
                    if "simulation_ids" in handle
                    else list(range(len(reference)))
                )
                if len(ids) != len(reference) or len(set(ids)) != len(ids):
                    raise ValueError("validation simulation IDs must be unique and match run count")
                if "times" in handle:
                    times = handle["times"][:]
                    if times.shape not in ((available + 1,), (len(reference), available + 1)):
                        raise ValueError("validation time grid shape disagrees with trajectories")
                    if not np.allclose(np.diff(times, axis=-1), self.dt, rtol=1e-7, atol=1e-12):
                        raise ValueError("validation time grid disagrees with dt")
            if file_sha256(self.val_path) != before:
                raise RuntimeError("validation source changed while reading")
            reference.setflags(write=False)
            self._reference = reference
            self.metadata = {
                "protocol": PROTOCOL,
                "selection_metric": METRIC,
                "val_path": str(self.val_path.resolve()),
                "val_sha256": before,
                "simulation_ids": ids,
                "runs": len(reference),
                "dt": self.dt,
                "steps": steps,
                "future_only": True,
                "initial_frame_index": 0,
                "reference_dtype": str(reference.dtype),
                "prediction_dtype": "float32",
                "torch_version": torch.__version__,
                "device": str(self.device),
                "nonfinite_policy": "reject_checkpoint_if_any_dynamic_value_nonfinite",
                "secondary_score": "legacy_log_ratio_formula_with_ordinary_median_and_failures_rejected",
            }
        return self._reference

    def _predict(self, model: nn.Module) -> np.ndarray:
        reference = self.reference
        predicted = np.empty(
            (len(reference), reference.shape[1] - 1, *reference.shape[2:]), dtype=np.float32
        )
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for start in range(0, len(reference), self.batch_size):
                    stop = min(start + self.batch_size, len(reference))
                    state = (
                        torch.from_numpy(reference[start:stop, 0].copy()).float().to(self.device)
                    )
                    batch = torch.empty(
                        (stop - start, *predicted.shape[1:]),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    for step in range(predicted.shape[1]):
                        state = model(state)
                        if state.shape != (stop - start, *reference.shape[2:]):
                            raise ValueError(
                                "model output shape disagrees with external state contract"
                            )
                        batch[:, step] = state
                    # Transfer/check once per trajectory batch, not once per GPU timestep.
                    snapshot = batch.cpu().numpy()
                    mass = reference[start:stop, :1, :, 4].astype(np.float32)
                    if not np.array_equal(
                        snapshot[..., 4], np.broadcast_to(mass, snapshot[..., 4].shape)
                    ):
                        raise ValueError("validation predictions must preserve masses exactly")
                    predicted[start:stop] = snapshot
        finally:
            model.train(was_training)
        return predicted

    def _summarize(self, predicted: np.ndarray) -> tuple[dict, np.ndarray]:
        reference = self.reference[:, 1:]
        prediction = predicted.astype(np.float64)
        failed = ~np.isfinite(prediction[..., :4]).all(axis=(2, 3))
        failed_runs = failed.any(axis=1)
        with np.errstate(over="ignore", invalid="ignore"):
            position_error = np.abs(prediction[..., :2] - reference[..., :2])
            state_mse_per_run_step = np.square(prediction[..., :4] - reference[..., :4]).mean(
                axis=(2, 3)
            )
            curve = np.median(state_mse_per_run_step, axis=0)
            mae = summarize(position_error.mean(axis=(1, 2, 3)))
            state_mse = float(state_mse_per_run_step.mean())
        windows = {}
        for name, (begin, end) in WINDOWS.items():
            stop = min(end, predicted.shape[1])
            count = max(0, stop - begin)
            windows[name] = {
                "start": begin,
                "stop": stop if count else begin,
                "available_steps": count,
                "expected_steps": end - begin,
                "complete": count == end - begin,
                "mae": summarize(position_error[:, begin:stop].mean(axis=(1, 2, 3)))
                if count
                else None,
            }
        finite = not failed_runs.any() and np.isfinite(mae["mean"]) and np.isfinite(state_mse)
        return {
            "position_mae": mae,
            "state4_mse": state_mse,
            "windows": windows,
            "failed_runs": int(failed_runs.sum()),
            "nonfinite_dynamic_values": int((~np.isfinite(prediction[..., :4])).sum()),
            "first_failed_step": [
                int(np.flatnonzero(row)[0] + 1) if row.any() else None for row in failed
            ],
            "selection_score": float(mae["mean"]) if finite else float("inf"),
            "eligible": bool(finite),
        }, curve

    def _prepare_baselines(self) -> None:
        if self._baseline_reports is not None:
            return
        # Freeze reference first, independent of which model is scored first.
        _ = self.reference
        before = file_sha256(self.train_path)
        if self.expected_train_sha256 is not None and before != self.expected_train_sha256:
            raise RuntimeError("training source changed since training setup")
        states = torch.from_numpy(read_states(self.train_path)).float()
        if self.n_train_trajectories is not None:
            if not 1 <= self.n_train_trajectories <= len(states):
                raise ValueError("invalid baseline training trajectory count")
            states = states[: self.n_train_trajectories]
        if file_sha256(self.train_path) != before:
            raise RuntimeError("training source changed while fitting baselines")
        self.metadata.update(
            {
                "train_path": str(self.train_path.resolve()),
                "train_sha256": before,
                "baseline_fit_trajectories": len(states),
                "baseline_fit": "training_only_all_saved_frames",
            }
        )
        baselines = {
            "persistence": PersistenceBaseline(),
            "constant_velocity": ConstantVelocityBaseline(self.dt),
            "mean_velocity": MeanVelocityBaseline(self.dt, states[..., 2:4].reshape(-1, 2).mean(0)),
            "mean_state": MeanStateBaseline(states[..., :2].reshape(-1, 2).mean(0)),
        }
        reports, curves = {}, []
        for name, baseline in baselines.items():
            reports[name], curve = self._summarize(self._predict(baseline.to(self.device)))
            if not reports[name]["eligible"]:
                raise ValueError(f"nonfinite validation baseline: {name}")
            curves.append(curve)
        self._envelope = np.stack(curves).min(axis=0)
        self._baseline_reports = reports

    def score(self, model: nn.Module) -> dict:
        """Return primary MAE, window metrics, failures and secondary baseline comparison."""
        self._prepare_baselines()
        report, curve = self._summarize(self._predict(model))
        secondary = compute_rollout_score(curve, self._envelope)
        report["baseline_normalized_state_mse"] = {
            "score": secondary.score if report["eligible"] else float("inf"),
            "dominance_horizon": secondary.dominance_horizon,
            "fraction_beating_baseline": secondary.fraction_beating_baseline,
            "final_ratio": secondary.final_ratio,
            "ratios_at_step": secondary.ratios_at_step,
        }
        return {
            "metadata": self.metadata.copy(),
            "model": report,
            "baselines": self._baseline_reports,
        }
