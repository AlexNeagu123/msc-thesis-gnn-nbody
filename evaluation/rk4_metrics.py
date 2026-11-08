"""NumPy implementation of Daniela's repository-defined evaluation reductions.

Primary compatibility energy intentionally compares unsoftened predicted energy
against softened initial energy, as 3_testing.ipynb currently does. Corrected
physical energy is reported separately, never substituted into legacy metrics.
"""

from __future__ import annotations

import numpy as np

WINDOWS = {"0-250": (0, 250), "250-1000": (250, 1000), "1000-2000": (1000, 2000)}


def invariants(states: np.ndarray, *, epsilon: float = 0.0) -> dict[str, np.ndarray]:
    """Calculate energy, linear momentum vector and signed planar angular momentum.

    Accept (..., body, [x,y,vx,vy,m]); G=1 matches the reference notebook.
    Zero epsilon deliberately retains singular/nonfinite unsoftened behaviour.
    """
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    pos, vel, mass = states[..., :2], states[..., 2:4], states[..., 4]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        energy = (0.5 * mass * np.square(vel).sum(axis=-1)).sum(axis=-1)
        for i in range(states.shape[-2]):
            for j in range(i + 1, states.shape[-2]):
                radius = np.sqrt(
                    np.square(pos[..., i, :] - pos[..., j, :]).sum(axis=-1) + epsilon**2
                )
                energy -= mass[..., i] * mass[..., j] / radius
        momentum = (mass[..., None] * vel).sum(axis=-2)
        angular = (mass * (pos[..., 0] * vel[..., 1] - pos[..., 1] * vel[..., 0])).sum(axis=-1)
    return {"energy": energy, "linear_momentum": momentum, "angular_momentum": angular}


def summarize(values: np.ndarray) -> dict:
    """Use population SD and ordinary mean, never silently drop divergent runs."""
    values = np.asarray(values, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        mean, std = float(np.mean(values)), float(np.std(values, ddof=0))
    return {
        "mean": mean,
        "std": std,
        "per_run": values.tolist(),
        "n_runs": len(values),
        "finite_runs": int(np.isfinite(values).sum()),
    }


def _temporal_summary(errors: np.ndarray, *, include_series: bool) -> dict:
    with np.errstate(over="ignore", invalid="ignore"):
        result = {
            "temporal_mean": summarize(errors.mean(axis=1)),
            "temporal_max": summarize(errors.max(axis=1)),
        }
    if include_series:
        result["per_run_series"] = errors.tolist()
    return result


def build_daniela_report(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    epsilon: float = 0.05,
    metadata: dict | None = None,
    include_series: bool = False,
    runtime_seconds: np.ndarray | None = None,
    runtime_scope: str | None = None,
) -> dict:
    """Score future predictions against a reference that INCLUDES the initial state.

    reference: (run, steps+1, body, 5), predicted: (run, steps, body, 5).
    Metric inputs are promoted to float64 for stable reductions. Missing windows
    remain explicitly empty, not mislabeled as complete. Nonfinite predictions
    propagate into summaries; strict JSON serialization maps these to null.
    """
    reference = np.asarray(reference, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if (
        reference.ndim != 4
        or predicted.ndim != 4
        or reference.shape[0] < 1
        or reference.shape[1] < 2
        or reference.shape[2] < 2
        or reference.shape[3] != 5
        or predicted.shape != (reference.shape[0], reference.shape[1] - 1, reference.shape[2], 5)
    ):
        raise ValueError(
            "expected reference (runs,steps+1,bodies,5) and predicted (runs,steps,bodies,5)"
        )
    if not np.isfinite(reference).all():
        raise ValueError("reference must be finite")
    masses = reference[:, :1, :, 4]
    if (
        np.any(masses <= 0)
        or not np.allclose(reference[..., 4], masses, rtol=0, atol=1e-12)
        or not np.allclose(predicted[..., 4], masses, rtol=1e-6, atol=1e-12)
    ):
        raise ValueError("positive, constant masses must be preserved in reference and predictions")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    n_runs, n_steps = predicted.shape[:2]
    trajectory = {}
    with np.errstate(over="ignore", invalid="ignore"):
        delta = predicted[..., :2] - reference[:, 1:, :, :2]
        for name, (start, end) in WINDOWS.items():
            actual_end = min(end, n_steps)
            count = max(0, actual_end - start)
            if count:
                section = delta[:, start:actual_end]
                mae = np.abs(section).mean(axis=(1, 2, 3))
                mse = np.square(section).mean(axis=(1, 2, 3))
            else:
                mae = mse = np.full(n_runs, np.nan)
            trajectory[name] = {
                "requested_future_steps": [start + 1, end],
                "n_steps_scored": count,
                "complete_window": count == end - start,
                "mae": summarize(mae),
                "mse": summarize(mse),
            }
    initial = invariants(reference[:, 0], epsilon=epsilon)
    legacy = invariants(predicted, epsilon=0.0)
    softened = invariants(predicted, epsilon=epsilon)
    with np.errstate(over="ignore", invalid="ignore"):
        energy_error = np.abs(legacy["energy"] - initial["energy"][:, None])
        soft_error = np.abs(softened["energy"] - initial["energy"][:, None])
        momentum_error = np.linalg.norm(
            legacy["linear_momentum"] - initial["linear_momentum"][:, None], axis=-1
        )
        angular_error = np.abs(legacy["angular_momentum"] - initial["angular_momentum"][:, None])
    errors = {
        "energy": energy_error,
        "linear_momentum": momentum_error,
        "angular_momentum": angular_error,
    }
    relative = {}
    for name, error in errors.items():
        magnitude = (
            np.linalg.norm(initial[name], axis=-1)
            if name == "linear_momentum"
            else np.abs(initial[name])
        )
        denominator = np.where(magnitude > 1e-15, magnitude, 1.0)
        relative[name] = _temporal_summary(
            error / denominator[:, None] * 100, include_series=include_series
        )
    finite = np.isfinite(predicted).all(axis=(2, 3))
    failures = [
        {"run_index": run, "first_nonfinite_future_step": int(np.flatnonzero(~valid)[0]) + 1}
        for run, valid in enumerate(finite)
        if not valid.all()
    ]
    runtime = None
    if runtime_seconds is not None:
        seconds = np.asarray(runtime_seconds, dtype=np.float64)
        if (
            seconds.shape != (n_runs,)
            or not np.isfinite(seconds).all()
            or np.any(seconds < 0)
            or not runtime_scope
        ):
            raise ValueError(
                "runtime requires one finite nonnegative value per run and an explicit scope"
            )
        runtime = {"seconds": summarize(seconds), "scope": runtime_scope}
    return {
        "schema_version": 1,
        "protocol": "daniela-repository-compat-v1",
        "metadata": dict(metadata or {}),
        "definitions": {
            "position_error": "componentwise x/y only; mean over time window, bodies and coordinates, then mean/std over runs",
            "std": "population std, ddof=0; no finite-only filtering",
            "energy_legacy": "abs(E_unsoftened(predicted_t)-E_softened(reference_initial)); G=1",
            "energy_physical": "abs(E_softened(predicted_t)-E_softened(reference_initial)); G=1",
            "epsilon": epsilon,
            "linear_momentum": "Euclidean norm of vector difference from reference initial momentum",
            "angular_momentum": "absolute signed planar-angular-momentum difference about original origin",
            "conservation": "temporal mean/max per run, then ensemble mean/std (max is mean of per-run maxima)",
            "relative_percent": "100*abs_error/initial_magnitude; denominator=1 when magnitude<=1e-15, as notebook",
            "alignment": "reference[:,0] initial; prediction[:,k] compares to reference[:,k+1]",
            "precision": "metrics float64; model-inference precision belongs in metadata",
            "nonfinite": "propagates through reductions; strict JSON uses null with explicit failure/finite counts",
        },
        "shape": {"n_runs": n_runs, "n_future_steps": n_steps, "n_bodies": predicted.shape[2]},
        "trajectory": trajectory,
        "conservation_legacy": {
            name: _temporal_summary(error, include_series=include_series)
            for name, error in errors.items()
        },
        "relative_percent_legacy": relative,
        "physical_softened_energy": _temporal_summary(soft_error, include_series=include_series),
        "runtime": runtime,
        "failures": {
            "nonfinite_runs": failures,
            "n_nonfinite_runs": len(failures),
            "finite_state_fraction_per_step": finite.mean(axis=0).tolist(),
        },
    }
