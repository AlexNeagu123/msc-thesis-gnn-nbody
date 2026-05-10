"""Typed contracts for the data pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationParams:
    """Physical and numerical parameters for a single trajectory."""

    n_particles: int
    t_end: float
    dt: float
    G: float
    mass: float
    min_distance: float
    max_position: float
    pos_scale: float
    vel_scale: float


@dataclass
class SplitConfig:
    """Configuration for a single data split (train, val, or test)."""

    name: str
    n_trajectories: int
    path: str
    seed: int


@dataclass
class DataGenConfig:
    """Top-level data generation configuration.

    `stratified` is None for uniform generation; when present, it has
    been fully validated by `StratifiedConfig.__post_init__`.
    """

    simulation: SimulationParams
    splits: list[SplitConfig]
    stratified: StratifiedConfig | None = None

    @staticmethod
    def from_dict(d: dict) -> DataGenConfig:
        """Build a DataGenConfig from a parsed YAML dict."""
        simulation = SimulationParams(
            n_particles=d["n_particles"],
            t_end=d["t_end"],
            dt=d["dt"],
            G=d["G"],
            mass=d["mass"],
            min_distance=d["min_distance"],
            max_position=d.get("max_position", float("inf")),
            pos_scale=d["pos_scale"],
            vel_scale=d["vel_scale"],
        )

        seed = d["seed"]
        splits = [
            SplitConfig("train", d["n_train"], d["train_path"], seed),
            SplitConfig("val", d["n_val"], d["val_path"], seed + 1000),
            SplitConfig("test", d["n_test"], d["test_path"], seed + 2000),
        ]

        stratified = StratifiedConfig.from_dict(d["stratified"]) if "stratified" in d else None

        return DataGenConfig(simulation=simulation, splits=splits, stratified=stratified)


@dataclass
class TrajectoryMetadata:
    """Provenance attributes attached to each trajectory HDF5 file."""

    n_trajectories: int
    n_particles: int
    n_steps: int
    t_end: float
    dt: float
    G: float
    mass: float
    min_distance: float
    pos_scale: float
    vel_scale: float
    seed: int
    rejection_rate: float


@dataclass(frozen=True)
class EncounterBin:
    """One bucket in a stratified encounter-severity binning.

    A trajectory belongs to this bin when its true minimum pairwise
    distance d_min satisfies `lo <= d_min < hi` (half-open interval).
    Top-of-range bins use `hi=float("inf")` to extend to the half-line.
    """

    name: str
    lo: float
    hi: float


@dataclass(frozen=True)
class StratifiedConfig:
    """Stratified-generation contract: bin definitions + per-split weights.

    Constructing this dataclass succeeds only when the bins form a
    contiguous half-open partition of `[0, inf)` and every distribution
    keys exactly match the bin names with finite, non-negative weights
    that sum positive. Block 4's generator can therefore treat the
    object as fully validated and complete.

    `max_attempts` optionally overrides the per-split candidate-attempt
    cap. None falls back to the generator's safe default (currently
    `max(10000, n_trajectories * 2000)`); set explicitly when a target
    bin is rare under the simulator's natural distribution.
    """

    bins: tuple[EncounterBin, ...]
    train_distribution: dict[str, float]
    val_distribution: dict[str, float]
    test_distribution: dict[str, float]
    max_attempts: int | None = None

    def __post_init__(self) -> None:
        """Enforce structural and per-split contracts."""
        self._validate_bins()
        self._validate_distribution("train_distribution", self.train_distribution)
        self._validate_distribution("val_distribution", self.val_distribution)
        self._validate_distribution("test_distribution", self.test_distribution)
        if self.max_attempts is not None and (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            # bools subclass int but `True` shouldn't read as a cap of 1; floats
            # (incl. nan/+inf) and strings would survive `< 1` (NaN comparisons are
            # always False), so reject them at the type level too.
            msg = f"max_attempts must be an integer >= 1 when set; got {self.max_attempts!r}"
            raise ValueError(msg)

    def _validate_bins(self) -> None:
        if not self.bins:
            msg = "bins must be non-empty"
            raise ValueError(msg)

        names = [b.name for b in self.bins]
        if len(set(names)) != len(names):
            msg = f"bin names must be unique; got {names}"
            raise ValueError(msg)

        for b in self.bins:
            if not isinstance(b.name, str):
                msg = f"bin names must be strings; got {type(b.name).__name__}"
                raise ValueError(msg)
            if not b.name.strip():
                msg = f"bin names must be non-blank; got {b.name!r}"
                raise ValueError(msg)
            if not math.isfinite(b.lo):
                msg = f"bin {b.name!r} has non-finite lo={b.lo}"
                raise ValueError(msg)
            if math.isnan(b.hi):
                msg = f"bin {b.name!r} has nan hi"
                raise ValueError(msg)
            if not b.lo < b.hi:
                msg = f"bin {b.name!r}: lo={b.lo} must be < hi={b.hi}"
                raise ValueError(msg)

        if self.bins[0].lo != 0.0:
            msg = f"first bin must start at 0.0; got lo={self.bins[0].lo}"
            raise ValueError(msg)

        if self.bins[-1].hi != float("inf"):
            msg = f"last bin must end at +inf; got hi={self.bins[-1].hi}"
            raise ValueError(msg)

        for a, b in zip(self.bins, self.bins[1:], strict=False):
            if a.hi != b.lo:
                msg = (
                    f"bins must be contiguous; gap between {a.name!r} (hi={a.hi}) "
                    f"and {b.name!r} (lo={b.lo})"
                )
                raise ValueError(msg)

    def _validate_distribution(self, label: str, dist: dict[str, float]) -> None:
        bin_names = {b.name for b in self.bins}
        if set(dist) != bin_names:
            missing = sorted(bin_names - set(dist))
            extra = sorted(set(dist) - bin_names)
            msg = f"{label} keys must match bin names exactly; missing={missing}, extra={extra}"
            raise ValueError(msg)
        for name, w in dist.items():
            if not math.isfinite(w):
                msg = f"{label}[{name!r}] must be finite; got {w}"
                raise ValueError(msg)
            if w < 0:
                msg = f"{label}[{name!r}] must be >= 0; got {w}"
                raise ValueError(msg)
        if sum(dist.values()) <= 0:
            msg = f"{label} weights must sum to > 0; got {dist}"
            raise ValueError(msg)

    @staticmethod
    def from_dict(d: dict) -> StratifiedConfig | None:
        """Parse a stratified-section dict; return None when disabled or absent.

        The `enabled` key acts as the on/off switch and is consumed here
        so downstream code only ever sees a fully validated config.
        """
        if not isinstance(d, dict):
            msg = f"stratified must be a mapping; got {type(d).__name__}"
            raise ValueError(msg)

        if not d.get("enabled", False):
            return None

        bins_raw = d.get("bins")
        if not bins_raw:
            msg = "stratified.bins is required when enabled"
            raise ValueError(msg)

        bins = tuple(StratifiedConfig._parse_bin(i, raw) for i, raw in enumerate(bins_raw))

        for key in ("train_distribution", "val_distribution", "test_distribution"):
            if key not in d:
                msg = f"stratified requires {key} when enabled"
                raise ValueError(msg)

        return StratifiedConfig(
            bins=bins,
            train_distribution=dict(d["train_distribution"]),
            val_distribution=dict(d["val_distribution"]),
            test_distribution=dict(d["test_distribution"]),
            max_attempts=d.get("max_attempts"),
        )

    @staticmethod
    def _parse_bin(idx: int, raw: object) -> EncounterBin:
        """Build one EncounterBin from a YAML bin entry with explicit shape errors."""
        if not isinstance(raw, dict):
            msg = f"stratified.bins[{idx}] must be a mapping; got {type(raw).__name__}"
            raise ValueError(msg)
        for key in ("name", "lo", "hi"):
            if key not in raw:
                msg = f"stratified.bins[{idx}] missing required key {key!r}"
                raise ValueError(msg)
        try:
            lo = float(raw["lo"])
            hi = float(raw["hi"])
        except (TypeError, ValueError) as e:
            msg = f"stratified.bins[{idx}] has non-numeric lo/hi: {e}"
            raise ValueError(msg) from e
        return EncounterBin(name=raw["name"], lo=lo, hi=hi)


@dataclass
class Trajectories:
    """Typed bundle for the contents of one trajectory HDF5 file.

    `metadata` is optional because test fixtures and older files may
    omit the metadata group; production files always include it.

    The four `encounter_*` / `min_pairwise_distance` / `encounter_bins`
    fields form an atomic stratification group: either all four are
    populated (a stratified dataset) or all four are None (uniform /
    non-stratified). The persistence boundary in `data/_io.py` enforces
    this contract; in-memory construction does not.
    """

    states: np.ndarray  # (n_trajectories, n_steps, n_particles, 5)
    energies: np.ndarray  # (n_trajectories, n_steps)
    metadata: TrajectoryMetadata | None = None
    encounter_bin_id: np.ndarray | None = None  # (n_trajectories,) int
    encounter_bin_name: np.ndarray | None = None  # (n_trajectories,) str
    min_pairwise_distance: np.ndarray | None = None  # (n_trajectories,) float
    encounter_bins: tuple[EncounterBin, ...] | None = None
