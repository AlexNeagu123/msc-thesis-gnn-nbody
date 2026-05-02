# Repository Architecture: Responsibility Boundaries

Author: Alexandru Neagu

## Purpose

This document describes how the thesis implementation is organized and why the current separation of files matters. The product specification explains what the software does. This document explains where each responsibility belongs.

The architecture follows one rule:

> Orchestration coordinates work, typed contracts define shape, I/O modules own persistence, numeric modules compute values, and notebooks visualize or launch experiments.

This matters because the repository is now an experimental evidence pipeline. When a thesis table cites a number, the path from configuration, to checkpoint, to `metrics.json`, to report should be easy to trace.

## Repository Map

```text
configs/      experiment contracts
data/         trajectory generation, HDF5 I/O, dataset loading
models/       EGNN and HGNN architectures
training/     training loop, checkpoints, diagnostics, sweeps
evaluation/   official metrics, reports, and diagnostic plots
colab/        remote execution launcher
docs/         canonical documentation
utils.py      small shared utilities
```

Generated artifacts are intentionally separate:

```text
data/output/          generated HDF5 datasets
checkpoints/          model checkpoints
logs/                 training metrics and diagnostics
results/evaluation/   local official evaluation reports
runs/                 downloaded or Colab-produced run folders
```

These artifact roots are ignored by git. They are thesis evidence, but not source code.

## Dependency Direction

The intended direction is:

```text
configs -> data -> models -> training -> evaluation -> reports / notebooks / Colab
```

The main restrictions are:

- `models/` should not depend on `training/` or `evaluation/`.
- `_types.py` files define contracts.
- `_io.py` files own filesystem and serialization boundaries.
- `evaluation/metrics.py` stays numeric and free of plotting, JSON, CSV, and notebook dependencies.
- `evaluation/plots.py` owns matplotlib and IPython-facing diagnostics.
- CLI modules coordinate existing components; they should not become schema owners.

This is the core of the refactor: the code is split by responsibility, not by file-size convenience.

## Module Ownership

| Module | Owns |
| --- | --- |
| `configs/` | YAML definitions for data generation and model training. |
| `data/` | Simulation parameters, generated trajectory schema, HDF5 read/write, and transition dataset loading. |
| `models/` | Pure PyTorch architecture definitions for EGNN and HGNN. |
| `training/` | Optimization, checkpoint lifecycle, metrics CSVs, diagnostics, scaling runs, and sweeps. |
| `evaluation/` | Official numeric evidence: report schema, metrics, rollout evaluation, energy drift, and markdown aggregation. |
| `colab/` | One-run remote launcher for `(MODEL, N_TRAIN)` experiments with Drive persistence. |
| `docs/` | Canonical documentation: product behavior and repository architecture. |

## File-Level Contracts

### Data

| File | Responsibility |
| --- | --- |
| `data/_types.py` | `DataGenConfig`, simulation parameters, split config, trajectory bundle, trajectory metadata. |
| `data/_io.py` | Data YAML loading and HDF5 trajectory persistence. |
| `data/generate.py` | REBOUND-based trajectory generation. |
| `data/dataset.py` | Conversion from trajectories to one-step supervised transitions. |

The HDF5 schema belongs only in `data/_io.py`. Training and evaluation consume typed loaders instead of reaching into raw HDF5 keys.

### Models

| File | Responsibility |
| --- | --- |
| `models/egnn.py` | E(n)-equivariant graph layers and direct next-state prediction. |
| `models/hgnn.py` | Kinetic branch, potential branch, learned Hamiltonian, and leapfrog-style step. |

Model files define the differentiable functions. They should not know where checkpoints, logs, or evaluation reports live.

### Training

| File | Responsibility |
| --- | --- |
| `training/_types.py` | Training config, checkpoint schema, training result, epoch metrics. |
| `training/_io.py` | Config loading, checkpoint read/write, metrics CSV read/write boundary. |
| `training/train.py` | Main trainer and CLI. |
| `training/diagnostics.py` | Outlier batch diagnostics. |
| `training/scaling.py` | Data-scaling training orchestration. |
| `training/sweep.py` | EGNN noise-factor and learning-rate sweep. |

Training owns optimization. It does not own official thesis metrics.

### Evaluation

| File | Responsibility |
| --- | --- |
| `evaluation/_types.py` | Evaluation report schema and flattened summary row. |
| `evaluation/_io.py` | `metrics.json` and `summary.csv` persistence. |
| `evaluation/metrics.py` | Single-step, rollout, divergence, distance, and energy computations. |
| `evaluation/evaluate.py` | Checkpoint evaluation and report construction. |
| `evaluation/scaling_report.py` | Markdown aggregation across evaluated runs. |
| `evaluation/plots.py` | Notebook diagnostics and animations. |

Evaluation owns the official numbers. `metrics.json` is the canonical artifact; `summary.csv` is a table-friendly projection.

## Documentation Ownership

The repository has one README policy:

- Root `README.md` is the entry point.
- Durable details live in `docs/`.
- Module-level READMEs are avoided unless there is a strong reason.

Canonical documents:

| Document | Purpose |
| --- | --- |
| `docs/product-specification.md` | Thesis framing, capabilities, commands, dataset contract, evaluation contract, references. |
| `docs/repository-architecture.md` | File structure, responsibility boundaries, artifact policy, maintenance rules. |

After each larger structural or behavioral change, explicitly check whether the root README or canonical docs need updates.

## Test Placement

Tests live next to the source they protect:

```text
data/test_*.py
models/test_*.py
training/test_*.py
evaluation/test_*.py
```

This keeps ownership visible. If a contract changes, the nearest tests should explain what behavior must still hold.

## Boundary Rules

The guardrails are:

- Add dataclasses and schema objects to `_types.py`.
- Add JSON, CSV, HDF5, YAML, and checkpoint persistence to `_io.py`.
- Keep model files free of filesystem, training-loop, and evaluation-report concerns.
- Keep numeric metric code free of plotting and notebook dependencies.
- Keep notebooks as launchers or diagnostics, not hidden product logic.
- Keep generated artifacts out of git.
- Tie thesis numbers to preserved `metrics.json`, `summary.csv`, or generated markdown reports.
- Add tests at the ownership level of the change.

These rules are practical. They prevent the evidence pipeline from becoming a collection of scripts that are hard to audit.

## Extension Rules

Add a new model:

1. Implement it under `models/`.
2. Add model construction in orchestration.
3. Add a config under `configs/`.
4. Add model tests.
5. Confirm training and evaluation artifacts remain compatible.

Add a new metric:

1. Implement numeric computation in `evaluation/metrics.py`.
2. Add report fields in `evaluation/_types.py`.
3. Add construction in `evaluation/evaluate.py`.
4. Persist through `evaluation/_io.py`.
5. Add metric and serialization tests.

Add a new artifact:

1. Define its shape in the relevant `_types.py`.
2. Add read/write behavior in the relevant `_io.py`.
3. Call it from orchestration.
4. Test the round trip.

The pattern is always the same: define the contract, centralize persistence, keep orchestration thin, and test the boundary.

## Generated Artifact Policy

Ignored artifact roots:

- `data/output/`
- `checkpoints/`
- `logs/`
- `runs/`
- `evaluation/runs/`
- `results/evaluation/`

Preserve important artifacts in Drive or local experiment storage when they are needed for thesis evidence. Commit the code, configs, docs, and reports that explain how those artifacts were produced.
