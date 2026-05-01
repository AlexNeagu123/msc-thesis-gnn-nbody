# N-Body GNN Workbench: Repository Architecture

Author: Alexandru Neagu

## Revision

| Revision | Date | Notes |
| --- | --- | --- |
| 1 | 2026-05-01 | Initial architecture and responsibility map for the repository. |

## Overview

This document describes how the repository is structured and why the files are separated the way they are. The product specification explains what the workbench can do and how to invoke it. This document explains where each responsibility lives so future changes stay local and the codebase remains readable.

The central design rule is simple: orchestration files should coordinate work, typed contracts should define data shape, I/O files should own persistence, and numeric modules should stay free of visualization or filesystem concerns.

This split matters for the thesis because the project is no longer just exploratory code. It now has to support repeatable experiments, preserve official metrics, and remain understandable when results are discussed months after a run was produced.

## Architecture Schematic

The repository is organized as a small experiment pipeline:

```text
configs/      -> experiment contracts
data/         -> trajectory generation and dataset loading
models/       -> EGNN and HGNN model definitions
training/     -> training orchestration, checkpointing, and sweeps
evaluation/   -> official metrics, evaluation reports, and diagnostics
colab/        -> remote execution launcher
docs/         -> canonical documentation
utils.py      -> small shared utilities
```

Generated artifacts are intentionally outside the source modules:

```text
data/output/          generated HDF5 datasets
checkpoints/          model checkpoints
logs/                 training metrics and diagnostics
results/evaluation/   local official evaluation reports
runs/                 downloaded or Colab-produced run folders
```

These artifact directories are ignored by git. The source tree should describe how to reproduce results, not store every large result file.

## Dependency Direction

The intended dependency direction is:

```text
configs
  -> data
  -> models
  -> training
  -> evaluation
  -> reports / notebooks / Colab
```

In practice:

- `models/` should not import from `training/` or `evaluation/`.
- `data/_io.py` owns HDF5 access so model, training, and evaluation code do not touch raw HDF5 keys.
- `training/_io.py` owns training config, checkpoint, and metrics-CSV persistence.
- `evaluation/_io.py` owns `metrics.json` and `summary.csv` persistence.
- `evaluation/metrics.py` owns pure numeric metric computation.
- `evaluation/plots.py` owns matplotlib and notebook visualization.
- CLI modules coordinate work but should not become schema or persistence owners.

This keeps the pipeline testable. If a report field changes, the change belongs in evaluation types and I/O. If a checkpoint field changes, the change belongs in training types and I/O. If a plot changes, the official evaluator should not need to change.

## Directory Responsibilities

| Path | Responsibility |
| --- | --- |
| `configs/` | YAML experiment contracts for data generation and model training. |
| `data/` | Simulation, HDF5 trajectory I/O, typed data contracts, and PyTorch dataset conversion. |
| `models/` | Neural network architectures only. |
| `training/` | Training loop, checkpoint persistence, metrics logging, diagnostics, and experiment sweeps. |
| `evaluation/` | Official numeric evaluation, report schema, report I/O, metric computation, plotting, and scaling report generation. |
| `colab/` | Notebook launcher for GPU runs with Google Drive persistence. |
| `docs/` | Canonical product and architecture documentation. |
| `utils.py` | Small cross-cutting helpers that do not justify a package. |

## Data Module

The data module owns the trajectory contract from generation to training input.

| File | Owns |
| --- | --- |
| `data/_types.py` | Data-generation config dataclasses and typed trajectory bundles. |
| `data/_io.py` | YAML loading for data configs and HDF5 read/write functions. |
| `data/generate.py` | REBOUND simulation orchestration and dataset split generation. |
| `data/dataset.py` | PyTorch `Dataset` that converts trajectories into consecutive state-transition pairs. |
| `data/visualize.ipynb` | Exploratory inspection of generated trajectories. |

Design notes:

- HDF5 dataset names and metadata keys belong in `data/_io.py`.
- `data/generate.py` should produce `Trajectories` and delegate persistence to `write_trajectories`.
- `data/dataset.py` should read states through `read_states`, not through direct HDF5 access.
- Training-size slicing belongs in `NBodyDataset`, because scaling experiments are a data-loading concern.

## Model Module

The model module owns neural network definitions and should stay independent from experiment orchestration.

| File | Owns |
| --- | --- |
| `models/egnn.py` | EGNN layers and forward prediction interface. |
| `models/hgnn.py` | HGNN energy networks, Hamiltonian computation, and leapfrog forward step. |

Design notes:

- Model files should expose PyTorch modules that accept and return state tensors.
- Model files should not write checkpoints, read configs, produce plots, or know about evaluation reports.
- Normalization buffers such as `pos_std` and `vel_std` belong in the model because they affect forward computation.
- The mass channel remains part of the state interface so both model families share the same input and output shape.

## Training Module

The training module owns optimization and training artifacts.

| File | Owns |
| --- | --- |
| `training/_types.py` | Training config dataclasses, checkpoint schema, training result, and CSV metric row. |
| `training/_io.py` | Training config loading, checkpoint read/write, and metrics-CSV persistence. |
| `training/train.py` | Main training orchestration and CLI entry point. |
| `training/diagnostics.py` | Batch-level diagnostics for outlier losses and unstable batches. |
| `training/scaling.py` | Repeated training runs over multiple `n_train_trajectories` values. |
| `training/sweep.py` | EGNN learning-rate and noise-factor sweep. |

Design notes:

- `training/train.py` should coordinate the training lifecycle, not own serialization details.
- Checkpoint shape belongs in `Checkpoint` from `training/_types.py`.
- `training/_io.py` is the only module that should call `torch.save` or normalize legacy checkpoint dictionaries.
- Training metrics are for optimization monitoring. Official thesis metrics belong to evaluation.
- Diagnostics should observe training behavior without changing the model contract.

## Evaluation Module

The evaluation module owns official numeric evidence.

| File | Owns |
| --- | --- |
| `evaluation/_types.py` | Evaluation report schema, rollout metric containers, and flattened summary rows. |
| `evaluation/_io.py` | `metrics.json` and `summary.csv` read/write behavior. |
| `evaluation/metrics.py` | Pure numeric metrics: energy, rollouts, single-step error, rollout MSE, and closest-pair distances. |
| `evaluation/plots.py` | Notebook-only plots and animations. |
| `evaluation/evaluate.py` | Official checkpoint evaluation CLI and report construction. |
| `evaluation/scaling_report.py` | Markdown report generation across evaluated runs. |
| `evaluation/*_visual_diagnostics.ipynb` | Interactive model-specific rollout inspection. |

Design notes:

- `metrics.json` is the canonical evaluation artifact.
- `summary.csv` is a flattened convenience view of the same report.
- `evaluation/evaluate.py` should build reports from typed metric outputs and delegate writing to `evaluation/_io.py`.
- `evaluation/metrics.py` should not import matplotlib, IPython, JSON, or CSV.
- `evaluation/plots.py` can depend on matplotlib and IPython because it is notebook-facing.
- Scaling reports should read typed `EvaluationReport` objects through `read_evaluation_report`.

## Config Module

The `configs/` directory owns experiment configuration, not generated results.

| File | Owns |
| --- | --- |
| `configs/data.yaml` | Dataset generation paths and simulation parameters. |
| `configs/egnn.yaml` | EGNN model, data, training, scheduler, checkpointing, and logging settings. |
| `configs/hgnn.yaml` | HGNN model, data, training, scheduler, checkpointing, and logging settings. |

Design notes:

- Config files should remain small enough to review in a thesis context.
- Data-scaling changes should prefer `--n-train` when only the number of training trajectories changes.
- New model families should get their own config file only when the model contract differs.

## Colab Module

The Colab module owns remote execution convenience.

| File | Owns |
| --- | --- |
| `colab/train_colab.ipynb` | One-run launcher for `(MODEL, N_TRAIN)` training and evaluation on Colab. |

Design notes:

- The notebook should orchestrate repository commands, not reimplement training or evaluation logic.
- Google Drive paths should be explicit.
- Long-running outputs should persist to Drive.
- Runtime validation, such as checking that `train.h5` contains enough trajectories, belongs in the notebook because it protects expensive remote runs.

## Documentation Module

The `docs/` directory contains the canonical documentation.

| File | Owns |
| --- | --- |
| `docs/product-specification.md` | Product story, supported use cases, command interfaces, configuration contracts, artifact contracts, and operating model. |
| `docs/repository-architecture.md` | File structure, responsibility boundaries, module ownership, and extension rules. |

README ownership is intentionally unitary. The root README is the entry point, and durable details belong in these canonical documents. Avoid module-level READMEs unless there is a strong reason to add one.

After each larger structural or behavioral change, explicitly check whether the root README or canonical docs need updates. Documentation drift should be treated as part of the change, not as cleanup for later.

## Test Layout

Tests live next to the module they validate.

Examples:

- `data/test_io.py` validates HDF5 I/O.
- `data/test_dataset.py` validates trajectory-to-transition loading.
- `models/test_egnn.py` and `models/test_hgnn.py` validate model behavior.
- `training/test_io.py` and `training/test_train.py` validate training contracts.
- `evaluation/test_types.py`, `evaluation/test_io.py`, `evaluation/test_metrics.py`, and `evaluation/test_evaluate.py` validate report schemas, persistence, metrics, and evaluator behavior.

This layout makes ownership visible. If a file changes, the nearest tests usually describe the contract that must continue to hold.

## Boundary Rules

The following rules are the architectural guardrails for future changes:

- Put typed dataclasses in `_types.py`.
- Put filesystem and serialization code in `_io.py`.
- Keep model files free of training and evaluation orchestration.
- Keep numeric metrics free of plotting dependencies.
- Keep notebooks and Colab cells as launchers or diagnostics, not hidden product logic.
- Keep generated artifacts out of git.
- Keep official thesis numbers tied to `metrics.json`, `summary.csv`, or generated markdown reports.
- Prefer small CLI modules that call reusable functions over scripts with embedded business logic.
- Add tests at the same ownership level as the change.

These rules are not aesthetic. They protect the evidence pipeline. When the thesis cites a number, the code path that produced it should be easy to trace.

## Extension Guide

### Add a New Model

1. Add the model implementation under `models/`.
2. Add or extend model construction in the training/evaluation orchestration.
3. Add a config under `configs/`.
4. Add model tests under `models/`.
5. Confirm training and evaluation still write the same artifact contracts.

The model implementation should not know where checkpoints or reports are stored.

### Add a New Evaluation Metric

1. Implement the numeric computation in `evaluation/metrics.py`.
2. Add typed fields to `evaluation/_types.py`.
3. Add report construction in `evaluation/evaluate.py`.
4. Add JSON and summary behavior through the typed report and `evaluation/_io.py`.
5. Add tests for the metric and report serialization.

If the metric needs a plot, add the plot separately in `evaluation/plots.py`.

### Add a New Artifact

1. Define the artifact schema in the relevant `_types.py`.
2. Add read/write functions in the relevant `_io.py`.
3. Call the I/O function from the orchestration module.
4. Add tests that prove the artifact can be written and loaded.

Do not write raw JSON, CSV, HDF5, or torch checkpoints directly from multiple modules.

### Add a New Experiment Sweep

1. Keep the base behavior in YAML config.
2. Add sweep orchestration under `training/` if it trains models.
3. Add report orchestration under `evaluation/` if it only consumes existing metrics.
4. Keep generated outputs under `runs/`, `checkpoints/`, `logs/`, or `results/evaluation/`.

Experiment sweeps should compose the existing training and evaluation APIs rather than duplicate them.

## Generated Artifact Policy

The repository distinguishes source files from experiment products.

Ignored artifact roots:

- `data/output/`
- `checkpoints/`
- `logs/`
- `runs/`
- `evaluation/runs/`
- `results/evaluation/`

These files are important, but they are not source code. Preserve them in Drive or local experiment storage when they are needed for thesis evidence. Commit the code and documentation that can reproduce or interpret them.

## Maintenance Checklist

Before merging a structural change, check:

- Does each new file have one clear owner?
- Did any module start writing raw JSON, CSV, HDF5, or checkpoints directly?
- Did a numeric module gain plotting or notebook dependencies?
- Did a model file gain training, evaluation, or filesystem concerns?
- Did an official report field change without a typed contract and tests?
- Did a README duplicate details that belong in the canonical docs?
- Can a future reader trace a thesis result from command, to checkpoint, to `metrics.json`, to report table?

If the answer is unclear, the responsibility boundary is probably not explicit enough yet.
