# Product Specification: Inductive Biases for Chaotic N-Body Dynamics

Author: Alexandru Neagu

## Purpose

This repository is the experimental software artifact for a master's thesis on **physical inductive biases in graph neural networks**. The thesis question is:

> How do geometric equivariance and Hamiltonian mechanics affect accuracy, stability, and interpretability when learning chaotic attractive gravitational dynamics?

The code exists to make that comparison reproducible. It generates a controlled 3-body gravitational dataset, trains two graph neural architectures under the same data and optimization protocol, evaluates them with the same numeric pipeline, and stores the evidence needed for thesis tables.

This is not a generic N-body simulator and not a model-serving application. It is a research workbench for comparing two architecture-level assumptions:

- **EGNN:** graph structure plus E(n) equivariance. The model predicts the next state directly.
- **HGNN:** graph structure plus Hamiltonian mechanics. The model learns a decomposed Hamiltonian, then derives the next state from it.

The important object of study is not only predictive error. The study asks where each inductive bias helps: local accuracy, autoregressive rollout stability, physical-energy behavior, and interpretability of the learned dynamics.

## Research Context

The project follows the progression in the thesis presentation:

1. Black-box neural networks can approximate chaotic 3-body trajectories, but do not encode conservation or relational structure by design.
2. Interaction Networks introduced graph-based relational inductive bias for physical systems.
3. Hamiltonian Neural Networks introduced conservation by learning an energy function and deriving dynamics from Hamilton's equations.
4. EGNN introduced E(n)-equivariant graph message passing, making rotations and translations structural symmetries of the model.
5. HGNN combined graph structure with a decomposed Hamiltonian, enabling energy-based dynamics and later symbolic inspection.

The gap targeted here is narrower and thesis-specific:

- EGNN and HGNN have not been compared on the same attractive chaotic gravitational benchmark.
- HGNN was tested on several physical systems, including repulsive gravity, but not this exact attractive 3-body setup.
- The relevant question is which inductive bias matters more under repeated rollout: geometric symmetry or Hamiltonian structure.

## System Capabilities

The software can:

- Generate train, validation, and test trajectory datasets for equal-mass 2D gravitational 3-body dynamics.
- Train EGNN and HGNN from YAML experiment configurations.
- Slice a larger training dataset into controlled `N_TRAIN` subsets for data-scaling studies.
- Run EGNN noise-injection and learning-rate sweeps.
- Evaluate checkpoints with state, position, velocity, divergence, and energy-drift metrics.
- Render scaling-study reports from saved `metrics.json` files.
- Run long training and evaluation jobs from Google Colab while persisting outputs to Google Drive.

The official thesis numbers should come from `metrics.json`, `summary.csv`, and generated scaling reports. Notebooks and plots are diagnostic surfaces, not the source of truth.

## Model Contracts

### EGNN

EGNN implements the geometric inductive bias.

Input:

```text
state_t: (n_particles, 5) = [x, y, vx, vy, mass]
```

Output:

```text
state_t+dt: (n_particles, 5)
```

Contract:

- Uses graph message passing over particles.
- Uses relative coordinate differences and squared distances.
- Preserves E(n) equivariance by construction.
- Predicts positions and velocities directly.
- Does not enforce physical-energy conservation.

### HGNN

HGNN implements the Hamiltonian inductive bias.

Input:

```text
state_t: (n_particles, 5) = [x, y, vx, vy, mass]
```

Output:

```text
state_t+dt: (n_particles, 5)
```

Contract:

- Uses graph message passing over pairwise distances for potential-energy structure.
- Learns a decomposed Hamiltonian `H = T + V`.
- Computes dynamics through automatic differentiation of the learned energy.
- Advances one step with a leapfrog-style integrator.
- Exposes a learned Hamiltonian quantity for evaluation and interpretation.

## Command Interface

Run commands from `impl/`. Prefer `uv run` locally.

### Generate Data

```bash
uv run python -m data.generate --config configs/data.yaml
```

Arguments:

| Argument | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--config` | No | `configs/data.yaml` | Data-generation YAML file. |

Outputs:

- `train_path`, `val_path`, and `test_path` from the config.
- HDF5 files containing trajectories, physical energies, and metadata.

### Train One Model

For a thesis-quality run, write artifacts under the canonical `runs/`
archive so checkpoints, metrics, and evaluation stay colocated:

```bash
uv run python -m training.train \
  --config configs/egnn_curriculum.yaml \
  --n-train 5000 \
  --artifact-dir runs/curriculum/egnn/n5000
```

The two training-only forms below still work and are useful for ad-hoc
local experiments. Without `--artifact-dir`, paths come from the YAML
(typically the legacy `checkpoints/` / `logs/` roots):

```bash
uv run python -m training.train --config configs/egnn.yaml
uv run python -m training.train --config configs/hgnn.yaml
```

Arguments:

| Argument | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--config` | Yes | None | Model training YAML file. |
| `--n-train` | No | Config value | Use only the first N training trajectories. |
| `--artifact-dir` | No | YAML values | Single directory under which both checkpoints and metrics are written, force-enabling both. The trainer appends `<run_id>` as a per-run subdirectory. |
| `--init-checkpoint` | No | None | Initialise model weights from a previous checkpoint; optimizer, scheduler, and run_id all start fresh. |

Outputs (canonical layout, with `--artifact-dir runs/curriculum/egnn/n5000`):

- `runs/curriculum/egnn/n5000/<run_id>/best.pt`
- `runs/curriculum/egnn/n5000/<run_id>/latest.pt`
- `runs/curriculum/egnn/n5000/<run_id>/metrics.csv`
- `runs/curriculum/egnn/n5000/<run_id>/diagnostics.log`

### Run a Data-Scaling Sweep

```bash
uv run python -m training.scaling --config configs/egnn.yaml --sizes 1000,2000,5000
```

Arguments:

| Argument | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--config` | Yes | None | Model training YAML file. |
| `--sizes` | No | `1000,2000,5000` | Comma-separated training-set sizes. |
| `--artifact-root` | No | `runs/scaling` | Parent directory for per-size run folders. Each size lands at `<root>/<model>/n<N>/<run_id>/`. |

Purpose:

- Produce matched checkpoints across training-set sizes.
- Keep architecture and optimization fixed while varying only data volume.

Outputs (default root):

- `runs/scaling/<model>/n<N>/<run_id>/{best.pt, latest.pt, metrics.csv, diagnostics.log}`

### Run a 2-D Hyperparameter Sweep (LR x noise)

```bash
uv run python -m training.sweep --config configs/egnn.yaml --epochs 200
```

Arguments:

| Argument | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--config` | No | `configs/egnn.yaml` | Base EGNN config. |
| `--epochs` | No | `200` | Epochs per sweep cell. |
| `--artifact-root` | No | `runs/sweep` | Parent directory for per-cell run folders. Each cell lands at `<root>/<model>/lr_<lr>_nf_<nf>/<run_id>/`. |

Current grid:

- learning rates: `5e-4`, `1e-3`, `2e-3`
- noise factors: `0.0`, `0.03`, `0.05`

Purpose:

- 2-D grid search; deliberately separate from the Colab single-axis noise sweep that lives under `runs/noise_sweep/...`.

### Evaluate a Checkpoint

For checkpoints under `runs/`, the evaluator defaults to writing the
report next to the checkpoint, so artifacts stay self-contained. Pass
`--output-dir` only when overriding that default:

```bash
uv run python -m evaluation.evaluate \
  --config configs/egnn_curriculum.yaml \
  --checkpoint runs/curriculum/egnn/n5000/<run_id>/best.pt \
  --test-path data/output/scaling/test.h5 \
  --device auto
```

This writes:

- `runs/curriculum/egnn/n5000/<run_id>/evaluation/metrics.json`
- `runs/curriculum/egnn/n5000/<run_id>/evaluation/summary.csv`

Arguments:

| Argument | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--config` | Yes | None | Model config used to rebuild the architecture. |
| `--checkpoint` | Yes | None | Checkpoint to evaluate. |
| `--test-path` | No | `data/output/test.h5` | Test HDF5 file. |
| `--output-dir` | No | See below | Evaluation output directory. |
| `--device` | No | `auto` | `auto`, `cuda`, `mps`, or `cpu`. |

`--output-dir` resolution:

1. Explicit value always wins.
2. Otherwise, if the checkpoint sits under any `runs/` ancestor, defaults
   to `<run_dir>/evaluation/` (canonical layout).
3. Otherwise (legacy `checkpoints/<model>/<run_id>/`), falls back to
   `results/evaluation/<model>/<run_id>/`.

Outputs:

- `metrics.json`: canonical evaluation report.
- `summary.csv`: flattened one-row summary for table drafting.

### Build a Scaling Report

```bash
uv run python -m evaluation.scaling_report \
  --manifest runs/scaling_runs.yaml \
  --output runs/scaling_report.md
```

Arguments:

| Argument | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--manifest` | Yes | None | YAML manifest pointing to evaluation reports. |
| `--output` | No | None | Optional markdown output path. |

Purpose:

- Read saved evaluation reports.
- Group results by model and training-set size.
- Render markdown tables for thesis discussion.

## Dataset Contract

Generated HDF5 files contain:

```text
trajectories: (n_trajectories, n_frames, n_particles, 5)
energies:     (n_trajectories, n_frames)
metadata:     simulation and provenance attributes
```

State channel order:

```text
[x, y, vx, vy, mass]
```

The training dataset converts each trajectory into consecutive supervised transitions:

```text
state_t -> state_t+dt
```

For a file with 200 frames, each trajectory contributes 199 transitions.

## Evaluation Contract

The evaluator measures:

| Metric family | Question answered |
| --- | --- |
| Single-step state MSE | How accurate is one learned transition over `[x, y, vx, vy]`? |
| Single-step position/velocity MSE | Which part of the transition error comes from geometry vs momentum? |
| Minimum pairwise distance | Are errors associated with close encounters? |
| Autoregressive rollout state MSE | What happens when predictions are fed back into the model? |
| Autoregressive rollout position MSE | Literature-facing trajectory error, separated from velocity error. |
| Finite-state fraction | How many rollouts remain numerically valid? |
| State/position divergence thresholds | At which step does error cross thesis-relevant limits? |
| Physical-energy drift | Does the predicted trajectory preserve the true conserved quantity? |
| Learned-Hamiltonian drift | For HGNN, is the learned energy internally stable? |

This split is deliberate. A model can be strong at one-step prediction and weak under rollout. Conversely, a model can sacrifice local accuracy while preserving a more stable long-horizon structure.

## Colab Contract

Colab entry point:

```text
colab/train_colab.ipynb
```

Notebook parameters:

| Parameter | Values | Meaning |
| --- | --- | --- |
| `GIT_REF` | Branch or tag | Checkout target for the code running on Colab. |
| `RUN_MODE` | `noise_sweep`, `single` | Run an EGNN noise sweep or one selected model. |
| `MODEL` | `egnn`, `hgnn` | Select architecture. |
| `N_TRAIN` | `1000`, `2000`, `5000`, `10000` | Select training subset size. |
| `EPOCHS` | Integer | Training duration. |
| `NOISE_FACTORS` | Comma-separated floats | Noise factors used when `RUN_MODE=noise_sweep`. |
| `RUN_TRAINING` | Boolean | Train selected model. |
| `RUN_EVALUATION` | Boolean | Evaluate selected checkpoint. |
| `SKIP_COMPLETED` | Boolean | Skip Drive run folders that already contain `evaluation/metrics.json`. |

Expected Drive input:

```text
MyDrive/masters-thesis/data/scaling/train.h5
MyDrive/masters-thesis/data/scaling/val.h5
MyDrive/masters-thesis/data/scaling/test.h5
```

Drive output:

```text
MyDrive/masters-thesis/runs/<model>/n<N_TRAIN>/<run_id>/
MyDrive/masters-thesis/runs/noise_sweep/egnn/n<N_TRAIN>_e<EPOCHS>/nf_<noise>/<run_id>/
```

The notebook validates that `train.h5` contains enough trajectories before training. This prevents a long run from silently using the wrong dataset.

## Boundaries

Current scope:

- 2D equal-mass attractive gravitational 3-body dynamics.
- EGNN and HGNN as the primary architecture comparison.
- One-step supervised training.
- Long-horizon behavior measured during evaluation through autoregressive rollout.
- Data-scaling study over fixed training-set sizes.
- EGNN noise-injection sweep as a stability intervention.

Out of scope unless explicitly added:

- General-purpose N-body simulation tooling.
- Serving trained models behind an API.
- Large-N astrophysical simulation.
- Symbolic regression execution pipeline.
- Additional baselines such as HOGN or plain Interaction Networks.

## Validation

Run the local validation suite:

```bash
uv run pytest
```

Run lint and formatting checks:

```bash
uv run ruff check .
uv run ruff format --check .
```

Validation means the software contracts hold. It does not by itself prove a thesis claim. Thesis claims must be tied to preserved `metrics.json` files, generated reports, and documented experimental conditions.

## References

- Battaglia et al. (2016), *Interaction Networks for Learning about Objects, Relations and Physics*, NeurIPS. https://arxiv.org/abs/1612.00222
- Greydanus et al. (2019), *Hamiltonian Neural Networks*, NeurIPS. https://arxiv.org/abs/1906.01563
- Breen et al. (2019), *Newton vs the Machine: Solving the Chaotic Three-Body Problem Using Deep Neural Networks*, MNRAS. https://arxiv.org/abs/1910.07291
- Sanchez-Gonzalez et al. (2019), *Hamiltonian Graph Networks with ODE Integrators*. https://arxiv.org/abs/1909.12790
- Satorras et al. (2021), *E(n) Equivariant Graph Neural Networks*, ICML. https://arxiv.org/abs/2102.09844
- Bishnoi et al. (2023), *Discovering Symbolic Laws Directly from Trajectories with Hamiltonian Graph Neural Networks*, ICML. https://arxiv.org/abs/2307.05299
- Cranmer (2023), *Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl*. https://arxiv.org/abs/2305.01582
- Rein and Liu (2012), *REBOUND: An Open-Source Multi-Purpose N-Body Code*, Astronomy and Astrophysics. https://doi.org/10.1051/0004-6361/201118085

Local thesis context:

- `edu/presentation/main.pdf`
- `edu/presentation/slides/09_gap.tex`
- `edu/research/papers/egnn.md`
- `edu/research/papers/hgnn.md`
