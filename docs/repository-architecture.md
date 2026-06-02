# Repository Architecture

Author: Alexandru Neagu

This repository is laid out around the final experiment workflow:

```text
configs -> data -> models -> training -> evaluation -> reports / notebooks / Colab
```

A reported number should be easy to trace back to the config, checkpoint, `metrics.json`, and report that produced it.

## Repository Map

```text
configs/      YAML configs for data, EGNN, HGNN
data/         REBOUND generation, stratification, HDF5 I/O, transition datasets
models/       EGNN, HGNN, and simple baselines
training/     trainer, curriculum schedule, checkpointing, validation scoring
evaluation/   checkpoint evaluation, reports, figures, chunked forecasts, animations
colab/        remote training/evaluation notebooks
docs/         short product and architecture documentation
runs/         ignored experiment outputs
```

## File Layout

Generated data:

```text
data/output/train.h5
data/output/val.h5
data/output/test.h5
```

Model runs:

```text
runs/egnn/<run_id>/
  best.pt
  latest.pt
  metrics.csv
  diagnostics.log
  evaluation/
    metrics.json
    summary.csv

runs/hgnn/<run_id>/
  best.pt
  latest.pt
  metrics.csv
  diagnostics.log
  evaluation/
    metrics.json
    summary.csv
```

Baseline evaluation:

```text
runs/baselines/constant_velocity/evaluation/
  metrics.json
  summary.csv
```

Comparison reports:

```text
runs/reports/<name>/
  report.md
  figures/
  tables/
  chunked/
  animations/
  selections/
```

`runs/` and `data/output/` are ignored by git. Keep important outputs in Drive or another experiment archive.

## Module Roles

### Data

| File | Role |
| --- | --- |
| `data/_types.py` | data-generation and trajectory dataclasses |
| `data/_io.py` | YAML loading and HDF5 read/write |
| `data/encounters.py` | encounter-distance bins and pure binning helpers |
| `data/generate.py` | REBOUND trajectory generation and stratified acceptance loop |
| `data/generate_eval_set.py` | one uniform test set per body count for generalization |
| `data/dataset.py` | transition dataset used by training |

HDF5 read/write details stay in `data/_io.py`. Training and evaluation use typed loaders instead of raw HDF5 keys.

### Models

| File | Role |
| --- | --- |
| `models/egnn.py` | E(n)-equivariant next-state predictor |
| `models/hgnn.py` | graph Hamiltonian model and learned-energy dynamics |
| `models/baselines.py` | simple baselines used by evaluation |

Model code should stay free of filesystem and report concerns.

### Training

| File | Role |
| --- | --- |
| `training/_types.py` | training config, checkpoint dataclasses, epoch metrics |
| `training/_io.py` | config loading, checkpoint read/write, and metrics CSV writing |
| `training/train.py` | trainer and CLI |
| `training/rollout_score.py` | validation rollout scoring, including per-group scoring |
| `training/diagnostics.py` | outlier and skipped-batch diagnostics |

Training selects checkpoints. Test-set results are produced later by evaluation.

### Evaluation

| File | Role |
| --- | --- |
| `evaluation/_types.py` | evaluation report dataclasses |
| `evaluation/_io.py` | `metrics.json` and `summary.csv` read/write |
| `evaluation/_binning.py` | per-bin mask helpers |
| `evaluation/_loader.py` | shared trained-model loading |
| `evaluation/metrics.py` | single-step, rollout, divergence, and energy metrics |
| `evaluation/evaluate.py` | trained-checkpoint evaluation |
| `evaluation/evaluate_baseline.py` | baseline evaluation |
| `evaluation/rollout_score.py` | per-bin baseline envelope used by evaluation |
| `evaluation/report.py` | comparison-report driver |
| `evaluation/report_tables.py` | report CSV and markdown tables |
| `evaluation/report_figures.py` | presentation-grade report figures |
| `evaluation/report_generalization.py` | per-N generalization report, no distance groups |
| `evaluation/evaluate_chunked.py` | periodically corrected short-horizon forecasting |
| `evaluation/animate_best.py` | representative trajectory animations |
| `evaluation/plots.py` | notebook-facing plotting utilities |
| `evaluation/visual_diagnostics.ipynb` | manual trajectory inspection and selection |

Evaluation produces the numbers used in the thesis. The report figures and tables are built from `metrics.json`.

### Colab

| File | Role |
| --- | --- |
| `colab/train_colab.ipynb` | train EGNN or HGNN from Drive data |
| `colab/evaluate_report_artifacts.ipynb` | evaluate checkpoints and generate reports/animations |

Colab notebooks are launchers. Product logic should remain in importable modules.

## Practical Rules

- Put dataclasses in `_types.py`.
- Put file read/write code in `_io.py`.
- Keep metric calculations out of plotting/report modules.
- Keep models independent of training and evaluation code.
- Keep notebooks thin; reusable behavior belongs in Python modules.
- Keep generated data and runs out of git.
- Tie thesis results to saved configs, checkpoints, `metrics.json`, and generated reports.

## Extension Pattern

Add a metric:

1. compute it in `evaluation/metrics.py`,
2. add dataclasses in `evaluation/_types.py`,
3. build it in `evaluation/evaluate.py`,
4. save/load it through `evaluation/_io.py`,
5. add tests near the changed layer.

Add a report figure:

1. read only from `EvaluationReport`,
2. implement plotting in `evaluation/report_figures.py`,
3. call it from `evaluation/report.py`,
4. add a focused figure test and a report-orchestrator test.

Add a training feature:

1. add config fields to `training/_types.py`,
2. wire behavior in `training/train.py`,
3. save/load any new epoch or checkpoint fields in `training/_io.py`,
4. add config and trainer tests.
