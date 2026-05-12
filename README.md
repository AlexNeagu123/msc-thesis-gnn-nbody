[![CI](https://github.com/AlexNeagu123/msc-thesis-gnn-nbody/actions/workflows/ci.yaml/badge.svg)](https://github.com/AlexNeagu123/msc-thesis-gnn-nbody/actions/workflows/ci.yaml)

# msc-thesis-gnn-nbody

Comparing inductive biases in graph neural networks for learning chaotic gravitational dynamics.

## Docs

- [Project Flow](docs/product-specification.md): data, training, evaluation, reports, chunked forecasts, and animations.
- [Repository Architecture](docs/repository-architecture.md): how the code and output folders are organized.

The root README is only the entry point. The practical details live in `docs/`.

## Quick Start

Generate datasets:

```bash
uv run python -m data.generate --config configs/data.yaml
```

Train EGNN and HGNN:

```bash
uv run python -m training.train \
  --config configs/egnn.yaml
uv run python -m training.train \
  --config configs/hgnn.yaml
```

Evaluate a checkpoint:

```bash
uv run python -m evaluation.evaluate \
  --config configs/egnn.yaml \
  --checkpoint runs/egnn/<run_id>/best.pt \
  --test-path data/output/test.h5
```

Generate the comparison report:

```bash
uv run python -m evaluation.report \
  --egnn runs/egnn/<egnn_run_id>/evaluation/metrics.json \
  --hgnn runs/hgnn/<hgnn_run_id>/evaluation/metrics.json \
  --baseline runs/baselines/constant_velocity/evaluation/metrics.json \
  --output runs/reports/official_1k
```

For the full workflow, read the project flow document.
