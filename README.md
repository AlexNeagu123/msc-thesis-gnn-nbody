[![CI](https://github.com/AlexNeagu123/msc-thesis-gnn-nbody/actions/workflows/ci.yaml/badge.svg)](https://github.com/AlexNeagu123/msc-thesis-gnn-nbody/actions/workflows/ci.yaml)

# msc-thesis-gnn-nbody

Comparing inductive biases in graph neural networks for learning chaotic gravitational dynamics.

## Canonical Documentation

- [Product Specification](docs/product-specification.md): what the workbench can do, how to invoke it, which artifacts it writes, and how official thesis metrics are produced.
- [Repository Architecture](docs/repository-architecture.md): how files are organized, which modules own which responsibilities, and where future changes should go.

These two documents are the source of truth for behavior and structure. Keep README ownership unitary: the root README is the entry point, and durable details belong in `docs/`.

## Quick Start

Generate datasets:

```bash
uv run python -m data.generate --config configs/data.yaml
```

Train a model:

```bash
uv run python -m training.train \
  --config configs/egnn.yaml \
  --artifact-dir runs/single/egnn/n1000
uv run python -m training.train \
  --config configs/hgnn.yaml \
  --artifact-dir runs/single/hgnn/n1000
```

Evaluate a checkpoint:

```bash
uv run python -m evaluation.evaluate \
  --config configs/egnn.yaml \
  --checkpoint runs/single/egnn/n1000/<run_id>/best.pt
```

For the complete command and artifact contract, read the product specification.
