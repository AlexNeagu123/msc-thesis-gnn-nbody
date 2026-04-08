"""
Shared training loop for EGNN and HGNN.

Usage:
    python train.py --model egnn --config ../config/egnn.yaml
    python train.py --model hgnn --config ../config/hgnn.yaml
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from data.dataset import NBodyDataset
from models.egnn import EGNN
from models.hgnn import HGNN


def train(model, loader, optimizer) -> float:
    """One epoch. Returns mean training loss."""
    pass


def validate(model, loader) -> float:
    """Returns mean validation loss."""
    pass


def main(model_name: str, config: dict) -> None:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["egnn", "hgnn"], required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    main(args.model, config)
