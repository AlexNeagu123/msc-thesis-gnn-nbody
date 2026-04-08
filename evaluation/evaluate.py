"""
Evaluation metrics for trained models.

Usage:
    python evaluate.py --model egnn --checkpoint path/to/checkpoint.pt --config ../config/egnn.yaml
"""

import argparse
import yaml
import torch


def trajectory_mse(model, initial_state: torch.Tensor, true_trajectory: torch.Tensor, rollout_steps: int) -> float:
    """
    Autoregressively rolls out the model and computes MSE against ground truth.
    initial_state: (n_particles, 4)
    true_trajectory: (rollout_steps, n_particles, 4)
    """
    pass


def energy_error(trajectory: torch.Tensor, G: float = 1.0) -> torch.Tensor:
    """
    Computes relative energy error |E(t) - E(0)| / |E(0)| at each step.
    trajectory: (T, n_particles, 4)
    Returns tensor of shape (T,).
    """
    pass


def zero_shot_mse(model, dataset_path: str, n_particles: int, rollout_steps: int) -> float:
    """Evaluates a model trained on 3 bodies on an N-body test set without retraining."""
    pass


def main(model_name: str, checkpoint_path: str, config: dict) -> None:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["egnn", "hgnn"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    main(args.model, args.checkpoint, config)
