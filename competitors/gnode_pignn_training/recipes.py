"""Declared shared-split recipes; not authorisation to launch full training."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Recipe:
    """Scientific recipe identity, including independently labelled corrections."""

    model_id: str
    kind: str
    train_position: str
    force_additive: float
    dt: float = 0.001
    physics_weight: float = 0.1
    seed: int = 42
    learning_rate: float = 0.0005
    adam_epsilon: float = 1e-7
    batch_size: int = 512
    epochs: int = 100

    def record(self) -> dict:
        """Return complete serialisable identity with shared selection policy."""
        return {
            **asdict(self),
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "weight_decay": None,
            "clipping": None,
            "scheduler": None,
            "normalization": "none",
            "initialization": "from_scratch_not_public_weights",
            "shuffle": self.kind == "gnode",
            "pair_order": "ascending system ID then saved time",
            "shuffle_implementation": "explicit saved NumPy Generator permutation each epoch; not recovered Keras RNG order",
            "primary_selection": "eligible minimum full1999-future position MAE on shared val80",
            "selection_epochs_proposed": list(range(4, 101, 4)),
            "secondary_checkpoint": "final epoch100; no choosing primary versus secondary from comparison100 scores",
            "inference_position": "expanded_constant_acceleration"
            if self.kind == "gnode"
            else "semiimplicit",
            "logical_updates_if_full_run_approved": 125000,
            "long_run_authorized": False,
        }


RECIPES = {
    "gnode_repository_shared_v1": Recipe(
        "gnode_repository_shared_v1", "gnode", "average_velocity", 0.0, physics_weight=0.0
    ),
    "pignn_repository_shared_v1": Recipe("pignn_repository_shared_v1", "pignn", "explicit", 0.05),
    "pignn_corrected_softening_semiimplicit_v1": Recipe(
        "pignn_corrected_softening_semiimplicit_v1", "pignn", "semiimplicit", 0.05**2
    ),
}
