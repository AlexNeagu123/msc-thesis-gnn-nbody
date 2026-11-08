"""Block A factorial cells, reusing immutable one-step engine without registration."""

from dataclasses import replace

from competitors.gnode_pignn_training.recipes import RECIPES, Recipe

BASE = RECIPES["pignn_repository_shared_v1"]
CELLS = {
    "A00": BASE,
    "A10": replace(BASE, model_id="pignn_softening_only_ablation_v1", force_additive=0.05**2),
    "A01": replace(
        BASE, model_id="pignn_projection_only_ablation_v1", train_position="semiimplicit"
    ),
    "A11": RECIPES["pignn_corrected_softening_semiimplicit_v1"],
}


def scientific_delta(left: Recipe, right: Recipe) -> dict:
    """Expose changed scientific fields separately from mandatory distinct identity."""
    a, b = left.record(), right.record()
    return {k: [a[k], b[k]] for k in a if a[k] != b[k] and k != "model_id"}
