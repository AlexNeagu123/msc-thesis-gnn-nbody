"""Explicit development cells. Registration is NOT production-run authorization."""
# ruff: noqa: ANN001, ANN201, ANN204, D101, D102, D103, D105

from dataclasses import asdict, dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class Cell:
    name: str
    track: str
    intervention: str
    comparator: str
    status: Literal["new", "existing_reference", "conditional"] = "new"
    seed: int = 42
    kinetic: Literal["learned", "exact"] = "learned"
    gravity: Literal["off", "scaled", "raw"] = "off"
    update: Literal["native", "semiimplicit"] = "native"
    shuffled: bool = False
    clip_norm: float | None = None
    architecture: Literal["native", "central_pair"] = "native"
    order: Literal["staged", "matched_tail"] = "staged"

    def __post_init__(self):
        if self.track not in {"hgnn", "gnode", "pi_order", "curriculum", "central_pair"}:
            raise ValueError("unknown track")
        if self.seed not in {42, 43}:
            raise ValueError("registered initialization seeds only")
        if self.clip_norm not in {None, 1.0}:
            raise ValueError("only preregistered clip1")

    def record(self):
        return {
            **asdict(self),
            "development_only": True,
            "production_authorized": False,
            "selection": "eligible minimum full1999-future VAL80 position MAE; earlier exact ties",
            "diagnostic_access": False,
            "automatic_retry": False,
        }


CELLS = {}


def add(cell):
    if cell.name in CELLS:
        raise ValueError("duplicate cell")
    CELLS[cell.name] = cell


for kinetic in ["learned", "exact"]:
    for gravity in ["off", "scaled"]:
        add(
            Cell(
                f"hgnn_{kinetic}_gravity_{gravity}",
                "hgnn",
                "kinetic prior x scaled generator-force loss",
                f"hgnn_learned_gravity_{gravity}",
                status="existing_reference" if kinetic == "learned" else "new",
                kinetic=kinetic,
                gravity=gravity,
            )
        )
add(
    Cell(
        "gnode_native",
        "gnode",
        "reference raw-loss staged GNODE",
        "gnode_native",
        status="existing_reference",
    )
)
add(
    Cell(
        "gnode_semiimplicit",
        "gnode",
        "change training and inference update only",
        "gnode_native",
        update="semiimplicit",
    )
)
for shuffled in [False, True]:
    for clip in [None, 1.0]:
        name = f"pi_{'shuffled' if shuffled else 'ordered'}_{'clip1' if clip else 'unclipped'}"
        add(
            Cell(
                name,
                "pi_order",
                "restorable epoch order x global norm clipping",
                "pi_ordered_unclipped",
                status="conditional"
                if shuffled and clip
                else "new"
                if shuffled
                else "existing_reference",
                gravity="raw",
                shuffled=shuffled,
                clip_norm=clip,
            )
        )
add(
    Cell(
        "hgnn_matched_tail",
        "curriculum",
        "permute first2000units; preserve final500h100units",
        "hgnn_learned_gravity_off",
        order="matched_tail",
    )
)
for gravity in ["off", "scaled"]:
    add(
        Cell(
            f"central_pair_gravity_{gravity}",
            "central_pair",
            "central pair architecture x normalized force supervision",
            "central_pair_gravity_off",
            gravity=gravity,
            architecture="central_pair",
            update="semiimplicit",
        )
    )


def validate_cell(cell):
    # Reject unsupported factor combinations rather than silently inventing a recipe.
    if cell.name not in CELLS or cell != replace(CELLS[cell.name], seed=cell.seed):
        raise ValueError("unregistered scientific cell")
    return cell
