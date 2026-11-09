"""Score explanations and the reference maps that give those scores meaning (D-04, D-15)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from humper import alignment, data, explain, maps, models, store

COARSEST_GRID = 7


def derangement(n: int, seed: int = 0) -> np.ndarray:
    """Return a seeded permutation of range(n) in which no index maps to itself."""
    order = np.random.default_rng(seed).permutation(n)
    partner = np.empty(n, dtype=int)
    partner[order] = np.roll(order, -1)
    return partner


@dataclass(frozen=True)
class Inputs:
    """Human data for every image, on the canvas, in table order."""

    image_ids: list[str]
    human: np.ndarray  # (n, canvas, canvas)
    fixations: dict[str, np.ndarray]
    human_sum: np.ndarray

    def center(self, index: int) -> np.ndarray:
        """Return the mean human map of all other images."""
        return (self.human_sum - self.human[index]) / (len(self.image_ids) - 1)


def build_inputs(table: pd.DataFrame) -> Inputs:
    fixations, human = {}, []
    for r in table.itertuples():
        size = (r.width, r.height)
        native = data.load_fixations(r.image_id)
        fixations[r.image_id] = maps.scale_fixations(native, size)
        human.append(maps.human_map_canvas(native, size))
    stack = np.stack(human)
    return Inputs(table.image_id.tolist(), stack, fixations, stack.sum(axis=0, dtype=np.float64))


def score_image(index: int, partner: int, inputs: Inputs, record: dict) -> list[dict]:
    """Return all score rows for one image. `partner` supplies the shuffled human data."""
    image_id = inputs.image_ids[index]
    human, fixations = inputs.human[index], inputs.fixations[image_id]
    negatives = alignment.other_fixations(inputs.fixations, image_id)
    partner_id = inputs.image_ids[partner]
    partner_negatives = alignment.other_fixations(inputs.fixations, partner_id)

    def row(name: str, kind: str, saliency: np.ndarray, shuffled: bool = False) -> dict:
        if shuffled:
            scores = alignment.score(
                saliency, inputs.human[partner], inputs.fixations[partner_id], partner_negatives
            )
        else:
            scores = alignment.score(saliency, human, fixations, negatives)
        return {"image_id": image_id, "explainer": name, "kind": kind, **scores}

    rows = [row("center", "reference", inputs.center(index))]
    for cells in sorted({spec.grid for spec in models.MODELS.values()}):
        rows.append(row(f"grid_{cells}x{cells}", "reference", maps.through_grid(human, cells)))
    for name, spec in explain.EXPLAINERS.items():
        canvas = maps.model_map_canvas(record[name])
        rows.append(row(name, "explainer", canvas))
        rows.append(row(f"shuffled_{name}", "shuffled", canvas, shuffled=True))
        if models.MODELS[spec.model].grid != COARSEST_GRID:
            reduced = maps.through_grid(canvas, COARSEST_GRID)
            rows.append(row(f"{name}_{COARSEST_GRID}x{COARSEST_GRID}", "reduced", reduced))
    return rows


def score_all(table: pd.DataFrame, seed: int = 0, progress=lambda items: items) -> pd.DataFrame:
    inputs = build_inputs(table)
    partner = derangement(len(table), seed)
    rows = []
    for index in progress(range(len(table))):
        record = store.load_record(inputs.image_ids[index])
        rows += score_image(index, int(partner[index]), inputs, record)
    return pd.DataFrame(rows)
