"""Build the blind labelling sheet (D-06): images and predicted classes, nothing else."""

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from humper import data

RANDOM_SAMPLE_SIZE = 100
RANDOM_SAMPLE_SEED = 2026
SHEET_ORDER_SEED = 2026
DISPLAY_LONG_SIDE = 768
TEMPLATE = Path(__file__).with_name("label_sheet.html")


def random_sample(table: pd.DataFrame) -> list[str]:
    """Return the images used for factor statistics. It never depends on scores."""
    return table.sample(n=RANDOM_SAMPLE_SIZE, random_state=RANDOM_SAMPLE_SEED).image_id.tolist()


def sheet_ids(table: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Return the random sample and candidates merged, in an order that hides which is which."""
    ids = sorted(set(random_sample(table)) | set(candidates))
    return [ids[i] for i in np.random.default_rng(SHEET_ORDER_SEED).permutation(len(ids))]


def _image_data_url(image_id: str) -> str:
    image = data.load_image(image_id)
    image.thumbnail((DISPLAY_LONG_SIDE, DISPLAY_LONG_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def sheet_entries(ids: list[str], predictions: pd.DataFrame) -> list[dict]:
    """Return what annotators see. Only classes: no scores, probabilities or heatmaps."""
    names = predictions.pivot(index="image_id", columns="model", values="top1_name")
    return [
        {
            "image_id": image_id,
            "resnet50": names.loc[image_id, "resnet50"],
            "vit_b16": names.loc[image_id, "vit_b16"],
        }
        for image_id in ids
    ]


def render(entries: list[dict]) -> str:
    """Return the self-contained HTML page."""
    items = [{**entry, "src": _image_data_url(entry["image_id"])} for entry in entries]
    payload = json.dumps(items).replace("<", "\\u003c")
    return TEMPLATE.read_text(encoding="utf-8").replace("/*__ITEMS__*/[]", payload)
