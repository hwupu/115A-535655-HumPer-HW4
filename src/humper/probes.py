"""Checks that go beyond heatmap appearance: occlusion and weight randomization (D-07)."""

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

from humper import explain, maps, models

MASK_FRACTION = 0.20
BLUR_RADIUS = 16
RANDOM_MASKS = 5
FILLS = ("blur", "mean")


def top_fraction_mask(saliency: np.ndarray, fraction: float = MASK_FRACTION) -> np.ndarray:
    """Return a boolean mask of exactly the highest `fraction` of pixels."""
    count = round(fraction * saliency.size)
    mask = np.zeros(saliency.shape, bool)
    mask.flat[np.argsort(saliency, axis=None, kind="stable")[-count:]] = True
    return mask


def random_region_masks(
    seed: int, count: int = RANDOM_MASKS, canvas: int = maps.CANVAS_SIZE, fraction: float = MASK_FRACTION
) -> list[np.ndarray]:
    """Return blob-shaped masks of the same area, so the control resembles a saliency region."""
    rng = np.random.default_rng(seed)
    return [
        top_fraction_mask(gaussian_filter(rng.random((canvas, canvas)), canvas / 14, mode="wrap"), fraction)
        for _ in range(count)
    ]


def occlude(image: Image.Image, mask: np.ndarray, fill: str) -> Image.Image:
    """Replace masked pixels with a heavy blur of the image, or with its mean color."""
    array = np.asarray(image).copy()
    if fill == "blur":
        replacement = np.asarray(image.filter(ImageFilter.GaussianBlur(BLUR_RADIUS)))
    elif fill == "mean":
        replacement = np.broadcast_to(array.mean(axis=(0, 1), keepdims=True).astype(np.uint8), array.shape)
    else:
        raise ValueError(fill)
    array[mask] = replacement[mask]
    return Image.fromarray(array)


@torch.no_grad()
def target_probabilities(model, spec, images: list[Image.Image], target: int, device) -> np.ndarray:
    """Return the probability of `target` for each image."""
    batch = torch.cat([models.preprocess(spec, image) for image in images]).to(device)
    return model(batch).softmax(dim=1)[:, target].cpu().numpy()


def occlusion_rows(
    image_id: str,
    canvas_image: Image.Image,
    human: np.ndarray,
    record: dict,
    loaded: dict,
    device,
    seed: int,
) -> list[dict]:
    """Return the drop in target probability for human, explainer and random masks."""
    randoms = random_region_masks(seed)
    rows = []
    for model_name, spec in models.MODELS.items():
        target = int(record[f"{model_name}__top_idx"][0])
        named = {"human": top_fraction_mask(human)}
        for name, explainer in explain.EXPLAINERS.items():
            if explainer.model == model_name:
                named[name] = top_fraction_mask(maps.model_map_canvas(record[name]))
        for fill in FILLS:
            base = target_probabilities(loaded[model_name], spec, [canvas_image], target, device)[0]
            masks = list(named.values()) + randoms
            after = target_probabilities(
                loaded[model_name], spec, [occlude(canvas_image, m, fill) for m in masks], target, device
            )
            for mask_name, value in zip(named, after):
                rows.append({
                    "image_id": image_id, "model": model_name, "fill": fill, "mask": mask_name,
                    "baseline_prob": float(base), "after_prob": float(value), "drop": float(base - value),
                })
            random_after = float(after[len(named) :].mean())
            rows.append({
                "image_id": image_id, "model": model_name, "fill": fill, "mask": "random",
                "baseline_prob": float(base), "after_prob": random_after, "drop": float(base) - random_after,
            })
    return rows


def rank_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the Spearman correlation between two grids, or NaN if either is constant."""
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(spearmanr(a.ravel(), b.ravel()).statistic)


def randomization_similarity(spec, trained, randomized, x, target: int) -> float:
    """Return how similar an explanation stays when the model's weights are random."""
    return rank_similarity(
        explain.explain_grid(spec, trained, x, target), explain.explain_grid(spec, randomized, x, target)
    )
