"""Alignment metrics between a saliency map and human attention (D-04).

All maps are on the canvas and fixations are integer (x, y) canvas pixels. A constant
map has no defined score, so every metric returns NaN for one.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from humper import maps


def _sample(saliency: np.ndarray, fixations: np.ndarray) -> np.ndarray:
    return saliency[fixations[:, 1], fixations[:, 0]]


def cc(saliency: np.ndarray, human: np.ndarray) -> float:
    """Pearson correlation between two maps."""
    return float(np.mean(maps.zscore(saliency) * maps.zscore(human)))


def nss(saliency: np.ndarray, fixations: np.ndarray) -> float:
    """Mean z-scored saliency at the fixated pixels."""
    if len(fixations) == 0:
        return float("nan")
    return float(np.mean(_sample(maps.zscore(saliency), fixations)))


def sauc(saliency: np.ndarray, fixations: np.ndarray, negatives: np.ndarray) -> float:
    """Shuffled AUC: how well the map ranks this image's fixations above `negatives`.

    `negatives` are fixations from other images (D-19), so a map that only captures
    the dataset-wide center bias scores about 0.5. Tied values count as half a win.
    """
    if len(fixations) == 0 or len(negatives) == 0 or saliency.std() < maps.MIN_STD:
        return float("nan")
    positive, negative = _sample(saliency, fixations), _sample(saliency, negatives)
    n_pos, n_neg = len(positive), len(negative)
    ranks = rankdata(np.concatenate([positive, negative]))
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@dataclass(frozen=True)
class MetricSpec:
    name: str
    fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]
    higher_is_better: bool
    kind: str
    note: str


METRICS = {
    spec.name: spec
    for spec in (
        MetricSpec("cc", lambda s, h, f, n: cc(s, h), True, "map", "depends on the human-map blur"),
        MetricSpec("nss", lambda s, h, f, n: nss(s, f), True, "fixation", "no human-map blur"),
        MetricSpec("sauc", lambda s, h, f, n: sauc(s, f, n), True, "fixation", "discounts center bias"),
    )
}


def score(
    saliency: np.ndarray, human: np.ndarray, fixations: np.ndarray, negatives: np.ndarray
) -> dict[str, float]:
    """Return every registered metric for one map."""
    return {name: spec.fn(saliency, human, fixations, negatives) for name, spec in METRICS.items()}
