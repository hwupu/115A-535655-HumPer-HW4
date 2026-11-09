"""Spatial pipeline: the one place for smoothing, resizing and normalization (D-03).

B-03 adds the human-side steps needed to validate the data. B-04 extends this module.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

CANVAS_SIZE = 224
# Best fit to the dataset's own fixMap (D-05, notebooks/01_data_check.ipynb).
HUMAN_SIGMA_PX = 24.0


def human_map_native(
    fixations: np.ndarray, size: tuple[int, int], sigma: float = HUMAN_SIGMA_PX
) -> np.ndarray:
    """Blur (x, y) fixation pixels into a float32 map of `size` = (width, height), peak 1.

    With no fixations the map is all zeros.
    """
    width, height = size
    points = np.zeros((height, width), dtype=np.float32)
    points[fixations[:, 1], fixations[:, 0]] = 1
    blurred = gaussian_filter(points, sigma=sigma, mode="constant")
    peak = blurred.max()
    return blurred / peak if peak > 0 else blurred


def scale_fixations(
    fixations: np.ndarray, size: tuple[int, int], canvas: int = CANVAS_SIZE
) -> np.ndarray:
    """Map native (x, y) pixels to integer canvas pixels, treating pixels as areas."""
    scale = np.array([canvas / size[0], canvas / size[1]])
    return np.clip(np.floor((fixations + 0.5) * scale).astype(int), 0, canvas - 1)
