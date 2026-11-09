"""Spatial pipeline: the one place for smoothing, resizing and normalization (D-03).

Every comparison happens on a square canvas. Resizing goes through PIL on float32
arrays, so the result does not depend on the torch device.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

CANVAS_SIZE = 224
# Best fit to the dataset's own fixMap (D-13, notebooks/01_data_check.ipynb).
HUMAN_SIGMA_PX = 24.0
# Below this standard deviation a map is treated as constant.
MIN_STD = 1e-8


def _resize(array: np.ndarray, size: tuple[int, int], resample: Image.Resampling) -> np.ndarray:
    """Resize a 2-D array to `size` = (width, height), returning float32."""
    resized = Image.fromarray(np.asarray(array, dtype=np.float32)).resize(size, resample)
    return np.asarray(resized, dtype=np.float32)


def model_input(image: Image.Image, canvas: int = CANVAS_SIZE) -> Image.Image:
    """Resize the whole image to the model input, with no crop (D-03)."""
    return image.resize((canvas, canvas), Image.Resampling.BICUBIC)


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


def human_map_canvas(
    fixations: np.ndarray,
    size: tuple[int, int],
    sigma: float = HUMAN_SIGMA_PX,
    canvas: int = CANVAS_SIZE,
) -> np.ndarray:
    """Blur at native resolution, then area-downsample to the canvas."""
    native = human_map_native(fixations, size, sigma)
    return _resize(native, (canvas, canvas), Image.Resampling.BOX)


def scale_fixations(
    fixations: np.ndarray, size: tuple[int, int], canvas: int = CANVAS_SIZE
) -> np.ndarray:
    """Map native (x, y) pixels to integer canvas pixels, treating pixels as areas."""
    scale = np.array([canvas / size[0], canvas / size[1]])
    return np.clip(np.floor((fixations + 0.5) * scale).astype(int), 0, canvas - 1)


def model_map_canvas(grid: np.ndarray, canvas: int = CANVAS_SIZE) -> np.ndarray:
    """Bilinearly upsample a model map from its grid (e.g. 7×7 or 14×14) to the canvas."""
    return _resize(grid, (canvas, canvas), Image.Resampling.BILINEAR)


def through_grid(saliency: np.ndarray, cells: int) -> np.ndarray:
    """Pass a canvas map through a `cells`×`cells` grid and back, as a model map would be.

    Applied to the human map, this estimates the best score a grid that coarse allows (D-15).
    """
    grid = _resize(saliency, (cells, cells), Image.Resampling.BOX)
    return model_map_canvas(grid, canvas=saliency.shape[0])


def zscore(saliency: np.ndarray) -> np.ndarray:
    """Return a zero-mean, unit-variance map, or all NaN if the map is constant.

    An all-zero Grad-CAM has no defined score, so NaN must reach `scores.csv`.
    """
    saliency = np.asarray(saliency, dtype=np.float64)
    std = saliency.std()
    if std < MIN_STD:
        return np.full_like(saliency, np.nan)
    return (saliency - saliency.mean()) / std


def to_display(saliency: np.ndarray) -> np.ndarray:
    """Min-max scale to [0, 1] for figures only. A constant map becomes all zeros."""
    saliency = np.asarray(saliency, dtype=np.float32)
    span = saliency.max() - saliency.min()
    return (saliency - saliency.min()) / span if span > 0 else np.zeros_like(saliency)


def warp_to_image(saliency: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Warp a canvas map back to the image's `size` = (width, height) for figures."""
    return _resize(saliency, size, Image.Resampling.BILINEAR)
