"""Center-crop geometry for the sensitivity check (D-18). The main analysis does not crop."""

import numpy as np
from PIL import Image

RESIZE_SHORT = 256
CROP = 224


def crop_geometry(size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Return (resized width, resized height, left, top) for Resize(256) + CenterCrop(224)."""
    width, height = size
    scale = RESIZE_SHORT / min(width, height)
    new_width, new_height = round(width * scale), round(height * scale)
    return new_width, new_height, (new_width - CROP) // 2, (new_height - CROP) // 2


def crop_image(image: Image.Image) -> Image.Image:
    new_width, new_height, left, top = crop_geometry(image.size)
    resized = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    return resized.crop((left, top, left + CROP, top + CROP))


def crop_map(native: np.ndarray) -> np.ndarray:
    """Apply the same crop to a native-resolution map, using area resampling."""
    height, width = native.shape
    new_width, new_height, left, top = crop_geometry((width, height))
    resized = np.asarray(
        Image.fromarray(native.astype(np.float32)).resize((new_width, new_height), Image.Resampling.BOX)
    )
    return resized[top : top + CROP, left : left + CROP]


def crop_fixations(fixations: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Map native (x, y) pixels into the crop, dropping those the crop removes."""
    new_width, new_height, left, top = crop_geometry(size)
    scale = np.array([new_width / size[0], new_height / size[1]])
    points = np.floor((fixations + 0.5) * scale).astype(int) - np.array([left, top])
    return points[((points >= 0) & (points < CROP)).all(axis=1)]
