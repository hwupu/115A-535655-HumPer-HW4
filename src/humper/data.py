"""MIT1003 loaders: images, fixation points and the dataset's own fixation maps.

Each `*_fixPts.jpg` marks one pixel per aggregated fixation. Per the dataset README
the maps use the top 6 fixations of each of the 15 viewers.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
MIT_ROOT = REPO_ROOT / "data" / "mit1003"
STIMULI = "ALLSTIMULI"
FIX_MAPS = "ALLFIXATIONMAPS"

# fixPts are JPEGs. Points are isolated and bright, so any mid-range cut recovers them.
FIX_THRESHOLD = 127


def _read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"))


def load_fixations(image_id: str, root: Path = MIT_ROOT) -> np.ndarray:
    """Return unique fixation pixels as an (N, 2) int array of (x, y), native resolution."""
    points = _read_gray(Path(root) / FIX_MAPS / f"{image_id}_fixPts.jpg") > FIX_THRESHOLD
    ys, xs = np.nonzero(points)
    return np.column_stack([xs, ys])


def load_fix_map(image_id: str, root: Path = MIT_ROOT) -> np.ndarray:
    """Return the dataset's own fixation map as float32 in [0, 1]."""
    return _read_gray(Path(root) / FIX_MAPS / f"{image_id}_fixMap.jpg").astype(np.float32) / 255


def load_image(image_id: str, root: Path = MIT_ROOT) -> Image.Image:
    """Return the stimulus as an RGB PIL image."""
    with Image.open(Path(root) / STIMULI / f"{image_id}.jpeg") as im:
        return im.convert("RGB")


def load_images(root: Path = MIT_ROOT, n: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Return one row per image, sorted by id, or a seeded sample of `n` rows.

    Columns: image_id, rel_path (POSIX, relative to `root`), width, height, n_fixations.
    """
    root = Path(root)
    rows = []
    for path in sorted((root / STIMULI).glob("*.jpeg")):
        with Image.open(path) as im:
            width, height = im.size
        rows.append({
            "image_id": path.stem,
            "rel_path": f"{STIMULI}/{path.name}",
            "width": width,
            "height": height,
            "n_fixations": len(load_fixations(path.stem, root)),
        })
    df = pd.DataFrame(rows)
    if n is not None:
        df = df.sample(n=n, random_state=seed).sort_values("image_id").reset_index(drop=True)
    return df
