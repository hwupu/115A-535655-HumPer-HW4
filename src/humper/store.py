"""Per-image storage of raw explanation grids and predictions (results/maps/, gitignored)."""

import os
from pathlib import Path

import numpy as np

from humper.data import REPO_ROOT

MAPS_DIR = REPO_ROOT / "results" / "maps"


def _path(image_id: str, directory: Path) -> Path:
    return Path(directory) / f"{image_id}.npz"


def save_record(image_id: str, arrays: dict[str, np.ndarray], directory: Path = MAPS_DIR) -> None:
    """Write one image's arrays atomically, so an interrupted run leaves no partial file."""
    path = _path(image_id, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(temporary, path)


def load_record(image_id: str, directory: Path = MAPS_DIR) -> dict[str, np.ndarray]:
    with np.load(_path(image_id, directory)) as record:
        return {key: record[key] for key in record.files}


def has_record(image_id: str, keys: set[str], directory: Path = MAPS_DIR) -> bool:
    """Return True if this image was stored with at least `keys`."""
    path = _path(image_id, directory)
    if not path.exists():
        return False
    with np.load(path) as record:
        return keys <= set(record.files)
