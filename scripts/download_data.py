"""Download MIT1003 into data/mit1003 and verify it. Safe to re-run."""

import hashlib
import urllib.request
import zipfile
from pathlib import Path

from humper.data import FIX_MAPS, MIT_ROOT, REPO_ROOT, STIMULI

BASE_URL = "https://people.csail.mit.edu/tjudd/WherePeopleLook"
DOWNLOAD_DIR = REPO_ROOT / "data" / "_download"
ARCHIVES = {
    f"{STIMULI}.zip": "2bd451ea3aed9422c463d5cd47f9bf65d1fec53b458afb5ef65640a2a8857788",
    f"{FIX_MAPS}.zip": "1c87b8c8023cd1e93fd3f8a341829cab54972190040699ba6dc8f04c0c997cd9",
}
N_IMAGES = 1003


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name: str, expected: str) -> Path:
    path = DOWNLOAD_DIR / name
    if not (path.exists() and sha256(path) == expected):
        print(f"downloading {name}")
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", path)
        if (actual := sha256(path)) != expected:
            raise RuntimeError(f"{name}: sha256 {actual} does not match the pinned value")
    return path


def extract(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if ".svn/" not in member and not member.startswith("__MACOSX"):
                archive.extract(member, MIT_ROOT)


def main() -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for name, expected in ARCHIVES.items():
        extract(fetch(name, expected))

    stems = {p.stem for p in (MIT_ROOT / STIMULI).glob("*.jpeg")}
    if len(stems) != N_IMAGES:
        raise RuntimeError(f"expected {N_IMAGES} images, found {len(stems)}")
    missing = [
        s for s in stems for kind in ("fixPts", "fixMap")
        if not (MIT_ROOT / FIX_MAPS / f"{s}_{kind}.jpg").exists()
    ]
    if missing:
        raise RuntimeError(f"{len(missing)} fixation files missing, e.g. {missing[0]}")
    print(f"ok: {len(stems)} images with fixation points and maps in {MIT_ROOT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
