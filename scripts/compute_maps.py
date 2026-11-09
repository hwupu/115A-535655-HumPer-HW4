"""Compute explanation grids and predictions for MIT1003. Resumable; safe to re-run."""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np
import pandas as pd
from tqdm import tqdm

from humper import data, explain, get_device, models, store

OUT = data.REPO_ROOT / "results"
TOP_K = 5


def compute_record(image, loaded, device) -> dict[str, np.ndarray]:
    """Return every explainer's grid, plus each model's top-k classes and probabilities."""
    record, tops, tensors = {}, {}, {}
    for name, spec in models.MODELS.items():
        tensors[name] = models.preprocess(spec, image).to(device)
        tops[name] = models.predict(loaded[name], tensors[name], k=TOP_K)
        record[f"{name}__top_idx"] = np.array([t[0] for t in tops[name]])
        record[f"{name}__top_prob"] = np.array([t[2] for t in tops[name]], np.float32)
    for name, spec in explain.EXPLAINERS.items():
        target = tops[spec.model][0][0]
        record[name] = explain.explain_grid(spec, loaded[spec.model], tensors[spec.model], target)
    return record


def expected_keys() -> set[str]:
    keys = set(explain.EXPLAINERS)
    for name in models.MODELS:
        keys |= {f"{name}__top_idx", f"{name}__top_prob"}
    return keys


def predictions_table(image_ids) -> pd.DataFrame:
    rows = []
    for image_id in image_ids:
        record = store.load_record(image_id)
        for name in models.MODELS:
            indices, probs = record[f"{name}__top_idx"], record[f"{name}__top_prob"]
            rows.append({
                "image_id": image_id,
                "model": name,
                "top1_idx": int(indices[0]),
                "top1_name": models.class_name(int(indices[0])),
                "top1_prob": float(probs[0]),
                "top5_idx": " ".join(str(int(i)) for i in indices),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="process only the first N images")
    args = parser.parse_args()

    table = data.load_images()
    image_ids = table.image_id.tolist()[: args.limit]
    keys = expected_keys()
    todo = [i for i in image_ids if not store.has_record(i, keys)]
    print(f"{len(image_ids) - len(todo)} already done, {len(todo)} to compute")

    device = get_device()
    start = time.time()
    if todo:
        loaded = {name: models.create(spec, device) for name, spec in models.MODELS.items()}
        for image_id in tqdm(todo):
            store.save_record(image_id, compute_record(data.load_image(image_id), loaded, device))
    elapsed = time.time() - start

    predictions_table(image_ids).to_csv(OUT / "predictions.csv", index=False)
    zero = {
        name: sum(float(store.load_record(i)[name].max()) == 0 for i in image_ids)
        for name in explain.EXPLAINERS
    }
    info = {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_images": len(image_ids),
        "computed_this_run": len(todo),
        "seconds_this_run": round(elapsed, 1),
        "device": device.type,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("torch", "timm", "grad-cam", "numpy", "pillow", "scipy")},
        "timm_ids": {name: spec.timm_id for name, spec in models.MODELS.items()},
        "all_zero_grids": zero,
    }
    (OUT / "maps_run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
