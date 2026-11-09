"""B-08: run the whole pipeline on a few images, on CPU and on the accelerator."""

import argparse

import numpy as np
import pandas as pd

from humper import alignment, data, explain, get_device, maps, models

OUT = data.REPO_ROOT / "results"


def run(images, negatives_by_id, device):
    """Return (rows of scores and predictions, {(image_id, explainer): canvas map})."""
    loaded = {name: models.create(spec, device) for name, spec in models.MODELS.items()}
    rows, canvases = [], {}
    for r in images.itertuples():
        image = data.load_image(r.image_id)
        size = (r.width, r.height)
        native = data.load_fixations(r.image_id)
        human = maps.human_map_canvas(native, size)
        fixations = maps.scale_fixations(native, size)
        for name, spec in explain.EXPLAINERS.items():
            model_spec = models.MODELS[spec.model]
            x = models.preprocess(model_spec, image).to(device)
            canvas = explain.explain(spec, loaded[spec.model], x)
            canvases[r.image_id, name] = canvas
            top = models.predict(loaded[spec.model], x)[0]
            rows.append({
                "image_id": r.image_id, "explainer": name, "device": device.type,
                "top1": top[0], "top1_prob": top[2],
                **alignment.score(canvas, human, fixations, negatives_by_id[r.image_id]),
            })
    return pd.DataFrame(rows), canvases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    table = data.load_images()
    canvas_fix = {
        r.image_id: maps.scale_fixations(data.load_fixations(r.image_id), (r.width, r.height))
        for r in table.itertuples()
    }
    images = table.sample(n=args.limit, random_state=0)
    negatives = {i: alignment.other_fixations(canvas_fix, i) for i in images.image_id}

    results = {}
    canvases = {}
    devices = ["cpu"] + ([get_device().type] if get_device().type != "cpu" else [])
    for name in devices:
        results[name], canvases[name] = run(images, negatives, get_device(name))
    scores = pd.concat(results.values(), ignore_index=True)
    scores.to_csv(OUT / "smoke_scores.csv", index=False)

    main_device = devices[-1]
    print(results[main_device].groupby("explainer")[list(alignment.METRICS)].agg(["mean", "std"]).round(3))
    print("undefined scores:", int(results[main_device][list(alignment.METRICS)].isna().sum().sum()))
    if len(devices) == 2:
        cpu, other = results["cpu"], results[main_device]
        print("top-1 agreement:", float((cpu.top1 == other.top1).mean()))
        print("max score difference:", (cpu[list(alignment.METRICS)] - other[list(alignment.METRICS)]).abs().max().round(6).to_dict())
        relative = []
        for key, a in canvases["cpu"].items():
            b = canvases[main_device][key]
            relative.append(np.abs(a - b).max() / max(a.max(), 1e-12))
        print("max map difference (fraction of map peak):", float(np.max(relative)))


if __name__ == "__main__":
    main()
