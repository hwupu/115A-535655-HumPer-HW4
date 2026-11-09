"""B-25: compare no-crop (D-03) with Resize(256) + CenterCrop(224) on a seeded sample."""

import numpy as np
import pandas as pd
from tqdm import tqdm

from humper import alignment, data, evaluate, explain, get_device, maps, models, sensitivity, store

OUT = data.REPO_ROOT / "results"
N_IMAGES = 200
SEED = 0


def main() -> None:
    sample = data.load_images().sample(N_IMAGES, random_state=SEED).reset_index(drop=True)
    device = get_device()
    loaded = {name: models.create(spec, device) for name, spec in models.MODELS.items()}

    humans, points, kept = {"no_crop": {}, "crop": {}}, {"no_crop": {}, "crop": {}}, []
    for r in sample.itertuples():
        size = (r.width, r.height)
        native = data.load_fixations(r.image_id)
        native_map = maps.human_map_native(native, size)
        humans["no_crop"][r.image_id] = maps.human_map_canvas(native, size)
        humans["crop"][r.image_id] = sensitivity.crop_map(native_map)
        points["no_crop"][r.image_id] = maps.scale_fixations(native, size)
        points["crop"][r.image_id] = sensitivity.crop_fixations(native, size)
        kept.append(len(points["crop"][r.image_id]) / len(native))

    rows = []
    for r in tqdm(sample.itertuples(), total=N_IMAGES):
        image = data.load_image(r.image_id)
        record = store.load_record(r.image_id)
        cropped = sensitivity.crop_image(image)
        for condition in ("no_crop", "crop"):
            human, fixations = humans[condition][r.image_id], points[condition][r.image_id]
            negatives = alignment.other_fixations(points[condition], r.image_id)
            center = np.mean([m for i, m in humans[condition].items() if i != r.image_id], axis=0)
            scored = {"center": center}
            for name, spec in explain.EXPLAINERS.items():
                if condition == "no_crop":
                    scored[name] = maps.model_map_canvas(record[name])
                else:
                    model_spec = models.MODELS[spec.model]
                    x = models.preprocess(model_spec, cropped).to(device)
                    scored[name] = explain.explain(spec, loaded[spec.model], x)
            for name, saliency in scored.items():
                rows.append({
                    "image_id": r.image_id, "explainer": f"{name}__{condition}", "kind": condition,
                    **alignment.score(saliency, human, fixations, negatives),
                })
    scores = pd.DataFrame(rows)

    result = []
    for name in [*explain.EXPLAINERS, "center"]:
        for metric in alignment.METRICS:
            comparison = evaluate.paired(scores, f"{name}__crop", f"{name}__no_crop", metric)
            means = scores[scores.explainer.isin([f"{name}__crop", f"{name}__no_crop"])].groupby("kind")[metric].mean()
            result.append({
                "explainer": name, "metric": metric, "n": comparison["n"],
                "no_crop": means["no_crop"], "crop": means["crop"],
                "crop_minus_no_crop": comparison["mean_difference"],
                "ci_low": comparison["ci_low"], "ci_high": comparison["ci_high"],
                "wilcoxon_p": comparison["wilcoxon_p"],
            })
    table = pd.DataFrame(result)
    table["holm_p"] = evaluate.holm(table.wilcoxon_p)
    table.to_csv(OUT / "tables" / "crop_sensitivity.csv", index=False)
    print(table.round(3).to_string())
    print("share of fixations kept by the crop:", round(float(np.mean(kept)), 3), "min", round(float(np.min(kept)), 3))


if __name__ == "__main__":
    main()
