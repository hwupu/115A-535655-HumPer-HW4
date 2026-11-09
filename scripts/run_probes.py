"""Occlusion on every image, and a weight-randomization check on a seeded sample."""

import pandas as pd
import torch
from tqdm import tqdm

from humper import data, evaluate, explain, get_device, maps, models, probes, store

OUT = data.REPO_ROOT / "results"
RANDOMIZATION_IMAGES = 100
SEED = 0


def main() -> None:
    table = data.load_images()
    device = get_device()
    loaded = {name: models.create(spec, device) for name, spec in models.MODELS.items()}

    rows = []
    for index, r in enumerate(tqdm(table.itertuples(), total=len(table))):
        size = (r.width, r.height)
        canvas_image = maps.model_input(data.load_image(r.image_id))
        human = maps.human_map_canvas(data.load_fixations(r.image_id), size)
        rows += probes.occlusion_rows(
            r.image_id, canvas_image, human, store.load_record(r.image_id), loaded, device, SEED + index
        )
    occlusion = pd.DataFrame(rows)
    occlusion.to_csv(OUT / "occlusion.csv", index=False)

    summary = []
    for (model, fill, mask), part in occlusion.groupby(["model", "fill", "mask"]):
        low, high = evaluate.bootstrap_ci(part["drop"].to_numpy())
        summary.append({
            "model": model, "fill": fill, "mask": mask, "n": len(part),
            "mean_drop": part["drop"].mean(), "ci_low": low, "ci_high": high,
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "tables" / "occlusion_summary.csv", index=False)
    print(summary.round(4).to_string())

    torch.manual_seed(SEED)
    randomized = {name: models.create(spec, device, pretrained=False) for name, spec in models.MODELS.items()}
    similarity = []
    for r in tqdm(table.sample(RANDOMIZATION_IMAGES, random_state=SEED).itertuples(), total=RANDOMIZATION_IMAGES):
        image = data.load_image(r.image_id)
        record = store.load_record(r.image_id)
        for name, spec in explain.EXPLAINERS.items():
            x = models.preprocess(models.MODELS[spec.model], image).to(device)
            target = int(record[f"{spec.model}__top_idx"][0])
            similarity.append({
                "image_id": r.image_id, "explainer": name,
                "spearman": probes.randomization_similarity(spec, loaded[spec.model], randomized[spec.model], x, target),
            })
    similarity = pd.DataFrame(similarity)
    result = similarity.groupby("explainer").spearman.agg(["count", "mean", "std"])
    result.to_csv(OUT / "tables" / "randomization_similarity.csv")
    print(result.round(3).to_string())


if __name__ == "__main__":
    main()
