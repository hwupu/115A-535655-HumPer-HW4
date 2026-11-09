"""Score the stored maps and all reference maps into results/scores.csv."""

import json

from tqdm import tqdm

from humper import alignment, data, evaluate

OUT = data.REPO_ROOT / "results"
SEED = 0


def main() -> None:
    table = data.load_images()
    scores = evaluate.score_all(table, seed=SEED, progress=tqdm)
    maps_info = json.loads((OUT / "maps_run_info.json").read_text())
    scores["device"] = maps_info["device"]
    scores.to_csv(OUT / "scores.csv", index=False)

    names = list(alignment.METRICS)
    info = {
        "seed": SEED,
        "n_images": len(table),
        "n_rows": len(scores),
        "undefined_by_metric": scores[names].isna().sum().to_dict(),
        "maps_date_utc": maps_info["date_utc"],
    }
    (OUT / "scores_run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(scores.groupby(["kind", "explainer"])[names].mean().round(3).to_string())
    print(info)


if __name__ == "__main__":
    main()
