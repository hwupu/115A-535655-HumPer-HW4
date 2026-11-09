"""Rank explainer weaknesses by category and draw the candidate cases."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from humper import categories, data, failures, viz

OUT = data.REPO_ROOT / "results"


def main() -> None:
    table = data.load_images()
    scores = pd.read_csv(OUT / "scores.csv")
    predictions = pd.read_csv(OUT / "predictions.csv")
    flags = categories.category_flags(pd.read_csv(OUT / "categories.csv"))

    effects = failures.category_effects(scores, flags)
    effects.to_csv(OUT / "tables" / "category_effects.csv", index=False)
    cases = failures.category_cases(scores, table, flags, effects)
    (OUT / "failures").mkdir(exist_ok=True)
    cases.to_csv(OUT / "failures" / "category_cases.csv", index=False)

    figures = OUT / "figures" / "failures"
    figures.mkdir(parents=True, exist_ok=True)
    for case in cases.itertuples():
        fig = viz.case_figure(case, scores, predictions)
        fig.savefig(figures / f"{case.category}__{case.weak}__{case.rank}.png", dpi=130)
        plt.close(fig)
    top = cases[cases["rank"] == 1]
    fig = viz.overview_figure(top, scores, predictions)
    fig.savefig(figures / "overview_top1.png", dpi=75)
    plt.close(fig)

    print(effects.round(3).to_string())
    print(cases.round(3).to_string())


if __name__ == "__main__":
    main()
