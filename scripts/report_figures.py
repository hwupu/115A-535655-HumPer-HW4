"""Draw the final-case figure for the report from annotations/case_notes.csv."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from humper import data, viz

OUT = data.REPO_ROOT / "results"


def main() -> None:
    cases = pd.read_csv(data.REPO_ROOT / "annotations" / "case_notes.csv")
    scores = pd.read_csv(OUT / "scores.csv")
    predictions = pd.read_csv(OUT / "predictions.csv")
    fig = viz.overview_figure(cases, scores, predictions)
    fig.savefig(OUT / "figures" / "failures" / "report_cases.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
