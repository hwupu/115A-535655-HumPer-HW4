"""Write the shortlist of low- and high-alignment images to results/failures/candidates.csv."""

import pandas as pd

from humper import data, failures

OUT = data.REPO_ROOT / "results"


def main() -> None:
    scores = pd.read_csv(OUT / "scores.csv")
    candidates = failures.select_candidates(scores, data.load_images())
    (OUT / "failures").mkdir(exist_ok=True)
    candidates.to_csv(OUT / "failures" / "candidates.csv", index=False)
    print(candidates.groupby(["reason", "pattern"]).size().to_string())


if __name__ == "__main__":
    main()
