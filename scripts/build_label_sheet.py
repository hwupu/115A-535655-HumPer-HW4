"""Write the candidate shortlist and the blind labelling page."""

import pandas as pd

from humper import data, failures, labeling

OUT = data.REPO_ROOT / "results"


def main() -> None:
    table = data.load_images()
    scores = pd.read_csv(OUT / "scores.csv")
    predictions = pd.read_csv(OUT / "predictions.csv")

    candidates = failures.select_candidates(scores, table)
    (OUT / "failures").mkdir(exist_ok=True)
    candidates.to_csv(OUT / "failures" / "candidates.csv", index=False)

    ids = labeling.sheet_ids(table, candidates.image_id.tolist())
    pd.DataFrame({"image_id": ids}).to_csv(data.REPO_ROOT / "annotations" / "sheet_ids.csv", index=False)

    directory = OUT / "labeling"
    directory.mkdir(exist_ok=True)
    page = directory / "label_sheet.html"
    page.write_text(labeling.render(labeling.sheet_entries(ids, predictions)), encoding="utf-8")
    print(f"{len(ids)} images on the sheet")
    print(f"Open {page.relative_to(data.REPO_ROOT)} in a browser")


if __name__ == "__main__":
    main()
