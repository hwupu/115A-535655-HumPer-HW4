"""Measure the image categories into results/categories.csv. Resumable."""

import csv

import pandas as pd
from tqdm import tqdm

from humper import categories, data

OUT = data.REPO_ROOT / "results" / "categories.csv"
COLUMNS = ["image_id", "face_count", "text_area", "attention_area", "edge_density", "texture_energy", "center_offset"]


def main() -> None:
    table = data.load_images()
    done = set(pd.read_csv(OUT).image_id) if OUT.exists() else set()
    todo = table[~table.image_id.isin(done)]
    print(f"{len(done)} already done, {len(todo)} to compute")
    if len(todo):
        detector, reader = categories.face_detector(), categories.text_reader()
        with OUT.open("a", newline="") as f:
            writer = csv.DictWriter(f, COLUMNS)
            if not done:
                writer.writeheader()
            for r in tqdm(todo.itertuples(), total=len(todo)):
                writer.writerow(categories.measure_image(r.image_id, (r.width, r.height), detector, reader))
                f.flush()
    flags = categories.category_flags(pd.read_csv(OUT))
    print(flags.sum().to_string())


if __name__ == "__main__":
    main()
