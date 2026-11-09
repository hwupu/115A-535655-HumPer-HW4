"""Candidate selection and blindness of the labelling sheet."""

import numpy as np
import pandas as pd
import pytest

from humper import explain, failures, labeling

NAMES = list(explain.EXPLAINERS)


@pytest.fixture
def table():
    return pd.DataFrame({
        "image_id": [f"i{n:03d}" for n in range(200)],
        "n_fixations": [10] * 5 + [60] * 195,
    })


@pytest.fixture
def scores(table):
    rng = np.random.default_rng(0)
    rows = []
    for image_id in table.image_id:
        rows.append({"image_id": image_id, "explainer": "center", "cc": 0.4, "nss": 1.0, "sauc": 0.5})
        for name in NAMES:
            cc, nss = rng.random(2)
            rows.append({"image_id": image_id, "explainer": name, "cc": cc, "nss": nss, "sauc": 0.5})
    frame = pd.DataFrame(rows)
    # i010 is the lowest for every explainer; i020 is the highest for every explainer.
    for image_id, value in (("i010", -5.0), ("i020", 5.0)):
        mask = (frame.image_id == image_id) & frame.explainer.isin(NAMES)
        frame.loc[mask, ["cc", "nss"]] = value
    # i000 has too few fixations, and is otherwise the lowest of all.
    mask = (frame.image_id == "i000") & frame.explainer.isin(NAMES)
    frame.loc[mask, ["cc", "nss"]] = -9.0
    return frame


def test_percentile_low_for_bad_map_and_flags_center(scores):
    table = failures.alignment_percentiles(scores)
    row = table[(table.image_id == "i010") & (table.explainer == NAMES[0])].iloc[0]
    assert row.alignment_pct < 0.02
    assert row.below_center


def test_low_fixation_images_are_excluded(scores, table):
    assert "i000" not in set(failures.select_candidates(scores, table).image_id)


def test_candidates_are_unique_and_contain_planted_cases(scores, table):
    selected = failures.select_candidates(scores, table, per_group=4, contrast=2).set_index("image_id")
    assert selected.index.is_unique
    assert selected.loc["i010", "reason"] == "shared_low"
    assert selected.loc["i010", "pattern"] == "shared"
    assert selected.loc["i020", "reason"] == "shared_high"
    assert len(selected) == 4 + 3 * 4 + 2


def test_random_sample_does_not_depend_on_candidates(table):
    table = pd.DataFrame({"image_id": [f"i{n:04d}" for n in range(1003)]})
    first = labeling.random_sample(table)
    assert first == labeling.random_sample(table)
    assert len(first) == labeling.RANDOM_SAMPLE_SIZE


def test_sheet_merges_without_duplicates_and_is_shuffled():
    table = pd.DataFrame({"image_id": [f"i{n:04d}" for n in range(1003)]})
    sample = labeling.random_sample(table)
    extra = ["i0000", sample[0]]
    ids = labeling.sheet_ids(table, extra)
    assert len(ids) == len(set(ids)) == len(set(sample) | set(extra))
    assert ids != sorted(ids)


def test_sheet_entries_reveal_only_classes():
    predictions = pd.DataFrame({
        "image_id": ["a", "a"], "model": ["resnet50", "vit_b16"], "top1_name": ["cat", "dog"],
        "top1_prob": [0.5, 0.6], "top1_idx": [1, 2], "top5_idx": ["1", "2"],
    })
    assert labeling.sheet_entries(["a"], predictions) == [
        {"image_id": "a", "resnet50": "cat", "vit_b16": "dog"}
    ]
