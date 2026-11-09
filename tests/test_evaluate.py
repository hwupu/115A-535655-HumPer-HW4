"""Reference-map logic and reproducibility of the committed scores."""

import numpy as np
import pandas as pd
import pytest

from humper import data, evaluate, store

SCORES = data.REPO_ROOT / "results" / "scores.csv"


def test_derangement_has_no_fixed_points_and_is_a_permutation():
    partner = evaluate.derangement(1003, seed=0)
    assert (partner != np.arange(1003)).all()
    assert sorted(partner.tolist()) == list(range(1003))


def test_derangement_is_seeded():
    assert (evaluate.derangement(50, 1) == evaluate.derangement(50, 1)).all()
    assert (evaluate.derangement(50, 1) != evaluate.derangement(50, 2)).any()


def test_center_excludes_the_image_itself():
    human = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    inputs = evaluate.Inputs(["a", "b", "c"], human, {}, human.sum(axis=0, dtype=np.float64))
    assert inputs.center(0) == pytest.approx((human[1] + human[2]) / 2)


@pytest.mark.skipif(
    not (SCORES.exists() and store.MAPS_DIR.is_dir() and (data.MIT_ROOT / data.STIMULI).is_dir()),
    reason="needs scores, stored maps and MIT1003",
)
def test_committed_scores_reproduce_from_stored_maps():
    """ARCHITECTURE §7 check 10."""
    committed = pd.read_csv(SCORES)
    table = data.load_images()
    inputs = evaluate.build_inputs(table)
    partner = evaluate.derangement(len(table), seed=0)
    for index in (0, 500, 1002):
        image_id = inputs.image_ids[index]
        rows = evaluate.score_image(index, int(partner[index]), inputs, store.load_record(image_id))
        expected = committed[committed.image_id == image_id].set_index("explainer")
        assert len(rows) == len(expected)
        for row in rows:
            for metric in ("cc", "nss", "sauc"):
                assert row[metric] == pytest.approx(expected.loc[row["explainer"], metric], abs=1e-9)
