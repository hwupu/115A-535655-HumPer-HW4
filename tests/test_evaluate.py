"""Reference-map logic and reproducibility of the committed scores."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import wilcoxon

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


def long_scores(a, b):
    n = len(a)
    return pd.DataFrame({
        "image_id": list(range(n)) * 2, "kind": ["explainer"] * 2 * n,
        "explainer": ["a"] * n + ["b"] * n,
        "cc": list(a) + list(b), "nss": list(a) + list(b), "sauc": list(a) + list(b),
    })


def test_bootstrap_interval_contains_the_mean_and_narrows_with_n():
    rng = np.random.default_rng(0)
    small, large = rng.normal(size=50), rng.normal(size=5000)
    low, high = evaluate.bootstrap_ci(small)
    assert low < small.mean() < high
    low2, high2 = evaluate.bootstrap_ci(large)
    assert high2 - low2 < high - low


def test_summary_reports_every_metric_and_ignores_nan():
    table = evaluate.summary(long_scores([1.0, 2.0, np.nan], [3.0, 3.0, 3.0]), n_boot=100)
    row = table[(table.explainer == "a") & (table.metric == "cc")].iloc[0]
    assert (row.n, row["mean"]) == (2, 1.5)
    assert len(table) == 6


def test_paired_difference_matches_scipy_and_drops_nan_pairs():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.5, 1, 200), rng.normal(0, 1, 200)
    a[0] = np.nan
    result = evaluate.paired(long_scores(a, b), "a", "b", "cc")
    assert result["n"] == 199
    assert result["mean_difference"] == pytest.approx((a[1:] - b[1:]).mean())
    assert result["wilcoxon_p"] == pytest.approx(wilcoxon(a[1:] - b[1:]).pvalue)


def test_holm_worked_example():
    adjusted = evaluate.holm([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


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
