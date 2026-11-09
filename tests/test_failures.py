"""Candidate selection."""

import numpy as np
import pandas as pd
import pytest

from humper import explain, failures

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


def planted_category(scores):
    """Category c holds i100..i139, where the first explainer is pushed down and the last up."""
    frame = scores.copy()
    members = [f"i{n}" for n in range(100, 140)]
    mask = frame.image_id.isin(members)
    frame.loc[mask & (frame.explainer == NAMES[0]), ["cc", "nss"]] -= 3
    frame.loc[mask & (frame.explainer == NAMES[-1]), ["cc", "nss"]] += 3
    ids = sorted(frame.image_id.unique())
    return frame, pd.DataFrame({"c": [i in members for i in ids]}, index=ids)


def test_category_effect_finds_the_harmed_explainer(scores):
    frame, flags = planted_category(scores)
    effects = failures.category_effects(frame, flags, n_boot=200)
    worst = effects.iloc[0]
    assert (worst.explainer, worst.category, worst.n_inside) == (NAMES[0], "c", 40)
    assert worst.difference < -0.3
    assert worst.ci_low < worst.difference < worst.ci_high < 0
    assert effects.iloc[-1].explainer == NAMES[-1]


def test_category_cases_pair_weak_and_strong(scores, table):
    frame, flags = planted_category(scores)
    effects = failures.category_effects(frame, flags, n_boot=50)
    cases = failures.category_cases(frame, table, flags, effects, per_pair=3)
    assert (cases.weak == NAMES[0]).all() and (cases.strong == NAMES[-1]).all()
    assert len(cases) == 3 and cases.image_id.is_unique
    assert cases["rank"].tolist() == [1, 2, 3]
    assert cases.gap.is_monotonic_decreasing
    assert (cases.holm_p < 0.05).all()
    assert set(cases.image_id) <= {f"i{n}" for n in range(100, 140)}


def test_category_cases_skip_unsupported_effects(scores, table):
    frame, flags = planted_category(scores)
    effects = failures.category_effects(frame, flags, n_boot=50)
    effects["holm_p"] = 1.0
    assert failures.category_cases(frame, table, flags, effects).empty


def test_low_fixation_images_are_excluded(scores, table):
    assert "i000" not in set(failures.select_candidates(scores, table).image_id)


def test_candidates_are_unique_and_contain_planted_cases(scores, table):
    selected = failures.select_candidates(scores, table, per_group=4, contrast=2).set_index("image_id")
    assert selected.index.is_unique
    assert selected.loc["i010", "reason"] == "shared_low"
    assert selected.loc["i010", "pattern"] == "shared"
    assert selected.loc["i020", "reason"] == "shared_high"
    assert len(selected) == 4 + 3 * 4 + 2
