"""Metric checks (ARCHITECTURE §7, checks 1 to 3)."""

import math

import numpy as np
import pytest
from scipy.stats import mannwhitneyu, pearsonr

from humper import alignment as al
from humper import data, maps

needs_data = pytest.mark.skipif(
    not (data.MIT_ROOT / data.STIMULI).is_dir(), reason="MIT1003 not downloaded"
)


@pytest.fixture
def a():
    return np.random.default_rng(0).random((32, 32))


@pytest.fixture
def b():
    return np.random.default_rng(1).random((32, 32))


def random_points(rng, n, size=32):
    return rng.integers(0, size, (n, 2))


def test_cc_identity_and_inverse(a):
    assert al.cc(a, a) == pytest.approx(1)
    assert al.cc(a, -a) == pytest.approx(-1)


def test_cc_is_invariant_to_scale_and_offset(a, b):
    assert al.cc(5 * a + 3, b) == pytest.approx(al.cc(a, b))


def test_cc_matches_scipy_and_is_symmetric(a, b):
    assert al.cc(a, b) == pytest.approx(pearsonr(a.ravel(), b.ravel())[0])
    assert al.cc(a, b) == pytest.approx(al.cc(b, a))


def test_nss_worked_example():
    # Map [[0, 2], [2, 4]]: mean 2, std sqrt(2). The fixation at x=1, y=1 has value 4.
    saliency = np.array([[0.0, 2.0], [2.0, 4.0]])
    assert al.nss(saliency, np.array([[1, 1]])) == pytest.approx(2 / math.sqrt(2))


def test_nss_reads_x_then_y():
    saliency = np.zeros((4, 8))
    saliency[1, 6] = 1  # row 1, column 6
    assert al.nss(saliency, np.array([[6, 1]])) > al.nss(saliency, np.array([[1, 3]]))


def test_nss_counts_repeated_fixations():
    saliency = np.array([[0.0, 2.0], [2.0, 4.0]])
    once = np.array([[1, 1], [0, 0]])
    twice = np.array([[1, 1], [1, 1], [0, 0]])
    assert al.nss(saliency, twice) > al.nss(saliency, once)


def test_nss_peaked_at_fixations_is_high():
    fixations = np.array([[3, 3], [20, 10], [28, 25]])
    saliency = np.zeros((32, 32))
    saliency[fixations[:, 1], fixations[:, 0]] = 1
    assert al.nss(saliency, fixations) > 5


def test_sauc_extremes():
    saliency = np.zeros((4, 4))
    saliency[0, 0] = 1
    high, low = np.array([[0, 0]]), np.array([[3, 3], [2, 2]])
    assert al.sauc(saliency, high, low) == 1
    assert al.sauc(saliency, low, high) == 0


def test_sauc_counts_ties_as_half():
    saliency = np.array([[1.0, 1.0], [1.0, 0.0]])
    positives = np.array([[0, 0], [1, 0]])
    negatives = np.array([[0, 1], [1, 1]])
    # Pairs: (1 vs 1) is a tie = 0.5 and (1 vs 0) is a win = 1, for each of 2 positives.
    assert al.sauc(saliency, positives, negatives) == pytest.approx(3 / 4)


def test_sauc_matches_mann_whitney():
    rng = np.random.default_rng(2)
    saliency = rng.integers(0, 6, (32, 32)).astype(float)
    pos, neg = random_points(rng, 60), random_points(rng, 500)
    u = mannwhitneyu(saliency[pos[:, 1], pos[:, 0]], saliency[neg[:, 1], neg[:, 0]]).statistic
    assert al.sauc(saliency, pos, neg) == pytest.approx(u / (len(pos) * len(neg)))


def test_sauc_of_random_map_is_chance():
    rng = np.random.default_rng(3)
    values = [al.sauc(rng.random((32, 32)), random_points(rng, 300), random_points(rng, 300))
              for _ in range(50)]
    assert np.mean(values) == pytest.approx(0.5, abs=0.02)


def test_center_prior_rewarded_by_nss_but_not_sauc():
    """Why sAUC is here: positives and negatives share one center-biased distribution."""
    rng = np.random.default_rng(4)
    size = 128
    ys, xs = np.indices((size, size))
    center = np.exp(-((xs - 64) ** 2 + (ys - 64) ** 2) / (2 * 16**2))
    draw = lambda n: np.clip(np.round(rng.normal(64, 16, (n, 2))).astype(int), 0, size - 1)
    positives, negatives = draw(3000), draw(3000)
    assert al.nss(center, positives) > 1
    assert al.sauc(center, positives, negatives) == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize("value", [0.0, 2.5])
def test_constant_map_is_undefined_for_every_metric(value, a):
    points = np.array([[1, 1], [2, 2]])
    result = al.score(np.full((32, 32), value), a, points, points)
    assert set(result) == {"cc", "nss", "sauc"}
    assert all(math.isnan(v) for v in result.values())


def test_no_fixations_is_undefined(a):
    empty = np.empty((0, 2), int)
    assert math.isnan(al.nss(a, empty))
    assert math.isnan(al.sauc(a, empty, np.array([[1, 1]])))


def test_registry_directions():
    assert all(spec.higher_is_better for spec in al.METRICS.values())


@needs_data
def test_human_map_scores_own_fixations_above_other_images():
    rows = list(data.load_images(n=21, seed=0).itertuples())
    canvas = [maps.scale_fixations(data.load_fixations(r.image_id), (r.width, r.height)) for r in rows]
    own, other, aucs = [], [], []
    for i, r in enumerate(rows):
        human = maps.human_map_canvas(data.load_fixations(r.image_id), (r.width, r.height))
        negatives = np.concatenate([c for j, c in enumerate(canvas) if j != i])
        own.append(al.nss(human, canvas[i]))
        other.append(al.nss(human, negatives))
        aucs.append(al.sauc(human, canvas[i], negatives))
    assert np.mean(own) > np.mean(other) + 0.5
    assert np.mean(aucs) > 0.6
