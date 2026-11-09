import math

import numpy as np
import pytest
import torch
from PIL import Image

from humper import models, probes

CPU = torch.device("cpu")


def test_top_fraction_mask_has_exact_area_and_picks_highest():
    saliency = np.arange(100, dtype=float).reshape(10, 10)
    mask = probes.top_fraction_mask(saliency, 0.2)
    assert mask.sum() == 20
    assert mask[9, 9] and not mask[0, 0]


def test_top_fraction_mask_handles_ties():
    assert probes.top_fraction_mask(np.zeros((10, 10)), 0.2).sum() == 20


def test_random_masks_are_seeded_and_same_area():
    first = probes.random_region_masks(3, count=2, canvas=64)
    again = probes.random_region_masks(3, count=2, canvas=64)
    other = probes.random_region_masks(4, count=2, canvas=64)
    assert all(m.sum() == round(0.2 * 64 * 64) for m in first)
    assert (first[0] == again[0]).all()
    assert not (first[0] == other[0]).all()


@pytest.mark.parametrize("fill", probes.FILLS)
def test_occlude_changes_only_masked_pixels(fill):
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, (64, 64, 3)).astype(np.uint8))
    mask = np.zeros((64, 64), bool)
    mask[10:20, 10:20] = True
    original, result = np.asarray(image), np.asarray(probes.occlude(image, mask, fill))
    assert (result[~mask] == original[~mask]).all()
    assert (result[mask] != original[mask]).any()


def test_unknown_fill_is_rejected():
    with pytest.raises(ValueError):
        probes.occlude(Image.new("RGB", (4, 4)), np.zeros((4, 4), bool), "noise")


def test_target_probabilities_are_valid_and_unchanged_by_empty_mask():
    spec = models.MODELS["resnet50"]
    torch.manual_seed(0)
    model = models.create(spec, CPU, pretrained=False)
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, (224, 224, 3)).astype(np.uint8))
    blank = probes.occlude(image, np.zeros((224, 224), bool), "blur")
    p = probes.target_probabilities(model, spec, [image, blank], target=7, device=CPU)
    assert p.shape == (2,) and (0 <= p).all() and (p <= 1).all()
    assert p[0] == pytest.approx(p[1], abs=1e-6)


def test_rank_similarity():
    a = np.arange(16.0).reshape(4, 4)
    assert probes.rank_similarity(a, a**3) == pytest.approx(1)
    assert probes.rank_similarity(a, -a) == pytest.approx(-1)
    assert math.isnan(probes.rank_similarity(a, np.ones((4, 4))))
