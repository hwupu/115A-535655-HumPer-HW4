"""Loader and human-map checks. Real-data tests skip when data/mit1003 is absent."""

import numpy as np
import pytest
from PIL import Image

from humper import data, maps

HAS_DATA = (data.MIT_ROOT / data.STIMULI).is_dir()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="MIT1003 not downloaded")


@pytest.fixture
def fake_root(tmp_path):
    (tmp_path / data.STIMULI).mkdir()
    (tmp_path / data.FIX_MAPS).mkdir()
    Image.new("RGB", (40, 20), "gray").save(tmp_path / data.STIMULI / "a.jpeg")
    points = np.zeros((20, 40), np.uint8)
    points[5, 30] = 255
    points[15, 10] = 255
    Image.fromarray(points).save(tmp_path / data.FIX_MAPS / "a_fixPts.jpg", quality=100)
    return tmp_path


def test_fixations_are_xy(fake_root):
    xy = data.load_fixations("a", fake_root)
    assert {tuple(p) for p in xy.tolist()} == {(30, 5), (10, 15)}


def test_load_images_table(fake_root):
    row = data.load_images(fake_root).iloc[0]
    assert (row.image_id, row.width, row.height, row.n_fixations) == ("a", 40, 20, 2)
    assert row.rel_path == "ALLSTIMULI/a.jpeg"


def test_human_map_shape_peak_and_location():
    m = maps.human_map_native(np.array([[30, 5]]), (40, 20), sigma=2)
    assert m.shape == (20, 40)
    assert m.max() == pytest.approx(1)
    assert np.unravel_index(m.argmax(), m.shape) == (5, 30)


def test_human_map_without_fixations_is_zero():
    assert maps.human_map_native(np.empty((0, 2), int), (40, 20)).sum() == 0


def test_scale_fixations_keeps_corners_and_range():
    xy = np.array([[0, 0], [1023, 767], [512, 384]])
    assert maps.scale_fixations(xy, (1024, 768)).tolist() == [[0, 0], [223, 223], [112, 112]]


@needs_data
def test_dataset_is_complete():
    assert len(list((data.MIT_ROOT / data.STIMULI).glob("*.jpeg"))) == 1003


@needs_data
def test_our_human_map_matches_dataset_map():
    """ARCHITECTURE §7 check 4, on a seeded sample."""
    for row in data.load_images(n=10, seed=0).itertuples():
        ours = maps.human_map_native(data.load_fixations(row.image_id), (row.width, row.height))
        theirs = data.load_fix_map(row.image_id)
        assert np.corrcoef(ours.ravel(), theirs.ravel())[0, 1] > 0.99
