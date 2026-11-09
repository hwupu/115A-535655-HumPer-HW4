import numpy as np
import pandas as pd
import pytest

from humper import categories as cat


def test_text_area_counts_overlap_once():
    assert cat.box_area_share([(0, 5, 0, 10), (0, 5, 0, 10)], 10, 10) == pytest.approx(0.5)
    assert cat.box_area_share([(0, 5, 0, 10), (5, 10, 0, 10)], 10, 10) == pytest.approx(1)


def test_text_boxes_outside_the_image_are_clipped():
    assert cat.box_area_share([(-5, 5, -5, 5)], 10, 10) == pytest.approx(0.25)


def test_edge_density_of_flat_and_checkerboard():
    flat = np.full((64, 64), 120, np.uint8)
    checker = (np.indices((64, 64)).sum(axis=0) // 8 % 2 * 255).astype(np.uint8)
    assert cat.edge_density(flat) == 0
    assert cat.edge_density(checker) > 0.1


def test_texture_energy_is_higher_for_noise():
    rng = np.random.default_rng(0)
    flat = np.full((64, 64), 120, np.uint8)
    noise = rng.integers(0, 255, (64, 64)).astype(np.uint8)
    assert cat.texture_energy(flat) == pytest.approx(0, abs=1e-3)
    assert cat.texture_energy(noise) > 30


def test_attention_area_small_for_a_peak_and_half_for_uniform():
    peak = np.zeros((10, 10))
    peak[3, 3] = 1
    assert cat.attention_area(peak) == pytest.approx(0.01)
    assert cat.attention_area(np.ones((10, 10))) == pytest.approx(0.5)


def test_center_offset_at_center_and_corner():
    middle = np.zeros((10, 10))
    middle[4:6, 4:6] = 1
    assert cat.center_offset(middle) == pytest.approx(0)
    corner = np.zeros((10, 10))
    corner[0, 0] = 1
    assert cat.center_offset(corner) == pytest.approx(np.hypot(4.5, 4.5) / np.hypot(5, 5))


def test_flags_use_the_right_tail():
    n = 100
    measures = pd.DataFrame({
        "image_id": [str(i) for i in range(n)],
        "face_count": [1] * 7 + [0] * 93,
        "text_area": [0.01] * 3 + [0.001] * 97,
        "attention_area": np.arange(n, dtype=float),
        "edge_density": np.arange(n, dtype=float),
        "texture_energy": np.arange(n, dtype=float),
        "center_offset": np.arange(n, dtype=float),
    })
    flags = cat.category_flags(measures)
    assert list(flags.columns) == cat.CATEGORIES
    assert flags.face.sum() == 7 and flags.text.sum() == 3
    assert flags.small_salient["0"] and not flags.small_salient["99"]
    assert flags.edge_texture_rich["99"] and not flags.edge_texture_rich["0"]
    assert flags.peripheral_attention["99"]
