"""Spatial pipeline checks. A geometry bug here would shift every alignment score."""

import numpy as np
import pytest
from PIL import Image

from humper import data, maps

needs_data = pytest.mark.skipif(
    not (data.MIT_ROOT / data.STIMULI).is_dir(), reason="MIT1003 not downloaded"
)


def center_of_mass(m):
    ys, xs = np.indices(m.shape)
    return np.array([(xs * m).sum(), (ys * m).sum()]) / m.sum()


def test_model_input_is_whole_image_resized():
    image = Image.new("RGB", (1024, 768))
    image.paste((255, 255, 255), (0, 0, 10, 768))  # a strip a center crop would remove
    out = maps.model_input(image)
    assert out.size == (224, 224)
    assert np.asarray(out)[112, 0, 0] > 200


def test_human_canvas_shape_and_constant_preservation():
    m = maps.human_map_canvas(np.array([[512, 100]]), (1024, 768), sigma=24)
    assert m.shape == (224, 224)
    flat = maps._resize(np.full((768, 1024), 0.5), (224, 224), Image.Resampling.BOX)
    assert flat == pytest.approx(np.full((224, 224), 0.5), abs=1e-5)


@pytest.mark.parametrize("size", [(1024, 768), (768, 1024), (1024, 576)])
@pytest.mark.parametrize("xy", [(200, 150), (500, 400), (700, 500)])
def test_human_canvas_peak_matches_scaled_fixation(size, xy):
    fixations = np.array([xy])
    m = maps.human_map_canvas(fixations, size)
    peak = np.array(np.unravel_index(m.argmax(), m.shape))[::-1]
    assert np.abs(peak - maps.scale_fixations(fixations, size)[0]).max() <= 1


def test_model_grid_center_stays_at_canvas_center():
    grid = np.zeros((7, 7))
    grid[3, 3] = 1
    m = maps.model_map_canvas(grid)
    assert m.shape == (224, 224)
    assert center_of_mass(m) == pytest.approx([111.5, 111.5], abs=0.05)


@pytest.mark.parametrize("cells", [7, 14])
def test_grid_round_trip_has_no_shift(cells):
    """A half-pixel grid error between the two paths would show up here."""
    human = maps.human_map_canvas(np.array([[250, 300], [700, 200]]), (1024, 768), sigma=60)
    back = maps.through_grid(human, cells)
    assert np.abs(center_of_mass(back) - center_of_mass(human)).max() < 1


def test_coarser_grid_loses_more_detail():
    human = maps.human_map_canvas(np.array([[250, 300], [700, 200]]), (1024, 768))
    cc = [np.corrcoef(maps.through_grid(human, c).ravel(), human.ravel())[0, 1] for c in (7, 14)]
    assert cc[0] < cc[1]


def test_zscore_normalizes():
    z = maps.zscore(np.arange(16.0).reshape(4, 4))
    assert z.mean() == pytest.approx(0)
    assert z.std() == pytest.approx(1)


@pytest.mark.parametrize("value", [0.0, 3.0])
def test_zscore_of_constant_map_is_nan(value):
    assert np.isnan(maps.zscore(np.full((7, 7), value))).all()


def test_to_display_range_and_constant():
    d = maps.to_display(np.array([[2.0, 4.0], [6.0, 10.0]]))
    assert (d.min(), d.max()) == (0, 1)
    assert maps.to_display(np.full((3, 3), 5.0)).sum() == 0


def test_warp_to_image_uses_width_height_order():
    assert maps.warp_to_image(np.ones((224, 224)), (1024, 768)).shape == (768, 1024)


@needs_data
def test_fixations_fall_on_high_canvas_values():
    for row in data.load_images(n=10, seed=0).itertuples():
        size = (row.width, row.height)
        fixations = data.load_fixations(row.image_id)
        m = maps.human_map_canvas(fixations, size)
        x, y = maps.scale_fixations(fixations, size).T
        assert m[y, x].mean() > 2 * m.mean()
