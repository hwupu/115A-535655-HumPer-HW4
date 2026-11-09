import numpy as np
import pandas as pd
import pytest
from PIL import Image

from humper import evaluate, sensitivity as sens


def test_crop_geometry_for_landscape():
    # Short side 768 becomes 256, so 1024 becomes 341; the 224 crop is centered.
    assert sens.crop_geometry((1024, 768)) == (341, 256, 58, 16)


def test_crop_image_and_map_shapes_match():
    assert sens.crop_image(Image.new("RGB", (1024, 768))).size == (224, 224)
    assert sens.crop_map(np.ones((768, 1024))).shape == (224, 224)


def test_fixations_map_to_the_crop_and_outside_ones_are_dropped():
    xy = np.array([[512, 384], [0, 384], [1023, 384]])
    assert sens.crop_fixations(xy, (1024, 768)).tolist() == [[112, 112]]


def test_crop_map_and_fixations_align():
    native = np.zeros((768, 1024), np.float32)
    native[300:320, 600:620] = 1
    cropped = sens.crop_map(native)
    ys, xs = np.nonzero(cropped > 0.5)
    point = sens.crop_fixations(np.array([[610, 310]]), (1024, 768))[0]
    assert abs(xs.mean() - point[0]) <= 1 and abs(ys.mean() - point[1]) <= 1


def test_spearman_factors_find_a_planted_relation():
    n = 300
    rng = np.random.default_rng(0)
    ids = [str(i) for i in range(n)]
    factor = pd.Series(rng.random(n), index=ids)
    scores = pd.DataFrame({
        "image_id": ids, "explainer": ["a"] * n, "cc": factor.values, "nss": -factor.values,
        "sauc": rng.random(n),
    })
    table = evaluate.spearman_factor(scores, "a", "f", factor, n_boot=100).set_index("metric")
    assert table.loc["cc", "rho"] == pytest.approx(1)
    assert table.loc["nss", "rho"] == pytest.approx(-1)
    assert abs(table.loc["sauc", "rho"]) < 0.2
    assert table.loc["cc", "n"] == n
