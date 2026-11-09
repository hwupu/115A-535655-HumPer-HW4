"""Model registry checks. Structural tests need no weights; the last one downloads them."""

import pytest
import torch
from PIL import Image

from humper import data, get_device, models

needs_data = pytest.mark.skipif(
    not (data.MIT_ROOT / data.STIMULI).is_dir(), reason="MIT1003 not downloaded"
)
SPECS = list(models.MODELS.values())


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_target_layer_gives_the_declared_grid(spec):
    model = models.create(spec, torch.device("cpu"), pretrained=False)
    seen = {}
    layer = model.get_submodule(spec.target_layer)
    handle = layer.register_forward_hook(lambda _m, _i, out: seen.update(out=out))
    with torch.no_grad():
        model(torch.zeros(1, 3, 224, 224))
    handle.remove()
    features = spec.reshape(seen["out"]) if spec.reshape else seen["out"]
    assert features.shape[-2:] == (spec.grid, spec.grid)


def test_each_model_uses_its_own_normalization():
    white = Image.new("RGB", (64, 48), "white")
    resnet = models.preprocess(models.MODELS["resnet50"], white)
    vit = models.preprocess(models.MODELS["vit_b16"], white)
    assert resnet.shape == vit.shape == (1, 3, 224, 224)
    assert vit[0, :, 0, 0].tolist() == pytest.approx([1, 1, 1])
    assert resnet[0, 0, 0, 0].item() == pytest.approx((1 - 0.485) / 0.229, abs=1e-4)


def test_vit_reshape_drops_cls_and_keeps_patch_order():
    tokens = torch.arange(1 + 4, dtype=torch.float32).reshape(1, 5, 1)
    grid = models.vit_tokens_to_grid(tokens)
    assert grid.shape == (1, 1, 2, 2)
    assert grid.flatten().tolist() == [1, 2, 3, 4]


def test_vit_reshape_rejects_non_square():
    with pytest.raises(ValueError):
        models.vit_tokens_to_grid(torch.zeros(1, 4, 8))


@needs_data
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_no_crop_predictions_are_sensible(spec):
    """Two images inspected by eye in B-05, chosen by a fixed seed and position."""
    ids = data.load_images(n=100, seed=0).image_id
    device = get_device()
    model = models.create(spec, device)
    for position, expected in ((8, "gondola"), (48, "crate")):
        image = data.load_image(ids.iloc[position])
        top = models.predict(model, models.preprocess(spec, image).to(device))
        assert top[0][1] == expected
