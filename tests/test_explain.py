"""Explainer checks. Untrained weights are enough, so none of these download anything."""

import numpy as np
import pytest
import torch
from torch import nn

from humper import explain as ex
from humper import models

CPU = torch.device("cpu")


class Toy(nn.Module):
    """A single 1×1 conv, then global average pooling, then a one-weight classifier."""

    def __init__(self, weight):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        self.fc = nn.Linear(1, 1, bias=False)
        nn.init.ones_(self.conv.weight)
        nn.init.constant_(self.fc.weight, weight)

    def forward(self, x):
        return self.fc(self.conv(x).mean(dim=(2, 3)))


@pytest.fixture(scope="module")
def resnet():
    torch.manual_seed(0)
    return models.create(models.MODELS["resnet50"], CPU, pretrained=False)


@pytest.fixture(scope="module")
def vit():
    torch.manual_seed(0)
    return models.create(models.MODELS["vit_b16"], CPU, pretrained=False)


@pytest.fixture(scope="module")
def x():
    torch.manual_seed(1)
    return torch.rand(1, 3, 224, 224)


def test_grad_cam_matches_worked_example():
    # d(output)/d(activation) = 2 / 4 per cell, so the CAM is 0.5 × activation.
    model = Toy(weight=2.0)
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    cam = ex.grad_cam_grid(model, model.conv, x, target=0)
    assert cam == pytest.approx(np.array([[0.5, 1.0], [1.5, 2.0]]))


def test_negative_evidence_gives_an_all_zero_map():
    model = Toy(weight=-2.0)
    cam = ex.grad_cam_grid(model, model.conv, torch.ones(1, 1, 2, 2), target=0)
    assert cam.sum() == 0


def test_grad_cam_matches_independent_implementation(resnet, x):
    store = {}
    layer = resnet.get_submodule("layer4")
    handle = layer.register_forward_hook(lambda _m, _i, out: store.update(a=out))
    output = resnet(x)
    handle.remove()
    gradient = torch.autograd.grad(output[0, 3], store["a"])[0]
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    expected = (weights * store["a"]).sum(dim=1).clamp(min=0)[0].detach().numpy()

    ours = ex.grad_cam_grid(resnet, layer, x, target=3)
    assert ours == pytest.approx(expected, rel=1e-4, abs=1e-6)


@pytest.mark.parametrize("name, cells", [("resnet50_gradcam", 7), ("vit_gradcam", 14)])
def test_grad_cam_depends_on_target_class(name, cells, resnet, vit, x):
    spec = ex.EXPLAINERS[name]
    model = resnet if spec.model == "resnet50" else vit
    a, b = ex.explain_grid(spec, model, x, target=1), ex.explain_grid(spec, model, x, target=900)
    assert a.shape == (cells, cells)
    assert not np.allclose(a, b)


def test_default_target_is_top1(resnet, x):
    spec = ex.EXPLAINERS["resnet50_gradcam"]
    top1 = int(resnet(x).argmax(dim=1))
    assert ex.explain_grid(spec, resnet, x) == pytest.approx(
        ex.explain_grid(spec, resnet, x, target=top1)
    )


def test_hooks_are_removed(resnet, x):
    layer = resnet.get_submodule("layer4")
    ex.grad_cam_grid(resnet, layer, x, target=0)
    assert len(layer._forward_hooks) == 0


def test_uniform_attention_gives_a_uniform_map():
    attention = torch.full((8, 5, 5), 1 / 5)
    grid = ex.attention_rollout([attention] * 3)
    assert grid.shape == (2, 2)
    assert grid.std() == pytest.approx(0, abs=1e-6)


def test_rollout_matches_worked_example_in_layer_order():
    first = torch.tensor([[0.6, 0.1, 0.1, 0.1, 0.1]] + [[0.2] * 5] * 4)
    second = torch.tensor([[0.1, 0.5, 0.2, 0.1, 0.1]] + [[0.3, 0.1, 0.2, 0.2, 0.2]] * 4)
    identity = torch.eye(5)
    expected = ((0.5 * second + 0.5 * identity) @ (0.5 * first + 0.5 * identity))[0, 1:]

    grid = ex.attention_rollout([first.unsqueeze(0), second.unsqueeze(0)])
    assert grid.ravel().tolist() == pytest.approx(expected.tolist(), rel=1e-5)


def test_captured_attention_is_valid_and_does_not_change_the_model(vit, x):
    fused = [block.attn.fused_attn for block in vit.blocks]
    with torch.no_grad():
        before = vit(x)
    attentions = ex.capture_attentions(vit, x)
    with torch.no_grad():
        after = vit(x)

    assert len(attentions) == 12
    assert attentions[0].shape == (12, 197, 197)
    assert attentions[5].sum(dim=-1) == pytest.approx(torch.ones(12, 197), abs=1e-4)
    assert [block.attn.fused_attn for block in vit.blocks] == fused
    assert torch.allclose(before, after)


def test_explain_returns_canvas(vit, x):
    canvas = ex.explain(ex.EXPLAINERS["vit_rollout"], vit, x)
    assert canvas.shape == (224, 224)
