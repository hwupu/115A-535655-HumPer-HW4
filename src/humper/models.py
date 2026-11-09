"""Pretrained model registry. Each model keeps its own normalization from timm."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import timm
import torch
from PIL import Image
from timm.data import ImageNetInfo

from humper import maps


def vit_tokens_to_grid(tokens: torch.Tensor, prefix_tokens: int = 1) -> torch.Tensor:
    """Turn ViT tokens (B, 1 + N, C) into a feature map (B, C, H, W), dropping the CLS token."""
    patches = tokens[:, prefix_tokens:]
    side = int(patches.shape[1] ** 0.5)
    if side**2 != patches.shape[1]:
        raise ValueError(f"{patches.shape[1]} patch tokens do not form a square grid")
    return patches.reshape(patches.shape[0], side, side, patches.shape[2]).permute(0, 3, 1, 2)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    timm_id: str
    family: str
    target_layer: str  # for Grad-CAM
    grid: int  # side of the explanation grid at 224×224 input
    reshape: Callable[[torch.Tensor], torch.Tensor] | None = None


MODELS = {
    spec.name: spec
    for spec in (
        ModelSpec("resnet50", "resnet50.tv2_in1k", "cnn", "layer4", 7),
        ModelSpec(
            "vit_b16", "vit_base_patch16_224.augreg2_in21k_ft_in1k", "vit", "blocks.11.norm1", 14,
            vit_tokens_to_grid,
        ),
    )
}

_imagenet = ImageNetInfo()


def create(spec: ModelSpec, device: torch.device, pretrained: bool = True) -> torch.nn.Module:
    """Return the model in eval mode on `device`."""
    return timm.create_model(spec.timm_id, pretrained=pretrained).eval().to(device)


def preprocess(spec: ModelSpec, image: Image.Image) -> torch.Tensor:
    """Return a (1, 3, 224, 224) tensor: whole-image resize, then this model's normalization."""
    cfg = timm.get_pretrained_cfg(spec.timm_id)
    array = np.asarray(maps.model_input(image.convert("RGB")), dtype=np.float32) / 255
    mean, std = np.array(cfg.mean, np.float32), np.array(cfg.std, np.float32)
    return torch.from_numpy((array - mean) / std).permute(2, 0, 1).unsqueeze(0)


def class_name(index: int) -> str:
    return _imagenet.index_to_description(index)


@torch.no_grad()
def predict(model: torch.nn.Module, x: torch.Tensor, k: int = 5) -> list[tuple[int, str, float]]:
    """Return the top-`k` (class index, class name, probability) for one image."""
    probs = model(x).softmax(dim=1)[0]
    top = probs.topk(k)
    return [
        (int(i), _imagenet.index_to_description(int(i)), float(p))
        for p, i in zip(top.values, top.indices)
    ]
