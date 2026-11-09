"""Explainer registry: Grad-CAM and Attention Rollout on each model's own grid (D-02)."""

from dataclasses import dataclass

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from humper import maps
from humper.models import MODELS


@dataclass(frozen=True)
class ExplainerSpec:
    name: str
    model: str  # key in MODELS
    method: str  # "gradcam" or "rollout"
    class_specific: bool


EXPLAINERS = {
    spec.name: spec
    for spec in (
        ExplainerSpec("resnet50_gradcam", "resnet50", "gradcam", True),
        ExplainerSpec("vit_gradcam", "vit_b16", "gradcam", True),
        ExplainerSpec("vit_rollout", "vit_b16", "rollout", False),
    )
}


def grad_cam_grid(model, layer, x, target, reshape=None) -> np.ndarray:
    """Return Grad-CAM for the `target` logit on the layer's grid, after ReLU.

    Stops before the package's min-max scaling and resizing, which would disguise
    an all-zero map as a valid one.
    """
    cam = GradCAM(model=model, target_layers=[layer], reshape_transform=reshape)
    try:
        outputs = cam.activations_and_grads(x)
        model.zero_grad()
        outputs[0, target].backward()
        activations = cam.activations_and_grads.activations[-1].numpy()
        gradients = cam.activations_and_grads.gradients[-1].numpy()
        raw = cam.get_cam_image(
            x, layer, [ClassifierOutputTarget(target)], activations, gradients
        )[0]
    finally:
        cam.activations_and_grads.release()
        model.zero_grad()
    return np.maximum(raw, 0).astype(np.float32)


def attention_rollout(attentions: list[torch.Tensor]) -> np.ndarray:
    """Return the CLS token's rollout onto the patches, as a square grid (D-08).

    Each entry is one layer's (heads, tokens, tokens) attention, earliest layer first.
    Heads are averaged and the residual connection is modelled as 0.5 A + 0.5 I.
    """
    tokens = attentions[0].shape[-1]
    identity = torch.eye(tokens, device=attentions[0].device)
    joint = identity
    for attention in attentions:
        layer = 0.5 * attention.mean(dim=0) + 0.5 * identity
        layer = layer / layer.sum(dim=-1, keepdim=True)
        joint = layer @ joint
    cls_to_patches = joint[0, 1:].cpu().numpy().astype(np.float32)
    side = int(len(cls_to_patches) ** 0.5)
    return cls_to_patches.reshape(side, side)


@torch.no_grad()
def capture_attentions(model, x) -> list[torch.Tensor]:
    """Return each block's (heads, tokens, tokens) attention for one image."""
    attentions = []
    handles = []
    fused = [block.attn.fused_attn for block in model.blocks]
    for block in model.blocks:
        # timm's fused path never materializes the attention matrix.
        block.attn.fused_attn = False
        handles.append(
            block.attn.attn_drop.register_forward_hook(
                lambda _module, _in, out: attentions.append(out[0].detach())
            )
        )
    try:
        model(x)
    finally:
        for handle, block, value in zip(handles, model.blocks, fused):
            handle.remove()
            block.attn.fused_attn = value
    return attentions


def explain_grid(spec: ExplainerSpec, model, x, target: int | None = None) -> np.ndarray:
    """Return the explanation on the model's grid. `target` defaults to the top-1 class."""
    if spec.method == "rollout":
        return attention_rollout(capture_attentions(model, x))
    if target is None:
        with torch.no_grad():
            target = int(model(x).argmax(dim=1))
    model_spec = MODELS[spec.model]
    layer = model.get_submodule(model_spec.target_layer)
    return grad_cam_grid(model, layer, x, target, model_spec.reshape)


def explain(spec: ExplainerSpec, model, x, target: int | None = None) -> np.ndarray:
    """Return the explanation on the comparison canvas."""
    return maps.model_map_canvas(explain_grid(spec, model, x, target))
