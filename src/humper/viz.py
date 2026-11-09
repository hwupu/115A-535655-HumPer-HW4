"""Figures for individual cases. Maps are normalized for display only."""

import matplotlib.pyplot as plt
import pandas as pd

from humper import data, explain, maps, store

LABELS = {
    "resnet50_gradcam": "ResNet-50 Grad-CAM",
    "vit_gradcam": "ViT Grad-CAM",
    "vit_rollout": "ViT Rollout",
}
MAP_ALPHA = 0.65


def _show_map(ax, image, saliency, size):
    ax.imshow(image)
    ax.imshow(maps.to_display(maps.warp_to_image(saliency, size)), cmap="magma", vmin=0, vmax=1, alpha=MAP_ALPHA)
    ax.axis("off")


def draw_case(axes, image_id: str, case, scores: pd.DataFrame, predictions: pd.DataFrame) -> None:
    """Fill five axes: input, human map, then each explainer with its scores."""
    image = data.load_image(image_id)
    size = image.size
    fixations = data.load_fixations(image_id)
    record = store.load_record(image_id)
    rows = scores[scores.image_id == image_id].set_index("explainer")
    top1 = predictions[predictions.image_id == image_id].set_index("model").top1_name

    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title(f"{case.category.replace('_', ' ')}\n{image_id[:26]}", fontsize=9)
    _show_map(axes[1], image, maps.human_map_native(fixations, size), size)
    axes[1].scatter(*fixations.T, s=5, c="#ffd21f", linewidths=0)
    axes[1].set_title("Human attention", fontsize=9)

    for ax, name in zip(axes[2:], explain.EXPLAINERS):
        spec = explain.EXPLAINERS[name]
        _show_map(ax, image, maps.model_map_canvas(record[name]), size)
        role = " (weak)" if name == case.weak else " (strong)" if name == case.strong else ""
        r = rows.loc[name]
        label = top1[spec.model].split(",")[0][:22]
        ax.set_title(
            f"{LABELS[name]}{role}\n{label}\nCC {r.cc:.2f}  NSS {r.nss:.2f}  sAUC {r.sauc:.2f}", fontsize=9
        )


def case_figure(case, scores: pd.DataFrame, predictions: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6))
    draw_case(axes, case.image_id, case, scores, predictions)
    fig.tight_layout()
    return fig


def overview_figure(cases: pd.DataFrame, scores: pd.DataFrame, predictions: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(len(cases), 5, figsize=(15, 3.4 * len(cases)), squeeze=False)
    for row, case in zip(axes, cases.itertuples()):
        draw_case(row, case.image_id, case, scores, predictions)
    fig.tight_layout()
    return fig

