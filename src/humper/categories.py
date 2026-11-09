"""Image categories for the misalignment ranking (D-22).

Every category comes from the image or the human data only, never from an explainer's
map, because a category built from the map would just restate its alignment score.
"""

import cv2
import numpy as np
import pandas as pd

from humper import data, maps

YUNET_PATH = data.REPO_ROOT / "data" / "_download" / "yunet" / "face_detection_yunet_2023mar.onnx"
FACE_SCORE = 0.75
# Small detections are often false hits, so text must cover a meaningful area.
MIN_TEXT_AREA = 0.005
TAIL_SHARE = 0.20
MASS_SHARE = 0.50
TEXTURE_WINDOW = 9
# Edge density and texture energy correlate at 0.88, so they form one category.
CATEGORIES = ["face", "text", "small_salient", "edge_texture_rich", "peripheral_attention"]


def face_detector():
    return cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), score_threshold=FACE_SCORE)


def text_reader():
    import easyocr

    return easyocr.Reader(["en"], gpu=False, verbose=False)


def count_faces(detector, rgb: np.ndarray) -> int:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    detector.setInputSize((bgr.shape[1], bgr.shape[0]))
    _, detections = detector.detect(bgr)
    return 0 if detections is None else len(detections)


def box_area_share(boxes: list, height: int, width: int) -> float:
    """Return the image share covered by (left, right, top, bottom) boxes; overlaps count once."""
    mask = np.zeros((height, width), bool)
    for left, right, top, bottom in boxes:
        mask[max(int(top), 0) : max(int(bottom), 0), max(int(left), 0) : max(int(right), 0)] = True
    return float(mask.mean())


def text_area_share(reader, rgb: np.ndarray) -> float:
    horizontal, _ = reader.detect(rgb, text_threshold=0.7, low_text=0.4)
    return box_area_share(horizontal[0], *rgb.shape[:2])


def edge_density(gray: np.ndarray) -> float:
    """Return the share of canvas pixels marked as edges."""
    return float((cv2.Canny(gray, 100, 200) > 0).mean())


def texture_energy(gray: np.ndarray) -> float:
    """Return the mean local standard deviation of brightness."""
    values = gray.astype(np.float32)
    size = (TEXTURE_WINDOW, TEXTURE_WINDOW)
    variance = cv2.blur(values**2, size) - cv2.blur(values, size) ** 2
    return float(np.sqrt(np.clip(variance, 0, None)).mean())


def attention_area(human: np.ndarray) -> float:
    """Return the canvas share needed to hold half of the human attention."""
    ordered = np.sort(human.ravel())[::-1]
    needed = np.searchsorted(np.cumsum(ordered), MASS_SHARE * ordered.sum()) + 1
    return needed / human.size


def center_offset(human: np.ndarray) -> float:
    """Return the distance from canvas center to the attention centroid, 0 at center, 1 at a corner."""
    height, width = human.shape
    ys, xs = np.indices(human.shape)
    x = ((xs + 0.5) * human).sum() / human.sum()
    y = ((ys + 0.5) * human).sum() / human.sum()
    return float(np.hypot(x - width / 2, y - height / 2) / np.hypot(width / 2, height / 2))


def measure_image(image_id: str, size: tuple[int, int], detector, reader) -> dict:
    """Return every category measure for one image."""
    image = data.load_image(image_id)
    rgb = np.asarray(image)
    gray = np.asarray(maps.model_input(image).convert("L"))
    human = maps.human_map_canvas(data.load_fixations(image_id), size)
    return {
        "image_id": image_id,
        "face_count": count_faces(detector, rgb),
        "text_area": text_area_share(reader, rgb),
        "attention_area": attention_area(human),
        "edge_density": edge_density(gray),
        "texture_energy": texture_energy(gray),
        "center_offset": center_offset(human),
    }


def category_flags(measures: pd.DataFrame) -> pd.DataFrame:
    """Return one boolean column per category, indexed by image_id."""
    m = measures.set_index("image_id")
    return pd.DataFrame({
        "face": m.face_count >= 1,
        "text": m.text_area >= MIN_TEXT_AREA,
        "small_salient": m.attention_area <= m.attention_area.quantile(TAIL_SHARE),
        "edge_texture_rich": m.texture_energy >= m.texture_energy.quantile(1 - TAIL_SHARE),
        "peripheral_attention": m.center_offset >= m.center_offset.quantile(1 - TAIL_SHARE),
    })
