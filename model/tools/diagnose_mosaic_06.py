"""One-off diagnostic for the bottom-left quadrant of mosaic #6 from the
post-affine-and-filter visualizer run.

Replicates the visualizer's exact seeding so the captured mosaic matches
the one the user inspected, then dumps:

  - 4 raw source images at native resolution with their ORIGINAL bbox
    overlays (these are the inputs to _build_mosaic).
  - The final mosaic with its bbox overlays.
  - A printed manifest mapping source path -> quadrant.

If a source's original bbox already sits on grass / non-damage pixels in
the source image, it's RDD2022 annotation noise, not a pipeline bug.

Output: checkpoints/aug_samples/post-affine-and-filter/mosaic_06_diagnostic.png
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage

from model.datasets.rdd2022 import RDD2022Dataset
from model.training.train_detection import RDD2022TorchDataset

ROOT = Path("model/data/rdd2022/complete")
INPUT_SIZE = 640
SEED = 42
MOSAIC_OFFSET = 6
OUT_PATH = Path(
    "checkpoints/aug_samples/post-affine-and-filter/mosaic_06_diagnostic.png"
)

CLASS_COLORS_BGR = [
    (255, 0, 0),    # blue
    (0, 255, 0),    # green
    (0, 0, 255),    # red
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
]

QUADRANT_LABELS = ["TL (primary)", "TR", "BL", "BR"]


def _draw_bboxes(image_rgb: np.ndarray, bboxes: List[List], class_names: List[str], title: str) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    for bbox in bboxes:
        x1 = max(0, min(w - 1, int(round(bbox[0] * w))))
        y1 = max(0, min(h - 1, int(round(bbox[1] * h))))
        x2 = max(0, min(w - 1, int(round(bbox[2] * w))))
        y2 = max(0, min(h - 1, int(round(bbox[3] * h))))
        label = bbox[4] if len(bbox) > 4 else ""
        try:
            idx = class_names.index(str(label)) if str(label) in class_names else 0
        except ValueError:
            idx = 0
        color = CLASS_COLORS_BGR[idx % len(CLASS_COLORS_BGR)]
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        if label:
            (tw, th), _ = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty1 = max(0, y1 - th - 4)
            cv2.rectangle(img_bgr, (x1, ty1), (min(w - 1, x1 + tw + 4), y1), color, -1)
            cv2.putText(img_bgr, str(label), (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    bar_h = 28
    bar = np.full((bar_h, w, 3), 32, dtype=np.uint8)
    cv2.putText(bar, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img_bgr])


def main() -> int:
    print(f"Loading RDD2022 train split from {ROOT}...")
    base = RDD2022Dataset(subset="train")
    base.load(ROOT)
    annotations = base.get_annotations()
    class_names = base.get_class_names()

    # Replicate visualizer seeding to find sample_indices[MOSAIC_OFFSET]
    random.seed(SEED)
    np.random.seed(SEED)
    candidates = [i for i, a in enumerate(annotations) if a.bounding_boxes]
    random.shuffle(candidates)
    sample_indices = candidates[:16]
    primary_idx = sample_indices[MOSAIC_OFFSET]
    print(f"Mosaic #{MOSAIC_OFFSET} primary annotation idx = {primary_idx}")
    print(f"Primary image path: {annotations[primary_idx].image_path}")

    # Build torch dataset (same params as visualizer mosaic path)
    ds = RDD2022TorchDataset(
        dataset=base,
        input_size=INPUT_SIZE,
        augmentation=None,
        mosaic=1.0,
        mixup=0.0,
    )

    # Capture source loads
    captured: List[Tuple[int, np.ndarray, List[List]]] = []
    real_load = ds._load_image_and_bboxes

    def capturing_load(img_idx):
        img_np, bboxes = real_load(img_idx)
        captured.append((img_idx, img_np.copy(), [list(b) for b in bboxes]))
        return img_np, bboxes

    ds._load_image_and_bboxes = capturing_load

    # Reproduce the exact mosaic seed used by the visualizer
    mosaic_seed = SEED + 9000 + MOSAIC_OFFSET
    random.seed(mosaic_seed)
    np.random.seed(mosaic_seed)
    mosaic_img, mosaic_bboxes = ds._build_mosaic(primary_idx)

    print()
    print("Captured sources:")
    for i, (img_idx, _, bboxes) in enumerate(captured):
        path = annotations[img_idx].image_path
        print(f"  Quadrant {QUADRANT_LABELS[i]} (idx={img_idx}): "
              f"{len(bboxes)} bbox(es)")
        print(f"    path: {path}")
        for b in bboxes:
            print(f"    bbox: x=[{b[0]:.3f},{b[2]:.3f}] "
                  f"y=[{b[1]:.3f},{b[3]:.3f}] label={b[4]!r}")
    print()
    print(f"Mosaic output: {len(mosaic_bboxes)} bbox(es) survived")

    # Render panels
    panels: List[np.ndarray] = []
    for i, (img_idx, img_np, bboxes) in enumerate(captured):
        path = annotations[img_idx].image_path
        title = f"src #{i}: {QUADRANT_LABELS[i]}  |  {Path(path).name}  |  {len(bboxes)} bbox"
        panel = _draw_bboxes(img_np, bboxes, class_names, title)
        panels.append(panel)

    sources_row = np.hstack(panels)  # 4 horizontal panels
    mosaic_panel = _draw_bboxes(
        mosaic_img, mosaic_bboxes, class_names,
        f"MOSAIC OUTPUT  |  {len(mosaic_bboxes)} bbox surviving",
    )
    # Pad mosaic panel to match sources_row width
    target_w = sources_row.shape[1]
    if mosaic_panel.shape[1] != target_w:
        scale = target_w / mosaic_panel.shape[1]
        new_h = int(round(mosaic_panel.shape[0] * scale))
        mosaic_panel = cv2.resize(mosaic_panel, (target_w, new_h),
                                  interpolation=cv2.INTER_LINEAR)
    composite = np.vstack([sources_row, mosaic_panel])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_PATH), composite)
    print(f"Wrote diagnostic to {OUT_PATH.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
