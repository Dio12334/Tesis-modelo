"""Visual sanity check for augmentation transforms.

Loads a small set of random samples from the configured RDD2022 train split,
applies each augmentation category in isolation (with deterministic seeds for
reproducibility), and writes annotated PNGs with bounding-box overlays so a
human can spot-check that bboxes still hug the damage after augmentation.

Output layout:
    {output_root}/<run_id>/raw/sample_<i>.png
    {output_root}/<run_id>/hflip/sample_<i>.png
    {output_root}/<run_id>/vflip/sample_<i>.png
    {output_root}/<run_id>/scale/sample_<i>.png
    {output_root}/<run_id>/translate/sample_<i>.png
    {output_root}/<run_id>/hsv/sample_<i>.png
    {output_root}/<run_id>/brightness/sample_<i>.png
    {output_root}/<run_id>/mosaic/sample_<i>.png
    {output_root}/<run_id>/mixup/sample_<i>.png

This tool is read-only with respect to the dataset and the model. It only
writes PNG samples to the output directory.

Usage (defaults):
    python -m model.tools.visualize_augmentation

With explicit options:
    python -m model.tools.visualize_augmentation \
        --root model/data/rdd2022/complete \
        --output-root checkpoints/aug_samples \
        --run-id manual-2026-06-07 \
        --num-samples 16 \
        --input-size 640 \
        --seed 42
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Tuple
from uuid import uuid4

import cv2
import numpy as np

from model.datasets.rdd2022 import RDD2022Dataset
from model.training.augmentation import (
    RandomBrightness,
    RandomHSV,
    RandomHorizontalFlip,
    RandomScale,
    RandomTranslate,
    RandomVerticalFlip,
)
from model.training.train_detection import RDD2022TorchDataset

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path("model/data/rdd2022/complete")
DEFAULT_OUTPUT_ROOT = Path("checkpoints/aug_samples")
DEFAULT_NUM_SAMPLES = 16
DEFAULT_INPUT_SIZE = 640
DEFAULT_SEED = 42

# BGR colors for class overlays (cv2 uses BGR)
CLASS_COLORS_BGR = [
    (255, 0, 0),    # blue
    (0, 255, 0),    # green
    (0, 0, 255),    # red
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
]


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _draw_bboxes(
    image_rgb: np.ndarray,
    bboxes: List[List],
    class_names: List[str],
    title: str,
) -> np.ndarray:
    """Return a BGR copy of ``image_rgb`` with bbox overlays + title bar.

    Bboxes are normalized [0,1] and may carry an arbitrary trailing class
    label (string or int). The returned image is suitable for ``cv2.imwrite``.
    """
    h, w = image_rgb.shape[:2]
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()

    for bbox in bboxes:
        x1 = int(round(bbox[0] * w))
        y1 = int(round(bbox[1] * h))
        x2 = int(round(bbox[2] * w))
        y2 = int(round(bbox[3] * h))
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        label = bbox[4] if len(bbox) > 4 else ""
        if isinstance(label, int) and 0 <= label < len(class_names):
            label_text = class_names[label]
            color = CLASS_COLORS_BGR[label % len(CLASS_COLORS_BGR)]
        else:
            label_text = str(label)
            try:
                idx = class_names.index(label_text) if label_text in class_names else 0
            except ValueError:
                idx = 0
            color = CLASS_COLORS_BGR[idx % len(CLASS_COLORS_BGR)]

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        if label_text:
            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            ty1 = max(0, y1 - th - 4)
            cv2.rectangle(
                img_bgr,
                (x1, ty1),
                (min(w - 1, x1 + tw + 4), y1),
                color,
                -1,
            )
            cv2.putText(
                img_bgr,
                label_text,
                (x1 + 2, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Title bar
    bar_h = 24
    bar = np.full((bar_h, w, 3), 32, dtype=np.uint8)
    cv2.putText(
        bar,
        title,
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([bar, img_bgr])


def _resize_keep_normalized(
    image: np.ndarray, size: int
) -> np.ndarray:
    h, w = image.shape[:2]
    if h == size and w == size:
        return image
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def _save_category(
    out_dir: Path,
    samples: List[Tuple[np.ndarray, List[List]]],
    class_names: List[str],
    title_prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (img, bboxes) in enumerate(samples):
        title = f"{title_prefix} #{i}  bboxes={len(bboxes)}"
        rendered = _draw_bboxes(img, bboxes, class_names, title)
        out_path = out_dir / f"sample_{i:02d}.png"
        cv2.imwrite(str(out_path), rendered)


def _apply_simple(
    transform: Callable[[np.ndarray, List[List]], Tuple[np.ndarray, List[List]]],
    raw_samples: List[Tuple[np.ndarray, List[List]]],
    seed_offset: int,
    base_seed: int,
) -> List[Tuple[np.ndarray, List[List]]]:
    """Apply a per-sample transform with deterministic per-sample seed."""
    out = []
    for i, (img, bboxes) in enumerate(raw_samples):
        random.seed(base_seed + seed_offset * 1000 + i)
        np.random.seed(base_seed + seed_offset * 1000 + i)
        new_img, new_bboxes = transform(img.copy(), [list(b) for b in bboxes])
        out.append((new_img, new_bboxes))
    return out


def _build_torch_dataset(
    root: Path, subset: str, input_size: int, mosaic: float, mixup: float
) -> RDD2022TorchDataset:
    base = RDD2022Dataset(subset=subset)
    base.load(root)
    return RDD2022TorchDataset(
        dataset=base,
        input_size=input_size,
        augmentation=None,
        mosaic=mosaic,
        mixup=mixup,
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.root.exists():
        print(f"ERROR: dataset root not found: {args.root}", file=sys.stderr)
        return 2

    run_id = args.run_id or (
        datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
    )
    out_dir = args.output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Visual sanity check -> {out_dir.resolve()}")

    # --- Load raw samples (no augmentation) -----------------------------
    _seed_all(args.seed)
    base_dataset = RDD2022Dataset(subset="train")
    base_dataset.load(args.root)
    annotations = base_dataset.get_annotations()
    if not annotations:
        print("ERROR: no annotations loaded from train split", file=sys.stderr)
        return 2

    class_names = base_dataset.get_class_names()
    print(f"Class names ({len(class_names)}): {class_names}")
    print(f"Total annotations: {len(annotations)}")

    # Pick samples that have at least one bbox so the overlays are meaningful
    candidates = [i for i, a in enumerate(annotations) if a.bounding_boxes]
    if len(candidates) < args.num_samples:
        print(
            f"WARN: only {len(candidates)} annotations have bboxes; "
            f"sampling without replacement.",
            file=sys.stderr,
        )
    random.shuffle(candidates)
    sample_indices = candidates[: args.num_samples]

    raw_samples: List[Tuple[np.ndarray, List[List]]] = []
    from PIL import Image as PILImage

    for idx in sample_indices:
        ann = annotations[idx]
        try:
            with PILImage.open(ann.image_path) as im:
                img_np = np.array(im.convert("RGB"))
        except (OSError, FileNotFoundError) as e:
            logger.warning("skip sample %s: %s", ann.image_path, e)
            continue
        img_np = _resize_keep_normalized(img_np, args.input_size)
        bboxes = [
            [b.x_min, b.y_min, b.x_max, b.y_max, b.class_label]
            for b in ann.bounding_boxes
        ]
        raw_samples.append((img_np, bboxes))

    if not raw_samples:
        print("ERROR: no samples could be loaded", file=sys.stderr)
        return 2

    print(f"Loaded {len(raw_samples)} raw samples at {args.input_size}x{args.input_size}")

    # --- Save raw -------------------------------------------------------
    _save_category(out_dir / "raw", raw_samples, class_names, "raw")

    # --- Per-image transforms -----------------------------------------
    transforms_to_run = [
        ("hflip", RandomHorizontalFlip(p=1.0)),
        ("vflip", RandomVerticalFlip(p=1.0)),
        ("scale", RandomScale(scale_range=(0.75, 1.25))),
        ("translate", RandomTranslate(translate=0.1)),
        ("hsv", RandomHSV(h_gain=0.015, s_gain=0.5, v_gain=0.4)),
        ("brightness", RandomBrightness(brightness_range=(0.8, 1.2))),
    ]
    for offset, (name, t) in enumerate(transforms_to_run, start=1):
        applied = _apply_simple(t, raw_samples, offset, args.seed)
        _save_category(out_dir / name, applied, class_names, name)
        print(f"  [{name}] saved {len(applied)} samples")

    # --- Multi-image (mosaic / mixup) -----------------------------------
    torch_ds = _build_torch_dataset(
        root=args.root,
        subset="train",
        input_size=args.input_size,
        mosaic=1.0,  # always-on for visualization
        mixup=0.0,
    )

    mosaic_samples: List[Tuple[np.ndarray, List[List]]] = []
    for offset, idx in enumerate(sample_indices[: len(raw_samples)]):
        random.seed(args.seed + 9000 + offset)
        np.random.seed(args.seed + 9000 + offset)
        img, bboxes = torch_ds._build_mosaic(idx)
        mosaic_samples.append((img, bboxes))
    _save_category(out_dir / "mosaic", mosaic_samples, class_names, "mosaic")
    print(f"  [mosaic] saved {len(mosaic_samples)} samples")

    # MixUp on top of mosaic
    mixup_samples: List[Tuple[np.ndarray, List[List]]] = []
    torch_ds_mixup = _build_torch_dataset(
        root=args.root,
        subset="train",
        input_size=args.input_size,
        mosaic=1.0,
        mixup=1.0,
    )
    for offset, idx in enumerate(sample_indices[: len(raw_samples)]):
        random.seed(args.seed + 7000 + offset)
        np.random.seed(args.seed + 7000 + offset)
        img1, bboxes1 = torch_ds_mixup._build_mosaic(idx)
        img, bboxes = torch_ds_mixup._apply_mixup(img1, bboxes1)
        mixup_samples.append((img, bboxes))
    _save_category(out_dir / "mixup", mixup_samples, class_names, "mixup")
    print(f"  [mixup] saved {len(mixup_samples)} samples")

    print()
    print(f"DONE. Inspect outputs under: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
