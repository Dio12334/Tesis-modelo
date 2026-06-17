"""Data augmentation transforms for object detection training.

Provides configurable image transforms that operate on images and bounding boxes.
Each transform accepts an image (numpy array) and a list of bounding boxes,
returning the transformed image and adjusted bounding boxes.

Transforms are composable via the Compose class and can be built from a
YAML-style augmentation config dict using build_augmentation_pipeline().

Bounding boxes are in normalized [0, 1] format: [x_min, y_min, x_max, y_max, ...].
Every spatial transform clips bboxes to [0, 1] and discards degenerate boxes
(area < MIN_BBOX_AREA) after transformation.
"""

import random
from typing import List, Tuple

import cv2
import numpy as np


# Type aliases for clarity
Image = np.ndarray  # HxWxC uint8 numpy array
BBoxes = List[List]  # List of [x_min, y_min, x_max, y_max, class_label, ...]

# Minimum bbox area (normalized) to keep after augmentation; smaller boxes are
# discarded as slivers produced by cropping/clipping.
#
# At input_size=640, area=0.0001 corresponds to roughly 41 px^2 (~6.4x6.4),
# which is just above the floor below which a road-damage box is unlikely to
# carry useful signal but well under the ~410 px^2 (~20x20) cutoff that the
# previous value of 0.001 imposed (which silently dropped legitimately small
# potholes after mosaic/scale shrinkage).
MIN_BBOX_AREA = 0.0001


def _clip_and_filter_bboxes(bboxes: BBoxes) -> BBoxes:
    """Clip bbox coordinates to [0, 1] and discard degenerate boxes.

    A box is degenerate if its area after clipping is below MIN_BBOX_AREA or
    if x_max <= x_min or y_max <= y_min.

    Args:
        bboxes: List of bounding boxes in normalized [0, 1] format.

    Returns:
        Filtered list of valid bounding boxes.
    """
    result = []
    for bbox in bboxes:
        x_min = max(0.0, min(1.0, bbox[0]))
        y_min = max(0.0, min(1.0, bbox[1]))
        x_max = max(0.0, min(1.0, bbox[2]))
        y_max = max(0.0, min(1.0, bbox[3]))
        w = x_max - x_min
        h = y_max - y_min
        if w > 0 and h > 0 and w * h >= MIN_BBOX_AREA:
            result.append([x_min, y_min, x_max, y_max] + bbox[4:])
    return result


class Compose:
    """Chains multiple transforms together into a pipeline.

    Each transform in the sequence is applied in order. Transforms must
    accept (image, bboxes) and return (image, bboxes).
    """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image: Image, bboxes: BBoxes) -> Tuple[Image, BBoxes]:
        for transform in self.transforms:
            image, bboxes = transform(image, bboxes)
        return image, bboxes

    def __repr__(self) -> str:
        transform_names = [repr(t) for t in self.transforms]
        return f"Compose([{', '.join(transform_names)}])"


class RandomHorizontalFlip:
    """Randomly flips the image horizontally with a given probability.

    Bounding box x-coordinates are mirrored accordingly.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image, bboxes: BBoxes) -> Tuple[Image, BBoxes]:
        if random.random() < self.p:
            image = np.fliplr(image).copy()
            new_bboxes = []
            for bbox in bboxes:
                x_min, y_min, x_max, y_max = bbox[:4]
                new_x_min = 1.0 - x_max
                new_x_max = 1.0 - x_min
                new_bboxes.append([new_x_min, y_min, new_x_max, y_max] + bbox[4:])
            bboxes = new_bboxes
        return image, bboxes

    def __repr__(self) -> str:
        return f"RandomHorizontalFlip(p={self.p})"


class RandomVerticalFlip:
    """Randomly flips the image vertically with a given probability.

    Bounding box y-coordinates are mirrored accordingly.
    Not recommended for road images (roads don't appear upside down).
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image, bboxes: BBoxes) -> Tuple[Image, BBoxes]:
        if random.random() < self.p:
            image = np.flipud(image).copy()
            new_bboxes = []
            for bbox in bboxes:
                x_min, y_min, x_max, y_max = bbox[:4]
                new_y_min = 1.0 - y_max
                new_y_max = 1.0 - y_min
                new_bboxes.append([x_min, new_y_min, x_max, new_y_max] + bbox[4:])
            bboxes = new_bboxes
        return image, bboxes

    def __repr__(self) -> str:
        return f"RandomVerticalFlip(p={self.p})"


class RandomBrightness:
    """Randomly adjusts image brightness by a factor within a given range.

    Bounding boxes are not affected by brightness changes.
    """

    def __init__(self, brightness_range: Tuple[float, float] = (0.8, 1.2)):
        self.low = brightness_range[0]
        self.high = brightness_range[1]

    def __call__(self, image: Image, bboxes: BBoxes) -> Tuple[Image, BBoxes]:
        factor = random.uniform(self.low, self.high)
        image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return image, bboxes

    def __repr__(self) -> str:
        return f"RandomBrightness(range=({self.low}, {self.high}))"


class RandomHSV:
    """Randomly adjusts image HSV channels.

    Applies random gains to Hue, Saturation, and Value channels independently.
    Bounding boxes are not affected (color-only transform).

    Args:
        h_gain: Maximum fractional hue shift (applied as ±h_gain * 180 degrees).
        s_gain: Maximum fractional saturation multiplier (range: [1-s_gain, 1+s_gain]).
        v_gain: Maximum fractional value multiplier (range: [1-v_gain, 1+v_gain]).
    """

    def __init__(self, h_gain: float = 0.015, s_gain: float = 0.7, v_gain: float = 0.4):
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain

    def __call__(self, image: Image, bboxes: BBoxes) -> Tuple[Image, BBoxes]:
        # Random gains
        h_delta = random.uniform(-self.h_gain, self.h_gain) * 180.0
        s_mult = random.uniform(1.0 - self.s_gain, 1.0 + self.s_gain)
        v_mult = random.uniform(1.0 - self.v_gain, 1.0 + self.v_gain)

        # Convert RGB to HSV (cv2 expects BGR, so convert)
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Apply gains
        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + h_delta) % 180.0
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * s_mult, 0, 255)
        img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * v_mult, 0, 255)

        # Convert back to RGB
        image = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return image, bboxes

    def __repr__(self) -> str:
        return f"RandomHSV(h={self.h_gain}, s={self.s_gain}, v={self.v_gain})"


class RandomScale:
    """Randomly scales the image and adjusts bounding boxes accordingly.

    When scale > 1 (zoom in): resize larger then random-crop back to original size.
    When scale < 1 (zoom out): resize smaller then place on gray-padded canvas.

    Args:
        scale_range: Tuple of (min_scale, max_scale). E.g., (0.5, 1.5).
    """

    def __init__(self, scale_range: Tuple[float, float] = (0.5, 1.5)):
        self.scale_min = scale_range[0]
        self.scale_max = scale_range[1]

    def __call__(self, image: Image, bboxes: BBoxes) -> Tuple[Image, BBoxes]:
        h, w = image.shape[:2]
        s = random.uniform(self.scale_min, self.scale_max)

        if abs(s - 1.0) < 1e-3:
            return image, bboxes

        new_h, new_w = int(h * s), int(w * s)

        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if s > 1.0:
            # Zoom in: random crop back to original size
            crop_x = random.randint(0, new_w - w)
            crop_y = random.randint(0, new_h - h)
            image = resized[crop_y:crop_y + h, crop_x:crop_x + w]

            # Bbox transform: shift by crop offset, scale by s
            # In normalized space of resized image, the crop covers:
            #   x: [crop_x/new_w, (crop_x+w)/new_w] = [crop_x/new_w, crop_x/new_w + 1/s]
            ox = crop_x / new_w  # offset in normalized resized space
            oy = crop_y / new_h
            new_bboxes = []
            for bbox in bboxes:
                x_min = (bbox[0] - ox) * s
                y_min = (bbox[1] - oy) * s
                x_max = (bbox[2] - ox) * s
                y_max = (bbox[3] - oy) * s
                new_bboxes.append([x_min, y_min, x_max, y_max] + bbox[4:])
            bboxes = _clip_and_filter_bboxes(new_bboxes)
        else:
            # Zoom out: place on gray canvas at random position
            canvas = np.full((h, w, 3), 114, dtype=np.uint8)
            # Random placement
            pad_x = random.randint(0, w - new_w)
            pad_y = random.randint(0, h - new_h)
            canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
            image = canvas

            # Bbox transform: scale down and offset by pad position
            px = pad_x / w  # normalized pad offset
            py = pad_y / h
            new_bboxes = []
            for bbox in bboxes:
                x_min = bbox[0] * s + px
                y_min = bbox[1] * s + py
                x_max = bbox[2] * s + px
                y_max = bbox[3] * s + py
                new_bboxes.append([x_min, y_min, x_max, y_max] + bbox[4:])
            bboxes = _clip_and_filter_bboxes(new_bboxes)

        return image, bboxes

    def __repr__(self) -> str:
        return f"RandomScale(range=({self.scale_min}, {self.scale_max}))"


class RandomTranslate:
    """Randomly translates the image and adjusts bounding boxes accordingly.

    Shifts the image by up to ±translate fraction in both x and y.
    Exposed areas are filled with gray (114, 114, 114).

    Args:
        translate: Maximum translation fraction. E.g., 0.1 means ±10%.
    """

    def __init__(self, translate: float = 0.1):
        self.translate = translate

    def __call__(self, image: Image, bboxes: BBoxes) -> Tuple[Image, BBoxes]:
        h, w = image.shape[:2]
        tx = random.uniform(-self.translate, self.translate)
        ty = random.uniform(-self.translate, self.translate)

        if abs(tx) < 1e-4 and abs(ty) < 1e-4:
            return image, bboxes

        # Pixel shifts
        dx = int(tx * w)
        dy = int(ty * h)

        # Affine translation matrix
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        image = cv2.warpAffine(
            image, M, (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(114, 114, 114),
        )

        # Bbox transform: shift in normalized space
        new_bboxes = []
        for bbox in bboxes:
            x_min = bbox[0] + tx
            y_min = bbox[1] + ty
            x_max = bbox[2] + tx
            y_max = bbox[3] + ty
            new_bboxes.append([x_min, y_min, x_max, y_max] + bbox[4:])
        bboxes = _clip_and_filter_bboxes(new_bboxes)

        return image, bboxes

    def __repr__(self) -> str:
        return f"RandomTranslate(translate={self.translate})"


class RandomIoUCrop:
    """SSD-style sample-IoU crop transform.

    Re-implementation of the canonical SSD/SSDLite augmentation
    (``torchvision.transforms.v2.RandomIoUCrop``) for the project's NumPy/cv2
    augmentation pipeline. At each call:

    1. Samples an option from ``min_jaccard_overlaps`` (None means "return
       the original image unchanged"; ``1.0`` would mean "keep only crops
       fully containing every box", which we exclude by default because it
       devolves to a no-op on most road-damage scenes).
    2. Tries up to ``trials`` random crops sized between ``min_scale`` and
       ``max_scale`` of the input area with aspect ratio in ``aspect_ratio_range``.
    3. Accepts a crop only when every retained bbox's **center** lies inside
       the crop AND the crop's minimum IoU with the boxes meets the sampled
       threshold.
    4. On a successful crop, resizes back to the original (H, W) so the rest
       of the pipeline stays size-stable.

    All bboxes use normalised ``[x_min, y_min, x_max, y_max, ...]`` form.
    Boxes whose centers fall outside the crop are dropped (matching the SSD
    paper's recipe).

    Args:
        min_jaccard_overlaps: Sampling options. ``None`` is a no-op option;
            float values are the minimum-IoU thresholds for the trial.
            Defaults to the SSD paper's ``(None, 0.1, 0.3, 0.5, 0.7, 0.9)``.
        trials: Maximum number of random crops attempted per sampled option.
        aspect_ratio_range: Allowed crop aspect ratios (``w/h``).
        min_scale: Minimum crop width/height as a fraction of the input
            width/height.
        max_scale: Maximum crop width/height as a fraction of the input
            width/height.
        p: Probability of applying the transform at all. ``p < 1`` lets the
            caller mix the original image into the training stream.
    """

    DEFAULT_OPTIONS = (None, 0.1, 0.3, 0.5, 0.7, 0.9)

    def __init__(
        self,
        min_jaccard_overlaps=DEFAULT_OPTIONS,
        trials: int = 40,
        aspect_ratio_range: Tuple[float, float] = (0.5, 2.0),
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        p: float = 1.0,
    ):
        if min_scale <= 0.0 or max_scale > 1.0 or min_scale > max_scale:
            raise ValueError(
                f"Invalid scale range ({min_scale}, {max_scale}); expect "
                f"0 < min_scale <= max_scale <= 1."
            )
        if (
            aspect_ratio_range[0] <= 0.0
            or aspect_ratio_range[1] < aspect_ratio_range[0]
        ):
            raise ValueError(
                f"Invalid aspect_ratio_range {aspect_ratio_range}; expect "
                f"0 < lo <= hi."
            )
        self.options = tuple(min_jaccard_overlaps)
        self.trials = int(trials)
        self.aspect_lo, self.aspect_hi = (
            float(aspect_ratio_range[0]),
            float(aspect_ratio_range[1]),
        )
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.p = float(p)

    def __call__(self, image: Image, bboxes: BBoxes) -> Tuple[Image, BBoxes]:
        # Probability gate: leave image untouched with prob (1 - p).
        if random.random() >= self.p:
            return image, bboxes

        h, w = image.shape[:2]
        if h <= 1 or w <= 1:
            return image, bboxes

        # SSD-style option sampling. ``None`` short-circuits to no-op.
        mode = random.choice(self.options)
        if mode is None:
            return image, bboxes

        # No-box edge case: random crop without IoU constraint.
        if not bboxes:
            crop = self._sample_crop(h, w)
            if crop is None:
                return image, bboxes
            x1, y1, x2, y2 = crop
            cropped = image[y1:y2, x1:x2]
            cropped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            return cropped, bboxes

        # Convert normalised bboxes once to absolute pixel xyxy + centers for
        # the IoU/center-in-crop checks.
        boxes_px = np.array(
            [[b[0] * w, b[1] * h, b[2] * w, b[3] * h] for b in bboxes],
            dtype=np.float32,
        )
        centers_x = (boxes_px[:, 0] + boxes_px[:, 2]) * 0.5
        centers_y = (boxes_px[:, 1] + boxes_px[:, 3]) * 0.5

        for _ in range(self.trials):
            crop = self._sample_crop(h, w)
            if crop is None:
                continue
            x1, y1, x2, y2 = crop

            # Filter boxes whose centers fall inside the crop (SSD recipe).
            mask = (
                (centers_x > x1)
                & (centers_x < x2)
                & (centers_y > y1)
                & (centers_y < y2)
            )
            if not bool(mask.any()):
                # Crop has no positives; reject and try another crop.
                continue

            kept = boxes_px[mask]
            ious = _box_iou_per_pair(kept, np.array([[x1, y1, x2, y2]], dtype=np.float32))
            if float(ious.min()) < float(mode):
                # IoU threshold not satisfied for at least one kept box.
                continue

            # Crop accepted: clip surviving boxes to the crop window and
            # remap to normalised coordinates of the (resized) crop.
            kept[:, 0] = np.clip(kept[:, 0], x1, x2) - x1
            kept[:, 1] = np.clip(kept[:, 1], y1, y2) - y1
            kept[:, 2] = np.clip(kept[:, 2], x1, x2) - x1
            kept[:, 3] = np.clip(kept[:, 3], y1, y2) - y1
            crop_w = max(x2 - x1, 1)
            crop_h = max(y2 - y1, 1)

            new_bboxes: BBoxes = []
            mask_list = mask.tolist()
            kept_idx = 0
            for keep, original in zip(mask_list, bboxes):
                if not keep:
                    continue
                box = kept[kept_idx]
                kept_idx += 1
                new_bboxes.append([
                    float(box[0]) / crop_w,
                    float(box[1]) / crop_h,
                    float(box[2]) / crop_w,
                    float(box[3]) / crop_h,
                ] + list(original[4:]))

            cropped = image[y1:y2, x1:x2]
            cropped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            return cropped, _clip_and_filter_bboxes(new_bboxes)

        # No accepted crop in ``trials`` attempts: return original (SSD default).
        return image, bboxes

    def _sample_crop(self, h: int, w: int):
        """Sample one candidate crop window as (x1, y1, x2, y2) pixel ints.

        Returns ``None`` if the sampled aspect ratio yields a window with
        non-positive width or height (rare; happens at extreme aspect ratios
        combined with small scales).
        """
        scale = random.uniform(self.min_scale, self.max_scale)
        # Sample log-uniformly in aspect ratio so the distribution is
        # symmetric in landscape vs portrait crops.
        ar_lo = np.log(self.aspect_lo)
        ar_hi = np.log(self.aspect_hi)
        ar = float(np.exp(random.uniform(ar_lo, ar_hi)))

        # Solve: cw * ch = scale^2 * w * h AND cw / ch = ar  (relative to
        # the input frame). Result clipped to the input size to avoid edge
        # cases where an extreme AR pushes either dimension above 1.0.
        area = scale * scale * w * h
        cw = int(round(np.sqrt(area * ar)))
        ch = int(round(np.sqrt(area / ar)))
        cw = min(cw, w)
        ch = min(ch, h)
        if cw < 2 or ch < 2:
            return None

        x1 = random.randint(0, w - cw)
        y1 = random.randint(0, h - ch)
        return x1, y1, x1 + cw, y1 + ch

    def __repr__(self) -> str:
        return (
            f"RandomIoUCrop(options={self.options}, trials={self.trials}, "
            f"aspect=({self.aspect_lo}, {self.aspect_hi}), "
            f"scale=({self.min_scale}, {self.max_scale}), p={self.p})"
        )


def _box_iou_per_pair(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorised IoU between every row of ``boxes_a`` and every row of ``boxes_b``.

    Shapes: ``boxes_a`` (N, 4), ``boxes_b`` (M, 4) in xyxy. Returns (N, M).
    """
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)
    a = boxes_a[:, None, :]  # (N, 1, 4)
    b = boxes_b[None, :, :]  # (1, M, 4)
    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)


def build_augmentation_pipeline(config: dict) -> Compose:
    """Build a composed augmentation pipeline from a configuration dict.

    Supported keys:

        augmentation:
            scale: [0.5, 1.5]        # random scale range
            translate: 0.1           # random translation fraction
            hsv_h: 0.015             # hue gain
            hsv_s: 0.7              # saturation gain
            hsv_v: 0.4              # value gain
            horizontal_flip: true
            brightness_range: [0.8, 1.2]  # ignored when HSV is active
            random_iou_crop: true    # SSD-style sample-IoU crop; can also be
                                     # a dict to override trials/scale/etc.

    Multi-image operations (mosaic, mixup) are handled at the Dataset level,
    not in this pipeline. Keys ``mosaic``, ``mixup``, ``mosaic_off_epochs``,
    ``rotation_range`` are accepted but ignored here for backward compatibility.

    Args:
        config: Augmentation configuration dictionary. Either the full config
            (with ``augmentation`` key) or the augmentation sub-dict directly.

    Returns:
        A Compose instance chaining the enabled transforms.
    """
    # Support both full config and augmentation sub-dict
    if "augmentation" in config:
        aug_config = config["augmentation"]
    else:
        aug_config = config

    transforms: list = []

    # SSD-style sample-IoU crop runs first because it changes both image
    # content (a random crop is resized back to the input size) and bbox
    # geometry. Downstream RandomScale/RandomTranslate then operate on the
    # already-cropped frame, which is exactly the order torchvision's
    # SSDLite pipeline uses.
    iou_crop_cfg = aug_config.get("random_iou_crop", None)
    if iou_crop_cfg:
        if iou_crop_cfg is True:
            transforms.append(RandomIoUCrop())
        elif isinstance(iou_crop_cfg, dict):
            transforms.append(RandomIoUCrop(
                min_jaccard_overlaps=tuple(
                    iou_crop_cfg.get(
                        "min_jaccard_overlaps", RandomIoUCrop.DEFAULT_OPTIONS
                    )
                ),
                trials=int(iou_crop_cfg.get("trials", 40)),
                aspect_ratio_range=tuple(
                    iou_crop_cfg.get("aspect_ratio_range", (0.5, 2.0))
                ),
                min_scale=float(iou_crop_cfg.get("min_scale", 0.3)),
                max_scale=float(iou_crop_cfg.get("max_scale", 1.0)),
                p=float(iou_crop_cfg.get("p", 1.0)),
            ))

    # Scale (applied first — changes spatial layout)
    scale = aug_config.get("scale", None)
    if scale is not None:
        if isinstance(scale, (list, tuple)) and len(scale) == 2:
            transforms.append(RandomScale(scale_range=tuple(scale)))

    # Translate
    translate = aug_config.get("translate", None)
    if translate is not None and float(translate) > 0:
        transforms.append(RandomTranslate(translate=float(translate)))

    # HSV color augmentation
    hsv_h = aug_config.get("hsv_h", None)
    hsv_s = aug_config.get("hsv_s", None)
    hsv_v = aug_config.get("hsv_v", None)
    hsv_active = any(v is not None and float(v) > 0 for v in [hsv_h, hsv_s, hsv_v])
    if hsv_active:
        transforms.append(RandomHSV(
            h_gain=float(hsv_h or 0),
            s_gain=float(hsv_s or 0),
            v_gain=float(hsv_v or 0),
        ))

    # Horizontal flip
    if aug_config.get("horizontal_flip", False):
        transforms.append(RandomHorizontalFlip(p=0.5))

    # Vertical flip (not recommended for road images)
    if aug_config.get("vertical_flip", False):
        transforms.append(RandomVerticalFlip(p=0.5))

    # Brightness (only if HSV is NOT active — HSV-V subsumes brightness)
    if not hsv_active:
        brightness_range = aug_config.get("brightness_range", None)
        if brightness_range is not None:
            if isinstance(brightness_range, (list, tuple)) and len(brightness_range) == 2:
                transforms.append(RandomBrightness(brightness_range=tuple(brightness_range)))

    return Compose(transforms)
