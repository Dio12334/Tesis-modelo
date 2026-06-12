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
