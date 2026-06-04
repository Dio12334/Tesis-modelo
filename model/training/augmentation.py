"""Data augmentation transforms for object detection training.

Provides configurable image transforms that operate on images and bounding boxes.
Each transform accepts an image (numpy array) and a list of bounding boxes,
returning the transformed image and adjusted bounding boxes.

Transforms are composable via the Compose class and can be built from a
YAML-style augmentation config dict using build_augmentation_pipeline().

Note: only cheap, well-tested transforms are provided. Mosaic and rotation
were intentionally removed because their CPU-side implementations were a
training bottleneck. Unknown keys in the config (e.g. legacy ``rotation_range``,
``mosaic``) are ignored for backward compatibility.
"""

import random
from typing import List, Tuple

import numpy as np


# Type aliases for clarity
Image = np.ndarray  # HxWxC uint8 numpy array
BBoxes = List[List[float]]  # List of [x_min, y_min, x_max, y_max] normalized [0,1]


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
                new_bbox = [new_x_min, y_min, new_x_max, y_max] + bbox[4:]
                new_bboxes.append(new_bbox)
            bboxes = new_bboxes
        return image, bboxes

    def __repr__(self) -> str:
        return f"RandomHorizontalFlip(p={self.p})"


class RandomVerticalFlip:
    """Randomly flips the image vertically with a given probability.

    Bounding box y-coordinates are mirrored accordingly.
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
                new_bbox = [x_min, new_y_min, x_max, new_y_max] + bbox[4:]
                new_bboxes.append(new_bbox)
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
        # Apply brightness factor and clip to valid range
        image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return image, bboxes

    def __repr__(self) -> str:
        return f"RandomBrightness(range=({self.low}, {self.high}))"


def build_augmentation_pipeline(config: dict) -> Compose:
    """Build a composed augmentation pipeline from a configuration dict.

    Supported keys:

        augmentation:
            horizontal_flip: true
            vertical_flip: false
            brightness_range: [0.8, 1.2]

    Legacy keys ``rotation_range`` and ``mosaic`` are accepted but ignored
    (those transforms were removed as a CPU-side training bottleneck).

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

    # Horizontal flip
    if aug_config.get("horizontal_flip", False):
        transforms.append(RandomHorizontalFlip(p=0.5))

    # Vertical flip
    if aug_config.get("vertical_flip", False):
        transforms.append(RandomVerticalFlip(p=0.5))

    # Brightness
    brightness_range = aug_config.get("brightness_range", None)
    if brightness_range is not None:
        if isinstance(brightness_range, (list, tuple)) and len(brightness_range) == 2:
            transforms.append(RandomBrightness(brightness_range=tuple(brightness_range)))

    return Compose(transforms)
