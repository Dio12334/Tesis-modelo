"""Data augmentation transforms for object detection training.

Provides configurable image transforms that operate on images and bounding boxes.
Each transform accepts an image (numpy array) and a list of bounding boxes,
returning the transformed image and adjusted bounding boxes.

Transforms are composable via the Compose class and can be built from a
YAML-style augmentation config dict using build_augmentation_pipeline().
"""

import math
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


class RandomRotation:
    """Randomly rotates the image within a degree range.

    Bounding boxes are rotated and then re-computed as axis-aligned
    bounding rectangles enclosing the rotated corners. Boxes are clipped
    to [0, 1] range after rotation.
    """

    def __init__(self, max_degrees: float = 15.0):
        self.max_degrees = max_degrees

    def __call__(self, image: Image, bboxes: BBoxes) -> Tuple[Image, BBoxes]:
        angle = random.uniform(-self.max_degrees, self.max_degrees)
        if abs(angle) < 1e-6:
            return image, bboxes

        h, w = image.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        # Rotation matrix
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        # Rotate image using affine transform (manual implementation)
        image = self._rotate_image(image, angle, cx, cy, cos_a, sin_a, w, h)

        # Rotate bounding boxes
        new_bboxes = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox[:4]
            # Convert normalized coords to pixel coords
            px_min, py_min = x_min * w, y_min * h
            px_max, py_max = x_max * w, y_max * h

            # Get four corners
            corners = [
                (px_min, py_min),
                (px_max, py_min),
                (px_max, py_max),
                (px_min, py_max),
            ]

            # Rotate each corner around center
            rotated_corners = []
            for px, py in corners:
                rx = cos_a * (px - cx) - sin_a * (py - cy) + cx
                ry = sin_a * (px - cx) + cos_a * (py - cy) + cy
                rotated_corners.append((rx, ry))

            # Get axis-aligned bounding box of rotated corners
            xs = [c[0] for c in rotated_corners]
            ys = [c[1] for c in rotated_corners]
            new_px_min = max(0.0, min(xs))
            new_py_min = max(0.0, min(ys))
            new_px_max = min(float(w), max(xs))
            new_py_max = min(float(h), max(ys))

            # Convert back to normalized coords
            new_x_min = new_px_min / w
            new_y_min = new_py_min / h
            new_x_max = new_px_max / w
            new_y_max = new_py_max / h

            # Only keep valid boxes
            if new_x_max > new_x_min and new_y_max > new_y_min:
                new_bbox = [new_x_min, new_y_min, new_x_max, new_y_max] + bbox[4:]
                new_bboxes.append(new_bbox)

        bboxes = new_bboxes
        return image, bboxes

    def _rotate_image(
        self,
        image: Image,
        angle: float,
        cx: float,
        cy: float,
        cos_a: float,
        sin_a: float,
        w: int,
        h: int,
    ) -> Image:
        """Rotate image around center using inverse mapping."""
        rotated = np.zeros_like(image)
        # Inverse rotation to map destination pixels to source
        for y in range(h):
            for x in range(w):
                # Map destination (x, y) back to source
                src_x = cos_a * (x - cx) + sin_a * (y - cy) + cx
                src_y = -sin_a * (x - cx) + cos_a * (y - cy) + cy
                src_xi = int(round(src_x))
                src_yi = int(round(src_y))
                if 0 <= src_xi < w and 0 <= src_yi < h:
                    rotated[y, x] = image[src_yi, src_xi]
        return rotated

    def __repr__(self) -> str:
        return f"RandomRotation(max_degrees={self.max_degrees})"


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


class Mosaic:
    """Mosaic augmentation placeholder.

    Mosaic augmentation combines 4 images into one by placing them in a 2x2 grid.
    Since this requires access to multiple images from the dataset, this class
    stores the configuration and provides a single-image fallback that applies
    a random crop and resize to simulate partial mosaic behavior.

    For full mosaic augmentation, the training pipeline should call
    `apply_mosaic()` with 4 images before individual transforms.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image, bboxes: BBoxes) -> Tuple[Image, BBoxes]:
        """Single-image fallback: random crop to simulate mosaic quadrant."""
        if random.random() >= self.p:
            return image, bboxes

        h, w = image.shape[:2]
        # Pick a random quadrant
        cx = random.uniform(0.3, 0.7)
        cy = random.uniform(0.3, 0.7)

        # Crop region in normalized coords
        crop_x_min = cx - 0.5
        crop_y_min = cy - 0.5
        crop_x_max = cx + 0.5
        crop_y_max = cy + 0.5

        # Clip to image bounds
        crop_x_min = max(0.0, crop_x_min)
        crop_y_min = max(0.0, crop_y_min)
        crop_x_max = min(1.0, crop_x_max)
        crop_y_max = min(1.0, crop_y_max)

        # Convert to pixel coords for cropping
        px_min = int(crop_x_min * w)
        py_min = int(crop_y_min * h)
        px_max = int(crop_x_max * w)
        py_max = int(crop_y_max * h)

        # Ensure valid crop dimensions
        if px_max <= px_min or py_max <= py_min:
            return image, bboxes

        cropped = image[py_min:py_max, px_min:px_max].copy()

        # Adjust bounding boxes to new crop region
        crop_w = crop_x_max - crop_x_min
        crop_h = crop_y_max - crop_y_min
        new_bboxes = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox[:4]
            # Shift and scale to crop region
            new_x_min = (x_min - crop_x_min) / crop_w
            new_y_min = (y_min - crop_y_min) / crop_h
            new_x_max = (x_max - crop_x_min) / crop_w
            new_y_max = (y_max - crop_y_min) / crop_h

            # Clip to [0, 1]
            new_x_min = max(0.0, new_x_min)
            new_y_min = max(0.0, new_y_min)
            new_x_max = min(1.0, new_x_max)
            new_y_max = min(1.0, new_y_max)

            # Keep only if box has valid area
            if new_x_max > new_x_min and new_y_max > new_y_min:
                new_bbox = [new_x_min, new_y_min, new_x_max, new_y_max] + bbox[4:]
                new_bboxes.append(new_bbox)

        return cropped, new_bboxes

    @staticmethod
    def apply_mosaic(
        images: List[Image], bboxes_list: List[BBoxes], target_size: Tuple[int, int]
    ) -> Tuple[Image, BBoxes]:
        """Combine 4 images into a 2x2 mosaic grid.

        Args:
            images: List of exactly 4 images (numpy arrays).
            bboxes_list: List of 4 bounding box lists corresponding to each image.
            target_size: (height, width) of the output mosaic image.

        Returns:
            Tuple of (mosaic_image, combined_bboxes) with adjusted coordinates.
        """
        if len(images) != 4 or len(bboxes_list) != 4:
            raise ValueError("Mosaic requires exactly 4 images and bbox lists")

        th, tw = target_size
        half_h, half_w = th // 2, tw // 2
        mosaic = np.zeros((th, tw, 3), dtype=np.uint8)
        combined_bboxes: BBoxes = []

        # Quadrant positions: (y_offset, x_offset, height, width)
        quadrants = [
            (0, 0, half_h, half_w),           # top-left
            (0, half_w, half_h, tw - half_w),  # top-right
            (half_h, 0, th - half_h, half_w),  # bottom-left
            (half_h, half_w, th - half_h, tw - half_w),  # bottom-right
        ]

        for i, (y_off, x_off, qh, qw) in enumerate(quadrants):
            img = images[i]
            ih, iw = img.shape[:2]

            # Resize image to fit quadrant
            resized = _resize_image(img, qh, qw)
            mosaic[y_off : y_off + qh, x_off : x_off + qw] = resized

            # Adjust bounding boxes for this quadrant
            for bbox in bboxes_list[i]:
                x_min, y_min, x_max, y_max = bbox[:4]
                # Scale to quadrant position in mosaic
                new_x_min = (x_min * qw + x_off) / tw
                new_y_min = (y_min * qh + y_off) / th
                new_x_max = (x_max * qw + x_off) / tw
                new_y_max = (y_max * qh + y_off) / th
                combined_bboxes.append(
                    [new_x_min, new_y_min, new_x_max, new_y_max] + bbox[4:]
                )

        return mosaic, combined_bboxes

    def __repr__(self) -> str:
        return f"Mosaic(p={self.p})"


def _resize_image(image: Image, target_h: int, target_w: int) -> Image:
    """Simple nearest-neighbor resize for numpy images."""
    h, w = image.shape[:2]
    if h == target_h and w == target_w:
        return image.copy()

    row_indices = (np.arange(target_h) * h / target_h).astype(int)
    col_indices = (np.arange(target_w) * w / target_w).astype(int)
    row_indices = np.clip(row_indices, 0, h - 1)
    col_indices = np.clip(col_indices, 0, w - 1)

    return image[row_indices][:, col_indices]


def build_augmentation_pipeline(config: dict) -> Compose:
    """Build a composed augmentation pipeline from a configuration dict.

    The config dict format matches the YAML training augmentation config:

        augmentation:
            horizontal_flip: true
            vertical_flip: false
            rotation_range: 15
            brightness_range: [0.8, 1.2]
            mosaic: true

    This function accepts either the full config (with 'augmentation' key)
    or just the augmentation sub-dict directly.

    Args:
        config: Augmentation configuration dictionary.

    Returns:
        A Compose instance chaining the enabled transforms.
    """
    # Support both full config and augmentation sub-dict
    if "augmentation" in config:
        aug_config = config["augmentation"]
    else:
        aug_config = config

    transforms = []

    # Horizontal flip
    if aug_config.get("horizontal_flip", False):
        transforms.append(RandomHorizontalFlip(p=0.5))

    # Vertical flip
    if aug_config.get("vertical_flip", False):
        transforms.append(RandomVerticalFlip(p=0.5))

    # Rotation
    rotation_range = aug_config.get("rotation_range", 0)
    if rotation_range and rotation_range > 0:
        transforms.append(RandomRotation(max_degrees=float(rotation_range)))

    # Brightness
    brightness_range = aug_config.get("brightness_range", None)
    if brightness_range is not None:
        if isinstance(brightness_range, (list, tuple)) and len(brightness_range) == 2:
            transforms.append(RandomBrightness(brightness_range=tuple(brightness_range)))

    # Mosaic
    if aug_config.get("mosaic", False):
        transforms.append(Mosaic(p=0.5))

    return Compose(transforms)
