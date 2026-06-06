"""Integration tests for RDD2022TorchDataset.__getitem__ bbox-image alignment.

These tests construct a synthetic image with a known bright rectangle at a
known location, run it through ``RDD2022TorchDataset.__getitem__`` (with and
without augmentation), and verify two things end-to-end:

1. The output bbox tensor has the expected pixel coordinates at ``input_size``.
2. The pixels of the returned image tensor *inside* the predicted bbox are
   actually bright, and *outside* are actually dark — proving the bbox is
   aligned with the visible content of the image even after augmentation.

The image-content check is the strongest assertion: it does not assume any
particular coordinate convention, only that the bbox matches the pixels.
"""

from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch
from PIL import Image as PILImage

from model.datasets.base import Annotation, BoundingBox
from model.training.augmentation import (
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)
from model.training.train_detection import RDD2022TorchDataset


# Image / box geometry used throughout
IMG_SIZE = 200            # synthetic source image is 200x200
INPUT_SIZE = 100          # dataset resizes to 100x100
# White rectangle in source pixel space: cols [40,120), rows [80,160)
RECT_X_MIN_PX = 40
RECT_X_MAX_PX = 120
RECT_Y_MIN_PX = 80
RECT_Y_MAX_PX = 160
# Normalized bbox derived from that rectangle
NORM_X_MIN = RECT_X_MIN_PX / IMG_SIZE   # 0.20
NORM_Y_MIN = RECT_Y_MIN_PX / IMG_SIZE   # 0.40
NORM_X_MAX = RECT_X_MAX_PX / IMG_SIZE   # 0.60
NORM_Y_MAX = RECT_Y_MAX_PX / IMG_SIZE   # 0.80


class _FakeDataset:
    """Minimal duck-typed stand-in exposing get_annotations / get_class_names."""

    def __init__(self, annotations: List[Annotation], class_names: List[str]):
        self._annotations = annotations
        self._class_names = class_names

    def get_annotations(self) -> List[Annotation]:
        return self._annotations

    def get_class_names(self) -> List[str]:
        return self._class_names


@pytest.fixture
def synthetic_image_path(tmp_path: Path) -> Path:
    """Create a 200x200 black image with a single white rectangle and save to disk."""
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    img[RECT_Y_MIN_PX:RECT_Y_MAX_PX, RECT_X_MIN_PX:RECT_X_MAX_PX] = 255
    path = tmp_path / "synthetic.png"
    PILImage.fromarray(img).save(path)
    return path


@pytest.fixture
def fake_dataset(synthetic_image_path: Path) -> _FakeDataset:
    """Build a dataset with one annotation referencing the synthetic image."""
    bbox = BoundingBox(
        x_min=NORM_X_MIN,
        y_min=NORM_Y_MIN,
        x_max=NORM_X_MAX,
        y_max=NORM_Y_MAX,
        class_label="fisura_longitudinal",
    )
    ann = Annotation(image_path=synthetic_image_path, bounding_boxes=[bbox])
    return _FakeDataset([ann], ["fisura_longitudinal"])


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _image_tensor_to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert (C,H,W) float[0,1] tensor -> (H,W,C) uint8 array."""
    arr = img_tensor.permute(1, 2, 0).cpu().numpy()
    return (arr * 255.0).round().astype(np.uint8)


def _mean_intensity_in_box(img_uint8: np.ndarray, box_xyxy_px: torch.Tensor) -> float:
    """Mean grayscale intensity of pixels strictly inside the box."""
    x1, y1, x2, y2 = [int(round(float(v))) for v in box_xyxy_px]
    crop = img_uint8[y1:y2, x1:x2]
    assert crop.size > 0, f"Empty crop for box {(x1, y1, x2, y2)}"
    return float(crop.mean())


def _mean_intensity_outside_box(img_uint8: np.ndarray, box_xyxy_px: torch.Tensor) -> float:
    """Mean grayscale intensity of pixels strictly outside the box."""
    x1, y1, x2, y2 = [int(round(float(v))) for v in box_xyxy_px]
    mask = np.ones(img_uint8.shape[:2], dtype=bool)
    mask[y1:y2, x1:x2] = False
    return float(img_uint8[mask].mean())


def _assert_box_aligned_with_white_rect(
    img_tensor: torch.Tensor,
    boxes: torch.Tensor,
    expected_xyxy_px: tuple,
    tol_px: float = 1.5,
) -> None:
    """Assert (a) coords match expected and (b) bright pixels live inside box."""
    assert boxes.shape == (1, 4), f"Expected exactly one box, got shape {boxes.shape}"
    box = boxes[0]

    # --- Coordinate assertion -------------------------------------------------
    expected = torch.tensor(expected_xyxy_px, dtype=box.dtype)
    diff = (box - expected).abs()
    assert torch.all(diff <= tol_px), (
        f"Box coords {box.tolist()} differ from expected {expected_xyxy_px} "
        f"by more than {tol_px}px (diffs={diff.tolist()})"
    )

    # --- Visual alignment assertion ------------------------------------------
    img_uint8 = _image_tensor_to_uint8(img_tensor)
    inside = _mean_intensity_in_box(img_uint8, box)
    outside = _mean_intensity_outside_box(img_uint8, box)
    assert inside > 200.0, (
        f"Box does not cover the white rectangle: mean intensity inside={inside:.1f} "
        f"(expected >200). Box coords={box.tolist()}."
    )
    assert outside < 50.0, (
        f"Box leaks: mean intensity outside={outside:.1f} (expected <50). "
        f"Box coords={box.tolist()}."
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


class TestDatasetBBoxAlignmentNoAug:
    """Without augmentation: pure resize from 200x200 -> 100x100."""

    def test_box_matches_white_rectangle(self, fake_dataset):
        ds = RDD2022TorchDataset(fake_dataset, input_size=INPUT_SIZE, augmentation=None)
        img_tensor, target = ds[0]

        # Image is at input_size, single channel-first tensor in [0,1]
        assert img_tensor.shape == (3, INPUT_SIZE, INPUT_SIZE)
        assert target["labels"].tolist() == [0]

        # Expected pixel xyxy at input_size = norm * INPUT_SIZE
        expected = (
            NORM_X_MIN * INPUT_SIZE,  # 20
            NORM_Y_MIN * INPUT_SIZE,  # 40
            NORM_X_MAX * INPUT_SIZE,  # 60
            NORM_Y_MAX * INPUT_SIZE,  # 80
        )
        _assert_box_aligned_with_white_rect(img_tensor, target["boxes"], expected)


class TestDatasetBBoxAlignmentHFlip:
    """Horizontal flip (deterministic via p=1.0)."""

    def test_box_follows_image_after_hflip(self, fake_dataset):
        aug = Compose([RandomHorizontalFlip(p=1.0)])
        ds = RDD2022TorchDataset(fake_dataset, input_size=INPUT_SIZE, augmentation=aug)
        img_tensor, target = ds[0]

        # Normalized coords mirror about x=0.5: [1-x_max, y_min, 1-x_min, y_max]
        # = [0.40, 0.40, 0.80, 0.80]  -> pixels (40, 40, 80, 80)
        expected = (
            (1.0 - NORM_X_MAX) * INPUT_SIZE,  # 40
            NORM_Y_MIN * INPUT_SIZE,          # 40
            (1.0 - NORM_X_MIN) * INPUT_SIZE,  # 80
            NORM_Y_MAX * INPUT_SIZE,          # 80
        )
        _assert_box_aligned_with_white_rect(img_tensor, target["boxes"], expected)


class TestDatasetBBoxAlignmentVFlip:
    """Vertical flip (deterministic via p=1.0)."""

    def test_box_follows_image_after_vflip(self, fake_dataset):
        aug = Compose([RandomVerticalFlip(p=1.0)])
        ds = RDD2022TorchDataset(fake_dataset, input_size=INPUT_SIZE, augmentation=aug)
        img_tensor, target = ds[0]

        # Normalized coords mirror about y=0.5: [x_min, 1-y_max, x_max, 1-y_min]
        # = [0.20, 0.20, 0.60, 0.60]  -> pixels (20, 20, 60, 60)
        expected = (
            NORM_X_MIN * INPUT_SIZE,          # 20
            (1.0 - NORM_Y_MAX) * INPUT_SIZE,  # 20
            NORM_X_MAX * INPUT_SIZE,          # 60
            (1.0 - NORM_Y_MIN) * INPUT_SIZE,  # 60
        )
        _assert_box_aligned_with_white_rect(img_tensor, target["boxes"], expected)


class TestDatasetBBoxAlignmentHFlipThenVFlip:
    """Composed flips: bbox should follow both image transforms."""

    def test_box_follows_image_after_hvflip(self, fake_dataset):
        aug = Compose([RandomHorizontalFlip(p=1.0), RandomVerticalFlip(p=1.0)])
        ds = RDD2022TorchDataset(fake_dataset, input_size=INPUT_SIZE, augmentation=aug)
        img_tensor, target = ds[0]

        # Both mirrors: x -> 1-x, y -> 1-y
        # = [0.40, 0.20, 0.80, 0.60]  -> pixels (40, 20, 80, 60)
        expected = (
            (1.0 - NORM_X_MAX) * INPUT_SIZE,  # 40
            (1.0 - NORM_Y_MAX) * INPUT_SIZE,  # 20
            (1.0 - NORM_X_MIN) * INPUT_SIZE,  # 80
            (1.0 - NORM_Y_MIN) * INPUT_SIZE,  # 60
        )
        _assert_box_aligned_with_white_rect(img_tensor, target["boxes"], expected)
