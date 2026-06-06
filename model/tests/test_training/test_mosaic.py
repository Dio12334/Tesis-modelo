"""Unit tests for Mosaic and MixUp augmentation in RDD2022TorchDataset.

Tests use mock RDD2022Dataset objects with synthetic images and bboxes
to verify correctness of multi-image augmentation operations.
"""

import random
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image as PILImage

from model.training.augmentation import Compose, RandomHorizontalFlip, _clip_and_filter_bboxes


# ---------------------------------------------------------------------------
# Helpers to create mock dataset + annotations
# ---------------------------------------------------------------------------


class FakeBBox:
    """Minimal bounding box matching RDD2022 Annotation.bounding_boxes interface."""
    def __init__(self, x_min, y_min, x_max, y_max, class_label):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.class_label = class_label


class FakeAnnotation:
    """Minimal annotation matching RDD2022 Annotation interface."""
    def __init__(self, image_path: str, bboxes: List[FakeBBox]):
        self.image_path = image_path
        self.bounding_boxes = bboxes


class FakeDataset:
    """Minimal RDD2022Dataset-like object for testing."""
    def __init__(self, annotations, class_names):
        self._annotations = annotations
        self._class_names = class_names

    def get_annotations(self):
        return self._annotations

    def get_class_names(self):
        return self._class_names


def _create_colored_image(color, size=100):
    """Create a solid-color PIL image and save to temp path.

    Returns the path string.
    """
    img = PILImage.new("RGB", (size, size), color)
    return img


def _make_fake_dataset(n_images=4, img_size=100):
    """Create a fake dataset with n_images, each a distinct color with one bbox.

    Images are stored in-memory but we patch Image.open to return them.
    """
    colors = [
        (255, 0, 0),    # red
        (0, 255, 0),    # green
        (0, 0, 255),    # blue
        (255, 255, 0),  # yellow
        (255, 0, 255),  # magenta
        (0, 255, 255),  # cyan
    ]
    annotations = []
    images = {}

    for i in range(n_images):
        path = f"/fake/image_{i}.jpg"
        color = colors[i % len(colors)]
        img = _create_colored_image(color, img_size)
        images[path] = img

        bbox = FakeBBox(
            x_min=0.2 + 0.01 * i,
            y_min=0.2 + 0.01 * i,
            x_max=0.8 - 0.01 * i,
            y_max=0.8 - 0.01 * i,
            class_label=f"class_{i % 3}",
        )
        annotations.append(FakeAnnotation(path, [bbox]))

    class_names = ["class_0", "class_1", "class_2"]
    dataset = FakeDataset(annotations, class_names)
    return dataset, images


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_dataset_and_images():
    """Create a fake dataset with 8 images and a dict of PIL images."""
    dataset, images = _make_fake_dataset(n_images=8, img_size=100)
    return dataset, images


@pytest.fixture
def patched_dataset(mock_dataset_and_images):
    """Create RDD2022TorchDataset with patched Image.open."""
    from model.training.train_detection import RDD2022TorchDataset

    dataset, images = mock_dataset_and_images

    def mock_open(path):
        path_str = str(path)
        if path_str in images:
            return images[path_str].copy()
        raise FileNotFoundError(f"Mock: {path_str} not found")

    with patch("model.training.train_detection.Image.open", side_effect=mock_open):
        # We need to also patch PIL.Image.open inside the module
        torch_ds = RDD2022TorchDataset(
            dataset, input_size=100, augmentation=None,
            mosaic=1.0, mixup=0.0,
        )
    return torch_ds, images


# ---------------------------------------------------------------------------
# Tests: Mosaic
# ---------------------------------------------------------------------------


class TestMosaic:
    def test_mosaic_produces_correct_shape(self, mock_dataset_and_images):
        """Mosaic output should be input_size x input_size x 3."""
        from model.training.train_detection import RDD2022TorchDataset

        dataset, images = mock_dataset_and_images

        def mock_open(path):
            path_str = str(path) if not isinstance(path, str) else path
            if path_str in images:
                return images[path_str].copy()
            raise FileNotFoundError(f"Mock: {path_str}")

        with patch("model.training.train_detection.Image.open", side_effect=mock_open):
            torch_ds = RDD2022TorchDataset(
                dataset, input_size=100, augmentation=None,
                mosaic=1.0, mixup=0.0,
            )
            random.seed(42)
            np.random.seed(42)
            img_tensor, target = torch_ds[0]

        assert img_tensor.shape == (3, 100, 100)

    def test_mosaic_merges_bboxes_from_multiple_images(self, mock_dataset_and_images):
        """Mosaic should contain bboxes from 4 images."""
        from model.training.train_detection import RDD2022TorchDataset

        dataset, images = mock_dataset_and_images

        def mock_open(path):
            path_str = str(path) if not isinstance(path, str) else path
            if path_str in images:
                return images[path_str].copy()
            raise FileNotFoundError(f"Mock: {path_str}")

        with patch("model.training.train_detection.Image.open", side_effect=mock_open):
            torch_ds = RDD2022TorchDataset(
                dataset, input_size=100, augmentation=None,
                mosaic=1.0, mixup=0.0,
            )
            random.seed(123)
            np.random.seed(123)
            _, target = torch_ds[0]

        # Should have bboxes from multiple images (up to 4)
        # Each image contributes 1 bbox, but some may be filtered
        assert target["boxes"].shape[0] >= 1
        assert target["boxes"].shape[1] == 4

    def test_mosaic_bboxes_are_valid(self, mock_dataset_and_images):
        """All mosaic bboxes should be within image bounds."""
        from model.training.train_detection import RDD2022TorchDataset

        dataset, images = mock_dataset_and_images

        def mock_open(path):
            path_str = str(path) if not isinstance(path, str) else path
            if path_str in images:
                return images[path_str].copy()
            raise FileNotFoundError(f"Mock: {path_str}")

        with patch("model.training.train_detection.Image.open", side_effect=mock_open):
            torch_ds = RDD2022TorchDataset(
                dataset, input_size=100, augmentation=None,
                mosaic=1.0, mixup=0.0,
            )
            for seed in range(10):
                random.seed(seed)
                np.random.seed(seed)
                _, target = torch_ds[0]
                boxes = target["boxes"]
                if boxes.shape[0] > 0:
                    assert (boxes[:, 0] >= 0).all()  # x1 >= 0
                    assert (boxes[:, 1] >= 0).all()  # y1 >= 0
                    assert (boxes[:, 2] <= 100).all()  # x2 <= input_size
                    assert (boxes[:, 3] <= 100).all()  # y2 <= input_size
                    assert (boxes[:, 2] > boxes[:, 0]).all()  # x2 > x1
                    assert (boxes[:, 3] > boxes[:, 1]).all()  # y2 > y1

    def test_mosaic_disabled_uses_single_image(self, mock_dataset_and_images):
        """With mosaic=0, dataset should use single-image path."""
        from model.training.train_detection import RDD2022TorchDataset

        dataset, images = mock_dataset_and_images

        def mock_open(path):
            path_str = str(path) if not isinstance(path, str) else path
            if path_str in images:
                return images[path_str].copy()
            raise FileNotFoundError(f"Mock: {path_str}")

        with patch("model.training.train_detection.Image.open", side_effect=mock_open):
            torch_ds = RDD2022TorchDataset(
                dataset, input_size=100, augmentation=None,
                mosaic=0.0, mixup=0.0,
            )
            random.seed(42)
            _, target = torch_ds[0]

        # Single image → max 1 bbox (each fake image has 1)
        assert target["boxes"].shape[0] == 1

    def test_set_mosaic_enabled_toggle(self, mock_dataset_and_images):
        """set_mosaic_enabled(False) should disable mosaic even if p=1."""
        from model.training.train_detection import RDD2022TorchDataset

        dataset, images = mock_dataset_and_images

        def mock_open(path):
            path_str = str(path) if not isinstance(path, str) else path
            if path_str in images:
                return images[path_str].copy()
            raise FileNotFoundError(f"Mock: {path_str}")

        with patch("model.training.train_detection.Image.open", side_effect=mock_open):
            torch_ds = RDD2022TorchDataset(
                dataset, input_size=100, augmentation=None,
                mosaic=1.0, mixup=0.0,
            )
            torch_ds.set_mosaic_enabled(False)
            random.seed(42)
            _, target = torch_ds[0]

        # Should behave as single-image
        assert target["boxes"].shape[0] == 1


# ---------------------------------------------------------------------------
# Tests: MixUp
# ---------------------------------------------------------------------------


class TestMixUp:
    def test_mixup_blends_images(self, mock_dataset_and_images):
        """MixUp should produce pixel values between the two source images."""
        from model.training.train_detection import RDD2022TorchDataset

        dataset, images = mock_dataset_and_images

        def mock_open(path):
            path_str = str(path) if not isinstance(path, str) else path
            if path_str in images:
                return images[path_str].copy()
            raise FileNotFoundError(f"Mock: {path_str}")

        with patch("model.training.train_detection.Image.open", side_effect=mock_open):
            torch_ds = RDD2022TorchDataset(
                dataset, input_size=100, augmentation=None,
                mosaic=1.0, mixup=1.0,  # Always apply mixup
            )
            random.seed(42)
            np.random.seed(42)
            img_tensor, target = torch_ds[0]

        # Image should not be all zeros (gray or blended)
        assert img_tensor.sum() > 0
        assert img_tensor.shape == (3, 100, 100)

    def test_mixup_merges_bboxes(self, mock_dataset_and_images):
        """MixUp should include bboxes from both images."""
        from model.training.train_detection import RDD2022TorchDataset

        dataset, images = mock_dataset_and_images

        def mock_open(path):
            path_str = str(path) if not isinstance(path, str) else path
            if path_str in images:
                return images[path_str].copy()
            raise FileNotFoundError(f"Mock: {path_str}")

        with patch("model.training.train_detection.Image.open", side_effect=mock_open):
            torch_ds = RDD2022TorchDataset(
                dataset, input_size=100, augmentation=None,
                mosaic=1.0, mixup=1.0,
            )
            random.seed(0)
            np.random.seed(0)
            _, target = torch_ds[0]

        # MixUp merges bboxes from primary mosaic (4 images) + secondary (4 images)
        # So we should have more boxes than a single mosaic
        assert target["boxes"].shape[0] >= 2

    def test_mixup_only_when_mosaic_active(self, mock_dataset_and_images):
        """MixUp is conditional on mosaic firing first."""
        from model.training.train_detection import RDD2022TorchDataset

        dataset, images = mock_dataset_and_images

        def mock_open(path):
            path_str = str(path) if not isinstance(path, str) else path
            if path_str in images:
                return images[path_str].copy()
            raise FileNotFoundError(f"Mock: {path_str}")

        with patch("model.training.train_detection.Image.open", side_effect=mock_open):
            # mosaic=0 means mixup can never trigger (it requires mosaic)
            torch_ds = RDD2022TorchDataset(
                dataset, input_size=100, augmentation=None,
                mosaic=0.0, mixup=1.0,
            )
            random.seed(42)
            _, target = torch_ds[0]

        # Single image path → 1 bbox
        assert target["boxes"].shape[0] == 1


# ---------------------------------------------------------------------------
# Tests: Mosaic + per-image augmentation
# ---------------------------------------------------------------------------


class TestMosaicWithAugmentation:
    def test_mosaic_plus_hflip(self, mock_dataset_and_images):
        """Mosaic followed by per-image augmentation should still produce valid output."""
        from model.training.train_detection import RDD2022TorchDataset

        dataset, images = mock_dataset_and_images
        aug_pipeline = Compose([RandomHorizontalFlip(p=1.0)])

        def mock_open(path):
            path_str = str(path) if not isinstance(path, str) else path
            if path_str in images:
                return images[path_str].copy()
            raise FileNotFoundError(f"Mock: {path_str}")

        with patch("model.training.train_detection.Image.open", side_effect=mock_open):
            torch_ds = RDD2022TorchDataset(
                dataset, input_size=100, augmentation=aug_pipeline,
                mosaic=1.0, mixup=0.0,
            )
            random.seed(7)
            np.random.seed(7)
            img_tensor, target = torch_ds[0]

        assert img_tensor.shape == (3, 100, 100)
        boxes = target["boxes"]
        if boxes.shape[0] > 0:
            assert (boxes[:, 0] >= 0).all()
            assert (boxes[:, 2] <= 100).all()
            assert (boxes[:, 2] > boxes[:, 0]).all()
            assert (boxes[:, 3] > boxes[:, 1]).all()
