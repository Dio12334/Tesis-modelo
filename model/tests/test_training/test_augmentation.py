"""Unit tests for data augmentation transforms."""

import numpy as np
import pytest

from model.training.augmentation import (
    Compose,
    RandomBrightness,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    build_augmentation_pipeline,
)


@pytest.fixture
def sample_image():
    """Create a sample 100x100 RGB image."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_bboxes():
    """Create sample normalized bounding boxes."""
    return [[0.1, 0.2, 0.5, 0.6], [0.3, 0.4, 0.8, 0.9]]


class TestBuildAugmentationPipeline:
    """Tests for build_augmentation_pipeline function."""

    def test_full_config_with_augmentation_key(self):
        # Legacy keys (rotation_range, mosaic) must be silently ignored.
        config = {
            "augmentation": {
                "horizontal_flip": True,
                "vertical_flip": False,
                "rotation_range": 15,
                "brightness_range": [0.8, 1.2],
                "mosaic": True,
            }
        }
        pipeline = build_augmentation_pipeline(config)
        assert isinstance(pipeline, Compose)
        # hflip + brightness only (rotation/mosaic removed, vflip disabled)
        assert len(pipeline.transforms) == 2
        assert isinstance(pipeline.transforms[0], RandomHorizontalFlip)
        assert isinstance(pipeline.transforms[1], RandomBrightness)

    def test_direct_augmentation_dict(self):
        config = {
            "horizontal_flip": True,
            "vertical_flip": True,
            "brightness_range": [0.9, 1.1],
        }
        pipeline = build_augmentation_pipeline(config)
        # hflip + vflip + brightness
        assert len(pipeline.transforms) == 3
        assert isinstance(pipeline.transforms[0], RandomHorizontalFlip)
        assert isinstance(pipeline.transforms[1], RandomVerticalFlip)
        assert isinstance(pipeline.transforms[2], RandomBrightness)

    def test_empty_config(self):
        pipeline = build_augmentation_pipeline({})
        assert isinstance(pipeline, Compose)
        assert len(pipeline.transforms) == 0

    def test_all_disabled(self):
        config = {
            "horizontal_flip": False,
            "vertical_flip": False,
        }
        pipeline = build_augmentation_pipeline(config)
        assert len(pipeline.transforms) == 0

    def test_only_horizontal_flip(self):
        config = {"horizontal_flip": True}
        pipeline = build_augmentation_pipeline(config)
        assert len(pipeline.transforms) == 1
        assert isinstance(pipeline.transforms[0], RandomHorizontalFlip)

    def test_legacy_keys_ignored(self):
        # rotation_range and mosaic should not produce any transform.
        config = {"rotation_range": 30, "mosaic": True}
        pipeline = build_augmentation_pipeline(config)
        assert len(pipeline.transforms) == 0


class TestRandomHorizontalFlip:
    """Tests for RandomHorizontalFlip transform."""

    def test_always_flip(self, sample_image, sample_bboxes):
        transform = RandomHorizontalFlip(p=1.0)
        result_img, result_bboxes = transform(sample_image.copy(), sample_bboxes[:])
        # x_min should become 1 - old x_max
        assert abs(result_bboxes[0][0] - (1.0 - 0.5)) < 1e-9
        assert abs(result_bboxes[0][2] - (1.0 - 0.1)) < 1e-9
        # y coords unchanged
        assert result_bboxes[0][1] == 0.2
        assert result_bboxes[0][3] == 0.6

    def test_never_flip(self, sample_image, sample_bboxes):
        transform = RandomHorizontalFlip(p=0.0)
        result_img, result_bboxes = transform(sample_image.copy(), sample_bboxes[:])
        assert np.array_equal(result_img, sample_image)
        assert result_bboxes == sample_bboxes

    def test_image_flipped(self, sample_image, sample_bboxes):
        transform = RandomHorizontalFlip(p=1.0)
        result_img, _ = transform(sample_image.copy(), sample_bboxes[:])
        expected = np.fliplr(sample_image)
        assert np.array_equal(result_img, expected)


class TestRandomVerticalFlip:
    """Tests for RandomVerticalFlip transform."""

    def test_always_flip(self, sample_image, sample_bboxes):
        transform = RandomVerticalFlip(p=1.0)
        result_img, result_bboxes = transform(sample_image.copy(), sample_bboxes[:])
        # y_min should become 1 - old y_max
        assert abs(result_bboxes[0][1] - (1.0 - 0.6)) < 1e-9
        assert abs(result_bboxes[0][3] - (1.0 - 0.2)) < 1e-9
        # x coords unchanged
        assert result_bboxes[0][0] == 0.1
        assert result_bboxes[0][2] == 0.5

    def test_never_flip(self, sample_image, sample_bboxes):
        transform = RandomVerticalFlip(p=0.0)
        result_img, result_bboxes = transform(sample_image.copy(), sample_bboxes[:])
        assert np.array_equal(result_img, sample_image)
        assert result_bboxes == sample_bboxes


class TestRandomBrightness:
    """Tests for RandomBrightness transform."""

    def test_no_change(self, sample_image, sample_bboxes):
        transform = RandomBrightness(brightness_range=(1.0, 1.0))
        result_img, result_bboxes = transform(sample_image.copy(), sample_bboxes[:])
        assert np.array_equal(result_img, sample_image)
        assert result_bboxes == sample_bboxes

    def test_bboxes_unchanged(self, sample_image, sample_bboxes):
        transform = RandomBrightness(brightness_range=(0.5, 1.5))
        _, result_bboxes = transform(sample_image.copy(), sample_bboxes[:])
        assert result_bboxes == sample_bboxes

    def test_output_clipped(self):
        # White image with brightness > 1 should still be <= 255
        white = np.full((10, 10, 3), 250, dtype=np.uint8)
        transform = RandomBrightness(brightness_range=(2.0, 2.0))
        result_img, _ = transform(white, [])
        assert result_img.max() <= 255


class TestCompose:
    """Tests for Compose pipeline."""

    def test_empty_compose(self, sample_image, sample_bboxes):
        pipeline = Compose([])
        result_img, result_bboxes = pipeline(sample_image.copy(), sample_bboxes[:])
        assert np.array_equal(result_img, sample_image)
        assert result_bboxes == sample_bboxes

    def test_chaining(self, sample_image, sample_bboxes):
        pipeline = Compose([
            RandomHorizontalFlip(p=1.0),
            RandomVerticalFlip(p=1.0),
        ])
        result_img, result_bboxes = pipeline(sample_image.copy(), sample_bboxes[:])
        # After both flips, check coordinates
        # hflip: x_min=1-0.5=0.5, x_max=1-0.1=0.9, y unchanged
        # vflip: y_min=1-0.6=0.4, y_max=1-0.2=0.8, x unchanged
        assert abs(result_bboxes[0][0] - 0.5) < 1e-9
        assert abs(result_bboxes[0][1] - 0.4) < 1e-9
        assert abs(result_bboxes[0][2] - 0.9) < 1e-9
        assert abs(result_bboxes[0][3] - 0.8) < 1e-9
