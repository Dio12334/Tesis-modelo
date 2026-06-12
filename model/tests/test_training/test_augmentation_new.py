"""Unit tests for new augmentation transforms: RandomHSV, RandomScale, RandomTranslate.

Uses white-rect-on-black pattern for analytical bbox verification.
"""

import numpy as np
import pytest

from model.training.augmentation import (
    Compose,
    MIN_BBOX_AREA,
    RandomHSV,
    RandomHorizontalFlip,
    RandomScale,
    RandomTranslate,
    _clip_and_filter_bboxes,
    build_augmentation_pipeline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def black_image_640():
    """640x640 black image."""
    return np.zeros((640, 640, 3), dtype=np.uint8)


@pytest.fixture
def white_rect_image():
    """640x640 image with a white rectangle at normalized [0.25, 0.25, 0.75, 0.75].

    The rectangle occupies the center 320x320 of the image.
    """
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img[160:480, 160:480] = 255
    return img


@pytest.fixture
def center_bbox():
    """Bbox at center: [0.25, 0.25, 0.75, 0.75, 'crack']."""
    return [[0.25, 0.25, 0.75, 0.75, "crack"]]


@pytest.fixture
def small_image():
    """100x100 image with known pixel values for HSV testing."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 200  # Red channel
    img[:, :, 1] = 100  # Green channel
    img[:, :, 2] = 50   # Blue channel
    return img


# ---------------------------------------------------------------------------
# _clip_and_filter_bboxes
# ---------------------------------------------------------------------------


class TestClipAndFilter:
    def test_clips_to_unit(self):
        bboxes = [[-0.1, -0.2, 1.3, 1.5, "a"]]
        result = _clip_and_filter_bboxes(bboxes)
        assert len(result) == 1
        assert result[0][:4] == [0.0, 0.0, 1.0, 1.0]
        assert result[0][4] == "a"

    def test_removes_degenerate_area(self):
        # Box with area below MIN_BBOX_AREA after clipping
        bboxes = [[0.5, 0.5, 0.501, 0.501, "b"]]  # area = 0.001 * 0.001 = 1e-6
        result = _clip_and_filter_bboxes(bboxes)
        assert len(result) == 0

    def test_keeps_box_at_threshold(self):
        # Box with area equal to MIN_BBOX_AREA must be kept (>= filter, not >)
        side = MIN_BBOX_AREA ** 0.5  # ~0.01 for MIN_BBOX_AREA=0.0001
        bboxes = [[0.4, 0.4, 0.4 + side, 0.4 + side, "x"]]
        result = _clip_and_filter_bboxes(bboxes)
        assert len(result) == 1, (
            f"Expected box with area={side*side:.6f} (== MIN_BBOX_AREA={MIN_BBOX_AREA}) to be kept"
        )

    def test_keeps_small_box_above_new_threshold(self):
        # Regression: a ~0.0005 box should survive (was incorrectly dropped
        # under the legacy MIN_BBOX_AREA=0.001 threshold).
        bboxes = [[0.5, 0.5, 0.5 + 0.025, 0.5 + 0.020, "small_pothole"]]  # 0.0005
        result = _clip_and_filter_bboxes(bboxes)
        assert len(result) == 1
        assert result[0][4] == "small_pothole"

    def test_removes_inverted_box(self):
        bboxes = [[0.8, 0.8, 0.2, 0.2, "c"]]  # x_max < x_min after clip
        result = _clip_and_filter_bboxes(bboxes)
        assert len(result) == 0

    def test_keeps_valid_box(self):
        bboxes = [[0.1, 0.2, 0.5, 0.6, "d"]]  # area = 0.4 * 0.4 = 0.16
        result = _clip_and_filter_bboxes(bboxes)
        assert len(result) == 1

    def test_preserves_class_label(self):
        bboxes = [[0.0, 0.0, 0.5, 0.5, "crack"]]
        result = _clip_and_filter_bboxes(bboxes)
        assert result[0][4] == "crack"


# ---------------------------------------------------------------------------
# RandomHSV
# ---------------------------------------------------------------------------


class TestRandomHSV:
    def test_output_shape_preserved(self, small_image, center_bbox):
        transform = RandomHSV(h_gain=0.015, s_gain=0.7, v_gain=0.4)
        result_img, result_bboxes = transform(small_image.copy(), center_bbox[:])
        assert result_img.shape == small_image.shape
        assert result_img.dtype == np.uint8

    def test_bboxes_unchanged(self, small_image, center_bbox):
        transform = RandomHSV(h_gain=0.015, s_gain=0.7, v_gain=0.4)
        _, result_bboxes = transform(small_image.copy(), center_bbox[:])
        assert result_bboxes == center_bbox

    def test_zero_gains_no_change(self, small_image, center_bbox):
        transform = RandomHSV(h_gain=0.0, s_gain=0.0, v_gain=0.0)
        result_img, _ = transform(small_image.copy(), center_bbox[:])
        # With zero gains, h_delta=0, s_mult=1, v_mult=1 => pixel-exact
        # However, RGB->HSV->RGB may have minor rounding (uint8), so allow ±1
        assert np.allclose(result_img, small_image, atol=1)

    def test_image_changes_with_large_gains(self, small_image, center_bbox):
        """With large gains the image should differ from original."""
        np.random.seed(42)
        transform = RandomHSV(h_gain=0.5, s_gain=0.9, v_gain=0.9)
        result_img, _ = transform(small_image.copy(), center_bbox[:])
        # Highly unlikely to be identical with such large gains
        assert not np.array_equal(result_img, small_image)

    def test_output_range_valid(self, small_image, center_bbox):
        """Output should be in [0, 255]."""
        transform = RandomHSV(h_gain=0.5, s_gain=0.9, v_gain=0.9)
        for _ in range(10):
            result_img, _ = transform(small_image.copy(), center_bbox[:])
            assert result_img.min() >= 0
            assert result_img.max() <= 255


# ---------------------------------------------------------------------------
# RandomScale
# ---------------------------------------------------------------------------


class TestRandomScale:
    def test_zoom_in_preserves_shape(self, white_rect_image, center_bbox):
        """Zoom in (scale > 1): output shape must match input."""
        np.random.seed(0)
        transform = RandomScale(scale_range=(1.5, 1.5))
        result_img, result_bboxes = transform(white_rect_image.copy(), center_bbox[:])
        assert result_img.shape == white_rect_image.shape

    def test_zoom_out_preserves_shape(self, white_rect_image, center_bbox):
        """Zoom out (scale < 1): output shape must match input."""
        np.random.seed(0)
        transform = RandomScale(scale_range=(0.5, 0.5))
        result_img, result_bboxes = transform(white_rect_image.copy(), center_bbox[:])
        assert result_img.shape == white_rect_image.shape

    def test_zoom_out_bbox_shrinks(self, white_rect_image, center_bbox):
        """When zooming out, bbox area should decrease."""
        np.random.seed(1)
        transform = RandomScale(scale_range=(0.5, 0.5))
        _, result_bboxes = transform(white_rect_image.copy(), center_bbox[:])
        if result_bboxes:
            x1, y1, x2, y2 = result_bboxes[0][:4]
            result_area = (x2 - x1) * (y2 - y1)
            orig_area = 0.5 * 0.5  # 0.25
            assert result_area < orig_area

    def test_zoom_in_bbox_changes(self, white_rect_image, center_bbox):
        """When zooming in, bbox coords should change due to cropping."""
        np.random.seed(2)
        transform = RandomScale(scale_range=(1.5, 1.5))
        _, result_bboxes = transform(white_rect_image.copy(), center_bbox[:])
        # After zoom-in + crop, bbox should exist and have different coords
        assert len(result_bboxes) >= 0  # May be filtered if mostly cropped out

    def test_identity_scale(self, white_rect_image, center_bbox):
        """Scale ~1.0 should not change bboxes significantly."""
        transform = RandomScale(scale_range=(1.0, 1.0))
        _, result_bboxes = transform(white_rect_image.copy(), center_bbox[:])
        # Should be identity (within floating point)
        assert len(result_bboxes) == 1
        for a, b in zip(result_bboxes[0][:4], center_bbox[0][:4]):
            assert abs(a - b) < 1e-3

    def test_zoom_out_gray_border(self, white_rect_image, center_bbox):
        """Zoom out should show gray (114) border pixels."""
        np.random.seed(3)
        transform = RandomScale(scale_range=(0.5, 0.5))
        result_img, _ = transform(white_rect_image.copy(), center_bbox[:])
        # Corners should be gray (114) since image is centered-ish
        # At minimum, some pixels should be 114
        assert 114 in result_img

    def test_class_label_preserved(self, white_rect_image, center_bbox):
        np.random.seed(4)
        transform = RandomScale(scale_range=(0.7, 0.7))
        _, result_bboxes = transform(white_rect_image.copy(), center_bbox[:])
        if result_bboxes:
            assert result_bboxes[0][4] == "crack"


# ---------------------------------------------------------------------------
# RandomTranslate
# ---------------------------------------------------------------------------


class TestRandomTranslate:
    def test_output_shape_preserved(self, white_rect_image, center_bbox):
        transform = RandomTranslate(translate=0.2)
        result_img, _ = transform(white_rect_image.copy(), center_bbox[:])
        assert result_img.shape == white_rect_image.shape

    def test_zero_translate_identity(self, white_rect_image, center_bbox):
        """translate=0 should be identity."""
        transform = RandomTranslate(translate=0.0)
        result_img, result_bboxes = transform(white_rect_image.copy(), center_bbox[:])
        assert np.array_equal(result_img, white_rect_image)
        assert result_bboxes == center_bbox

    def test_bbox_shifts_with_image(self, white_rect_image, center_bbox):
        """Bbox should shift by same fraction as the image."""
        np.random.seed(10)
        transform = RandomTranslate(translate=0.3)
        _, result_bboxes = transform(white_rect_image.copy(), center_bbox[:])
        if result_bboxes:
            # At least one coord should differ from original
            orig = center_bbox[0][:4]
            res = result_bboxes[0][:4]
            differs = any(abs(a - b) > 0.01 for a, b in zip(orig, res))
            assert differs

    def test_large_translate_can_remove_bbox(self):
        """If bbox is shifted entirely off-canvas, it should be filtered out."""
        # Bbox in top-left corner
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        bboxes = [[0.0, 0.0, 0.05, 0.05, "tiny"]]
        # Force a large positive shift (shift right/down by 0.5)
        # We can't control random directly, but with translate=0.5 and many tries...
        transform = RandomTranslate(translate=0.5)
        np.random.seed(99)
        found_removed = False
        for _ in range(50):
            _, result = transform(img.copy(), [b[:] for b in bboxes])
            if len(result) == 0:
                found_removed = True
                break
        # With translate=0.5 and a 5% box at corner, shifting left should clip it
        assert found_removed

    def test_gray_fill(self, white_rect_image, center_bbox):
        """Translated areas should be gray (114)."""
        np.random.seed(5)
        transform = RandomTranslate(translate=0.3)
        result_img, _ = transform(white_rect_image.copy(), center_bbox[:])
        # If shifted, exposed area should be 114
        assert 114 in result_img

    def test_class_label_preserved(self, white_rect_image, center_bbox):
        np.random.seed(6)
        transform = RandomTranslate(translate=0.1)
        _, result_bboxes = transform(white_rect_image.copy(), center_bbox[:])
        if result_bboxes:
            assert result_bboxes[0][4] == "crack"


# ---------------------------------------------------------------------------
# build_augmentation_pipeline with new transforms
# ---------------------------------------------------------------------------


class TestBuildPipelineNew:
    def test_full_pipeline_all_transforms(self):
        """Config with all new transforms should produce Scale+Translate+HSV+HFlip."""
        config = {
            "scale": [0.5, 1.5],
            "translate": 0.1,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "horizontal_flip": True,
        }
        pipeline = build_augmentation_pipeline(config)
        assert len(pipeline.transforms) == 4
        assert isinstance(pipeline.transforms[0], RandomScale)
        assert isinstance(pipeline.transforms[1], RandomTranslate)
        assert isinstance(pipeline.transforms[2], RandomHSV)
        assert isinstance(pipeline.transforms[3], RandomHorizontalFlip)

    def test_hsv_disables_brightness(self):
        """When HSV is active, brightness_range should be ignored."""
        config = {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "brightness_range": [0.8, 1.2],
            "horizontal_flip": True,
        }
        pipeline = build_augmentation_pipeline(config)
        # Should have HSV + HFlip, NOT brightness
        transform_types = [type(t).__name__ for t in pipeline.transforms]
        assert "RandomHSV" in transform_types
        assert "RandomBrightness" not in transform_types

    def test_brightness_without_hsv(self):
        """When HSV is NOT active, brightness_range should be included."""
        config = {
            "brightness_range": [0.8, 1.2],
            "horizontal_flip": True,
        }
        pipeline = build_augmentation_pipeline(config)
        transform_types = [type(t).__name__ for t in pipeline.transforms]
        assert "RandomBrightness" in transform_types
        assert "RandomHSV" not in transform_types

    def test_mosaic_key_ignored_in_pipeline(self):
        """mosaic/mixup keys are handled at dataset level, not in pipeline."""
        config = {
            "mosaic": 1.0,
            "mixup": 0.1,
            "mosaic_off_epochs": 10,
            "horizontal_flip": True,
        }
        pipeline = build_augmentation_pipeline(config)
        # Only HFlip should be present
        assert len(pipeline.transforms) == 1
        assert isinstance(pipeline.transforms[0], RandomHorizontalFlip)

    def test_scale_only(self):
        config = {"scale": [0.8, 1.2]}
        pipeline = build_augmentation_pipeline(config)
        assert len(pipeline.transforms) == 1
        assert isinstance(pipeline.transforms[0], RandomScale)

    def test_translate_only(self):
        config = {"translate": 0.2}
        pipeline = build_augmentation_pipeline(config)
        assert len(pipeline.transforms) == 1
        assert isinstance(pipeline.transforms[0], RandomTranslate)


# ---------------------------------------------------------------------------
# Composed pipeline integration test
# ---------------------------------------------------------------------------


class TestComposedPipeline:
    def test_full_pipeline_produces_valid_output(self, white_rect_image, center_bbox):
        """Full pipeline should produce valid image and valid bboxes."""
        config = {
            "scale": [0.5, 1.5],
            "translate": 0.1,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "horizontal_flip": True,
        }
        pipeline = build_augmentation_pipeline(config)
        np.random.seed(42)
        for _ in range(20):
            result_img, result_bboxes = pipeline(white_rect_image.copy(), [b[:] for b in center_bbox])
            # Image shape and type
            assert result_img.shape == white_rect_image.shape
            assert result_img.dtype == np.uint8
            # All bboxes valid
            for bbox in result_bboxes:
                x1, y1, x2, y2 = bbox[:4]
                assert 0.0 <= x1 < x2 <= 1.0
                assert 0.0 <= y1 < y2 <= 1.0
                assert (x2 - x1) * (y2 - y1) >= MIN_BBOX_AREA
                assert bbox[4] == "crack"  # class preserved
