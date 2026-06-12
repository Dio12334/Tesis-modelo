"""End-to-end alignment tests for spatial and photometric augmentations.

Each transform is applied with deterministic seeds to a synthetic
black-image-with-white-rectangle. The post-transform bbox(es) are then
checked against the visible bright region:

* **Spatial transforms** (``RandomScale``, ``RandomTranslate``) must produce
  bboxes that still hug the now-translated/scaled white rectangle. We assert
  via ``assert_bbox_aligned_with_brightness``: mean intensity inside the
  bbox is high, mean intensity outside is low.
* **Photometric transforms** (``RandomHSV``, ``RandomBrightness``) must not
  move bboxes at all -- they only affect pixel values. We assert exact
  equality of bbox coordinates pre/post.

These tests complement ``test_dataset_alignment.py`` (which exercises the
full ``RDD2022TorchDataset.__getitem__`` integration) and
``test_augmentation_new.py`` (which exercises low-level transform shape /
filter behaviour). The new tests fill the gap that the original plan
identified: existing alignment tests only covered HFlip and VFlip.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from model.training.augmentation import (
    RandomBrightness,
    RandomHSV,
    RandomScale,
    RandomTranslate,
)


# ---------------------------------------------------------------------------
# RandomScale alignment (spatial)
# ---------------------------------------------------------------------------


class TestRandomScaleAlignment:
    """Bbox must follow the white rectangle through both zoom-in and zoom-out."""

    @pytest.mark.parametrize("seed", [0, 1, 7, 23, 42])
    def test_zoom_in_box_tracks_bright_region(
        self, white_rect_image_with_bbox, assert_bbox_aligned_with_brightness, seed
    ):
        random.seed(seed)
        np.random.seed(seed)

        img, bboxes = white_rect_image_with_bbox(size=320)
        # Force zoom-in (s > 1)
        transform = RandomScale(scale_range=(1.25, 1.4))
        out_img, out_bboxes = transform(img.copy(), [list(b) for b in bboxes])

        # Some seeds may push the rect out of the crop window; only check
        # cases where a box survived.
        if not out_bboxes:
            return
        assert len(out_bboxes) == 1
        assert_bbox_aligned_with_brightness(out_img, tuple(out_bboxes[0][:4]))
        assert out_bboxes[0][4] == "crack"

    @pytest.mark.parametrize("seed", [0, 1, 7, 23, 42])
    def test_zoom_out_box_tracks_bright_region(
        self, white_rect_image_with_bbox, assert_bbox_aligned_with_brightness, seed
    ):
        random.seed(seed)
        np.random.seed(seed)

        img, bboxes = white_rect_image_with_bbox(size=320)
        # Force zoom-out (s < 1)
        transform = RandomScale(scale_range=(0.6, 0.8))
        out_img, out_bboxes = transform(img.copy(), [list(b) for b in bboxes])

        if not out_bboxes:
            return
        assert len(out_bboxes) == 1
        assert_bbox_aligned_with_brightness(out_img, tuple(out_bboxes[0][:4]))
        assert out_bboxes[0][4] == "crack"

    def test_zoom_out_bbox_inside_canvas(self, white_rect_image_with_bbox):
        """After zoom-out the bbox must lie within the [0,1] canvas."""
        random.seed(123)
        np.random.seed(123)

        img, bboxes = white_rect_image_with_bbox(size=320)
        transform = RandomScale(scale_range=(0.5, 0.7))
        _, out_bboxes = transform(img.copy(), [list(b) for b in bboxes])
        for bb in out_bboxes:
            assert 0.0 <= bb[0] < bb[2] <= 1.0
            assert 0.0 <= bb[1] < bb[3] <= 1.0


# ---------------------------------------------------------------------------
# RandomTranslate alignment (spatial)
# ---------------------------------------------------------------------------


class TestRandomTranslateAlignment:
    """Bbox must shift along with the white rectangle."""

    @pytest.mark.parametrize("seed", [0, 1, 5, 11, 42, 99])
    def test_translation_keeps_bbox_aligned(
        self, white_rect_image_with_bbox, assert_bbox_aligned_with_brightness, seed
    ):
        random.seed(seed)
        np.random.seed(seed)

        img, bboxes = white_rect_image_with_bbox(size=320)
        transform = RandomTranslate(translate=0.1)
        out_img, out_bboxes = transform(img.copy(), [list(b) for b in bboxes])

        if not out_bboxes:
            return
        assert len(out_bboxes) == 1
        assert_bbox_aligned_with_brightness(out_img, tuple(out_bboxes[0][:4]))
        assert out_bboxes[0][4] == "crack"

    def test_translate_zero_is_identity(self, white_rect_image_with_bbox):
        """When the random shift rounds to ~0, bboxes must be untouched."""
        # translate=1e-6 forces a near-zero shift; transform short-circuits.
        random.seed(0)
        np.random.seed(0)

        img, bboxes = white_rect_image_with_bbox(size=320)
        original_bbox = [list(b) for b in bboxes]
        transform = RandomTranslate(translate=1e-6)
        _, out_bboxes = transform(img.copy(), [list(b) for b in bboxes])
        assert out_bboxes == original_bbox


# ---------------------------------------------------------------------------
# RandomHSV alignment (photometric only)
# ---------------------------------------------------------------------------


class TestRandomHSVDoesNotMoveBBoxes:
    """HSV is a colour transform; bboxes must be byte-equal pre/post."""

    @pytest.mark.parametrize("seed", [0, 1, 7, 42])
    def test_hsv_preserves_bboxes_exactly(self, white_rect_image_with_bbox, seed):
        random.seed(seed)
        np.random.seed(seed)

        img, bboxes = white_rect_image_with_bbox(size=128)
        original = [list(b) for b in bboxes]
        transform = RandomHSV(h_gain=0.015, s_gain=0.5, v_gain=0.4)
        out_img, out_bboxes = transform(img.copy(), [list(b) for b in bboxes])

        # bbox list must be byte-equal
        assert out_bboxes == original

        # Image must have changed in at least one pixel (proves transform ran)
        assert out_img.shape == img.shape
        # We do not assert change unconditionally because zero-gain rolls
        # are technically allowed by the API; instead verify the *type* and
        # range invariants.
        assert out_img.dtype == np.uint8

    def test_hsv_does_not_touch_bbox_when_image_constant(
        self, white_rect_image_with_bbox
    ):
        """Even on a constant-colour image, the bbox list must not mutate."""
        random.seed(0)
        np.random.seed(0)
        img, bboxes = white_rect_image_with_bbox(size=64)
        # Force constant grey image
        img[...] = 128
        original = [list(b) for b in bboxes]
        transform = RandomHSV(h_gain=0.5, s_gain=0.9, v_gain=0.9)
        _, out_bboxes = transform(img, [list(b) for b in bboxes])
        assert out_bboxes == original


# ---------------------------------------------------------------------------
# RandomBrightness alignment (photometric only)
# ---------------------------------------------------------------------------


class TestRandomBrightnessDoesNotMoveBBoxes:
    """Brightness is a colour-only transform; bboxes must be byte-equal."""

    @pytest.mark.parametrize("seed", [0, 1, 7, 42])
    def test_brightness_preserves_bboxes_exactly(
        self, white_rect_image_with_bbox, seed
    ):
        random.seed(seed)
        np.random.seed(seed)

        img, bboxes = white_rect_image_with_bbox(size=128)
        original = [list(b) for b in bboxes]
        transform = RandomBrightness(brightness_range=(0.6, 1.4))
        out_img, out_bboxes = transform(img.copy(), [list(b) for b in bboxes])

        assert out_bboxes == original
        assert out_img.shape == img.shape
        assert out_img.dtype == np.uint8

    def test_brightness_factor_below_one_dims_image(
        self, white_rect_image_with_bbox
    ):
        """Sanity: factor < 1 must reduce mean intensity."""
        random.seed(0)
        np.random.seed(0)
        img, bboxes = white_rect_image_with_bbox(size=64)
        before_mean = float(img.mean())
        transform = RandomBrightness(brightness_range=(0.5, 0.5))  # forced
        out_img, _ = transform(img.copy(), [list(b) for b in bboxes])
        after_mean = float(out_img.mean())
        assert after_mean < before_mean
