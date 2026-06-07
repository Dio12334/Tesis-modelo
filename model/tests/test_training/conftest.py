"""Shared fixtures for ``model/tests/test_training/`` alignment tests.

Centralises the synthetic ``white-rect-on-black`` pattern used by the
augmentation/alignment suites. A small black image with a single white
rectangle has the property that its bounding box is *exactly* the bright
region: any transform that breaks bbox alignment will leave the box covering
mostly-black pixels (or leave bright pixels uncovered), which is trivially
detectable via mean intensity.

Fixture API:

    white_rect_image_with_bbox(
        size: int = 256,
        rect_norm_xyxy: tuple = (0.25, 0.25, 0.75, 0.75),
    ) -> tuple[np.ndarray, list]

Returns ``(image_uint8_HxWx3, [[x_min, y_min, x_max, y_max, class_label]])``,
matching the ``(image, bboxes)`` contract of every transform in
``model.training.augmentation``.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest


def _build_white_rect(
    size: int, rect_norm_xyxy: Tuple[float, float, float, float], class_label: str
) -> Tuple[np.ndarray, List[List]]:
    if not (0.0 <= rect_norm_xyxy[0] < rect_norm_xyxy[2] <= 1.0):
        raise ValueError(
            f"Invalid rect x range: {rect_norm_xyxy[0]}..{rect_norm_xyxy[2]}"
        )
    if not (0.0 <= rect_norm_xyxy[1] < rect_norm_xyxy[3] <= 1.0):
        raise ValueError(
            f"Invalid rect y range: {rect_norm_xyxy[1]}..{rect_norm_xyxy[3]}"
        )

    img = np.zeros((size, size, 3), dtype=np.uint8)
    x_min_px = int(round(rect_norm_xyxy[0] * size))
    y_min_px = int(round(rect_norm_xyxy[1] * size))
    x_max_px = int(round(rect_norm_xyxy[2] * size))
    y_max_px = int(round(rect_norm_xyxy[3] * size))
    img[y_min_px:y_max_px, x_min_px:x_max_px] = 255

    bbox = list(rect_norm_xyxy) + [class_label]
    return img, [bbox]


@pytest.fixture
def white_rect_image_with_bbox():
    """Factory fixture producing a black image with a centered white rectangle.

    Usage:
        def test_foo(white_rect_image_with_bbox):
            img, bboxes = white_rect_image_with_bbox(size=320)
            # img is 320x320x3 uint8; bboxes is one [0.25, 0.25, 0.75, 0.75, 'crack']

        def test_bar(white_rect_image_with_bbox):
            img, bboxes = white_rect_image_with_bbox(
                size=128, rect_norm_xyxy=(0.1, 0.1, 0.5, 0.5)
            )
    """

    def _factory(
        size: int = 256,
        rect_norm_xyxy: Tuple[float, float, float, float] = (0.25, 0.25, 0.75, 0.75),
        class_label: str = "crack",
    ) -> Tuple[np.ndarray, List[List]]:
        return _build_white_rect(size, rect_norm_xyxy, class_label)

    return _factory


@pytest.fixture
def assert_bbox_aligned_with_brightness():
    """Helper fixture: assert ``bbox`` covers bright pixels and excludes them outside.

    Uses *bright-pixel coverage* rather than mean intensity so the assertion
    is robust to grey padding introduced by zoom-out / translate (canvas
    fill = 114), which would otherwise dominate the outside-region mean.

    Specifically, with bright = pixels >= ``bright_threshold`` (default 220):

      * fraction-of-inside-pixels-that-are-bright >= ``inside_bright_frac_min``
      * fraction-of-outside-pixels-that-are-bright <= ``outside_bright_frac_max``

    Skips assertion (returns gracefully) if the bbox area fell to zero, since
    that case is handled elsewhere (a transform may legitimately drop the
    box if the white rect was scrolled off-screen).
    """

    def _check(
        image: np.ndarray,
        bbox_xyxy_norm: Tuple[float, float, float, float],
        inside_bright_frac_min: float = 0.6,
        outside_bright_frac_max: float = 0.02,
        bright_threshold: int = 220,
        require_min_area_norm: float = 0.001,
    ) -> None:
        h, w = image.shape[:2]
        x1 = max(0, min(w - 1, int(round(bbox_xyxy_norm[0] * w))))
        y1 = max(0, min(h - 1, int(round(bbox_xyxy_norm[1] * h))))
        x2 = max(0, min(w, int(round(bbox_xyxy_norm[2] * w))))
        y2 = max(0, min(h, int(round(bbox_xyxy_norm[3] * h))))

        # Skip if box collapsed to zero area (e.g., transform pushed rect out)
        if x2 <= x1 or y2 <= y1:
            return
        norm_area = ((x2 - x1) / w) * ((y2 - y1) / h)
        if norm_area < require_min_area_norm:
            return

        # Single-channel intensity proxy (max across colour channels)
        intensity = image.max(axis=2) if image.ndim == 3 else image
        bright = intensity >= bright_threshold

        crop_bright = bright[y1:y2, x1:x2]
        inside_frac = float(crop_bright.mean()) if crop_bright.size > 0 else 0.0

        outside_mask = np.ones(image.shape[:2], dtype=bool)
        outside_mask[y1:y2, x1:x2] = False
        if outside_mask.any():
            outside_frac = float(bright[outside_mask].mean())
        else:
            outside_frac = 0.0

        assert inside_frac >= inside_bright_frac_min, (
            f"Bbox does not enclose enough bright pixels: "
            f"inside_bright_frac={inside_frac:.3f} "
            f"(expected >= {inside_bright_frac_min}). "
            f"bbox(norm)={bbox_xyxy_norm} bbox(px)=({x1},{y1},{x2},{y2})"
        )
        assert outside_frac <= outside_bright_frac_max, (
            f"Bright pixels leak outside bbox: "
            f"outside_bright_frac={outside_frac:.3f} "
            f"(expected <= {outside_bright_frac_max}). "
            f"bbox(norm)={bbox_xyxy_norm} bbox(px)=({x1},{y1},{x2},{y2})"
        )

    return _check
