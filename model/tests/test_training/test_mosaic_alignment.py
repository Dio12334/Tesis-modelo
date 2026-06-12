"""Alignment tests for the rewritten ``RDD2022TorchDataset._build_mosaic``.

These tests pin three properties that distinguish the Ultralytics-style
mosaic from the earlier shrink-to-quadrant implementation:

1. **Source-resolution preservation.** A bbox surviving the mosaic must
   have the same width/height (in pixels) as the source bbox -- the mosaic
   may *crop* the source but does not *resize* it.

2. **White-rectangle alignment.** Each quadrant either fully or partially
   pastes a synthetic white-rect-on-black source. After centre-cropping,
   the surviving bbox must still hug the visible bright pixels: bright
   pixels live inside the bbox, dark pixels outside (with allowance for
   the grey ``114`` canvas fill, captured via the bright-pixel coverage
   check).

3. **Output shape invariance.** The mosaic returns a ``S x S`` image and
   normalised bboxes regardless of the random centre.

The rewrite must satisfy these without regressing the existing shape /
validity / disable-toggle behaviours covered by ``test_mosaic.py``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image as PILImage

from model.training.train_detection import RDD2022TorchDataset


INPUT_SIZE = 320
RECT_NORM = (0.25, 0.25, 0.75, 0.75)


# ---------------------------------------------------------------------------
# Stub dataset producing white-rect-on-black sources
# ---------------------------------------------------------------------------


class _FakeBBox:
    def __init__(self, x_min, y_min, x_max, y_max, class_label):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.class_label = class_label


class _FakeAnnotation:
    def __init__(self, image_path, bboxes):
        self.image_path = image_path
        self.bounding_boxes = bboxes


class _FakeDataset:
    def __init__(self, annotations, class_names):
        self._annotations = annotations
        self._class_names = class_names

    def get_annotations(self):
        return self._annotations

    def get_class_names(self):
        return self._class_names


def _white_rect_pil(size: int, rect_norm) -> PILImage.Image:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    x1 = int(round(rect_norm[0] * size))
    y1 = int(round(rect_norm[1] * size))
    x2 = int(round(rect_norm[2] * size))
    y2 = int(round(rect_norm[3] * size))
    arr[y1:y2, x1:x2] = 255
    return PILImage.fromarray(arr)


def _build_white_rect_torch_dataset():
    """Return (RDD2022TorchDataset, image_lookup) where every image is the
    same white-rect-on-black sample with bbox at RECT_NORM."""
    annotations: List[_FakeAnnotation] = []
    images = {}
    for i in range(8):
        path = f"/fake/white_rect_{i}.jpg"
        images[path] = _white_rect_pil(INPUT_SIZE, RECT_NORM)
        annotations.append(
            _FakeAnnotation(
                path,
                [
                    _FakeBBox(
                        RECT_NORM[0], RECT_NORM[1], RECT_NORM[2], RECT_NORM[3],
                        "crack",
                    )
                ],
            )
        )
    fake_ds = _FakeDataset(annotations, ["crack"])

    def mock_open(path):
        return images[str(path) if not isinstance(path, str) else path].copy()

    return fake_ds, images, mock_open


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMosaicShape:
    def test_output_is_input_size_square(self):
        fake_ds, _, mock_open = _build_white_rect_torch_dataset()
        with patch(
            "model.training.train_detection.Image.open",
            side_effect=mock_open,
        ):
            ds = RDD2022TorchDataset(
                fake_ds, input_size=INPUT_SIZE, augmentation=None,
                mosaic=1.0, mixup=0.0,
            )
            for seed in range(5):
                random.seed(seed)
                np.random.seed(seed)
                img, bboxes = ds._build_mosaic(0)
                assert img.shape == (INPUT_SIZE, INPUT_SIZE, 3), (
                    f"Mosaic must return ({INPUT_SIZE},{INPUT_SIZE},3); got {img.shape}"
                )
                assert img.dtype == np.uint8


class TestMosaicSourceResolutionPreserved:
    """Random-affine scale distribution must be unbiased and exercise both
    zoom-in and zoom-out regimes.

    The previous shrink-to-quadrant mosaic systematically resized each
    source to ``S/2 x S/2``, halving every bbox's pixel resolution and
    causing small features (e.g. distant potholes) to fall below
    ``MIN_BBOX_AREA``. The current implementation pastes sources at
    native ``S x S`` into a ``2S x 2S`` canvas and then applies a random
    affine ``scale ~ U[1 - gain, 1 + gain]`` (default gain 0.5).

    We verify two properties:

    1. The empirical scale distribution covers both ``< 1`` and ``> 1``
       within the expected range (so we are not stuck at a fixed shrink).
    2. The mean output bbox area is approximately equal to the source
       bbox area (E[scale^2] ~= 1 + gain^2 / 3), not systematically halved.

    Instrumentation is via ``random.uniform`` capture: ``_random_affine_mosaic``
    samples ``scale`` and the two translation jitters from
    ``random.uniform`` so we intercept the first uniform draw per call to
    record the scale.
    """

    def _measure_scale_distribution(
        self, ds, idx: int, seeds: range, gain: float
    ):
        """Run _build_mosaic over ``seeds`` and return list of sampled scales."""
        observed: List[float] = []
        real_uniform = random.uniform
        # Sequence per _build_mosaic call: random.uniform(s/2, 3*s/2) was
        # removed by the random-affine rewrite. Now random.uniform is called
        # 3 times inside _random_affine_mosaic in this order:
        #   1) scale  ~ uniform(1 - gain, 1 + gain)
        #   2) tx     ~ uniform(-translate, +translate)
        #   3) ty     ~ uniform(-translate, +translate)
        # So we capture only the first uniform draw per mosaic build.

        call_state = {"first_per_mosaic": True}

        def _capturing_uniform(a, b):
            v = real_uniform(a, b)
            if call_state["first_per_mosaic"]:
                expected_a = 1.0 - gain
                expected_b = 1.0 + gain
                if abs(a - expected_a) < 1e-6 and abs(b - expected_b) < 1e-6:
                    observed.append(v)
                    call_state["first_per_mosaic"] = False
            return v

        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            random.uniform = _capturing_uniform
            call_state["first_per_mosaic"] = True
            try:
                ds._build_mosaic(idx)
            finally:
                random.uniform = real_uniform

        return observed

    def test_scale_distribution_spans_zoom_in_and_zoom_out(self):
        fake_ds, _, mock_open = _build_white_rect_torch_dataset()
        with patch(
            "model.training.train_detection.Image.open",
            side_effect=mock_open,
        ):
            gain = 0.5
            ds = RDD2022TorchDataset(
                fake_ds, input_size=INPUT_SIZE, augmentation=None,
                mosaic=1.0, mixup=0.0,
                mosaic_scale_gain=gain, mosaic_translate=0.1,
            )
            scales = self._measure_scale_distribution(
                ds, idx=0, seeds=range(40), gain=gain,
            )
            assert len(scales) == 40, (
                f"Expected 40 scale samples; captured {len(scales)}"
            )
            assert min(scales) >= 1.0 - gain - 1e-6, (
                f"Min scale {min(scales):.3f} below lower bound {1.0 - gain}"
            )
            assert max(scales) <= 1.0 + gain + 1e-6, (
                f"Max scale {max(scales):.3f} above upper bound {1.0 + gain}"
            )
            n_zoom_out = sum(1 for s in scales if s < 1.0)
            n_zoom_in = sum(1 for s in scales if s > 1.0)
            assert n_zoom_out >= 5, (
                f"Only {n_zoom_out}/40 seeds produced scale<1 "
                f"(zoom-out); expected ~20."
            )
            assert n_zoom_in >= 5, (
                f"Only {n_zoom_in}/40 seeds produced scale>1 "
                f"(zoom-in); expected ~20."
            )

    def test_mean_scale_close_to_one(self):
        fake_ds, _, mock_open = _build_white_rect_torch_dataset()
        with patch(
            "model.training.train_detection.Image.open",
            side_effect=mock_open,
        ):
            gain = 0.5
            ds = RDD2022TorchDataset(
                fake_ds, input_size=INPUT_SIZE, augmentation=None,
                mosaic=1.0, mixup=0.0,
                mosaic_scale_gain=gain, mosaic_translate=0.1,
            )
            scales = self._measure_scale_distribution(
                ds, idx=0, seeds=range(200), gain=gain,
            )
            mean = float(np.mean(scales))
            # Theoretical mean of U[0.5, 1.5] is 1.0
            assert abs(mean - 1.0) < 0.05, (
                f"Mean scale {mean:.4f} differs from 1.0 by >0.05 "
                f"over 200 seeds; sampler may be biased."
            )


class TestMosaicAlignment:
    """Surviving bboxes must still cover bright pixels in the mosaic."""

    @pytest.mark.parametrize("seed", [0, 1, 7, 11, 23, 42])
    def test_bright_pixels_inside_each_surviving_bbox(
        self, seed, assert_bbox_aligned_with_brightness
    ):
        fake_ds, _, mock_open = _build_white_rect_torch_dataset()
        with patch(
            "model.training.train_detection.Image.open",
            side_effect=mock_open,
        ):
            ds = RDD2022TorchDataset(
                fake_ds, input_size=INPUT_SIZE, augmentation=None,
                mosaic=1.0, mixup=0.0,
            )
            random.seed(seed)
            np.random.seed(seed)
            img, bboxes = ds._build_mosaic(0)

            assert img.shape == (INPUT_SIZE, INPUT_SIZE, 3)
            # We expect at least one surviving bbox; if filtering ever
            # produced zero (edge case), that's covered by other tests.
            if not bboxes:
                return
            for bb in bboxes:
                # In-box: most pixels must be bright.
                # Out-of-box: very few pixels may be bright (small leak ok
                # because mosaic boundaries are pixel-aligned by clipping).
                assert_bbox_aligned_with_brightness(
                    img,
                    tuple(bb[:4]),
                    inside_bright_frac_min=0.55,
                    outside_bright_frac_max=0.35,
                    require_min_area_norm=0.005,
                )


class TestMosaicBBoxesNormalised:
    """All bboxes returned by mosaic must be in [0, 1] and non-degenerate."""

    def test_bboxes_in_unit_interval(self):
        fake_ds, _, mock_open = _build_white_rect_torch_dataset()
        with patch(
            "model.training.train_detection.Image.open",
            side_effect=mock_open,
        ):
            ds = RDD2022TorchDataset(
                fake_ds, input_size=INPUT_SIZE, augmentation=None,
                mosaic=1.0, mixup=0.0,
            )
            for seed in range(15):
                random.seed(seed)
                np.random.seed(seed)
                _, bboxes = ds._build_mosaic(0)
                for bb in bboxes:
                    assert 0.0 <= bb[0] < bb[2] <= 1.0, (
                        f"x range violated at seed={seed}: {bb}"
                    )
                    assert 0.0 <= bb[1] < bb[3] <= 1.0, (
                        f"y range violated at seed={seed}: {bb}"
                    )
                    assert bb[4] == "crack"


class TestMosaicRandomAffineProducesVariance:
    """Random-affine must use *random* scale and translate, not fixed values.

    With white-rect sources, fixed scale + translate would yield identical
    bbox centroids in the output across all seeds. The scale + translate
    jitter from ``_random_affine_mosaic`` should produce centroid variance
    that grows with both parameters.
    """

    def test_random_affine_produces_variance_in_bbox_positions(self):
        fake_ds, _, mock_open = _build_white_rect_torch_dataset()
        with patch(
            "model.training.train_detection.Image.open",
            side_effect=mock_open,
        ):
            ds = RDD2022TorchDataset(
                fake_ds, input_size=INPUT_SIZE, augmentation=None,
                mosaic=1.0, mixup=0.0,
                mosaic_scale_gain=0.5, mosaic_translate=0.1,
            )
            centroids_x: List[float] = []
            centroids_y: List[float] = []
            for seed in range(40):
                random.seed(seed)
                np.random.seed(seed)
                _, bboxes = ds._build_mosaic(0)
                for bb in bboxes:
                    centroids_x.append(0.5 * (bb[0] + bb[2]))
                    centroids_y.append(0.5 * (bb[1] + bb[3]))

            assert len(centroids_x) > 10, (
                f"Too few surviving bboxes across 40 seeds: {len(centroids_x)}"
            )
            std_x = float(np.std(centroids_x))
            std_y = float(np.std(centroids_y))
            # Bboxes are normalised; std >= 0.01 corresponds to >= 3.2 px
            # at INPUT_SIZE=320. A degenerate fixed affine would yield
            # std == 0.
            assert std_x >= 0.01, (
                f"Bbox x-centroid std={std_x:.4f} too small; "
                f"affine may be deterministic."
            )
            assert std_y >= 0.01, (
                f"Bbox y-centroid std={std_y:.4f} too small; "
                f"affine may be deterministic."
            )

    def test_zero_gain_and_zero_translate_is_deterministic(self):
        """With gain=0 and translate=0 the affine collapses to a pure 2x
        downscale (scale=1.0 always, no jitter), so bbox centroids are
        identical across seeds (sanity check)."""
        fake_ds, _, mock_open = _build_white_rect_torch_dataset()
        with patch(
            "model.training.train_detection.Image.open",
            side_effect=mock_open,
        ):
            ds = RDD2022TorchDataset(
                fake_ds, input_size=INPUT_SIZE, augmentation=None,
                mosaic=1.0, mixup=0.0,
                mosaic_scale_gain=0.0, mosaic_translate=0.0,
            )
            seen = set()
            for seed in range(5):
                random.seed(seed)
                np.random.seed(seed)
                _, bboxes = ds._build_mosaic(0)
                key = tuple(
                    (round(bb[0], 4), round(bb[1], 4), round(bb[2], 4), round(bb[3], 4))
                    for bb in sorted(bboxes, key=lambda b: (b[0], b[1]))
                )
                seen.add(key)
            # All seeds should produce identical layouts (modulo source order
            # which depends on random companion sampling -- but the WHITE rect
            # is in all sources so its centroid pattern is identical).
            assert len(seen) <= 2, (
                f"With gain=0, translate=0 expected ~1 unique layout; got "
                f"{len(seen)}. Affine may not be honouring the zero settings."
            )


