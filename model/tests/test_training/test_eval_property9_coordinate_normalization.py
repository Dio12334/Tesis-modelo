"""Property-based tests for coordinate normalization.

Feature: generic-evaluation-script
Property 9: Coordinates entering metrics are normalized and non-degenerate

For any set of detector output boxes (with coordinates in pixel scale,
normalized scale, or out of range) and any positive inference input size, after
the normalization stage every coordinate of every box stored in predictions and
ground truths lies in the closed interval ``[0.0, 1.0]`` (pixel boxes are
divided by the input size, out-of-range values are clamped), and no box with
``x_min > x_max`` or ``y_min > y_max`` survives into the metrics input.

The normalization stage is the composition the inference loop applies to every
box: ``normalize_box`` (pixel-vs-normalized scale conversion) followed by
``clamp_and_filter`` (clamp into ``[0, 1]`` and drop degenerate boxes). These
tests exercise the real functions in ``model/training/evaluate_detection.py``.

Three coordinate regimes are generated to cover Req 6.1/6.2/6.4:

* ``normalized`` -- every coordinate already in ``[0, 1]`` (pass-through);
* ``pixel`` -- every coordinate ``> 1.0`` (divided by ``input_size``);
* ``out_of_range`` -- arbitrary coordinates including negatives and values far
  above ``input_size`` (must be clamped, possibly turned degenerate).

**Validates: Requirements 6.1, 6.2, 6.4, 6.5**
"""

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from model.training.evaluate_detection import clamp_and_filter, normalize_box


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Finite coordinate in [0, 1]: already normalized (Req 6.2 pass-through).
_NORMALIZED_COORD = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Finite coordinate strictly above 1.0: pixel scale (Req 6.1 -> divide).
_PIXEL_COORD = st.floats(
    min_value=1.0,
    max_value=1.0e4,
    allow_nan=False,
    allow_infinity=False,
    exclude_min=True,
)

# Arbitrary finite coordinate, including negatives and large values, exercising
# clamping (Req 6.4) and potential clamp-induced degeneracy (Req 6.5).
_ANY_COORD = st.floats(
    min_value=-1.0e4, max_value=1.0e4, allow_nan=False, allow_infinity=False
)

# Positive inference input size (image side length in pixels).
_INPUT_SIZE = st.integers(min_value=1, max_value=4096)

# Image identifiers used in warnings; kept simple and printable.
_IMAGE_IDS = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-/.", min_size=1, max_size=40
)


def _box_from(coord_strategy):
    """Build a four-coordinate box strategy from a single-coordinate strategy."""
    return st.lists(coord_strategy, min_size=4, max_size=4)


@st.composite
def _boxes(draw):
    """Draw a list of boxes spanning all three coordinate regimes.

    Each box independently picks one of the ``normalized`` / ``pixel`` /
    ``out_of_range`` regimes, so a single example mixes scales the way a real
    detector batch might.
    """
    regime = st.sampled_from(["normalized", "pixel", "out_of_range"])

    def _one_box():
        kind = draw(regime)
        if kind == "normalized":
            return draw(_box_from(_NORMALIZED_COORD))
        if kind == "pixel":
            return draw(_box_from(_PIXEL_COORD))
        return draw(_box_from(_ANY_COORD))

    count = draw(st.integers(min_value=0, max_value=10))
    return [_one_box() for _ in range(count)]


# ---------------------------------------------------------------------------
# Property 9
# ---------------------------------------------------------------------------


class TestProperty9CoordinateNormalization:
    """Property 9: Coordinates entering metrics are normalized and non-degenerate.

    **Validates: Requirements 6.1, 6.2, 6.4, 6.5**
    """

    @given(boxes=_boxes(), input_size=_INPUT_SIZE, image_id=_IMAGE_IDS)
    @settings(max_examples=100)
    def test_surviving_boxes_normalized_and_non_degenerate(
        self, boxes, input_size, image_id
    ):
        # Feature: generic-evaluation-script, Property 9: Coordinate normalization
        """Every surviving box has coords in [0, 1] and is non-degenerate.

        The normalization stage is ``normalize_box`` then ``clamp_and_filter``.
        A box that survives (``clamp_and_filter`` returns a list) must have all
        four coordinates within ``[0.0, 1.0]`` (Req 6.1/6.2/6.4) and satisfy
        ``x_min <= x_max`` and ``y_min <= y_max`` (Req 6.5). A box that is
        excluded yields ``None``.

        **Validates: Requirements 6.1, 6.2, 6.4, 6.5**
        """
        for box in boxes:
            normalized, mode = normalize_box(box, input_size)

            # normalize_box reports the regime it detected (Req 6.1/6.2).
            assert mode in ("pixel", "normalized")
            assert len(normalized) == 4
            # Scale conversion never introduces non-finite values.
            assert all(math.isfinite(c) for c in normalized)

            result = clamp_and_filter(normalized, image_id)

            if result is None:
                # Degenerate box excluded from metrics (Req 6.5). Nothing more
                # to assert: it does not survive into the metrics input.
                continue

            # Surviving box: all coordinates clamped into [0, 1] (Req 6.4).
            assert len(result) == 4
            for coord in result:
                assert 0.0 <= coord <= 1.0

            # Surviving box is non-degenerate (Req 6.5).
            x_min, y_min, x_max, y_max = result
            assert x_min <= x_max
            assert y_min <= y_max

    @given(box=_box_from(_PIXEL_COORD), input_size=_INPUT_SIZE)
    @settings(max_examples=100)
    def test_pixel_boxes_are_divided_by_input_size(self, box, input_size):
        # Feature: generic-evaluation-script, Property 9: Coordinate normalization
        """A box with any coordinate > 1.0 is treated as pixel scale (Req 6.1).

        Every coordinate is divided by ``input_size`` and the reported mode is
        ``"pixel"``.

        **Validates: Requirements 6.1**
        """
        normalized, mode = normalize_box(box, input_size)

        assert mode == "pixel"
        for original, converted in zip(box, normalized):
            assert converted == original / input_size

    @given(box=_box_from(_NORMALIZED_COORD), input_size=_INPUT_SIZE)
    @settings(max_examples=100)
    def test_normalized_boxes_pass_through_unchanged(self, box, input_size):
        # Feature: generic-evaluation-script, Property 9: Coordinate normalization
        """A box whose coords are all in [0, 1] is passed through (Req 6.2).

        The reported mode is ``"normalized"`` and the coordinates are unchanged.

        **Validates: Requirements 6.2**
        """
        normalized, mode = normalize_box(box, input_size)

        assert mode == "normalized"
        assert normalized == list(box)


# ---------------------------------------------------------------------------
# Example-based unit tests complementing Property 9
# ---------------------------------------------------------------------------


class TestCoordinateNormalizationExamples:
    """Concrete examples complementing Property 9.

    **Validates: Requirements 6.1, 6.2, 6.4, 6.5**
    """

    def test_pixel_box_divided_then_within_range(self):
        """A 640-scale pixel box divided by 640 lands in [0, 1]. (Req 6.1)"""
        normalized, mode = normalize_box([64.0, 128.0, 320.0, 640.0], 640)
        assert mode == "pixel"
        result = clamp_and_filter(normalized, "img-1")
        assert result == [0.1, 0.2, 0.5, 1.0]
        for coord in result:
            assert 0.0 <= coord <= 1.0

    def test_normalized_box_passes_through(self):
        """An already-normalized box is unchanged and survives. (Req 6.2)"""
        normalized, mode = normalize_box([0.1, 0.2, 0.3, 0.4], 640)
        assert mode == "normalized"
        assert normalized == [0.1, 0.2, 0.3, 0.4]
        assert clamp_and_filter(normalized, "img-2") == [0.1, 0.2, 0.3, 0.4]

    def test_out_of_range_negative_is_clamped(self):
        """A negative coordinate is clamped to 0.0. (Req 6.4)"""
        result = clamp_and_filter([-0.5, 0.2, 0.3, 0.4], "img-3")
        assert result == [0.0, 0.2, 0.3, 0.4]
        for coord in result:
            assert 0.0 <= coord <= 1.0

    def test_above_one_after_pixel_division_is_clamped(self):
        """A coordinate exceeding input_size stays > 1 after division, then clamps.

        ``[320, 0, 1280, 320]`` with input_size 640 is pixel mode, so every
        coordinate is divided by 640 -> ``[0.5, 0.0, 2.0, 0.5]``. The ``x_max``
        of ``2.0`` is then clamped to ``1.0`` (Req 6.4), leaving a
        non-degenerate box.
        """
        normalized, mode = normalize_box([320.0, 0.0, 1280.0, 320.0], 640)
        assert mode == "pixel"
        result = clamp_and_filter(normalized, "img-4")
        assert result == [0.5, 0.0, 1.0, 0.5]

    def test_degenerate_box_excluded(self):
        """A box with x_min > x_max is dropped (returns None). (Req 6.5)"""
        assert clamp_and_filter([0.8, 0.2, 0.3, 0.4], "img-5") is None

    def test_degenerate_y_box_excluded(self):
        """A box with y_min > y_max is dropped (returns None). (Req 6.5)"""
        assert clamp_and_filter([0.1, 0.9, 0.5, 0.4], "img-6") is None

    def test_clamp_induced_degeneracy_excluded(self):
        """A box that becomes degenerate only after clamping is excluded. (Req 6.5)

        ``x_min = 1.5`` clamps to ``1.0`` and ``x_max = 0.5`` stays, so the
        clamped box has ``x_min (1.0) > x_max (0.5)`` and is dropped.
        """
        assert clamp_and_filter([1.5, 0.2, 0.5, 0.4], "img-7") is None
