"""Property-based tests for inference error accounting and format.

Feature: generic-evaluation-script
Property 11: Error accounting and format

For any evaluation run, every failed image contributes exactly one error string
of the form ``<image_id>: <exception text>`` to the run's error list, and the
assembled report's error count equals the length of the error list, which
equals the number of failed images.

These tests exercise the real ``run_inference`` function in
``model/training/evaluate_detection.py``. A *failed* image is one whose image
cannot be decoded (the load/decode stage raises) or whose forward pass raises
(Req 15.1, 14.2). For each failed image the loop appends exactly one
``"<image_id>: <exception text>"`` entry to the run's error list (Req 15.2), and
the run's error list is what the report's ``errors`` block reports (``count`` ==
``len(items)`` == number of failed images -- Req 15.3).

To make the property independent of GPUs, real checkpoints, and real datasets,
the test drives ``run_inference`` with:

* a conforming **fake split dataset** whose ``get_annotations()`` returns a list
  of fake annotations (one ground-truth box each, unique image ids), and
* a conforming **fake detector** whose ``forward`` is parameterized by a
  per-image outcome pattern, plus a patched ``PIL.Image.open`` that raises for
  *decode-failure* images and returns a real tiny image otherwise.

Four per-image outcomes are generated:

* ``success`` -- image decodes, ``forward`` returns one detection (non-empty
  prediction entry); contributes **no** error;
* ``empty_success`` -- image decodes, ``forward`` returns zero detections (empty
  prediction entry, but a *successful* no-detection result); contributes **no**
  error;
* ``forward_exception`` -- image decodes, ``forward`` raises; a *failed* image
  that contributes exactly one error;
* ``decode_failure`` -- ``Image.open`` raises; a *failed* image that contributes
  exactly one error.

Including ``empty_success`` is deliberate: it guards against confusing "empty
prediction entry" with "error". A no-detection success produces an empty
prediction entry yet must contribute **no** error, so the error count tracks the
number of *failed* images, not the number of *empty* predictions.

**Validates: Requirements 15.2, 15.3**
"""

from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image as PILImage
from hypothesis import given, settings
from hypothesis import strategies as st

from model.training.evaluate_detection import run_inference


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fixed exception messages so the produced "<image_id>: <text>" strings are
# fully predictable. Neither contains the ": " separator, so the first ": " in
# an error string is always the image-id/exception-text boundary.
DECODE_MSG = "could not decode image"
FORWARD_MSG = "forward pass failed"

# Small input size keeps the real Resize+ToTensor transform cheap; coordinate
# scale is irrelevant to error accounting (success boxes are already in [0, 1]).
INPUT_SIZE = 8

# Outcomes that count as a *failed* image (each contributes exactly one error).
_FAILURE_OUTCOMES = ("decode_failure", "forward_exception")

# Outcomes that produce an *empty* prediction entry (failures plus the
# successful no-detection case).
_EMPTY_PRED_OUTCOMES = ("decode_failure", "forward_exception", "empty_success")


# ---------------------------------------------------------------------------
# Conforming fakes
# ---------------------------------------------------------------------------


class _FakeBBox:
    """Minimal ground-truth box (the fields ``_build_ground_truth`` reads)."""

    def __init__(self, x_min, y_min, x_max, y_max, class_label):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.class_label = class_label


class _FakeAnnotation:
    """Minimal annotation (the fields ``run_inference`` reads)."""

    def __init__(self, image_path: Path, bounding_boxes):
        self.image_path = image_path
        self.bounding_boxes = bounding_boxes


class _FakeSplit:
    """Conforming split dataset exposing ``get_annotations()``."""

    def __init__(self, annotations):
        self._annotations = annotations

    def get_annotations(self):
        return self._annotations


class _FakeDetector:
    """Conforming detector whose ``forward`` follows a per-image outcome list.

    ``forward`` is called once per image that passes the load/decode stage, in
    annotation order, so the outcome list omits ``decode_failure`` images (which
    never reach the forward pass).

    * ``forward_exception`` -> raise ``RuntimeError(FORWARD_MSG)``;
    * ``empty_success`` -> return one no-detection output (empty tensors);
    * ``success`` -> return one output with a single valid detection.
    """

    def __init__(self, forward_outcomes):
        self._outcomes = list(forward_outcomes)
        self._i = 0

    def forward(self, image_tensor):
        outcome = self._outcomes[self._i]
        self._i += 1
        if outcome == "forward_exception":
            raise RuntimeError(FORWARD_MSG)
        if outcome == "empty_success":
            return [
                {
                    "boxes": torch.zeros((0, 4)),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                    "scores": torch.zeros((0,)),
                }
            ]
        # success: a single in-range, non-degenerate detection.
        return [
            {
                "boxes": torch.tensor([[0.1, 0.1, 0.5, 0.5]]),
                "labels": torch.tensor([1], dtype=torch.int64),
                "scores": torch.tensor([0.9]),
            }
        ]


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _run(outcomes, input_size: int = INPUT_SIZE):
    """Run ``run_inference`` for a per-image ``outcomes`` pattern.

    Returns ``(predictions, ground_truths, errors, image_ids, outcomes)`` where
    ``image_ids[i]`` is the image id of the ``i``-th annotation.
    """
    annotations = []
    image_ids = []
    for i, _outcome in enumerate(outcomes):
        image_path = Path(f"/fake/images/img_{i}.jpg")
        image_id = str(image_path)
        image_ids.append(image_id)
        bbox = _FakeBBox(0.1, 0.1, 0.5, 0.5, "crack")
        annotations.append(_FakeAnnotation(image_path, [bbox]))

    # Map image id -> outcome so the patched Image.open can decide per image.
    outcome_by_id = dict(zip(image_ids, outcomes))

    # forward() is reached only by images that decode successfully, in order.
    forward_outcomes = [o for o in outcomes if o != "decode_failure"]
    detector = _FakeDetector(forward_outcomes)
    split = _FakeSplit(annotations)

    def _fake_open(path, *args, **kwargs):
        outcome = outcome_by_id[str(path)]
        if outcome == "decode_failure":
            raise ValueError(DECODE_MSG)
        # A real tiny RGB image so the real transform pipeline works.
        return PILImage.new("RGB", (16, 16))

    with patch(
        "model.training.evaluate_detection.Image.open", side_effect=_fake_open
    ):
        predictions, ground_truths, errors = run_inference(
            detector,
            split,
            torch.device("cpu"),
            input_size,
            {1: "crack"},
        )

    return predictions, ground_truths, errors, image_ids, outcomes


def _expected_error(image_id: str, outcome: str) -> str:
    """The exact error string a failed image should contribute."""
    if outcome == "decode_failure":
        return f"{image_id}: {DECODE_MSG}"
    if outcome == "forward_exception":
        return f"{image_id}: {FORWARD_MSG}"
    raise AssertionError(f"{outcome!r} is not a failure outcome")


def _is_empty_prediction(pred: dict) -> bool:
    return (
        len(pred["boxes"]) == 0
        and len(pred["labels"]) == 0
        and len(pred["scores"]) == 0
    )


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_OUTCOME = st.sampled_from(
    ["success", "empty_success", "forward_exception", "decode_failure"]
)

# A run is a list of per-image outcomes; empty runs are valid (no images).
_OUTCOME_LISTS = st.lists(_OUTCOME, min_size=0, max_size=15)

# A list guaranteed to contain at least one failure, to exercise the non-empty
# error-list regime densely.
_OUTCOME_LISTS_WITH_FAILURE = st.lists(_OUTCOME, min_size=1, max_size=15).filter(
    lambda os: any(o in _FAILURE_OUTCOMES for o in os)
)


# ---------------------------------------------------------------------------
# Property 11
# ---------------------------------------------------------------------------


class TestProperty11ErrorAccounting:
    """Property 11: Error accounting and format.

    **Validates: Requirements 15.2, 15.3**
    """

    @given(outcomes=_OUTCOME_LISTS)
    @settings(max_examples=100, deadline=None)
    def test_one_error_per_failed_image_in_order_and_format(self, outcomes):
        # Feature: generic-evaluation-script, Property 11: Error accounting
        """Every failed image contributes exactly one correctly-formatted error.

        The run's error list equals, in order, one ``"<image_id>: <text>"``
        string per failed image (decode failure or forward exception) and
        nothing for successful (or no-detection) images (Req 15.2).

        **Validates: Requirements 15.2**
        """
        predictions, _gts, errors, image_ids, outcomes = _run(outcomes)

        # The exact expected error list, in annotation order, for failed images.
        expected_errors = [
            _expected_error(image_id, outcome)
            for image_id, outcome in zip(image_ids, outcomes)
            if outcome in _FAILURE_OUTCOMES
        ]
        assert errors == expected_errors

        # Each error is "<image_id>: <exception text>": starts with a real image
        # id followed by ": " and a non-empty exception text.
        for error in errors:
            image_id, sep, text = error.partition(": ")
            assert sep == ": "
            assert image_id in image_ids
            assert text != ""

    @given(outcomes=_OUTCOME_LISTS)
    @settings(max_examples=100, deadline=None)
    def test_error_count_equals_number_of_failed_images(self, outcomes):
        # Feature: generic-evaluation-script, Property 11: Error accounting
        """len(errors) == number of failed images, and exactly one id each.

        The set of image ids appearing in the error list equals exactly the set
        of failed image ids, and since image ids are unique there is exactly one
        error per failed image (Req 15.2).

        **Validates: Requirements 15.2**
        """
        _preds, _gts, errors, image_ids, outcomes = _run(outcomes)

        failed_ids = {
            image_id
            for image_id, outcome in zip(image_ids, outcomes)
            if outcome in _FAILURE_OUTCOMES
        }
        num_failed = sum(1 for o in outcomes if o in _FAILURE_OUTCOMES)

        assert len(errors) == num_failed

        error_ids = [error.partition(": ")[0] for error in errors]
        # Exactly one error per failed image: ids unique, set matches, no dupes.
        assert set(error_ids) == failed_ids
        assert len(error_ids) == len(set(error_ids)) == len(failed_ids)

    @given(outcomes=_OUTCOME_LISTS)
    @settings(max_examples=100, deadline=None)
    def test_report_error_count_matches_error_list_and_failures(self, outcomes):
        # Feature: generic-evaluation-script, Property 11: Error accounting
        """Report error count == len(error list) == number of failed images.

        The report's ``errors`` block (per the design contract:
        ``{"count": len(items), "items": <error list>}``) reports a count equal
        to the length of the run's error list, which equals the number of failed
        images (Req 15.3).

        **Validates: Requirements 15.3**
        """
        _preds, _gts, errors, image_ids, outcomes = _run(outcomes)

        num_failed = sum(1 for o in outcomes if o in _FAILURE_OUTCOMES)

        # Assemble the report's errors block exactly as the design specifies.
        report_errors = {"count": len(errors), "items": list(errors)}

        assert report_errors["count"] == len(errors) == num_failed
        assert report_errors["items"] == errors

    @given(outcomes=_OUTCOME_LISTS)
    @settings(max_examples=100, deadline=None)
    def test_failed_images_have_empty_predictions_and_alignment_holds(
        self, outcomes
    ):
        # Feature: generic-evaluation-script, Property 11: Error accounting
        """Failed images contribute empty predictions; emptiness != error count.

        Predictions and ground truths stay 1:1 with the annotations; every
        failed image (and every successful no-detection image) has an empty
        prediction entry. The number of empty predictions equals failures plus
        no-detection successes -- confirming the error count tracks *failures*,
        not *emptiness* (Req 15.2 secondary accounting check).

        **Validates: Requirements 15.2**
        """
        predictions, ground_truths, errors, _image_ids, outcomes = _run(outcomes)

        # 1:1 alignment with the split's annotations.
        assert len(predictions) == len(ground_truths) == len(outcomes)

        # Each failed image's prediction entry is empty.
        for pred, outcome in zip(predictions, outcomes):
            if outcome in _FAILURE_OUTCOMES:
                assert _is_empty_prediction(pred)

        # Empty-prediction count == failures + no-detection successes, which is
        # in general strictly greater than the error count.
        empty_predictions = sum(1 for p in predictions if _is_empty_prediction(p))
        expected_empty = sum(1 for o in outcomes if o in _EMPTY_PRED_OUTCOMES)
        assert empty_predictions == expected_empty

        num_empty_success = sum(1 for o in outcomes if o == "empty_success")
        # The error count never counts no-detection successes.
        assert len(errors) == empty_predictions - num_empty_success

    @given(outcomes=_OUTCOME_LISTS_WITH_FAILURE)
    @settings(max_examples=100, deadline=None)
    def test_each_error_maps_to_a_distinct_failed_image(self, outcomes):
        # Feature: generic-evaluation-script, Property 11: Error accounting
        """In runs with failures, errors map one-to-one onto failed images.

        **Validates: Requirements 15.2, 15.3**
        """
        _preds, _gts, errors, image_ids, outcomes = _run(outcomes)

        failed = [
            image_id
            for image_id, outcome in zip(image_ids, outcomes)
            if outcome in _FAILURE_OUTCOMES
        ]
        assert len(errors) == len(failed) >= 1

        # The i-th error corresponds to the i-th failed image, in order.
        for error, image_id, outcome in zip(
            errors,
            [i for i, o in zip(image_ids, outcomes) if o in _FAILURE_OUTCOMES],
            [o for o in outcomes if o in _FAILURE_OUTCOMES],
        ):
            assert error == _expected_error(image_id, outcome)


# ---------------------------------------------------------------------------
# Example-based unit tests complementing Property 11
# ---------------------------------------------------------------------------


class TestErrorAccountingExamples:
    """Concrete examples complementing Property 11.

    **Validates: Requirements 15.2, 15.3**
    """

    def test_all_success_produces_no_errors(self):
        """A run of only successful images has an empty error list. (Req 15.2)"""
        _preds, _gts, errors, _ids, _outcomes = _run(["success", "success"])
        assert errors == []

    def test_empty_success_contributes_no_error(self):
        """A no-detection success is empty but contributes no error. (Req 15.2)"""
        predictions, _gts, errors, _ids, _outcomes = _run(["empty_success"])
        assert errors == []
        assert _is_empty_prediction(predictions[0])

    def test_decode_failure_records_one_error(self):
        """A decode failure records exactly one "<id>: <text>" entry. (Req 15.2)"""
        _preds, _gts, errors, ids, _outcomes = _run(["decode_failure"])
        assert errors == [f"{ids[0]}: {DECODE_MSG}"]

    def test_forward_exception_records_one_error(self):
        """A forward exception records exactly one entry. (Req 15.2)"""
        _preds, _gts, errors, ids, _outcomes = _run(["forward_exception"])
        assert errors == [f"{ids[0]}: {FORWARD_MSG}"]

    def test_mixed_run_error_count_and_order(self):
        """A mixed run records one error per failed image, in order. (Req 15.3)"""
        outcomes = [
            "success",
            "decode_failure",
            "empty_success",
            "forward_exception",
            "success",
        ]
        _preds, _gts, errors, ids, _outcomes = _run(outcomes)

        assert errors == [
            f"{ids[1]}: {DECODE_MSG}",
            f"{ids[3]}: {FORWARD_MSG}",
        ]
        # Report count mirrors the error list length and the failure count.
        assert {"count": len(errors), "items": errors}["count"] == 2

    def test_empty_run_has_no_errors(self):
        """A run with zero images has an empty error list. (Req 15.3)"""
        preds, gts, errors, _ids, _outcomes = _run([])
        assert preds == []
        assert gts == []
        assert errors == []
