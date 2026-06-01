"""Property-based tests for prediction/ground-truth 1:1 alignment.

Feature: generic-evaluation-script
Property 10: Predictions and ground truths stay 1:1 aligned

For any evaluation split and any pattern of per-image inference outcomes
(success, empty success, image-decode failure, or forward-pass exception), the
inference loop produces exactly one prediction entry per ground-truth
annotation in the same order, such that ``len(predictions) ==
len(ground_truths)`` and ``predictions[i]["image_id"] ==
ground_truths[i]["image_id"]`` for every ``i``, with failed images contributing
an empty prediction entry (empty ``boxes``, ``labels``, and ``scores``).

These tests exercise the real ``run_inference`` function in
``model/training/evaluate_detection.py``. To keep the property independent of
GPUs, real checkpoints, and real image files:

* a *conforming fake detector* implements ``forward`` per the Base_Detector
  contract (returns a length-``B`` list of dicts with ``boxes``/``labels``/
  ``scores`` tensors), and is driven by a generated per-image outcome plan so it
  can return detections, return no detections, or raise on demand;
* a *fake split dataset* exposes ``get_annotations()`` returning
  :class:`~model.datasets.base.Annotation` objects with unique image paths and
  generated ground-truth bounding boxes;
* ``PIL.Image.open`` (as referenced by the evaluation module) is monkeypatched
  so that "decode failure" images raise on open while every other image yields a
  small in-memory RGB image the real torchvision transform can consume.

The four generated outcomes cover Req 7.2 (success), Req 7.3/15.1 (forward-pass
exception -> empty entry + error), Req 14.2 (image-decode failure -> empty entry
+ error), and the empty-detection success case (empty entry, *not* an error).

**Validates: Requirements 7.1, 7.2, 7.3, 14.2, 15.1**
"""

from pathlib import Path
from unittest.mock import patch

import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image as PILImage

from model.datasets.base import Annotation, BoundingBox
from model.training.evaluate_detection import run_inference


# ---------------------------------------------------------------------------
# Per-image outcome plan
# ---------------------------------------------------------------------------

# The four per-image inference outcomes Property 10 must cover:
#   "success"      -> forward returns >=1 detection (Req 7.2)
#   "empty"        -> forward returns 0 detections (success, but empty entry)
#   "decode_fail"  -> Image.open raises (Req 14.2: empty entry + error)
#   "forward_fail" -> forward raises (Req 15.1: empty entry + error)
_OUTCOMES = ("success", "empty", "decode_fail", "forward_fail")

# Outcomes that must appear in the run's error list (failed images).
_FAILURE_OUTCOMES = ("decode_fail", "forward_fail")


@st.composite
def _image_plans(draw):
    """Draw a list of ``(outcome, gt_count, pred_count)`` per-image plans.

    ``gt_count`` is the number of ground-truth boxes on the annotation;
    ``pred_count`` is the number of detections the fake detector returns for a
    ``success`` image (1-3) and ``0`` for every other outcome. The list may be
    empty so the zero-image edge case is exercised.
    """
    count = draw(st.integers(min_value=0, max_value=6))
    plans = []
    for _ in range(count):
        outcome = draw(st.sampled_from(_OUTCOMES))
        gt_count = draw(st.integers(min_value=0, max_value=3))
        pred_count = draw(st.integers(min_value=1, max_value=3)) if outcome == "success" else 0
        plans.append((outcome, gt_count, pred_count))
    return plans


# ---------------------------------------------------------------------------
# Conforming fakes
# ---------------------------------------------------------------------------


class _FakeSplit:
    """A minimal evaluation-split dataset exposing ``get_annotations()``."""

    def __init__(self, annotations):
        self._annotations = annotations

    def get_annotations(self):
        return self._annotations

    def __len__(self):
        return len(self._annotations)


class _FakeDetector:
    """A conforming Base_Detector test double driven by a forward plan.

    ``forward`` is called exactly once per image that decoded successfully, in
    annotation order, so a simple positional cursor over ``forward_plan``
    (outcomes for non-decode-failed images) selects the right behavior:

    * ``forward_fail`` -> raise (Req 15.1);
    * ``empty`` -> return a single dict with zero-length tensors (Req 2.5);
    * ``success`` -> return a single dict with ``pred_count`` valid, normalized,
      non-degenerate detections (so they survive normalization/clamping).
    """

    def __init__(self, forward_plan):
        self._forward_plan = forward_plan
        self._cursor = 0
        self.forward_calls = 0

    def forward(self, images):
        outcome, pred_count = self._forward_plan[self._cursor]
        self._cursor += 1
        self.forward_calls += 1

        # The Base_Detector contract returns one dict per image in the batch;
        # run_inference feeds a single image (B == 1) and reads outputs[0].
        if outcome == "forward_fail":
            raise RuntimeError("simulated forward-pass failure")

        if outcome == "empty":
            return [
                {
                    "boxes": torch.zeros((0, 4)),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                    "scores": torch.zeros((0,)),
                }
            ]

        # "success": pred_count valid normalized boxes in [0, 1], non-degenerate.
        boxes = torch.tensor([[0.1, 0.1, 0.5, 0.5]] * pred_count, dtype=torch.float32)
        labels = torch.tensor([1] * pred_count, dtype=torch.int64)
        scores = torch.tensor([0.9] * pred_count, dtype=torch.float32)
        return [{"boxes": boxes, "labels": labels, "scores": scores}]


def _build_annotations(plans):
    """Build one Annotation per plan with a unique path and ground-truth boxes.

    The ``.png`` suffix on each unique ``img_<i>`` path keeps image ids from
    being prefixes of one another, so error-string membership checks are exact.
    """
    annotations = []
    for i, (_outcome, gt_count, _pred_count) in enumerate(plans):
        boxes = [
            BoundingBox(0.1, 0.1, 0.4, 0.4, "crack") for _ in range(gt_count)
        ]
        annotations.append(
            Annotation(image_path=Path(f"/fake/img_{i}.png"), bounding_boxes=boxes)
        )
    return annotations


def _make_fake_open(outcome_by_id):
    """Return an ``Image.open`` replacement keyed by the requested image path.

    Decode-failure images raise on open (Req 14.2); every other image yields a
    small in-memory RGB image whose ``.convert("RGB")`` the real torchvision
    transform consumes. ``PILImage.new`` is unaffected by patching ``open``.
    """

    def _fake_open(path, *args, **kwargs):
        if outcome_by_id.get(str(path)) == "decode_fail":
            raise OSError(f"cannot identify image file {path!r}")
        return PILImage.new("RGB", (16, 16))

    return _fake_open


def _run(plans, input_size):
    """Run ``run_inference`` against fakes built from ``plans``.

    Returns ``(predictions, ground_truths, errors, annotations)``.
    """
    annotations = _build_annotations(plans)
    outcome_by_id = {
        str(ann.image_path): plan[0] for ann, plan in zip(annotations, plans)
    }
    # Forward is reached only by images that decode successfully, in order.
    forward_plan = [
        (outcome, pred_count)
        for (outcome, _gt, pred_count) in plans
        if outcome != "decode_fail"
    ]

    split_ds = _FakeSplit(annotations)
    detector = _FakeDetector(forward_plan)
    device = torch.device("cpu")
    idx_to_class = {1: "crack", 2: "pothole"}

    with patch(
        "model.training.evaluate_detection.Image.open",
        _make_fake_open(outcome_by_id),
    ):
        predictions, ground_truths, errors = run_inference(
            detector, split_ds, device, input_size, idx_to_class
        )

    return predictions, ground_truths, errors, annotations


# ---------------------------------------------------------------------------
# Property 10
# ---------------------------------------------------------------------------


class TestProperty10Alignment:
    """Property 10: Predictions and ground truths stay 1:1 aligned.

    **Validates: Requirements 7.1, 7.2, 7.3, 14.2, 15.1**
    """

    @given(plans=_image_plans(), input_size=st.integers(min_value=8, max_value=64))
    @settings(max_examples=100, deadline=None)
    def test_predictions_and_ground_truths_stay_aligned(self, plans, input_size):
        # Feature: generic-evaluation-script, Property 10: 1:1 alignment
        """One entry per annotation, same order, same image_id; failures empty.

        **Validates: Requirements 7.1, 7.2, 7.3, 14.2, 15.1**
        """
        predictions, ground_truths, errors, annotations = _run(plans, input_size)

        # Req 7.1: exactly one prediction and one ground-truth per annotation.
        assert len(predictions) == len(ground_truths) == len(annotations)

        for i, (annotation, plan) in enumerate(zip(annotations, plans)):
            image_id = str(annotation.image_path)
            outcome = plan[0]

            # Req 7.1: same order, matching image_id across both lists.
            assert predictions[i]["image_id"] == image_id
            assert ground_truths[i]["image_id"] == image_id

            pred = predictions[i]
            # The three prediction lists are always mutually consistent.
            assert len(pred["boxes"]) == len(pred["labels"]) == len(pred["scores"])

            if outcome in _FAILURE_OUTCOMES:
                # Req 7.3, 14.2, 15.1: a failed image contributes an empty entry.
                assert pred["boxes"] == []
                assert pred["labels"] == []
                assert pred["scores"] == []
            elif outcome == "empty":
                # Empty-detection success: empty entry but NOT a failure.
                assert pred["boxes"] == []
                assert pred["labels"] == []
                assert pred["scores"] == []
            else:  # "success" (Req 7.2)
                assert len(pred["boxes"]) == plan[2]
                assert len(pred["boxes"]) >= 1

    @given(plans=_image_plans(), input_size=st.integers(min_value=8, max_value=64))
    @settings(max_examples=100, deadline=None)
    def test_errors_correspond_exactly_to_failed_images(self, plans, input_size):
        # Feature: generic-evaluation-script, Property 10: 1:1 alignment
        """Every failed image (and only those) yields an empty entry + an error.

        Distinguishes a genuine failure (decode/forward) from an empty-detection
        success: both produce an empty prediction entry, but only failures are
        recorded in the error list.

        **Validates: Requirements 7.3, 14.2, 15.1**
        """
        predictions, _ground_truths, errors, annotations = _run(plans, input_size)

        failed_ids = [
            str(ann.image_path)
            for ann, plan in zip(annotations, plans)
            if plan[0] in _FAILURE_OUTCOMES
        ]

        # One error string per failed image (Req 15.1 / 14.2).
        assert len(errors) == len(failed_ids)

        # Each failed image is named in the errors, and no successful image is.
        for ann, plan in zip(annotations, plans):
            image_id = str(ann.image_path)
            named = any(err.startswith(f"{image_id}:") for err in errors)
            if plan[0] in _FAILURE_OUTCOMES:
                assert named
            else:
                assert not named


# ---------------------------------------------------------------------------
# Example-based tests complementing Property 10
# ---------------------------------------------------------------------------


class TestAlignmentExamples:
    """Concrete scenarios complementing Property 10.

    **Validates: Requirements 7.1, 7.2, 7.3, 14.2, 15.1**
    """

    def test_empty_split_yields_empty_aligned_results(self):
        """A split with no annotations yields empty, aligned results. (Req 7.1)"""
        predictions, ground_truths, errors, annotations = _run([], 32)
        assert predictions == []
        assert ground_truths == []
        assert errors == []
        assert annotations == []

    def test_all_success_entries_are_populated_and_error_free(self):
        """Every successful image gets a populated entry and no errors. (Req 7.2)"""
        plans = [("success", 1, 2), ("success", 0, 1), ("success", 2, 3)]
        predictions, ground_truths, errors, annotations = _run(plans, 32)

        assert len(predictions) == len(ground_truths) == 3
        assert errors == []
        for i, plan in enumerate(plans):
            assert len(predictions[i]["boxes"]) == plan[2]
            assert predictions[i]["image_id"] == str(annotations[i].image_path)
            assert ground_truths[i]["image_id"] == str(annotations[i].image_path)

    def test_decode_failure_yields_empty_entry_and_error(self):
        """An image that fails to decode gets an empty entry + an error. (Req 14.2)"""
        plans = [("decode_fail", 1, 0)]
        predictions, _gts, errors, annotations = _run(plans, 32)

        assert predictions[0]["boxes"] == []
        assert predictions[0]["labels"] == []
        assert predictions[0]["scores"] == []
        assert len(errors) == 1
        assert errors[0].startswith(f"{annotations[0].image_path}:")

    def test_forward_failure_yields_empty_entry_and_error(self):
        """A forward-pass exception gets an empty entry + an error. (Req 15.1)"""
        plans = [("forward_fail", 2, 0)]
        predictions, _gts, errors, annotations = _run(plans, 32)

        assert predictions[0]["boxes"] == []
        assert predictions[0]["labels"] == []
        assert predictions[0]["scores"] == []
        assert len(errors) == 1
        assert errors[0].startswith(f"{annotations[0].image_path}:")

    def test_empty_detection_success_is_not_an_error(self):
        """A no-detection success gets an empty entry but is NOT an error."""
        plans = [("empty", 1, 0)]
        predictions, _gts, errors, annotations = _run(plans, 32)

        assert predictions[0]["boxes"] == []
        assert predictions[0]["labels"] == []
        assert predictions[0]["scores"] == []
        assert errors == []

    def test_mixed_outcomes_stay_aligned_in_order(self):
        """A mix of all four outcomes stays 1:1 aligned in annotation order."""
        plans = [
            ("success", 1, 2),
            ("decode_fail", 0, 0),
            ("empty", 2, 0),
            ("forward_fail", 1, 0),
            ("success", 0, 1),
        ]
        predictions, ground_truths, errors, annotations = _run(plans, 32)

        assert len(predictions) == len(ground_truths) == len(annotations) == 5
        for i, ann in enumerate(annotations):
            assert predictions[i]["image_id"] == str(ann.image_path)
            assert ground_truths[i]["image_id"] == str(ann.image_path)

        # Failures are the decode_fail (index 1) and forward_fail (index 3).
        assert predictions[1]["boxes"] == []
        assert predictions[3]["boxes"] == []
        assert len(errors) == 2
        assert len(predictions[0]["boxes"]) == 2
        assert len(predictions[4]["boxes"]) == 1
        assert predictions[2]["boxes"] == []  # empty success
