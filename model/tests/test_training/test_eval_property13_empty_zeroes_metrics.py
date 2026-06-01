"""Property-based tests for empty-prediction zeroed metrics.

Feature: generic-evaluation-script
Property 13: Empty prediction entry forces zeroed scalar metrics

For any aligned predictions and ground truths in which at least one prediction
entry is empty, metrics computation reports ``map_50``, ``map_50_95``,
``precision``, ``recall``, and ``f1_score`` all equal to ``0.0``, while still
producing a confusion matrix of shape ``(C, C)`` where ``C`` is the number of
configured classes.

These tests exercise the real ``compute_all_metrics`` function in
``model/training/evaluate_detection.py``. The function delegates to the
already-tested metrics collaborators (``compute_map``,
``compute_precision_recall_f1``, ``compute_confusion_matrix``) and applies the
Req 8.5 zeroing rule: when **any** image has an empty prediction entry (no
boxes, labels, or scores), the five scalar metrics are forced to ``0.0`` while
the confusion matrix is still computed and returned sized to the number of
classes.

To make the property independent of GPUs, real checkpoints, and real datasets,
the test generates:

* a list of class names (1–5 classes);
* aligned predictions and ground truths where at least one prediction entry is
  empty (empty ``boxes``, ``labels``, and ``scores``);
* valid confidence and IoU thresholds in ``[0.0, 1.0]``.

The property asserts that:

1. ``map_50``, ``map_50_95``, ``precision``, ``recall``, and ``f1_score`` are
   all exactly ``0.0``;
2. ``confusion_matrix`` has shape ``(C, C)`` where ``C == len(class_names)``.

**Validates: Requirements 8.5**
"""

from typing import List

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from model.training.evaluate_detection import compute_all_metrics


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


@st.composite
def _class_names(draw) -> List[str]:
    """Draw a list of 1–5 unique class names."""
    count = draw(st.integers(min_value=1, max_value=5))
    names = [f"class_{i}" for i in range(count)]
    return names


@st.composite
def _non_empty_prediction(draw, class_names: List[str]) -> dict:
    """Draw a non-empty prediction entry with 1–3 detections.

    Boxes are normalized (in [0, 1]), non-degenerate, and labels are drawn from
    the provided class names. Scores are in [0.0, 1.0].
    """
    num_detections = draw(st.integers(min_value=1, max_value=3))
    boxes = []
    labels = []
    scores = []
    for _ in range(num_detections):
        # Generate a valid, non-degenerate box in [0, 1]
        x_min = draw(st.floats(min_value=0.0, max_value=0.4))
        y_min = draw(st.floats(min_value=0.0, max_value=0.4))
        x_max = draw(st.floats(min_value=x_min + 0.1, max_value=1.0))
        y_max = draw(st.floats(min_value=y_min + 0.1, max_value=1.0))
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(draw(st.sampled_from(class_names)))
        scores.append(draw(st.floats(min_value=0.0, max_value=1.0)))
    return {"boxes": boxes, "labels": labels, "scores": scores}


@st.composite
def _ground_truth(draw, class_names: List[str]) -> dict:
    """Draw a ground-truth entry with 0–3 boxes.

    Boxes are normalized (in [0, 1]), non-degenerate, and labels are drawn from
    the provided class names.
    """
    num_boxes = draw(st.integers(min_value=0, max_value=3))
    boxes = []
    labels = []
    for _ in range(num_boxes):
        x_min = draw(st.floats(min_value=0.0, max_value=0.4))
        y_min = draw(st.floats(min_value=0.0, max_value=0.4))
        x_max = draw(st.floats(min_value=x_min + 0.1, max_value=1.0))
        y_max = draw(st.floats(min_value=y_min + 0.1, max_value=1.0))
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(draw(st.sampled_from(class_names)))
    return {"boxes": boxes, "labels": labels}


def _empty_prediction() -> dict:
    """Return an empty prediction entry (no boxes, labels, or scores)."""
    return {"boxes": [], "labels": [], "scores": []}


@st.composite
def _aligned_data_with_empty(draw):
    """Draw aligned predictions and ground truths with at least one empty pred.

    Returns a tuple of (predictions, ground_truths, class_names) where:
    - predictions and ground_truths are aligned (same length, same image_ids);
    - at least one prediction entry is empty;
    - class_names is the list of class names used.
    """
    class_names = draw(_class_names())
    num_images = draw(st.integers(min_value=1, max_value=6))

    # Decide which images have empty predictions (at least one must be empty)
    # Generate a list of booleans, then ensure at least one is True
    is_empty = draw(st.lists(st.booleans(), min_size=num_images, max_size=num_images))
    if not any(is_empty):
        # Force at least one empty prediction
        empty_idx = draw(st.integers(min_value=0, max_value=num_images - 1))
        is_empty[empty_idx] = True

    predictions = []
    ground_truths = []

    for i in range(num_images):
        image_id = f"image_{i}"

        if is_empty[i]:
            pred = _empty_prediction()
        else:
            pred = draw(_non_empty_prediction(class_names))
        pred["image_id"] = image_id
        predictions.append(pred)

        gt = draw(_ground_truth(class_names))
        gt["image_id"] = image_id
        ground_truths.append(gt)

    return predictions, ground_truths, class_names


@st.composite
def _thresholds(draw):
    """Draw valid confidence and IoU thresholds in [0.0, 1.0]."""
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    iou = draw(st.floats(min_value=0.0, max_value=1.0))
    return confidence, iou


# ---------------------------------------------------------------------------
# Property 13
# ---------------------------------------------------------------------------


class TestProperty13EmptyZeroesMetrics:
    """Property 13: Empty prediction entry forces zeroed scalar metrics.

    **Validates: Requirements 8.5**
    """

    @given(
        data=_aligned_data_with_empty(),
        thresholds=_thresholds(),
    )
    @settings(max_examples=100, deadline=None)
    def test_empty_prediction_zeroes_scalar_metrics(self, data, thresholds):
        # Feature: generic-evaluation-script, Property 13: Empty zeroes metrics
        """Any empty prediction entry forces all five scalar metrics to 0.0.

        When at least one prediction entry is empty (no boxes, labels, or
        scores), ``map_50``, ``map_50_95``, ``precision``, ``recall``, and
        ``f1_score`` are all exactly ``0.0`` (Req 8.5).

        **Validates: Requirements 8.5**
        """
        predictions, ground_truths, class_names = data
        confidence_threshold, iou_threshold = thresholds

        metrics = compute_all_metrics(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=class_names,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )

        # Req 8.5: all five scalar metrics must be exactly 0.0
        assert metrics["map_50"] == 0.0, f"map_50 should be 0.0, got {metrics['map_50']}"
        assert metrics["map_50_95"] == 0.0, f"map_50_95 should be 0.0, got {metrics['map_50_95']}"
        assert metrics["precision"] == 0.0, f"precision should be 0.0, got {metrics['precision']}"
        assert metrics["recall"] == 0.0, f"recall should be 0.0, got {metrics['recall']}"
        assert metrics["f1_score"] == 0.0, f"f1_score should be 0.0, got {metrics['f1_score']}"

    @given(
        data=_aligned_data_with_empty(),
        thresholds=_thresholds(),
    )
    @settings(max_examples=100, deadline=None)
    def test_confusion_matrix_has_correct_shape(self, data, thresholds):
        # Feature: generic-evaluation-script, Property 13: Empty zeroes metrics
        """Confusion matrix has shape (C, C) even with empty predictions.

        Even when scalar metrics are zeroed due to empty predictions, the
        confusion matrix is still computed and has shape ``(C, C)`` where
        ``C == len(class_names)`` (Req 8.5).

        **Validates: Requirements 8.5**
        """
        predictions, ground_truths, class_names = data
        confidence_threshold, iou_threshold = thresholds

        metrics = compute_all_metrics(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=class_names,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )

        num_classes = len(class_names)
        confusion_matrix = metrics["confusion_matrix"]

        # Req 8.5: confusion matrix must have shape (C, C)
        assert isinstance(confusion_matrix, np.ndarray), "confusion_matrix should be a numpy array"
        assert confusion_matrix.shape == (num_classes, num_classes), (
            f"confusion_matrix should have shape ({num_classes}, {num_classes}), "
            f"got {confusion_matrix.shape}"
        )

    @given(
        data=_aligned_data_with_empty(),
        thresholds=_thresholds(),
    )
    @settings(max_examples=100, deadline=None)
    def test_per_class_ap_keys_match_class_names(self, data, thresholds):
        # Feature: generic-evaluation-script, Property 13: Empty zeroes metrics
        """per_class_ap contains an entry for every class name.

        The ``per_class_ap`` dict should have keys matching the configured
        class names, even when scalar metrics are zeroed.

        **Validates: Requirements 8.5**
        """
        predictions, ground_truths, class_names = data
        confidence_threshold, iou_threshold = thresholds

        metrics = compute_all_metrics(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=class_names,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )

        per_class_ap = metrics["per_class_ap"]
        assert set(per_class_ap.keys()) == set(class_names), (
            f"per_class_ap keys {set(per_class_ap.keys())} should match "
            f"class_names {set(class_names)}"
        )


# ---------------------------------------------------------------------------
# Example-based tests complementing Property 13
# ---------------------------------------------------------------------------


class TestEmptyZeroesMetricsExamples:
    """Concrete examples complementing Property 13.

    **Validates: Requirements 8.5**
    """

    def test_single_empty_prediction_zeroes_metrics(self):
        """A single empty prediction entry zeroes all scalar metrics. (Req 8.5)"""
        predictions = [
            {"image_id": "img_0", "boxes": [], "labels": [], "scores": []},
        ]
        ground_truths = [
            {"image_id": "img_0", "boxes": [[0.1, 0.1, 0.5, 0.5]], "labels": ["crack"]},
        ]
        class_names = ["crack", "pothole"]

        metrics = compute_all_metrics(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=class_names,
            confidence_threshold=0.5,
            iou_threshold=0.5,
        )

        assert metrics["map_50"] == 0.0
        assert metrics["map_50_95"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1_score"] == 0.0
        assert metrics["confusion_matrix"].shape == (2, 2)

    def test_mixed_empty_and_non_empty_zeroes_metrics(self):
        """A mix of empty and non-empty predictions still zeroes metrics. (Req 8.5)"""
        predictions = [
            {"image_id": "img_0", "boxes": [[0.1, 0.1, 0.5, 0.5]], "labels": ["crack"], "scores": [0.9]},
            {"image_id": "img_1", "boxes": [], "labels": [], "scores": []},  # empty
            {"image_id": "img_2", "boxes": [[0.2, 0.2, 0.6, 0.6]], "labels": ["pothole"], "scores": [0.8]},
        ]
        ground_truths = [
            {"image_id": "img_0", "boxes": [[0.1, 0.1, 0.5, 0.5]], "labels": ["crack"]},
            {"image_id": "img_1", "boxes": [[0.3, 0.3, 0.7, 0.7]], "labels": ["pothole"]},
            {"image_id": "img_2", "boxes": [[0.2, 0.2, 0.6, 0.6]], "labels": ["pothole"]},
        ]
        class_names = ["crack", "pothole"]

        metrics = compute_all_metrics(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=class_names,
            confidence_threshold=0.5,
            iou_threshold=0.5,
        )

        # Even with some valid predictions, the presence of one empty entry
        # forces all scalar metrics to 0.0
        assert metrics["map_50"] == 0.0
        assert metrics["map_50_95"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1_score"] == 0.0
        assert metrics["confusion_matrix"].shape == (2, 2)

    def test_all_empty_predictions_zeroes_metrics(self):
        """All empty predictions zero all scalar metrics. (Req 8.5)"""
        predictions = [
            {"image_id": "img_0", "boxes": [], "labels": [], "scores": []},
            {"image_id": "img_1", "boxes": [], "labels": [], "scores": []},
        ]
        ground_truths = [
            {"image_id": "img_0", "boxes": [[0.1, 0.1, 0.5, 0.5]], "labels": ["crack"]},
            {"image_id": "img_1", "boxes": [[0.2, 0.2, 0.6, 0.6]], "labels": ["pothole"]},
        ]
        class_names = ["crack", "pothole", "spalling"]

        metrics = compute_all_metrics(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=class_names,
            confidence_threshold=0.25,
            iou_threshold=0.5,
        )

        assert metrics["map_50"] == 0.0
        assert metrics["map_50_95"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1_score"] == 0.0
        # Confusion matrix should be 3x3 for 3 classes
        assert metrics["confusion_matrix"].shape == (3, 3)

    def test_empty_prediction_with_empty_ground_truth(self):
        """Empty prediction with empty ground truth still zeroes metrics. (Req 8.5)"""
        predictions = [
            {"image_id": "img_0", "boxes": [], "labels": [], "scores": []},
        ]
        ground_truths = [
            {"image_id": "img_0", "boxes": [], "labels": []},
        ]
        class_names = ["crack"]

        metrics = compute_all_metrics(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=class_names,
            confidence_threshold=0.5,
            iou_threshold=0.5,
        )

        assert metrics["map_50"] == 0.0
        assert metrics["map_50_95"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1_score"] == 0.0
        assert metrics["confusion_matrix"].shape == (1, 1)

    def test_confusion_matrix_shape_with_many_classes(self):
        """Confusion matrix has correct shape with many classes. (Req 8.5)"""
        predictions = [
            {"image_id": "img_0", "boxes": [], "labels": [], "scores": []},
        ]
        ground_truths = [
            {"image_id": "img_0", "boxes": [[0.1, 0.1, 0.5, 0.5]], "labels": ["class_0"]},
        ]
        class_names = [f"class_{i}" for i in range(10)]

        metrics = compute_all_metrics(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=class_names,
            confidence_threshold=0.5,
            iou_threshold=0.5,
        )

        assert metrics["map_50"] == 0.0
        assert metrics["map_50_95"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1_score"] == 0.0
        # Confusion matrix should be 10x10 for 10 classes
        assert metrics["confusion_matrix"].shape == (10, 10)

    def test_per_class_ap_present_with_empty_predictions(self):
        """per_class_ap is present and has correct keys with empty predictions. (Req 8.5)"""
        predictions = [
            {"image_id": "img_0", "boxes": [], "labels": [], "scores": []},
        ]
        ground_truths = [
            {"image_id": "img_0", "boxes": [[0.1, 0.1, 0.5, 0.5]], "labels": ["crack"]},
        ]
        class_names = ["crack", "pothole", "spalling"]

        metrics = compute_all_metrics(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=class_names,
            confidence_threshold=0.5,
            iou_threshold=0.5,
        )

        # per_class_ap should have entries for all class names
        assert "per_class_ap" in metrics
        assert set(metrics["per_class_ap"].keys()) == set(class_names)
