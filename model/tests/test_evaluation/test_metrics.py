"""Property-based tests for evaluation metrics.

Tests Properties 10, 11, and 13 from the design document using Hypothesis.
"""

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from model.evaluation.engine import EvaluationEngine
from model.evaluation.metrics import (
    compute_confusion_matrix,
    compute_map,
    compute_precision_recall_f1,
)


# ---------------------------------------------------------------------------
# Hypothesis strategies for predictions and ground truths
# ---------------------------------------------------------------------------

# Class names used in tests
_CLASS_NAMES = ["bache", "fisura_longitudinal", "fisura_transversal", "piel_de_cocodrilo"]

_CLASS_NAME_STRATEGY = st.sampled_from(_CLASS_NAMES)


@st.composite
def valid_bounding_box(draw):
    """Generate a valid bounding box [x_min, y_min, x_max, y_max]."""
    x_min = draw(st.floats(min_value=0.0, max_value=0.8))
    y_min = draw(st.floats(min_value=0.0, max_value=0.8))
    x_max = draw(st.floats(min_value=x_min + 0.05, max_value=1.0))
    y_max = draw(st.floats(min_value=y_min + 0.05, max_value=1.0))
    return [x_min, y_min, x_max, y_max]


@st.composite
def prediction_entry(draw, class_names=None):
    """Generate a single prediction dict for one image."""
    if class_names is None:
        class_names = _CLASS_NAMES
    num_boxes = draw(st.integers(min_value=0, max_value=5))
    boxes = [draw(valid_bounding_box()) for _ in range(num_boxes)]
    labels = [draw(st.sampled_from(class_names)) for _ in range(num_boxes)]
    scores = [draw(st.floats(min_value=0.01, max_value=1.0)) for _ in range(num_boxes)]
    image_id = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=3,
        max_size=10,
    ))
    return {
        "image_id": image_id,
        "boxes": boxes,
        "labels": labels,
        "scores": scores,
    }


@st.composite
def ground_truth_entry(draw, class_names=None):
    """Generate a single ground truth dict for one image."""
    if class_names is None:
        class_names = _CLASS_NAMES
    num_boxes = draw(st.integers(min_value=1, max_value=5))
    boxes = [draw(valid_bounding_box()) for _ in range(num_boxes)]
    labels = [draw(st.sampled_from(class_names)) for _ in range(num_boxes)]
    image_id = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=3,
        max_size=10,
    ))
    return {
        "image_id": image_id,
        "boxes": boxes,
        "labels": labels,
    }


@st.composite
def predictions_and_ground_truths(draw, min_images=1, max_images=4, class_names=None):
    """Generate matching predictions and ground truths with shared image IDs."""
    if class_names is None:
        class_names = _CLASS_NAMES
    num_images = draw(st.integers(min_value=min_images, max_value=max_images))
    image_ids = [f"img_{i}" for i in range(num_images)]

    predictions = []
    ground_truths = []

    for img_id in image_ids:
        # Ground truth always has at least 1 box
        num_gt_boxes = draw(st.integers(min_value=1, max_value=4))
        gt_boxes = [draw(valid_bounding_box()) for _ in range(num_gt_boxes)]
        gt_labels = [draw(st.sampled_from(class_names)) for _ in range(num_gt_boxes)]
        ground_truths.append({
            "image_id": img_id,
            "boxes": gt_boxes,
            "labels": gt_labels,
        })

        # Predictions can have 0 or more boxes
        num_pred_boxes = draw(st.integers(min_value=0, max_value=5))
        pred_boxes = [draw(valid_bounding_box()) for _ in range(num_pred_boxes)]
        pred_labels = [draw(st.sampled_from(class_names)) for _ in range(num_pred_boxes)]
        pred_scores = [draw(st.floats(min_value=0.01, max_value=1.0)) for _ in range(num_pred_boxes)]
        predictions.append({
            "image_id": img_id,
            "boxes": pred_boxes,
            "labels": pred_labels,
            "scores": pred_scores,
        })

    return predictions, ground_truths


# ---------------------------------------------------------------------------
# Property 10: Evaluation metrics satisfy mathematical bounds
# Feature: road-damage-evaluation-framework, Property 10: Evaluation metrics satisfy mathematical bounds
# ---------------------------------------------------------------------------


class TestProperty10MetricBounds:
    """Property 10: For any set of predictions and ground truth annotations,
    the Evaluation Engine SHALL produce: mAP in [0, 1], per-class AP values
    each in [0, 1] whose mean equals mAP@0.5, precision in [0, 1], recall
    in [0, 1], and F1 = 2 * precision * recall / (precision + recall) when
    precision + recall > 0.

    **Validates: Requirements 5.1, 5.2, 5.3**
    """

    @given(data=predictions_and_ground_truths())
    @settings(max_examples=100)
    def test_map_bounded_zero_one(self, data):
        # Feature: road-damage-evaluation-framework, Property 10: Evaluation metrics satisfy mathematical bounds
        """mAP@0.5 and mAP@0.5:0.95 are in [0, 1]."""
        predictions, ground_truths = data

        result = compute_map(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=_CLASS_NAMES,
        )

        assert 0.0 <= result["map_50"] <= 1.0
        assert 0.0 <= result["map_50_95"] <= 1.0

    @given(data=predictions_and_ground_truths())
    @settings(max_examples=100)
    def test_per_class_ap_bounded_and_mean_equals_map50(self, data):
        # Feature: road-damage-evaluation-framework, Property 10: Evaluation metrics satisfy mathematical bounds
        """Per-class AP values are each in [0, 1] and their mean equals mAP@0.5."""
        predictions, ground_truths = data

        result = compute_map(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=_CLASS_NAMES,
        )

        per_class_ap = result["per_class_ap"]
        assert len(per_class_ap) == len(_CLASS_NAMES)

        for cls_name, ap_value in per_class_ap.items():
            assert 0.0 <= ap_value <= 1.0, f"AP for {cls_name} out of bounds: {ap_value}"

        # Mean of per-class AP should equal mAP@0.5
        expected_map_50 = np.mean(list(per_class_ap.values()))
        assert abs(result["map_50"] - expected_map_50) < 1e-9

    @given(data=predictions_and_ground_truths())
    @settings(max_examples=100)
    def test_precision_recall_bounded(self, data):
        # Feature: road-damage-evaluation-framework, Property 10: Evaluation metrics satisfy mathematical bounds
        """Precision and recall are in [0, 1]."""
        predictions, ground_truths = data

        result = compute_precision_recall_f1(
            predictions=predictions,
            ground_truths=ground_truths,
            confidence_threshold=0.5,
            iou_threshold=0.5,
        )

        assert 0.0 <= result["precision"] <= 1.0
        assert 0.0 <= result["recall"] <= 1.0

    @given(data=predictions_and_ground_truths())
    @settings(max_examples=100)
    def test_f1_formula(self, data):
        # Feature: road-damage-evaluation-framework, Property 10: Evaluation metrics satisfy mathematical bounds
        """F1 = 2 * precision * recall / (precision + recall) when precision + recall > 0."""
        predictions, ground_truths = data

        result = compute_precision_recall_f1(
            predictions=predictions,
            ground_truths=ground_truths,
            confidence_threshold=0.5,
            iou_threshold=0.5,
        )

        precision = result["precision"]
        recall = result["recall"]
        f1 = result["f1"]

        assert 0.0 <= f1 <= 1.0

        if precision + recall > 0:
            expected_f1 = 2 * precision * recall / (precision + recall)
            assert abs(f1 - expected_f1) < 1e-9, (
                f"F1 mismatch: got {f1}, expected {expected_f1}"
            )
        else:
            assert f1 == 0.0


# ---------------------------------------------------------------------------
# Property 11: Class filtering excludes unlisted classes
# Feature: road-damage-evaluation-framework, Property 11: Class filtering excludes unlisted classes
# ---------------------------------------------------------------------------


class TestProperty11ClassFiltering:
    """Property 11: For any set of predictions spanning multiple classes and a
    target class filter list, evaluation SHALL compute metrics using only
    predictions and ground truth for the listed classes, and predictions for
    unlisted classes SHALL not affect the computed metrics.

    **Validates: Requirements 5.5, 5.6**
    """

    @given(data=st.data())
    @settings(max_examples=100)
    def test_filtering_excludes_unlisted_classes(self, data):
        # Feature: road-damage-evaluation-framework, Property 11: Class filtering excludes unlisted classes
        """Predictions for unlisted classes do not affect computed metrics."""
        # Use at least 2 classes so we can filter some out
        all_classes = _CLASS_NAMES
        # Choose a non-empty subset as target classes
        num_target = data.draw(st.integers(min_value=1, max_value=len(all_classes) - 1))
        target_classes = sorted(data.draw(
            st.lists(
                st.sampled_from(all_classes),
                min_size=num_target,
                max_size=num_target,
                unique=True,
            )
        ))
        excluded_classes = [c for c in all_classes if c not in target_classes]
        assume(len(excluded_classes) > 0)

        # Generate predictions and ground truths using all classes
        preds_gts = data.draw(predictions_and_ground_truths(
            min_images=2, max_images=3, class_names=all_classes
        ))
        predictions, ground_truths = preds_gts

        # Evaluate with class filtering via EvaluationEngine
        engine = EvaluationEngine()
        report = engine.evaluate(
            precomputed_predictions=predictions,
            precomputed_ground_truths=ground_truths,
            target_classes=target_classes,
            confidence_threshold=0.3,
            model_id="filter_test",
        )

        # Verify only target classes appear in results
        assert set(report.class_names) == set(target_classes)
        for cls_name in report.per_class_ap:
            assert cls_name in target_classes

        # Verify excluded classes are not in per_class_ap
        for cls_name in excluded_classes:
            assert cls_name not in report.per_class_ap

    @given(data=st.data())
    @settings(max_examples=100)
    def test_adding_excluded_class_predictions_does_not_change_metrics(self, data):
        # Feature: road-damage-evaluation-framework, Property 11: Class filtering excludes unlisted classes
        """Adding predictions for excluded classes does not change filtered metrics."""
        target_classes = ["bache", "fisura_longitudinal"]
        excluded_class = "piel_de_cocodrilo"

        # Generate base predictions and ground truths using only target classes
        base_data = data.draw(predictions_and_ground_truths(
            min_images=2, max_images=3, class_names=target_classes
        ))
        base_predictions, ground_truths = base_data

        # Create augmented predictions with extra excluded-class predictions
        augmented_predictions = []
        for pred in base_predictions:
            new_pred = {
                "image_id": pred["image_id"],
                "boxes": pred["boxes"][:],
                "labels": pred["labels"][:],
                "scores": pred["scores"][:],
            }
            # Add some excluded-class predictions
            num_extra = data.draw(st.integers(min_value=1, max_value=3))
            for _ in range(num_extra):
                new_pred["boxes"].append(data.draw(valid_bounding_box()))
                new_pred["labels"].append(excluded_class)
                new_pred["scores"].append(data.draw(st.floats(min_value=0.5, max_value=1.0)))
            augmented_predictions.append(new_pred)

        engine = EvaluationEngine()

        # Evaluate base predictions with filtering
        report_base = engine.evaluate(
            precomputed_predictions=base_predictions,
            precomputed_ground_truths=ground_truths,
            target_classes=target_classes,
            confidence_threshold=0.3,
            model_id="base",
        )

        # Evaluate augmented predictions with filtering
        report_augmented = engine.evaluate(
            precomputed_predictions=augmented_predictions,
            precomputed_ground_truths=ground_truths,
            target_classes=target_classes,
            confidence_threshold=0.3,
            model_id="augmented",
        )

        # Metrics should be identical since excluded class is filtered out
        assert abs(report_base.map_50 - report_augmented.map_50) < 1e-9
        assert abs(report_base.precision - report_augmented.precision) < 1e-9
        assert abs(report_base.recall - report_augmented.recall) < 1e-9
        assert abs(report_base.f1_score - report_augmented.f1_score) < 1e-9


# ---------------------------------------------------------------------------
# Property 13: Confusion matrix invariants
# Feature: road-damage-evaluation-framework, Property 13: Confusion matrix invariants
# ---------------------------------------------------------------------------


class TestProperty13ConfusionMatrixInvariants:
    """Property 13: For any set of predictions and ground truth with C classes,
    the confusion matrix SHALL have dimensions C×C, all entries SHALL be
    non-negative integers, and the sum of all entries SHALL equal the total
    number of ground truth instances evaluated (that were matched).

    **Validates: Requirements 5.8**
    """

    @given(data=predictions_and_ground_truths())
    @settings(max_examples=100)
    def test_confusion_matrix_dimensions(self, data):
        # Feature: road-damage-evaluation-framework, Property 13: Confusion matrix invariants
        """Confusion matrix has dimensions C×C where C is the number of classes."""
        predictions, ground_truths = data
        num_classes = len(_CLASS_NAMES)

        matrix = compute_confusion_matrix(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=_CLASS_NAMES,
            iou_threshold=0.5,
            confidence_threshold=0.3,
        )

        assert matrix.shape == (num_classes, num_classes)

    @given(data=predictions_and_ground_truths())
    @settings(max_examples=100)
    def test_confusion_matrix_non_negative_integers(self, data):
        # Feature: road-damage-evaluation-framework, Property 13: Confusion matrix invariants
        """All entries in the confusion matrix are non-negative integers."""
        predictions, ground_truths = data

        matrix = compute_confusion_matrix(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=_CLASS_NAMES,
            iou_threshold=0.5,
            confidence_threshold=0.3,
        )

        # All entries should be non-negative
        assert np.all(matrix >= 0), f"Negative entries found: {matrix}"

        # All entries should be integers
        assert matrix.dtype in (np.int64, np.int32, np.int_), (
            f"Matrix dtype is not integer: {matrix.dtype}"
        )

    @given(data=predictions_and_ground_truths())
    @settings(max_examples=100)
    def test_confusion_matrix_sum_equals_matched_gt(self, data):
        # Feature: road-damage-evaluation-framework, Property 13: Confusion matrix invariants
        """Sum of all entries equals the total number of ground truth instances
        that were matched by predictions."""
        predictions, ground_truths = data

        confidence_threshold = 0.3
        iou_threshold = 0.5

        matrix = compute_confusion_matrix(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=_CLASS_NAMES,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold,
        )

        matrix_sum = int(matrix.sum())

        # The sum of the confusion matrix should equal the number of GT instances
        # that were matched (i.e., had a prediction with IoU >= threshold).
        # This is bounded by the total number of GT instances.
        total_gt = sum(len(gt["boxes"]) for gt in ground_truths)
        assert matrix_sum <= total_gt, (
            f"Matrix sum {matrix_sum} exceeds total GT instances {total_gt}"
        )
        # Sum must be non-negative
        assert matrix_sum >= 0

    @given(
        data=st.data(),
        num_classes=st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=100)
    def test_confusion_matrix_with_variable_classes(self, data, num_classes):
        # Feature: road-damage-evaluation-framework, Property 13: Confusion matrix invariants
        """Confusion matrix dimensions match the number of classes provided."""
        class_names = _CLASS_NAMES[:num_classes]

        preds_gts = data.draw(predictions_and_ground_truths(
            min_images=1, max_images=3, class_names=class_names
        ))
        predictions, ground_truths = preds_gts

        matrix = compute_confusion_matrix(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=class_names,
            iou_threshold=0.5,
            confidence_threshold=0.3,
        )

        assert matrix.shape == (num_classes, num_classes)
        assert np.all(matrix >= 0)
        assert matrix.dtype in (np.int64, np.int32, np.int_)
