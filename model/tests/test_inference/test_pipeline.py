"""Property-based tests for inference pipeline filtering and NMS.

Tests Properties 18 and 19 from the design document:
- Property 18: Confidence threshold filtering
- Property 19: NMS post-condition
"""

from itertools import combinations

from hypothesis import given, settings
from hypothesis import strategies as st

from model.datasets.base import BoundingBox
from model.inference.pipeline import apply_nms, compute_iou, filter_by_confidence


# --- Strategies ---

# Generate valid bounding boxes with random confidences
bounding_box_strategy = st.builds(
    BoundingBox,
    x_min=st.floats(min_value=0.0, max_value=0.9),
    y_min=st.floats(min_value=0.0, max_value=0.9),
    x_max=st.floats(min_value=0.1, max_value=1.0),
    y_max=st.floats(min_value=0.1, max_value=1.0),
    class_label=st.sampled_from(["bache", "fisura_longitudinal", "fisura_transversal", "piel_de_cocodrilo"]),
    confidence=st.floats(min_value=0.0, max_value=1.0),
).filter(lambda bb: bb.x_min < bb.x_max and bb.y_min < bb.y_max)

# Generate lists of bounding boxes for confidence filtering tests
predictions_strategy = st.lists(bounding_box_strategy, min_size=0, max_size=20)

# Generate confidence thresholds
threshold_strategy = st.floats(min_value=0.0, max_value=1.0)

# Generate bounding box tuples for NMS tests (same class, varying positions)
box_tuple_strategy = st.tuples(
    st.floats(min_value=0.0, max_value=0.9),
    st.floats(min_value=0.0, max_value=0.9),
    st.floats(min_value=0.1, max_value=1.0),
    st.floats(min_value=0.1, max_value=1.0),
).filter(lambda b: b[0] < b[2] and b[1] < b[3])

# Generate scores for NMS
score_strategy = st.floats(min_value=0.01, max_value=1.0)


# --- Property 18: Confidence threshold filtering ---
# Feature: road-damage-evaluation-framework, Property 18: Confidence threshold filtering


class TestConfidenceThresholdFiltering:
    """Property 18: Confidence threshold filtering.

    For any set of predictions and any confidence threshold T, filtering SHALL
    return only predictions with confidence >= T, and all predictions with
    confidence < T SHALL be excluded.

    **Validates: Requirements 8.5**
    """

    @given(predictions=predictions_strategy, threshold=threshold_strategy)
    @settings(max_examples=100)
    def test_all_returned_predictions_meet_threshold(self, predictions, threshold):
        """All predictions returned by filter_by_confidence have confidence >= threshold.

        **Validates: Requirements 8.5**
        """
        # Feature: road-damage-evaluation-framework, Property 18: Confidence threshold filtering
        result = filter_by_confidence(predictions, threshold)

        for pred in result:
            assert pred.confidence >= threshold, (
                f"Prediction with confidence {pred.confidence} should not pass "
                f"threshold {threshold}"
            )

    @given(predictions=predictions_strategy, threshold=threshold_strategy)
    @settings(max_examples=100)
    def test_all_excluded_predictions_below_threshold(self, predictions, threshold):
        """All predictions excluded by filter_by_confidence have confidence < threshold.

        **Validates: Requirements 8.5**
        """
        # Feature: road-damage-evaluation-framework, Property 18: Confidence threshold filtering
        result = filter_by_confidence(predictions, threshold)
        result_set = set(id(p) for p in result)

        for pred in predictions:
            if id(pred) not in result_set:
                assert pred.confidence < threshold, (
                    f"Prediction with confidence {pred.confidence} should not be "
                    f"excluded at threshold {threshold}"
                )

    @given(predictions=predictions_strategy, threshold=threshold_strategy)
    @settings(max_examples=100)
    def test_filter_preserves_qualifying_predictions(self, predictions, threshold):
        """Every prediction with confidence >= threshold appears in the result.

        **Validates: Requirements 8.5**
        """
        # Feature: road-damage-evaluation-framework, Property 18: Confidence threshold filtering
        result = filter_by_confidence(predictions, threshold)

        expected_count = sum(1 for p in predictions if p.confidence >= threshold)
        assert len(result) == expected_count, (
            f"Expected {expected_count} predictions with confidence >= {threshold}, "
            f"got {len(result)}"
        )


# --- Property 19: NMS post-condition ---
# Feature: road-damage-evaluation-framework, Property 19: NMS post-condition


class TestNMSPostCondition:
    """Property 19: NMS post-condition.

    For any set of bounding box predictions for the same class and any IoU
    threshold T, after applying Non-Maximum Suppression, no two remaining
    boxes SHALL have IoU > T.

    **Validates: Requirements 8.6**
    """

    @given(
        boxes=st.lists(box_tuple_strategy, min_size=0, max_size=15),
        iou_threshold=st.floats(min_value=0.1, max_value=0.9),
    )
    @settings(max_examples=100)
    def test_no_remaining_pair_exceeds_iou_threshold(self, boxes, iou_threshold):
        """After NMS, no two remaining boxes have IoU > threshold.

        **Validates: Requirements 8.6**
        """
        # Feature: road-damage-evaluation-framework, Property 19: NMS post-condition
        if not boxes:
            assert apply_nms([], [], iou_threshold) == []
            return

        # Generate scores for each box
        scores = [1.0 / (i + 1) for i in range(len(boxes))]

        keep_indices = apply_nms(boxes, scores, iou_threshold)

        # Verify post-condition: no two kept boxes have IoU > threshold
        kept_boxes = [boxes[i] for i in keep_indices]
        for i, j in combinations(range(len(kept_boxes)), 2):
            iou = compute_iou(kept_boxes[i], kept_boxes[j])
            assert iou <= iou_threshold, (
                f"NMS post-condition violated: boxes {i} and {j} have "
                f"IoU={iou:.4f} > threshold={iou_threshold:.4f}"
            )

    @given(
        boxes=st.lists(box_tuple_strategy, min_size=1, max_size=15),
        scores=st.lists(score_strategy, min_size=1, max_size=15),
        iou_threshold=st.floats(min_value=0.1, max_value=0.9),
    )
    @settings(max_examples=100)
    def test_nms_postcondition_with_random_scores(self, boxes, scores, iou_threshold):
        """NMS post-condition holds with randomly generated scores.

        **Validates: Requirements 8.6**
        """
        # Feature: road-damage-evaluation-framework, Property 19: NMS post-condition
        # Ensure boxes and scores have the same length
        min_len = min(len(boxes), len(scores))
        boxes = boxes[:min_len]
        scores = scores[:min_len]

        if not boxes:
            return

        keep_indices = apply_nms(boxes, scores, iou_threshold)

        # Verify post-condition: no two kept boxes have IoU > threshold
        kept_boxes = [boxes[i] for i in keep_indices]
        for i, j in combinations(range(len(kept_boxes)), 2):
            iou = compute_iou(kept_boxes[i], kept_boxes[j])
            assert iou <= iou_threshold, (
                f"NMS post-condition violated: boxes {i} and {j} have "
                f"IoU={iou:.4f} > threshold={iou_threshold:.4f}"
            )

    @given(
        boxes=st.lists(box_tuple_strategy, min_size=0, max_size=15),
        iou_threshold=st.floats(min_value=0.1, max_value=0.9),
    )
    @settings(max_examples=100)
    def test_nms_keeps_subset_of_input(self, boxes, iou_threshold):
        """NMS result indices are a valid subset of input indices.

        **Validates: Requirements 8.6**
        """
        # Feature: road-damage-evaluation-framework, Property 19: NMS post-condition
        if not boxes:
            assert apply_nms([], [], iou_threshold) == []
            return

        scores = [1.0 / (i + 1) for i in range(len(boxes))]
        keep_indices = apply_nms(boxes, scores, iou_threshold)

        # All kept indices must be valid
        for idx in keep_indices:
            assert 0 <= idx < len(boxes), (
                f"Invalid index {idx} returned by NMS for input of size {len(boxes)}"
            )

        # No duplicate indices
        assert len(keep_indices) == len(set(keep_indices)), (
            "NMS returned duplicate indices"
        )
