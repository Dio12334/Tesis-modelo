"""Unit tests for the inference pipeline."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from model.datasets.base import BoundingBox
from model.inference.pipeline import (
    InferencePipeline,
    apply_nms,
    apply_nms_to_predictions,
    compute_iou,
    filter_by_confidence,
)


class TestComputeIoU:
    """Tests for the compute_iou function."""

    def test_identical_boxes(self):
        """Identical boxes should have IoU of 1.0."""
        box = (0.0, 0.0, 1.0, 1.0)
        assert compute_iou(box, box) == 1.0

    def test_no_overlap(self):
        """Non-overlapping boxes should have IoU of 0.0."""
        box_a = (0.0, 0.0, 0.5, 0.5)
        box_b = (0.6, 0.6, 1.0, 1.0)
        assert compute_iou(box_a, box_b) == 0.0

    def test_partial_overlap(self):
        """Partially overlapping boxes should have IoU between 0 and 1."""
        box_a = (0.0, 0.0, 0.5, 0.5)
        box_b = (0.25, 0.25, 0.75, 0.75)
        iou = compute_iou(box_a, box_b)
        assert 0.0 < iou < 1.0

    def test_one_inside_other(self):
        """A box fully inside another should have IoU = area_small / area_large."""
        box_a = (0.0, 0.0, 1.0, 1.0)
        box_b = (0.25, 0.25, 0.75, 0.75)
        iou = compute_iou(box_a, box_b)
        # Inner area = 0.25, outer area = 1.0, union = 1.0
        assert abs(iou - 0.25) < 1e-6

    def test_zero_area_box(self):
        """A zero-area box should have IoU of 0.0."""
        box_a = (0.5, 0.5, 0.5, 0.5)
        box_b = (0.0, 0.0, 1.0, 1.0)
        assert compute_iou(box_a, box_b) == 0.0


class TestFilterByConfidence:
    """Tests for the filter_by_confidence function."""

    def test_all_above_threshold(self):
        """All predictions above threshold should be kept."""
        preds = [
            BoundingBox(0.1, 0.1, 0.5, 0.5, "A", 0.9),
            BoundingBox(0.2, 0.2, 0.6, 0.6, "B", 0.8),
        ]
        result = filter_by_confidence(preds, 0.5)
        assert len(result) == 2

    def test_all_below_threshold(self):
        """All predictions below threshold should be removed."""
        preds = [
            BoundingBox(0.1, 0.1, 0.5, 0.5, "A", 0.3),
            BoundingBox(0.2, 0.2, 0.6, 0.6, "B", 0.2),
        ]
        result = filter_by_confidence(preds, 0.5)
        assert len(result) == 0

    def test_mixed_confidences(self):
        """Only predictions at or above threshold should be kept."""
        preds = [
            BoundingBox(0.1, 0.1, 0.5, 0.5, "A", 0.9),
            BoundingBox(0.2, 0.2, 0.6, 0.6, "B", 0.3),
            BoundingBox(0.3, 0.3, 0.7, 0.7, "C", 0.5),
        ]
        result = filter_by_confidence(preds, 0.5)
        assert len(result) == 2
        assert all(p.confidence >= 0.5 for p in result)

    def test_exact_threshold(self):
        """Predictions exactly at threshold should be kept."""
        preds = [BoundingBox(0.1, 0.1, 0.5, 0.5, "A", 0.5)]
        result = filter_by_confidence(preds, 0.5)
        assert len(result) == 1

    def test_empty_list(self):
        """Empty input should return empty output."""
        result = filter_by_confidence([], 0.5)
        assert result == []


class TestApplyNMS:
    """Tests for the apply_nms function."""

    def test_empty_input(self):
        """Empty input should return empty output."""
        assert apply_nms([], [], 0.5) == []

    def test_single_box(self):
        """Single box should always be kept."""
        boxes = [(0.0, 0.0, 0.5, 0.5)]
        scores = [0.9]
        result = apply_nms(boxes, scores, 0.5)
        assert result == [0]

    def test_non_overlapping_boxes(self):
        """Non-overlapping boxes should all be kept."""
        boxes = [
            (0.0, 0.0, 0.3, 0.3),
            (0.5, 0.5, 0.8, 0.8),
            (0.0, 0.5, 0.3, 0.8),
        ]
        scores = [0.9, 0.8, 0.7]
        result = apply_nms(boxes, scores, 0.5)
        assert len(result) == 3

    def test_identical_boxes_suppresses_lower(self):
        """Identical boxes should suppress lower-confidence duplicates."""
        boxes = [
            (0.0, 0.0, 0.5, 0.5),
            (0.0, 0.0, 0.5, 0.5),
        ]
        scores = [0.9, 0.7]
        result = apply_nms(boxes, scores, 0.5)
        assert len(result) == 1
        assert result[0] == 0  # Higher confidence kept

    def test_high_overlap_suppresses(self):
        """Highly overlapping boxes should be suppressed."""
        boxes = [
            (0.0, 0.0, 0.5, 0.5),
            (0.05, 0.05, 0.55, 0.55),
        ]
        scores = [0.9, 0.8]
        result = apply_nms(boxes, scores, 0.5)
        # These boxes have high IoU, lower confidence should be suppressed
        assert len(result) == 1

    def test_low_overlap_keeps_both(self):
        """Boxes with IoU below threshold should both be kept."""
        boxes = [
            (0.0, 0.0, 0.5, 0.5),
            (0.4, 0.4, 0.9, 0.9),
        ]
        scores = [0.9, 0.8]
        iou = compute_iou(boxes[0], boxes[1])
        # Use a threshold higher than the actual IoU
        result = apply_nms(boxes, scores, iou + 0.01)
        assert len(result) == 2


class TestApplyNMSToPredictions:
    """Tests for apply_nms_to_predictions."""

    def test_empty_predictions(self):
        """Empty input should return empty output."""
        result = apply_nms_to_predictions([], 0.5)
        assert result == []

    def test_different_classes_not_suppressed(self):
        """Overlapping boxes of different classes should not suppress each other."""
        preds = [
            BoundingBox(0.0, 0.0, 0.5, 0.5, "A", 0.9),
            BoundingBox(0.0, 0.0, 0.5, 0.5, "B", 0.8),
        ]
        result = apply_nms_to_predictions(preds, 0.5)
        assert len(result) == 2

    def test_same_class_overlapping_suppressed(self):
        """Overlapping boxes of same class should be suppressed."""
        preds = [
            BoundingBox(0.0, 0.0, 0.5, 0.5, "A", 0.9),
            BoundingBox(0.0, 0.0, 0.5, 0.5, "A", 0.7),
        ]
        result = apply_nms_to_predictions(preds, 0.5)
        assert len(result) == 1
        assert result[0].confidence == 0.9


class TestInferencePipeline:
    """Tests for the InferencePipeline class."""

    def test_init_stores_parameters(self):
        """Constructor should store all parameters."""
        model = MagicMock()
        pipeline = InferencePipeline(
            model=model,
            confidence_threshold=0.6,
            nms_iou_threshold=0.4,
            batch_size=16,
        )
        assert pipeline.model is model
        assert pipeline.confidence_threshold == 0.6
        assert pipeline.nms_iou_threshold == 0.4
        assert pipeline.batch_size == 16

    def test_init_default_parameters(self):
        """Constructor should use default parameters when not specified."""
        model = MagicMock()
        pipeline = InferencePipeline(model=model)
        assert pipeline.confidence_threshold == 0.5
        assert pipeline.nms_iou_threshold == 0.45
        assert pipeline.batch_size == 8

    def test_predict_image_file_not_found(self):
        """predict_image should raise FileNotFoundError for missing files."""
        model = MagicMock()
        pipeline = InferencePipeline(model=model)
        with pytest.raises(FileNotFoundError):
            pipeline.predict_image(Path("/nonexistent/image.jpg"))

    def test_predict_directory_not_found(self):
        """predict_directory should raise FileNotFoundError for missing dirs."""
        model = MagicMock()
        pipeline = InferencePipeline(model=model)
        with pytest.raises(FileNotFoundError):
            pipeline.predict_directory(Path("/nonexistent/directory"))

    def test_predict_directory_empty(self):
        """predict_directory on empty dir should return empty dict."""
        model = MagicMock()
        pipeline = InferencePipeline(model=model)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = pipeline.predict_directory(Path(tmpdir))
            assert result == {}

    def test_predict_image_with_mock_model(self):
        """predict_image should process model output correctly."""
        model = MagicMock()
        model.forward.return_value = [{
            "boxes": [[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6]],
            "labels": ["crack", "pothole"],
            "scores": [0.9, 0.3],
        }]

        pipeline = InferencePipeline(model=model, confidence_threshold=0.5)

        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = Path(f.name)

        try:
            # Mock _load_image to avoid needing a real image
            with patch.object(pipeline, '_load_image', return_value=None):
                result = pipeline.predict_image(tmp_path)

            # Only the high-confidence prediction should remain
            assert len(result) == 1
            assert result[0].class_label == "crack"
            assert result[0].confidence == 0.9
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_predict_image_applies_nms(self):
        """predict_image should apply NMS to overlapping same-class boxes."""
        model = MagicMock()
        model.forward.return_value = [{
            "boxes": [
                [0.1, 0.1, 0.5, 0.5],
                [0.1, 0.1, 0.5, 0.5],
            ],
            "labels": ["crack", "crack"],
            "scores": [0.9, 0.8],
        }]

        pipeline = InferencePipeline(
            model=model, confidence_threshold=0.5, nms_iou_threshold=0.5
        )

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = Path(f.name)

        try:
            with patch.object(pipeline, '_load_image', return_value=None):
                result = pipeline.predict_image(tmp_path)

            # NMS should suppress the lower-confidence duplicate
            assert len(result) == 1
            assert result[0].confidence == 0.9
        finally:
            tmp_path.unlink(missing_ok=True)
