"""Unit tests for the EvaluationEngine."""

import numpy as np
import pytest

from model.evaluation.engine import EvaluationEngine
from model.evaluation.report import EvaluationReport


class TestEvaluationEngine:
    """Tests for EvaluationEngine.evaluate method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = EvaluationEngine()

    def test_evaluate_with_precomputed_data(self):
        """Test evaluation with pre-computed predictions and ground truths."""
        predictions = [
            {
                "image_id": "img1",
                "boxes": [[0.1, 0.1, 0.5, 0.5]],
                "labels": ["crack"],
                "scores": [0.9],
            },
            {
                "image_id": "img2",
                "boxes": [[0.2, 0.2, 0.6, 0.6]],
                "labels": ["pothole"],
                "scores": [0.8],
            },
        ]
        ground_truths = [
            {
                "image_id": "img1",
                "boxes": [[0.1, 0.1, 0.5, 0.5]],
                "labels": ["crack"],
            },
            {
                "image_id": "img2",
                "boxes": [[0.2, 0.2, 0.6, 0.6]],
                "labels": ["pothole"],
            },
        ]

        report = self.engine.evaluate(
            precomputed_predictions=predictions,
            precomputed_ground_truths=ground_truths,
            confidence_threshold=0.5,
            model_id="test_model",
        )

        assert isinstance(report, EvaluationReport)
        assert report.model_id == "test_model"
        assert 0.0 <= report.map_50 <= 1.0
        assert 0.0 <= report.map_50_95 <= 1.0
        assert 0.0 <= report.precision <= 1.0
        assert 0.0 <= report.recall <= 1.0
        assert 0.0 <= report.f1_score <= 1.0
        assert report.class_names == ["crack", "pothole"]
        # (N+1)x(N+1): includes background row/col for FP/missed detections
        assert report.confusion_matrix.shape == (3, 3)
        assert report.timestamp  # Non-empty timestamp

    def test_evaluate_perfect_predictions(self):
        """Test evaluation with perfect predictions yields high metrics."""
        predictions = [
            {
                "image_id": "img1",
                "boxes": [[0.1, 0.1, 0.5, 0.5]],
                "labels": ["crack"],
                "scores": [0.95],
            },
        ]
        ground_truths = [
            {
                "image_id": "img1",
                "boxes": [[0.1, 0.1, 0.5, 0.5]],
                "labels": ["crack"],
            },
        ]

        report = self.engine.evaluate(
            precomputed_predictions=predictions,
            precomputed_ground_truths=ground_truths,
            confidence_threshold=0.5,
            model_id="perfect_model",
        )

        assert report.map_50 == 1.0
        assert report.precision == 1.0
        assert report.recall == 1.0
        assert report.f1_score == 1.0

    def test_evaluate_no_predictions(self):
        """Test evaluation with no predictions yields zero metrics."""
        predictions = [
            {
                "image_id": "img1",
                "boxes": [],
                "labels": [],
                "scores": [],
            },
        ]
        ground_truths = [
            {
                "image_id": "img1",
                "boxes": [[0.1, 0.1, 0.5, 0.5]],
                "labels": ["crack"],
            },
        ]

        report = self.engine.evaluate(
            precomputed_predictions=predictions,
            precomputed_ground_truths=ground_truths,
            confidence_threshold=0.5,
            model_id="empty_model",
        )

        assert report.map_50 == 0.0
        assert report.precision == 0.0
        assert report.recall == 0.0
        assert report.f1_score == 0.0

    def test_evaluate_class_filtering(self):
        """Test that target_classes filters predictions and ground truths."""
        predictions = [
            {
                "image_id": "img1",
                "boxes": [[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]],
                "labels": ["crack", "pothole"],
                "scores": [0.9, 0.8],
            },
        ]
        ground_truths = [
            {
                "image_id": "img1",
                "boxes": [[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]],
                "labels": ["crack", "pothole"],
            },
        ]

        # Evaluate only "crack" class
        report = self.engine.evaluate(
            precomputed_predictions=predictions,
            precomputed_ground_truths=ground_truths,
            confidence_threshold=0.5,
            target_classes=["crack"],
            model_id="filtered_model",
        )

        assert report.class_names == ["crack"]
        assert "crack" in report.per_class_ap
        assert "pothole" not in report.per_class_ap
        # (N+1)x(N+1): includes background row/col for FP/missed detections
        assert report.confusion_matrix.shape == (2, 2)

    def test_evaluate_class_filtering_excludes_unlisted(self):
        """Test that unlisted classes don't affect metrics."""
        # Predictions include a wrong class that would lower metrics
        predictions = [
            {
                "image_id": "img1",
                "boxes": [[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.7, 0.7]],
                "labels": ["crack", "noise_class"],
                "scores": [0.9, 0.9],
            },
        ]
        ground_truths = [
            {
                "image_id": "img1",
                "boxes": [[0.1, 0.1, 0.5, 0.5]],
                "labels": ["crack"],
            },
        ]

        # Evaluate only "crack" - noise_class should be excluded
        report = self.engine.evaluate(
            precomputed_predictions=predictions,
            precomputed_ground_truths=ground_truths,
            confidence_threshold=0.5,
            target_classes=["crack"],
            model_id="filtered_model",
        )

        # With filtering, only the correct crack prediction remains
        assert report.precision == 1.0
        assert report.recall == 1.0

    def test_evaluate_raises_without_model_or_predictions(self):
        """Test that ValueError is raised when no model or predictions provided."""
        with pytest.raises(ValueError, match="model.*precomputed_predictions"):
            self.engine.evaluate(
                dataset="dummy",
                precomputed_ground_truths=[],
            )

    def test_evaluate_raises_without_dataset_or_ground_truths(self):
        """Test that ValueError is raised when no dataset or ground truths provided."""
        with pytest.raises(ValueError, match="dataset.*precomputed_ground_truths"):
            self.engine.evaluate(
                model="dummy",
                precomputed_predictions=[],
            )

    def test_evaluate_report_config(self):
        """Test that the report contains the evaluation config."""
        predictions = [
            {
                "image_id": "img1",
                "boxes": [[0.1, 0.1, 0.5, 0.5]],
                "labels": ["crack"],
                "scores": [0.9],
            },
        ]
        ground_truths = [
            {
                "image_id": "img1",
                "boxes": [[0.1, 0.1, 0.5, 0.5]],
                "labels": ["crack"],
            },
        ]

        report = self.engine.evaluate(
            precomputed_predictions=predictions,
            precomputed_ground_truths=ground_truths,
            confidence_threshold=0.7,
            target_classes=["crack"],
            model_id="config_test",
        )

        assert report.config["confidence_threshold"] == 0.7
        assert report.config["target_classes"] == ["crack"]

    def test_evaluate_model_id_defaults(self):
        """Test model_id defaults to 'unknown' when using precomputed data."""
        predictions = [
            {
                "image_id": "img1",
                "boxes": [],
                "labels": [],
                "scores": [],
            },
        ]
        ground_truths = [
            {
                "image_id": "img1",
                "boxes": [],
                "labels": [],
            },
        ]

        report = self.engine.evaluate(
            precomputed_predictions=predictions,
            precomputed_ground_truths=ground_truths,
        )

        assert report.model_id == "unknown"

    def test_evaluate_multiple_classes(self):
        """Test evaluation with multiple classes computes per-class AP."""
        predictions = [
            {
                "image_id": "img1",
                "boxes": [[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]],
                "labels": ["crack", "pothole"],
                "scores": [0.9, 0.85],
            },
            {
                "image_id": "img2",
                "boxes": [[0.0, 0.0, 0.4, 0.4]],
                "labels": ["crack"],
                "scores": [0.7],
            },
        ]
        ground_truths = [
            {
                "image_id": "img1",
                "boxes": [[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]],
                "labels": ["crack", "pothole"],
            },
            {
                "image_id": "img2",
                "boxes": [[0.0, 0.0, 0.4, 0.4]],
                "labels": ["crack"],
            },
        ]

        report = self.engine.evaluate(
            precomputed_predictions=predictions,
            precomputed_ground_truths=ground_truths,
            confidence_threshold=0.5,
            model_id="multi_class",
        )

        assert "crack" in report.per_class_ap
        assert "pothole" in report.per_class_ap
        assert len(report.class_names) == 2
