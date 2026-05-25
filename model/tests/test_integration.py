"""Integration tests for end-to-end workflows.

Tests exercise the full pipeline end-to-end:
1. Training workflow: mock model + synthetic dataset → TrainingPipeline.train()
2. Evaluation workflow: pre-computed predictions/ground truths → EvaluationEngine.evaluate()
3. Inference workflow: mock model returning predictions → filtering + NMS

Validates: Requirements 4.1, 5.4, 8.2
"""

import random
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import pytest

from model.datasets.base import Annotation, BaseDataset, BoundingBox
from model.evaluation.engine import EvaluationEngine
from model.evaluation.report import EvaluationReport
from model.inference.pipeline import (
    InferencePipeline,
    apply_nms_to_predictions,
    filter_by_confidence,
)
from model.models.registry import BaseDetector
from model.training.pipeline import TrainingPipeline


# ---------------------------------------------------------------------------
# Test helpers: ConcreteDataset and MockDetector
# ---------------------------------------------------------------------------


class ConcreteDataset(BaseDataset):
    """Minimal concrete BaseDataset subclass for integration testing."""

    def __init__(self, annotations: List[Annotation] | None = None):
        self._annotations: List[Annotation] = annotations or []

    def load(self, path: Path) -> None:
        pass

    def get_annotations(self) -> List[Annotation]:
        return list(self._annotations)

    def split(
        self,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int = 42,
    ) -> Tuple["ConcreteDataset", "ConcreteDataset", "ConcreteDataset"]:
        annotations = list(self._annotations)
        n = len(annotations)
        rng = random.Random(seed)
        rng.shuffle(annotations)

        train_end = int(round(n * train_ratio))
        val_end = train_end + int(round(n * val_ratio))

        return (
            ConcreteDataset(annotations[:train_end]),
            ConcreteDataset(annotations[train_end:val_end]),
            ConcreteDataset(annotations[val_end:]),
        )

    def __iter__(self) -> Iterator[Annotation]:
        return iter(self._annotations)

    def __len__(self) -> int:
        return len(self._annotations)

    def get_class_names(self) -> List[str]:
        classes = set()
        for ann in self._annotations:
            for bb in ann.bounding_boxes:
                classes.add(bb.class_label)
        return sorted(classes)


class MockDetector(BaseDetector):
    """Mock BaseDetector for integration testing."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def forward(self, images):
        """Return mock predictions with bounding boxes."""
        return [
            {
                "boxes": [
                    [0.1, 0.2, 0.4, 0.5],
                    [0.3, 0.3, 0.6, 0.7],
                    [0.5, 0.1, 0.8, 0.4],
                ],
                "labels": ["bache", "fisura_longitudinal", "bache"],
                "scores": [0.9, 0.75, 0.4],
            }
        ]

    def get_config_schema(self) -> dict:
        return {}

    def load_checkpoint(self, path: Path) -> None:
        pass

    def save_checkpoint(self, path: Path) -> None:
        pass


def _make_synthetic_dataset(n: int = 30) -> ConcreteDataset:
    """Create a synthetic dataset with n annotations across multiple classes."""
    classes = ["bache", "fisura_longitudinal", "fisura_transversal", "piel_de_cocodrilo"]
    annotations = []
    for i in range(n):
        num_boxes = random.randint(1, 3)
        boxes = []
        for _ in range(num_boxes):
            x_min = random.uniform(0.0, 0.6)
            y_min = random.uniform(0.0, 0.6)
            boxes.append(
                BoundingBox(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_min + random.uniform(0.1, 0.3),
                    y_max=y_min + random.uniform(0.1, 0.3),
                    class_label=random.choice(classes),
                    confidence=1.0,
                )
            )
        annotations.append(
            Annotation(
                image_path=Path(f"image_{i:04d}.jpg"),
                bounding_boxes=boxes,
                metadata={"index": i},
            )
        )
    return ConcreteDataset(annotations)


# ---------------------------------------------------------------------------
# Integration Test: Training Workflow
# ---------------------------------------------------------------------------


class TestTrainingWorkflowIntegration:
    """End-to-end training workflow: model + dataset → pipeline → metrics + checkpoints.

    Validates: Requirement 4.1
    """

    def test_full_training_workflow(self, tmp_path):
        """Complete training workflow produces valid metrics and checkpoints."""
        # Setup
        dataset = _make_synthetic_dataset(30)
        model = MockDetector()
        config = {
            "training": {
                "epochs": 5,
                "batch_size": 8,
                "learning_rate": 0.01,
                "optimizer": "SGD",
                "weight_decay": 0.0005,
                "momentum": 0.9,
                "scheduler": "cosine",
                "warmup_epochs": 1,
                "val_split": 0.2,
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "log_interval": 5,
            }
        }

        # Execute
        pipeline = TrainingPipeline(model, dataset, config)
        metrics = pipeline.train()

        # Verify metrics structure
        assert isinstance(metrics, dict)
        assert "final_train_loss" in metrics
        assert "final_val_loss" in metrics
        assert "best_val_loss" in metrics
        assert "best_epoch" in metrics
        assert "total_epochs_trained" in metrics
        assert "checkpoint_dir" in metrics

        # Verify metric values are valid
        assert isinstance(metrics["final_train_loss"], float)
        assert metrics["final_train_loss"] >= 0.0
        assert isinstance(metrics["final_val_loss"], float)
        assert metrics["final_val_loss"] >= 0.0
        assert isinstance(metrics["best_val_loss"], float)
        assert metrics["best_val_loss"] >= 0.0
        assert metrics["total_epochs_trained"] == 5
        assert 0 <= metrics["best_epoch"] < 5

        # Verify checkpoints were created
        checkpoint_dir = tmp_path / "checkpoints"
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "checkpoint_best.json").exists()
        assert (checkpoint_dir / "checkpoint_final.json").exists()

    def test_training_with_different_optimizers(self, tmp_path):
        """Training completes successfully with all supported optimizers."""
        dataset = _make_synthetic_dataset(20)
        model = MockDetector()

        for optimizer in ["SGD", "Adam", "AdamW"]:
            config = {
                "training": {
                    "epochs": 2,
                    "batch_size": 4,
                    "learning_rate": 0.01,
                    "optimizer": optimizer,
                    "weight_decay": 0.0005,
                    "momentum": 0.9,
                    "scheduler": "cosine",
                    "warmup_epochs": 0,
                    "val_split": 0.2,
                    "checkpoint_dir": str(tmp_path / f"checkpoints_{optimizer}"),
                    "log_interval": 10,
                }
            }

            pipeline = TrainingPipeline(model, dataset, config)
            metrics = pipeline.train()
            assert metrics["total_epochs_trained"] == 2

    def test_training_resume_workflow(self, tmp_path):
        """Training can be resumed from a checkpoint and completes correctly."""
        import json

        dataset = _make_synthetic_dataset(20)
        model = MockDetector()
        config = {
            "training": {
                "epochs": 6,
                "batch_size": 4,
                "learning_rate": 0.01,
                "optimizer": "SGD",
                "weight_decay": 0.0005,
                "momentum": 0.9,
                "scheduler": "cosine",
                "warmup_epochs": 1,
                "val_split": 0.2,
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "log_interval": 5,
            }
        }

        # Create a recovery checkpoint simulating interruption at epoch 2
        recovery_path = tmp_path / "recovery_checkpoint.json"
        recovery_data = {
            "type": "recovery",
            "epoch": 2,
            "metrics": {
                "metrics_history": [
                    {"epoch": 0, "train_loss": 1.8, "val_loss": 2.0},
                    {"epoch": 1, "train_loss": 1.5, "val_loss": 1.7},
                    {"epoch": 2, "train_loss": 1.2, "val_loss": 1.4},
                ],
            },
        }
        with open(recovery_path, "w") as f:
            json.dump(recovery_data, f)

        # Resume training
        pipeline = TrainingPipeline(model, dataset, config)
        metrics = pipeline.resume(recovery_path)

        # Should complete all 6 epochs
        assert metrics["total_epochs_trained"] == 6
        assert "final_train_loss" in metrics
        assert "best_val_loss" in metrics


# ---------------------------------------------------------------------------
# Integration Test: Evaluation Workflow
# ---------------------------------------------------------------------------


class TestEvaluationWorkflowIntegration:
    """End-to-end evaluation workflow: predictions + ground truths → report.

    Validates: Requirement 5.4
    """

    def test_full_evaluation_workflow(self):
        """Evaluation produces a valid report with all expected fields."""
        # Setup pre-computed predictions and ground truths
        class_names = ["bache", "fisura_longitudinal", "fisura_transversal"]

        ground_truths = [
            {
                "image_id": "img_001",
                "boxes": [[0.1, 0.2, 0.4, 0.5], [0.5, 0.5, 0.8, 0.9]],
                "labels": ["bache", "fisura_longitudinal"],
            },
            {
                "image_id": "img_002",
                "boxes": [[0.2, 0.1, 0.6, 0.4]],
                "labels": ["fisura_transversal"],
            },
            {
                "image_id": "img_003",
                "boxes": [[0.3, 0.3, 0.7, 0.7], [0.1, 0.1, 0.3, 0.3]],
                "labels": ["bache", "fisura_longitudinal"],
            },
        ]

        predictions = [
            {
                "image_id": "img_001",
                "boxes": [[0.1, 0.2, 0.4, 0.5], [0.5, 0.5, 0.8, 0.9]],
                "labels": ["bache", "fisura_longitudinal"],
                "scores": [0.95, 0.88],
            },
            {
                "image_id": "img_002",
                "boxes": [[0.2, 0.1, 0.6, 0.4], [0.7, 0.7, 0.9, 0.9]],
                "labels": ["fisura_transversal", "bache"],
                "scores": [0.82, 0.3],
            },
            {
                "image_id": "img_003",
                "boxes": [[0.3, 0.3, 0.7, 0.7]],
                "labels": ["bache"],
                "scores": [0.91],
            },
        ]

        # Execute
        engine = EvaluationEngine()
        report = engine.evaluate(
            precomputed_predictions=predictions,
            precomputed_ground_truths=ground_truths,
            confidence_threshold=0.5,
            model_id="test_model_v1",
        )

        # Verify report type and structure
        assert isinstance(report, EvaluationReport)
        assert report.model_id == "test_model_v1"
        assert report.timestamp is not None and len(report.timestamp) > 0

        # Verify metric bounds
        assert 0.0 <= report.map_50 <= 1.0
        assert 0.0 <= report.map_50_95 <= 1.0
        assert 0.0 <= report.precision <= 1.0
        assert 0.0 <= report.recall <= 1.0
        assert 0.0 <= report.f1_score <= 1.0

        # Verify per-class AP
        assert isinstance(report.per_class_ap, dict)
        for class_name, ap in report.per_class_ap.items():
            assert 0.0 <= ap <= 1.0

        # Verify confusion matrix
        assert isinstance(report.confusion_matrix, np.ndarray)
        assert report.confusion_matrix.ndim == 2
        num_classes = len(report.class_names)
        assert report.confusion_matrix.shape[0] == num_classes
        assert report.confusion_matrix.shape[1] == num_classes
        assert np.all(report.confusion_matrix >= 0)

        # Verify class names
        assert isinstance(report.class_names, list)
        assert len(report.class_names) > 0

        # Verify config
        assert isinstance(report.config, dict)

    def test_evaluation_with_class_filtering(self):
        """Evaluation with target_classes only reports on specified classes."""
        ground_truths = [
            {
                "image_id": "img_001",
                "boxes": [[0.1, 0.2, 0.4, 0.5], [0.5, 0.5, 0.8, 0.9]],
                "labels": ["bache", "fisura_longitudinal"],
            },
            {
                "image_id": "img_002",
                "boxes": [[0.2, 0.1, 0.6, 0.4]],
                "labels": ["fisura_transversal"],
            },
        ]

        predictions = [
            {
                "image_id": "img_001",
                "boxes": [[0.1, 0.2, 0.4, 0.5], [0.5, 0.5, 0.8, 0.9]],
                "labels": ["bache", "fisura_longitudinal"],
                "scores": [0.9, 0.85],
            },
            {
                "image_id": "img_002",
                "boxes": [[0.2, 0.1, 0.6, 0.4]],
                "labels": ["fisura_transversal"],
                "scores": [0.8],
            },
        ]

        engine = EvaluationEngine()
        report = engine.evaluate(
            precomputed_predictions=predictions,
            precomputed_ground_truths=ground_truths,
            target_classes=["bache"],
            model_id="filtered_model",
        )

        # Only the filtered class should appear
        assert report.class_names == ["bache"]
        assert "bache" in report.per_class_ap
        assert report.confusion_matrix.shape == (1, 1)

    def test_evaluation_report_serialization(self, tmp_path):
        """Evaluation report can be serialized to JSON and loaded back."""
        ground_truths = [
            {
                "image_id": "img_001",
                "boxes": [[0.1, 0.2, 0.5, 0.6]],
                "labels": ["bache"],
            },
        ]
        predictions = [
            {
                "image_id": "img_001",
                "boxes": [[0.1, 0.2, 0.5, 0.6]],
                "labels": ["bache"],
                "scores": [0.9],
            },
        ]

        engine = EvaluationEngine()
        report = engine.evaluate(
            precomputed_predictions=predictions,
            precomputed_ground_truths=ground_truths,
            model_id="serialization_test",
        )

        # Save and reload
        report_path = tmp_path / "report.json"
        report.save(report_path)
        loaded_report = EvaluationReport.load(report_path)

        # Verify round-trip
        assert loaded_report.model_id == report.model_id
        assert loaded_report.timestamp == report.timestamp
        assert loaded_report.map_50 == report.map_50
        assert loaded_report.map_50_95 == report.map_50_95
        assert loaded_report.precision == report.precision
        assert loaded_report.recall == report.recall
        assert loaded_report.f1_score == report.f1_score
        assert loaded_report.per_class_ap == report.per_class_ap
        assert loaded_report.class_names == report.class_names
        assert np.array_equal(loaded_report.confusion_matrix, report.confusion_matrix)

    def test_evaluation_with_no_predictions(self):
        """Evaluation handles the case where no predictions are made."""
        ground_truths = [
            {
                "image_id": "img_001",
                "boxes": [[0.1, 0.2, 0.5, 0.6]],
                "labels": ["bache"],
            },
        ]
        predictions = [
            {
                "image_id": "img_001",
                "boxes": [],
                "labels": [],
                "scores": [],
            },
        ]

        engine = EvaluationEngine()
        report = engine.evaluate(
            precomputed_predictions=predictions,
            precomputed_ground_truths=ground_truths,
            model_id="no_preds_model",
        )

        assert isinstance(report, EvaluationReport)
        assert report.map_50 == 0.0
        assert report.precision == 0.0
        assert report.recall == 0.0


# ---------------------------------------------------------------------------
# Integration Test: Inference Workflow
# ---------------------------------------------------------------------------


class TestInferenceWorkflowIntegration:
    """End-to-end inference workflow: model → confidence filter → NMS → predictions.

    Validates: Requirement 8.2
    """

    def test_full_inference_pipeline_filtering(self):
        """Inference pipeline applies confidence filtering and NMS correctly."""
        # Create predictions simulating model output
        raw_predictions = [
            BoundingBox(x_min=0.1, y_min=0.2, x_max=0.4, y_max=0.5,
                        class_label="bache", confidence=0.95),
            BoundingBox(x_min=0.12, y_min=0.21, x_max=0.41, y_max=0.51,
                        class_label="bache", confidence=0.7),
            BoundingBox(x_min=0.5, y_min=0.5, x_max=0.8, y_max=0.9,
                        class_label="fisura_longitudinal", confidence=0.85),
            BoundingBox(x_min=0.3, y_min=0.3, x_max=0.6, y_max=0.6,
                        class_label="bache", confidence=0.3),
            BoundingBox(x_min=0.0, y_min=0.0, x_max=0.2, y_max=0.2,
                        class_label="fisura_transversal", confidence=0.6),
        ]

        # Step 1: Confidence filtering at threshold 0.5
        filtered = filter_by_confidence(raw_predictions, threshold=0.5)

        # Should exclude the prediction with confidence 0.3
        assert len(filtered) == 4
        assert all(p.confidence >= 0.5 for p in filtered)

        # Step 2: Apply NMS with IoU threshold 0.5
        final_predictions = apply_nms_to_predictions(filtered, iou_threshold=0.5)

        # The two overlapping "bache" boxes should be reduced to one
        bache_preds = [p for p in final_predictions if p.class_label == "bache"]
        assert len(bache_preds) >= 1
        # The highest confidence bache prediction should be kept
        assert any(p.confidence == 0.95 for p in bache_preds)

        # Non-overlapping predictions from other classes should remain
        other_preds = [p for p in final_predictions if p.class_label != "bache"]
        assert len(other_preds) >= 1

    def test_inference_pipeline_with_mock_model(self, tmp_path):
        """InferencePipeline with mock model produces predictions on a real image file."""
        # Create a minimal image file (just needs to exist for the pipeline)
        # The mock model returns fixed predictions regardless of input
        image_path = tmp_path / "test_image.jpg"

        # Create a minimal valid JPEG-like file (PIL can open it)
        try:
            from PIL import Image
            img = Image.new("RGB", (100, 100), color=(128, 128, 128))
            img.save(image_path)
        except ImportError:
            # If PIL not available, create a dummy file and skip the full test
            image_path.write_bytes(b"\x00" * 100)
            pytest.skip("PIL not available for creating test image")

        # Create pipeline with mock model
        model = MockDetector()
        pipeline = InferencePipeline(
            model=model,
            confidence_threshold=0.5,
            nms_iou_threshold=0.5,
            batch_size=1,
        )

        # Run inference
        predictions = pipeline.predict_image(image_path)

        # Verify predictions are returned
        assert isinstance(predictions, list)
        # All predictions should have confidence >= threshold
        for pred in predictions:
            assert isinstance(pred, BoundingBox)
            assert pred.confidence >= 0.5

    def test_inference_confidence_threshold_variations(self):
        """Different confidence thresholds produce different filtering results."""
        predictions = [
            BoundingBox(x_min=0.1, y_min=0.1, x_max=0.3, y_max=0.3,
                        class_label="bache", confidence=0.9),
            BoundingBox(x_min=0.4, y_min=0.4, x_max=0.6, y_max=0.6,
                        class_label="bache", confidence=0.6),
            BoundingBox(x_min=0.7, y_min=0.7, x_max=0.9, y_max=0.9,
                        class_label="bache", confidence=0.3),
        ]

        # High threshold: only highest confidence kept
        high_filtered = filter_by_confidence(predictions, threshold=0.8)
        assert len(high_filtered) == 1
        assert high_filtered[0].confidence == 0.9

        # Medium threshold: two kept
        med_filtered = filter_by_confidence(predictions, threshold=0.5)
        assert len(med_filtered) == 2

        # Low threshold: all kept
        low_filtered = filter_by_confidence(predictions, threshold=0.1)
        assert len(low_filtered) == 3

    def test_inference_nms_removes_overlapping_boxes(self):
        """NMS correctly removes highly overlapping boxes of the same class."""
        # Create overlapping boxes for the same class
        predictions = [
            BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5,
                        class_label="bache", confidence=0.9),
            BoundingBox(x_min=0.12, y_min=0.12, x_max=0.52, y_max=0.52,
                        class_label="bache", confidence=0.8),
            BoundingBox(x_min=0.11, y_min=0.11, x_max=0.51, y_max=0.51,
                        class_label="bache", confidence=0.7),
            # Non-overlapping box of same class
            BoundingBox(x_min=0.7, y_min=0.7, x_max=0.9, y_max=0.9,
                        class_label="bache", confidence=0.85),
            # Different class - should not be affected by same-class NMS
            BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5,
                        class_label="fisura_longitudinal", confidence=0.75),
        ]

        result = apply_nms_to_predictions(predictions, iou_threshold=0.5)

        # The three overlapping bache boxes should be reduced
        bache_results = [p for p in result if p.class_label == "bache"]
        # At minimum, the highest confidence overlapping box and the non-overlapping one
        assert len(bache_results) == 2
        assert any(p.confidence == 0.9 for p in bache_results)
        assert any(p.confidence == 0.85 for p in bache_results)

        # The fisura_longitudinal box should remain untouched
        fisura_results = [p for p in result if p.class_label == "fisura_longitudinal"]
        assert len(fisura_results) == 1
        assert fisura_results[0].confidence == 0.75

    def test_inference_empty_predictions(self):
        """Pipeline handles empty prediction lists gracefully."""
        filtered = filter_by_confidence([], threshold=0.5)
        assert filtered == []

        nms_result = apply_nms_to_predictions([], iou_threshold=0.5)
        assert nms_result == []

    def test_inference_directory_workflow(self, tmp_path):
        """InferencePipeline processes a directory of images."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available for creating test images")

        # Create multiple test images
        for i in range(3):
            img = Image.new("RGB", (64, 64), color=(i * 50, i * 50, i * 50))
            img.save(tmp_path / f"image_{i}.jpg")

        model = MockDetector()
        pipeline = InferencePipeline(
            model=model,
            confidence_threshold=0.5,
            nms_iou_threshold=0.5,
            batch_size=2,
        )

        results = pipeline.predict_directory(tmp_path)

        # Should have results for each image
        assert isinstance(results, dict)
        assert len(results) == 3
        for filename, preds in results.items():
            assert isinstance(preds, list)
            for pred in preds:
                assert isinstance(pred, BoundingBox)
                assert pred.confidence >= 0.5
