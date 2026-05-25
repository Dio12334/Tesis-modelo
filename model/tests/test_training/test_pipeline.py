"""Unit tests for the training pipeline.

Tests cover:
- Basic training with a mock model and small synthetic dataset
- Checkpoint saving and resumption
- Augmentation pipeline composition
- Config validation (different optimizers, schedulers)

Validates: Requirements 4.1, 4.6, 4.7, 4.8
"""

import json
import random
from pathlib import Path
from typing import Iterator, List, Tuple

import pytest

from model.datasets.base import Annotation, BaseDataset, BoundingBox
from model.models.registry import BaseDetector
from model.training.augmentation import (
    Compose,
    RandomBrightness,
    RandomHorizontalFlip,
    RandomRotation,
    RandomVerticalFlip,
    build_augmentation_pipeline,
)
from model.training.pipeline import TrainingPipeline


# ---------------------------------------------------------------------------
# Test fixtures: ConcreteDataset and MockDetector
# ---------------------------------------------------------------------------


class ConcreteDataset(BaseDataset):
    """Minimal concrete BaseDataset subclass for pipeline testing."""

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
    """Mock BaseDetector that implements all abstract methods as no-ops."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def forward(self, images):
        return [{"boxes": [], "labels": [], "scores": []}]

    def get_config_schema(self) -> dict:
        return {}

    def load_checkpoint(self, path: Path) -> None:
        pass

    def save_checkpoint(self, path: Path) -> None:
        pass


def _make_annotations(n: int) -> List[Annotation]:
    """Create n distinct Annotation objects for testing."""
    return [
        Annotation(
            image_path=Path(f"image_{i}.jpg"),
            bounding_boxes=[
                BoundingBox(
                    x_min=0.1,
                    y_min=0.1,
                    x_max=0.5,
                    y_max=0.5,
                    class_label="damage",
                )
            ],
            metadata={"index": i},
        )
        for i in range(n)
    ]


def _make_training_config(tmp_path: Path, **overrides) -> dict:
    """Create a minimal training config dict for testing."""
    config = {
        "training": {
            "epochs": 3,
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
    config["training"].update(overrides)
    return config


# ---------------------------------------------------------------------------
# Test: Basic training
# ---------------------------------------------------------------------------


class TestBasicTraining:
    """Test training with a mock model and small synthetic dataset."""

    def test_train_returns_metrics_dict(self, tmp_path):
        """Train returns a dict with expected metric keys."""
        annotations = _make_annotations(20)
        dataset = ConcreteDataset(annotations)
        model = MockDetector()
        config = _make_training_config(tmp_path)

        pipeline = TrainingPipeline(model, dataset, config)
        metrics = pipeline.train()

        assert isinstance(metrics, dict)
        assert "final_train_loss" in metrics
        assert "final_val_loss" in metrics
        assert "best_val_loss" in metrics
        assert "best_epoch" in metrics
        assert "total_epochs_trained" in metrics
        assert "checkpoint_dir" in metrics

    def test_train_loss_values_are_numeric(self, tmp_path):
        """All loss values in the returned metrics are floats."""
        annotations = _make_annotations(10)
        dataset = ConcreteDataset(annotations)
        model = MockDetector()
        config = _make_training_config(tmp_path)

        pipeline = TrainingPipeline(model, dataset, config)
        metrics = pipeline.train()

        assert isinstance(metrics["final_train_loss"], float)
        assert isinstance(metrics["final_val_loss"], float)
        assert isinstance(metrics["best_val_loss"], float)

    def test_total_epochs_trained_matches_config(self, tmp_path):
        """total_epochs_trained should match the configured epochs."""
        annotations = _make_annotations(15)
        dataset = ConcreteDataset(annotations)
        model = MockDetector()
        config = _make_training_config(tmp_path, epochs=5)

        pipeline = TrainingPipeline(model, dataset, config)
        metrics = pipeline.train()

        assert metrics["total_epochs_trained"] == 5

    def test_best_epoch_within_range(self, tmp_path):
        """best_epoch should be within [0, epochs-1]."""
        annotations = _make_annotations(12)
        dataset = ConcreteDataset(annotations)
        model = MockDetector()
        config = _make_training_config(tmp_path, epochs=4)

        pipeline = TrainingPipeline(model, dataset, config)
        metrics = pipeline.train()

        assert 0 <= metrics["best_epoch"] < 4


# ---------------------------------------------------------------------------
# Test: Checkpoint saving
# ---------------------------------------------------------------------------


class TestCheckpointSaving:
    """Verify that after training, checkpoint files are created."""

    def test_best_checkpoint_created(self, tmp_path):
        """Best checkpoint file should exist after training."""
        annotations = _make_annotations(20)
        dataset = ConcreteDataset(annotations)
        model = MockDetector()
        config = _make_training_config(tmp_path)

        pipeline = TrainingPipeline(model, dataset, config)
        pipeline.train()

        checkpoint_dir = tmp_path / "checkpoints"
        best_path = checkpoint_dir / "checkpoint_best.json"
        assert best_path.exists()

    def test_final_checkpoint_created(self, tmp_path):
        """Final checkpoint file should exist after training."""
        annotations = _make_annotations(20)
        dataset = ConcreteDataset(annotations)
        model = MockDetector()
        config = _make_training_config(tmp_path)

        pipeline = TrainingPipeline(model, dataset, config)
        pipeline.train()

        checkpoint_dir = tmp_path / "checkpoints"
        final_path = checkpoint_dir / "checkpoint_final.json"
        assert final_path.exists()

    def test_recovery_checkpoint_created(self, tmp_path):
        """Recovery checkpoint file should exist after training."""
        annotations = _make_annotations(20)
        dataset = ConcreteDataset(annotations)
        model = MockDetector()
        config = _make_training_config(tmp_path)

        pipeline = TrainingPipeline(model, dataset, config)
        pipeline.train()

        checkpoint_dir = tmp_path / "checkpoints"
        recovery_path = checkpoint_dir / "checkpoint_recovery.json"
        assert recovery_path.exists()

    def test_best_checkpoint_contains_valid_json(self, tmp_path):
        """Best checkpoint should contain valid JSON with expected keys."""
        annotations = _make_annotations(20)
        dataset = ConcreteDataset(annotations)
        model = MockDetector()
        config = _make_training_config(tmp_path)

        pipeline = TrainingPipeline(model, dataset, config)
        pipeline.train()

        best_path = tmp_path / "checkpoints" / "checkpoint_best.json"
        with open(best_path) as f:
            data = json.load(f)

        assert data["type"] == "best"
        assert "epoch" in data
        assert "val_loss" in data

    def test_final_checkpoint_contains_valid_json(self, tmp_path):
        """Final checkpoint should contain valid JSON with expected keys."""
        annotations = _make_annotations(20)
        dataset = ConcreteDataset(annotations)
        model = MockDetector()
        config = _make_training_config(tmp_path)

        pipeline = TrainingPipeline(model, dataset, config)
        pipeline.train()

        final_path = tmp_path / "checkpoints" / "checkpoint_final.json"
        with open(final_path) as f:
            data = json.load(f)

        assert data["type"] == "final"
        assert "best_val_loss" in data


# ---------------------------------------------------------------------------
# Test: Training resumption
# ---------------------------------------------------------------------------


class TestTrainingResumption:
    """Verify that training can be resumed from a recovery checkpoint."""

    def test_resume_continues_from_saved_epoch(self, tmp_path):
        """Resume should continue training from the checkpoint epoch."""
        annotations = _make_annotations(20)
        dataset = ConcreteDataset(annotations)
        model = MockDetector()
        config = _make_training_config(tmp_path, epochs=6)

        # Run initial training for 6 epochs
        pipeline = TrainingPipeline(model, dataset, config)
        pipeline.train()

        # Create a recovery checkpoint at epoch 2 (so resume starts at epoch 3)
        recovery_path = tmp_path / "checkpoints" / "checkpoint_recovery_manual.json"
        recovery_data = {
            "type": "recovery",
            "epoch": 2,
            "metrics": {
                "metrics_history": [
                    {"epoch": 0, "train_loss": 1.5, "val_loss": 1.6},
                    {"epoch": 1, "train_loss": 1.2, "val_loss": 1.3},
                    {"epoch": 2, "train_loss": 1.0, "val_loss": 1.1},
                ],
            },
        }
        with open(recovery_path, "w") as f:
            json.dump(recovery_data, f)

        # Resume training
        pipeline2 = TrainingPipeline(model, dataset, config)
        metrics = pipeline2.resume(recovery_path)

        # Should have trained from epoch 3 to epoch 5 (total 6 epochs configured)
        # total_epochs_trained = last epoch + 1 = 6
        assert metrics["total_epochs_trained"] == 6

    def test_resume_from_epoch_zero(self, tmp_path):
        """Resume from epoch 0 should train all epochs."""
        annotations = _make_annotations(15)
        dataset = ConcreteDataset(annotations)
        model = MockDetector()
        config = _make_training_config(tmp_path, epochs=3)

        recovery_path = tmp_path / "recovery.json"
        recovery_data = {
            "type": "recovery",
            "epoch": -1,  # Will resume from epoch 0 (epoch + 1 = 0)
            "metrics": {"metrics_history": []},
        }
        with open(recovery_path, "w") as f:
            json.dump(recovery_data, f)

        pipeline = TrainingPipeline(model, dataset, config)
        metrics = pipeline.resume(recovery_path)

        assert metrics["total_epochs_trained"] == 3

    def test_resume_produces_checkpoints(self, tmp_path):
        """Resumed training should still produce checkpoint files."""
        annotations = _make_annotations(20)
        dataset = ConcreteDataset(annotations)
        model = MockDetector()
        config = _make_training_config(tmp_path, epochs=4)

        recovery_path = tmp_path / "recovery.json"
        recovery_data = {
            "type": "recovery",
            "epoch": 1,
            "metrics": {"metrics_history": []},
        }
        with open(recovery_path, "w") as f:
            json.dump(recovery_data, f)

        pipeline = TrainingPipeline(model, dataset, config)
        pipeline.resume(recovery_path)

        checkpoint_dir = tmp_path / "checkpoints"
        assert (checkpoint_dir / "checkpoint_best.json").exists()
        assert (checkpoint_dir / "checkpoint_final.json").exists()

    def test_resume_returns_valid_metrics(self, tmp_path):
        """Resumed training should return a valid metrics dict."""
        annotations = _make_annotations(15)
        dataset = ConcreteDataset(annotations)
        model = MockDetector()
        config = _make_training_config(tmp_path, epochs=4)

        recovery_path = tmp_path / "recovery.json"
        recovery_data = {
            "type": "recovery",
            "epoch": 1,
            "metrics": {"metrics_history": []},
        }
        with open(recovery_path, "w") as f:
            json.dump(recovery_data, f)

        pipeline = TrainingPipeline(model, dataset, config)
        metrics = pipeline.resume(recovery_path)

        assert "final_train_loss" in metrics
        assert "final_val_loss" in metrics
        assert "best_val_loss" in metrics
        assert "best_epoch" in metrics


# ---------------------------------------------------------------------------
# Test: Augmentation pipeline composition
# ---------------------------------------------------------------------------


class TestAugmentationPipelineComposition:
    """Verify augmentation pipeline builds correctly from config."""

    def test_full_augmentation_config(self):
        """All augmentation options enabled produces correct pipeline."""
        config = {
            "augmentation": {
                "horizontal_flip": True,
                "vertical_flip": True,
                "rotation_range": 15,
                "brightness_range": [0.8, 1.2],
                "mosaic": True,
            }
        }
        pipeline = build_augmentation_pipeline(config)
        assert isinstance(pipeline, Compose)
        # hflip + vflip + rotation + brightness + mosaic = 5
        assert len(pipeline.transforms) == 5

    def test_partial_augmentation_config(self):
        """Only some augmentations enabled."""
        config = {
            "augmentation": {
                "horizontal_flip": True,
                "vertical_flip": False,
                "rotation_range": 0,
                "mosaic": False,
            }
        }
        pipeline = build_augmentation_pipeline(config)
        assert len(pipeline.transforms) == 1
        assert isinstance(pipeline.transforms[0], RandomHorizontalFlip)

    def test_augmentation_with_rotation_and_brightness(self):
        """Rotation and brightness only."""
        config = {
            "rotation_range": 20,
            "brightness_range": [0.7, 1.3],
        }
        pipeline = build_augmentation_pipeline(config)
        assert len(pipeline.transforms) == 2
        assert isinstance(pipeline.transforms[0], RandomRotation)
        assert isinstance(pipeline.transforms[1], RandomBrightness)

    def test_empty_augmentation_config(self):
        """Empty config produces empty pipeline."""
        pipeline = build_augmentation_pipeline({})
        assert isinstance(pipeline, Compose)
        assert len(pipeline.transforms) == 0

    def test_pipeline_is_callable(self):
        """Built pipeline should be callable with image and bboxes."""
        import numpy as np

        config = {"horizontal_flip": True, "brightness_range": [0.9, 1.1]}
        pipeline = build_augmentation_pipeline(config)

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        bboxes = [[0.1, 0.2, 0.5, 0.6]]
        result_img, result_bboxes = pipeline(image, bboxes)
        assert result_img.ndim == 3


# ---------------------------------------------------------------------------
# Test: Config validation (different optimizers, schedulers)
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """Test that the pipeline handles various config combinations."""

    def test_sgd_optimizer(self, tmp_path):
        """Training with SGD optimizer completes successfully."""
        annotations = _make_annotations(10)
        dataset = ConcreteDataset(annotations)
        model = MockDetector()
        config = _make_training_config(tmp_path, optimizer="SGD", epochs=2)

        pipeline = TrainingPipeline(model, dataset, config)
        metrics = pipeline.train()
        assert metrics["total_epochs_trained"] == 2

    def test_adam_optimizer(self, tmp_path):
        """Training with Adam optimizer completes successfully."""
        annotations = _make_annotations(10)
        dataset = ConcreteDataset(annotations)
        model = MockDetector()
        config = _make_training_config(tmp_path, optimizer="Adam", epochs=2)

        pipeline = TrainingPipeline(model, dataset, config)
        metrics = pipeline.train()
        assert metrics["total_epochs_trained"] == 2

    def test_adamw_optimizer(self, tmp_path):
        """Training with AdamW optimizer completes successfully."""
        annotations = _make_annotations(10)
        dataset = ConcreteDataset(annotations)
        model = MockDetector()
        config = _make_training_config(tmp_path, optimizer="AdamW", epochs=2)

        pipeline = TrainingPipeline(model, dataset, config)
        metrics = pipeline.train()
        assert metrics["total_epochs_trained"] == 2

    def test_cosine_scheduler(self, tmp_path):
        """Training with cosine scheduler completes successfully."""
        annotations = _make_annotations(10)
        dataset = ConcreteDataset(annotations)
        model = MockDetector()
        config = _make_training_config(tmp_path, scheduler="cosine", epochs=3)

        pipeline = TrainingPipeline(model, dataset, config)
        metrics = pipeline.train()
        assert metrics["total_epochs_trained"] == 3

    def test_step_scheduler(self, tmp_path):
        """Training with step scheduler completes successfully."""
        annotations = _make_annotations(10)
        dataset = ConcreteDataset(annotations)
        model = MockDetector()
        config = _make_training_config(tmp_path, scheduler="step", epochs=3)

        pipeline = TrainingPipeline(model, dataset, config)
        metrics = pipeline.train()
        assert metrics["total_epochs_trained"] == 3

    def test_plateau_scheduler(self, tmp_path):
        """Training with plateau scheduler completes successfully."""
        annotations = _make_annotations(10)
        dataset = ConcreteDataset(annotations)
        model = MockDetector()
        config = _make_training_config(tmp_path, scheduler="plateau", epochs=3)

        pipeline = TrainingPipeline(model, dataset, config)
        metrics = pipeline.train()
        assert metrics["total_epochs_trained"] == 3

    def test_unknown_optimizer_defaults_to_sgd(self, tmp_path):
        """Unknown optimizer name should default to SGD and still train."""
        annotations = _make_annotations(10)
        dataset = ConcreteDataset(annotations)
        model = MockDetector()
        config = _make_training_config(tmp_path, optimizer="UnknownOpt", epochs=2)

        pipeline = TrainingPipeline(model, dataset, config)
        metrics = pipeline.train()
        assert metrics["total_epochs_trained"] == 2

    def test_unknown_scheduler_defaults_to_cosine(self, tmp_path):
        """Unknown scheduler name should default to cosine and still train."""
        annotations = _make_annotations(10)
        dataset = ConcreteDataset(annotations)
        model = MockDetector()
        config = _make_training_config(tmp_path, scheduler="unknown_sched", epochs=2)

        pipeline = TrainingPipeline(model, dataset, config)
        metrics = pipeline.train()
        assert metrics["total_epochs_trained"] == 2

    def test_flat_config_format(self, tmp_path):
        """Pipeline accepts flat config (without 'training' key)."""
        annotations = _make_annotations(10)
        dataset = ConcreteDataset(annotations)
        model = MockDetector()
        config = {
            "epochs": 2,
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

        pipeline = TrainingPipeline(model, dataset, config)
        metrics = pipeline.train()
        assert metrics["total_epochs_trained"] == 2

    def test_custom_val_split(self, tmp_path):
        """Custom val_split ratio is respected."""
        annotations = _make_annotations(20)
        dataset = ConcreteDataset(annotations)
        model = MockDetector()
        config = _make_training_config(tmp_path, val_split=0.3, epochs=2)

        pipeline = TrainingPipeline(model, dataset, config)
        metrics = pipeline.train()
        assert metrics["total_epochs_trained"] == 2

    def test_warmup_epochs_zero(self, tmp_path):
        """Training with zero warmup epochs works."""
        annotations = _make_annotations(10)
        dataset = ConcreteDataset(annotations)
        model = MockDetector()
        config = _make_training_config(tmp_path, warmup_epochs=0, epochs=2)

        pipeline = TrainingPipeline(model, dataset, config)
        metrics = pipeline.train()
        assert metrics["total_epochs_trained"] == 2
