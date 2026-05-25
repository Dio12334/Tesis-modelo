"""Unit tests for YOLOv6Detector wrapper."""

from pathlib import Path

import pytest

from model.exceptions import ConfigurationError
from model.models.registry import ModelRegistry
from model.models.yolov6_wrapper import VALID_BACKBONE_SIZES, YOLOv6Detector

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the model registry before each test, ensuring yolov6 is registered."""
    # Save existing registrations
    saved = dict(ModelRegistry._models)
    # Ensure yolov6 is registered for this test module
    ModelRegistry._models["yolov6"] = YOLOv6Detector
    yield
    ModelRegistry._models = saved


class TestYOLOv6DetectorInit:
    """Tests for YOLOv6Detector initialization."""

    def test_valid_backbone_sizes(self):
        """Test that all valid backbone sizes are accepted."""
        for size in VALID_BACKBONE_SIZES:
            config = {"backbone_size": size, "num_classes": 4}
            detector = YOLOv6Detector(config)
            assert detector.backbone_size == size
            assert detector.num_classes == 4

    def test_invalid_backbone_size_raises_error(self):
        """Test that invalid backbone size raises ConfigurationError."""
        config = {"backbone_size": "tiny", "num_classes": 4}
        with pytest.raises(ConfigurationError) as exc_info:
            YOLOv6Detector(config)
        assert "tiny" in exc_info.value.violations[0]

    def test_missing_backbone_size_raises_error(self):
        """Test that missing backbone_size raises ConfigurationError."""
        config = {"num_classes": 4}
        with pytest.raises(ConfigurationError):
            YOLOv6Detector(config)

    def test_config_stored(self):
        """Test that config is stored on the instance."""
        config = {"backbone_size": "nano", "num_classes": 10}
        detector = YOLOv6Detector(config)
        assert detector.config == config


class TestYOLOv6DetectorRegistration:
    """Tests for YOLOv6Detector registry integration."""

    def test_registered_as_yolov6(self):
        """Test that YOLOv6Detector is registered as 'yolov6'."""
        assert "yolov6" in ModelRegistry.list_models()

    def test_create_via_registry(self):
        """Test creating YOLOv6Detector through the registry."""
        config = {"backbone_size": "small", "num_classes": 4}
        model = ModelRegistry.create("yolov6", config)
        assert isinstance(model, YOLOv6Detector)
        assert model.backbone_size == "small"

    def test_registry_validates_missing_params(self):
        """Test that registry catches missing required params."""
        with pytest.raises(ConfigurationError) as exc_info:
            ModelRegistry.create("yolov6", {})
        violations = exc_info.value.violations
        assert any("backbone_size" in v for v in violations)


class TestYOLOv6DetectorForward:
    """Tests for YOLOv6Detector forward pass."""

    @requires_torch
    def test_forward_returns_predictions_per_image(self):
        """Test forward returns one prediction dict per image in batch."""
        config = {"backbone_size": "nano", "num_classes": 4}
        detector = YOLOv6Detector(config)

        batch = torch.randn(3, 3, 640, 640)
        predictions = detector.forward(batch)

        assert len(predictions) == 3

    @requires_torch
    def test_forward_prediction_structure(self):
        """Test that each prediction has boxes, labels, scores keys."""
        config = {"backbone_size": "nano", "num_classes": 4}
        detector = YOLOv6Detector(config)

        batch = torch.randn(1, 3, 640, 640)
        predictions = detector.forward(batch)

        pred = predictions[0]
        assert "boxes" in pred
        assert "labels" in pred
        assert "scores" in pred

    @requires_torch
    def test_forward_empty_predictions_shape(self):
        """Test that placeholder returns empty tensors with correct shapes."""
        config = {"backbone_size": "large", "num_classes": 4}
        detector = YOLOv6Detector(config)

        batch = torch.randn(2, 3, 320, 320)
        predictions = detector.forward(batch)

        for pred in predictions:
            assert pred["boxes"].shape == (0, 4)
            assert pred["labels"].shape == (0,)
            assert pred["scores"].shape == (0,)


class TestYOLOv6DetectorCheckpoint:
    """Tests for YOLOv6Detector checkpoint save/load."""

    @requires_torch
    def test_save_and_load_checkpoint(self, tmp_path):
        """Test saving and loading a checkpoint."""
        config = {"backbone_size": "nano", "num_classes": 4}
        detector = YOLOv6Detector(config)
        detector._state_dict = {"layer1.weight": torch.randn(10, 10)}

        checkpoint_path = tmp_path / "checkpoint.pt"
        detector.save_checkpoint(checkpoint_path)

        assert checkpoint_path.exists()

        # Load into a new detector
        detector2 = YOLOv6Detector(config)
        detector2.load_checkpoint(checkpoint_path)

        assert "layer1.weight" in detector2._state_dict
        assert torch.equal(
            detector._state_dict["layer1.weight"],
            detector2._state_dict["layer1.weight"],
        )

    @requires_torch
    def test_save_creates_parent_directories(self, tmp_path):
        """Test that save_checkpoint creates parent directories."""
        config = {"backbone_size": "nano", "num_classes": 4}
        detector = YOLOv6Detector(config)

        nested_path = tmp_path / "deep" / "nested" / "checkpoint.pt"
        detector.save_checkpoint(nested_path)

        assert nested_path.exists()

    def test_load_nonexistent_checkpoint_raises_error(self, tmp_path):
        """Test that loading a non-existent checkpoint raises FileNotFoundError."""
        config = {"backbone_size": "nano", "num_classes": 4}
        detector = YOLOv6Detector(config)

        with pytest.raises((FileNotFoundError, RuntimeError)):
            detector.load_checkpoint(tmp_path / "nonexistent.pt")


class TestYOLOv6DetectorConfigSchema:
    """Tests for YOLOv6Detector config schema."""

    def test_schema_requires_backbone_size(self):
        """Test that schema marks backbone_size as required."""
        config = {"backbone_size": "nano", "num_classes": 4}
        detector = YOLOv6Detector(config)
        schema = detector.get_config_schema()

        assert "backbone_size" in schema
        assert schema["backbone_size"]["required"] is True
        assert schema["backbone_size"]["type"] == "str"

    def test_schema_requires_num_classes(self):
        """Test that schema marks num_classes as required."""
        config = {"backbone_size": "nano", "num_classes": 4}
        detector = YOLOv6Detector(config)
        schema = detector.get_config_schema()

        assert "num_classes" in schema
        assert schema["num_classes"]["required"] is True
        assert schema["num_classes"]["type"] == "int"
