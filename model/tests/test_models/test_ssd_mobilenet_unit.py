"""Unit tests for SSDMobileNetV3 wrapper."""

from pathlib import Path

import pytest

from model.exceptions import ConfigurationError
from model.models.registry import ModelRegistry
from model.models.ssd_mobilenet import SSDMobileNetV3, VALID_INPUT_SIZES

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the model registry before each test, ensuring ssd_mobilenetv3 is registered."""
    saved = dict(ModelRegistry._models)
    ModelRegistry._models["ssd_mobilenetv3"] = SSDMobileNetV3
    yield
    ModelRegistry._models = saved


class TestSSDMobileNetV3Init:
    """Tests for SSDMobileNetV3 initialization."""

    def test_valid_input_sizes(self):
        """Test that all valid input sizes are accepted."""
        for size in VALID_INPUT_SIZES:
            config = {"input_size": size, "num_classes": 4}
            detector = SSDMobileNetV3(config)
            assert detector.input_size == size
            assert detector.num_classes == 4

    def test_invalid_input_size_raises_error(self):
        """Test that invalid input size raises ConfigurationError."""
        config = {"input_size": 416, "num_classes": 4}
        with pytest.raises(ConfigurationError) as exc_info:
            SSDMobileNetV3(config)
        assert "416" in exc_info.value.violations[0]

    def test_missing_input_size_raises_error(self):
        """Test that missing input_size raises ConfigurationError."""
        config = {"num_classes": 4}
        with pytest.raises(ConfigurationError):
            SSDMobileNetV3(config)

    def test_config_stored(self):
        """Test that config is stored on the instance."""
        config = {"input_size": 320, "num_classes": 10}
        detector = SSDMobileNetV3(config)
        assert detector.config == config


class TestSSDMobileNetV3Registration:
    """Tests for SSDMobileNetV3 registry integration."""

    def test_registered_as_ssd_mobilenetv3(self):
        """Test that SSDMobileNetV3 is registered as 'ssd_mobilenetv3'."""
        assert "ssd_mobilenetv3" in ModelRegistry.list_models()

    def test_create_via_registry(self):
        """Test creating SSDMobileNetV3 through the registry."""
        config = {"input_size": 320, "num_classes": 4}
        model = ModelRegistry.create("ssd_mobilenetv3", config)
        assert isinstance(model, SSDMobileNetV3)
        assert model.input_size == 320

    def test_registry_validates_missing_params(self):
        """Test that registry catches missing required params."""
        with pytest.raises(ConfigurationError) as exc_info:
            ModelRegistry.create("ssd_mobilenetv3", {})
        violations = exc_info.value.violations
        assert any("input_size" in v for v in violations)


class TestSSDMobileNetV3Forward:
    """Tests for SSDMobileNetV3 forward pass."""

    @requires_torch
    def test_forward_returns_predictions_per_image(self):
        """Test forward returns one prediction dict per image in batch."""
        config = {"input_size": 320, "num_classes": 4}
        detector = SSDMobileNetV3(config)

        batch = torch.randn(3, 3, 320, 320)
        predictions = detector.forward(batch)

        assert len(predictions) == 3

    @requires_torch
    def test_forward_prediction_structure(self):
        """Test that each prediction has boxes, labels, scores keys."""
        config = {"input_size": 640, "num_classes": 4}
        detector = SSDMobileNetV3(config)

        batch = torch.randn(1, 3, 640, 640)
        predictions = detector.forward(batch)

        pred = predictions[0]
        assert "boxes" in pred
        assert "labels" in pred
        assert "scores" in pred

    @requires_torch
    def test_forward_empty_predictions_shape(self):
        """Test that placeholder returns empty tensors with correct shapes."""
        config = {"input_size": 320, "num_classes": 4}
        detector = SSDMobileNetV3(config)

        batch = torch.randn(2, 3, 320, 320)
        predictions = detector.forward(batch)

        for pred in predictions:
            assert pred["boxes"].shape == (0, 4)
            assert pred["labels"].shape == (0,)
            assert pred["scores"].shape == (0,)


class TestSSDMobileNetV3Checkpoint:
    """Tests for SSDMobileNetV3 checkpoint save/load."""

    @requires_torch
    def test_save_and_load_checkpoint(self, tmp_path):
        """Test saving and loading a checkpoint."""
        config = {"input_size": 320, "num_classes": 4}
        detector = SSDMobileNetV3(config)
        detector._state_dict = {"layer1.weight": torch.randn(10, 10)}

        checkpoint_path = tmp_path / "checkpoint.pt"
        detector.save_checkpoint(checkpoint_path)

        assert checkpoint_path.exists()

        # Load into a new detector
        detector2 = SSDMobileNetV3(config)
        detector2.load_checkpoint(checkpoint_path)

        assert "layer1.weight" in detector2._state_dict
        assert torch.equal(
            detector._state_dict["layer1.weight"],
            detector2._state_dict["layer1.weight"],
        )

    @requires_torch
    def test_save_creates_parent_directories(self, tmp_path):
        """Test that save_checkpoint creates parent directories."""
        config = {"input_size": 320, "num_classes": 4}
        detector = SSDMobileNetV3(config)

        nested_path = tmp_path / "deep" / "nested" / "checkpoint.pt"
        detector.save_checkpoint(nested_path)

        assert nested_path.exists()

    def test_load_nonexistent_checkpoint_raises_error(self, tmp_path):
        """Test that loading a non-existent checkpoint raises FileNotFoundError."""
        config = {"input_size": 320, "num_classes": 4}
        detector = SSDMobileNetV3(config)

        with pytest.raises((FileNotFoundError, RuntimeError)):
            detector.load_checkpoint(tmp_path / "nonexistent.pt")


class TestSSDMobileNetV3ConfigSchema:
    """Tests for SSDMobileNetV3 config schema."""

    def test_schema_requires_input_size(self):
        """Test that schema marks input_size as required."""
        config = {"input_size": 320, "num_classes": 4}
        detector = SSDMobileNetV3(config)
        schema = detector.get_config_schema()

        assert "input_size" in schema
        assert schema["input_size"]["required"] is True
        assert schema["input_size"]["type"] == "int"

    def test_schema_requires_num_classes(self):
        """Test that schema marks num_classes as required."""
        config = {"input_size": 320, "num_classes": 4}
        detector = SSDMobileNetV3(config)
        schema = detector.get_config_schema()

        assert "num_classes" in schema
        assert schema["num_classes"]["required"] is True
        assert schema["num_classes"]["type"] == "int"
