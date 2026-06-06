"""Unit tests for YOLO26Detector wrapper registration and configuration schema.

Tests cover:
- Model registration in ModelRegistry
- Configuration schema structure
- Model size to file mapping
- Module importability without ultralytics
- Default values for optional parameters
- Checkpoint management (save/load round-trip, corrupted files, directory creation)

Requirements: 1.1, 1.4, 2.1, 2.2, 2.3, 2.4, 2.5, 4.1, 4.2, 4.4, 4.5, 8.1, 8.2, 8.3, 8.4, 8.5
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

import pytest

from model.exceptions import ConfigurationError
from model.models.registry import BaseDetector, ModelRegistry
from model.models.yolo26_wrapper import YOLO26Detector


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the model registry before each test, ensuring yolo26 is registered."""
    saved = dict(ModelRegistry._models)
    ModelRegistry._models["yolo26"] = YOLO26Detector
    yield
    ModelRegistry._models = saved


class TestYOLO26Registration:
    """Tests for YOLO26Detector registry integration (Requirements 1.1, 1.4)."""

    def test_yolo26_in_list_models(self):
        """Test that 'yolo26' appears in ModelRegistry.list_models()."""
        models = ModelRegistry.list_models()
        assert "yolo26" in models

    def test_list_models_is_sorted(self):
        """Test that list_models returns a sorted list."""
        models = ModelRegistry.list_models()
        assert models == sorted(models)

    @patch("model.models.yolo26_wrapper.ultralytics", new=MagicMock())
    def test_create_via_registry(self):
        """Test creating YOLO26Detector through the registry."""
        config = {"model_size": "m", "num_classes": 5}
        model = ModelRegistry.create("yolo26", config)
        assert isinstance(model, YOLO26Detector)
        assert isinstance(model, BaseDetector)

    def test_yolo26_is_base_detector_subclass(self):
        """Test that YOLO26Detector is a subclass of BaseDetector."""
        assert issubclass(YOLO26Detector, BaseDetector)


class TestYOLO26ConfigSchema:
    """Tests for YOLO26Detector.get_config_schema() (Requirements 2.1-2.5)."""

    @patch("model.models.yolo26_wrapper.ultralytics", new=MagicMock())
    def test_schema_contains_all_parameters(self):
        """Test that schema contains all expected configuration parameters."""
        config = {"model_size": "n", "num_classes": 4}
        detector = YOLO26Detector(config)
        schema = detector.get_config_schema()

        expected_keys = {
            "model_size",
            "num_classes",
            "end2end",
            "confidence_threshold",
            "iou_threshold",
        }
        assert set(schema.keys()) == expected_keys

    @patch("model.models.yolo26_wrapper.ultralytics", new=MagicMock())
    def test_schema_model_size_required(self):
        """Test that model_size is marked as required with type str."""
        config = {"model_size": "s", "num_classes": 4}
        detector = YOLO26Detector(config)
        schema = detector.get_config_schema()

        assert schema["model_size"]["type"] == "str"
        assert schema["model_size"]["required"] is True

    @patch("model.models.yolo26_wrapper.ultralytics", new=MagicMock())
    def test_schema_num_classes_required(self):
        """Test that num_classes is marked as required with type int."""
        config = {"model_size": "n", "num_classes": 4}
        detector = YOLO26Detector(config)
        schema = detector.get_config_schema()

        assert schema["num_classes"]["type"] == "int"
        assert schema["num_classes"]["required"] is True

    @patch("model.models.yolo26_wrapper.ultralytics", new=MagicMock())
    def test_schema_end2end_optional(self):
        """Test that end2end is marked as optional with type bool."""
        config = {"model_size": "n", "num_classes": 4}
        detector = YOLO26Detector(config)
        schema = detector.get_config_schema()

        assert schema["end2end"]["type"] == "bool"
        assert schema["end2end"]["required"] is False

    @patch("model.models.yolo26_wrapper.ultralytics", new=MagicMock())
    def test_schema_confidence_threshold_optional(self):
        """Test that confidence_threshold is marked as optional with type float."""
        config = {"model_size": "n", "num_classes": 4}
        detector = YOLO26Detector(config)
        schema = detector.get_config_schema()

        assert schema["confidence_threshold"]["type"] == "float"
        assert schema["confidence_threshold"]["required"] is False

    @patch("model.models.yolo26_wrapper.ultralytics", new=MagicMock())
    def test_schema_iou_threshold_optional(self):
        """Test that iou_threshold is marked as optional with type float."""
        config = {"model_size": "n", "num_classes": 4}
        detector = YOLO26Detector(config)
        schema = detector.get_config_schema()

        assert schema["iou_threshold"]["type"] == "float"
        assert schema["iou_threshold"]["required"] is False


class TestYOLO26ModelFileMap:
    """Tests for MODEL_FILE_MAP mapping model sizes to files (Requirements 8.1-8.5)."""

    def test_nano_maps_to_yolo26n(self):
        """Test that model_size 'n' maps to 'yolo26n.pt'."""
        assert YOLO26Detector.MODEL_FILE_MAP["n"] == "yolo26n.pt"

    def test_small_maps_to_yolo26s(self):
        """Test that model_size 's' maps to 'yolo26s.pt'."""
        assert YOLO26Detector.MODEL_FILE_MAP["s"] == "yolo26s.pt"

    def test_medium_maps_to_yolo26m(self):
        """Test that model_size 'm' maps to 'yolo26m.pt'."""
        assert YOLO26Detector.MODEL_FILE_MAP["m"] == "yolo26m.pt"

    def test_large_maps_to_yolo26l(self):
        """Test that model_size 'l' maps to 'yolo26l.pt'."""
        assert YOLO26Detector.MODEL_FILE_MAP["l"] == "yolo26l.pt"

    def test_xlarge_maps_to_yolo26x(self):
        """Test that model_size 'x' maps to 'yolo26x.pt'."""
        assert YOLO26Detector.MODEL_FILE_MAP["x"] == "yolo26x.pt"

    def test_all_valid_sizes_have_mapping(self):
        """Test that every valid model size has a corresponding file mapping."""
        for size in YOLO26Detector.VALID_MODEL_SIZES:
            assert size in YOLO26Detector.MODEL_FILE_MAP
            assert YOLO26Detector.MODEL_FILE_MAP[size].startswith("yolo26")
            assert YOLO26Detector.MODEL_FILE_MAP[size].endswith(".pt")


class TestYOLO26ImportWithoutUltralytics:
    """Tests for module importability without ultralytics (Requirements 7.2, 7.4)."""

    def test_module_importable_without_ultralytics(self):
        """Test that the yolo26_wrapper module can be imported even if ultralytics is None."""
        # The module is already imported successfully at the top of this file.
        # The try/except guard sets ultralytics=None when not installed.
        # Verify the module-level guard works by checking the class exists.
        from model.models import yolo26_wrapper

        assert hasattr(yolo26_wrapper, "YOLO26Detector")
        assert hasattr(yolo26_wrapper, "ultralytics")

    def test_instantiation_raises_import_error_without_ultralytics(self):
        """Test that instantiation raises ImportError when ultralytics is None."""
        config = {"model_size": "n", "num_classes": 4}
        with patch("model.models.yolo26_wrapper.ultralytics", new=None):
            with pytest.raises(ImportError) as exc_info:
                YOLO26Detector(config)
            assert "ultralytics" in str(exc_info.value)

    def test_class_attributes_accessible_without_ultralytics(self):
        """Test that class-level attributes are accessible regardless of ultralytics."""
        # These should be accessible even when ultralytics is not installed
        assert YOLO26Detector.VALID_MODEL_SIZES == ("n", "s", "m", "l", "x")
        assert isinstance(YOLO26Detector.MODEL_FILE_MAP, dict)


class TestYOLO26DefaultValues:
    """Tests for default values of optional parameters (Requirements 2.3, 2.4, 2.5)."""

    @patch("model.models.yolo26_wrapper.ultralytics", new=MagicMock())
    def test_default_end2end_is_true(self):
        """Test that end2end defaults to True when not specified."""
        config = {"model_size": "n", "num_classes": 4}
        detector = YOLO26Detector(config)
        assert detector.end2end is True

    @patch("model.models.yolo26_wrapper.ultralytics", new=MagicMock())
    def test_default_confidence_threshold(self):
        """Test that confidence_threshold defaults to 0.25 when not specified."""
        config = {"model_size": "n", "num_classes": 4}
        detector = YOLO26Detector(config)
        assert detector.confidence_threshold == 0.25

    @patch("model.models.yolo26_wrapper.ultralytics", new=MagicMock())
    def test_default_iou_threshold(self):
        """Test that iou_threshold defaults to 0.7 when not specified."""
        config = {"model_size": "n", "num_classes": 4}
        detector = YOLO26Detector(config)
        assert detector.iou_threshold == 0.7

    @patch("model.models.yolo26_wrapper.ultralytics", new=MagicMock())
    def test_explicit_values_override_defaults(self):
        """Test that explicitly provided values override defaults."""
        config = {
            "model_size": "l",
            "num_classes": 10,
            "end2end": False,
            "confidence_threshold": 0.5,
            "iou_threshold": 0.45,
        }
        detector = YOLO26Detector(config)
        assert detector.end2end is False
        assert detector.confidence_threshold == 0.5
        assert detector.iou_threshold == 0.45



class TestYOLO26CheckpointManagement:
    """Tests for checkpoint save/load functionality (Requirements 4.1, 4.2, 4.4, 4.5)."""

    def _make_detector_with_mock_model(self):
        """Create a YOLO26Detector with a mocked ultralytics model attached.

        Uses MagicMock for ultralytics so that __init__ can call
        ultralytics.YOLO() and _build_loss_fn() without errors, then
        replaces the internal _model with a real nn.Module for checkpoint tests.
        """
        mock_ul = MagicMock()
        # Make YOLO() return a mock with a model attribute (MagicMock)
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.model = MagicMock()
        # _build_loss_fn checks hasattr(model, 'criterion')
        mock_yolo_instance.model.criterion = None
        mock_ul.YOLO.return_value = mock_yolo_instance

        with patch("model.models.yolo26_wrapper.ultralytics", new=mock_ul):
            detector = YOLO26Detector({"model_size": "n", "num_classes": 5})

        # Now replace _model with a mock that has a real nn.Module as .model
        # so that save_checkpoint can call .state_dict() on it
        inner_model = torch.nn.Linear(10, 5)
        real_mock_yolo = MagicMock()
        real_mock_yolo.model = inner_model
        detector._model = real_mock_yolo
        return detector

    def _make_mock_ultralytics(self):
        """Create a properly configured MagicMock for ultralytics module."""
        mock_ul = MagicMock()
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.model = MagicMock()
        mock_yolo_instance.model.criterion = None
        mock_ul.YOLO.return_value = mock_yolo_instance
        return mock_ul

    def test_save_load_round_trip(self, tmp_path):
        """Test that saving and loading a checkpoint preserves model state.

        Validates: Requirements 4.1, 4.2, 4.5
        """
        detector = self._make_detector_with_mock_model()

        # Capture original state dict values
        original_state = {
            k: v.clone() for k, v in detector._model.model.state_dict().items()
        }

        # Save checkpoint
        ckpt_path = tmp_path / "checkpoint.pt"
        detector.save_checkpoint(ckpt_path)

        # Verify file was created
        assert ckpt_path.exists()
        assert ckpt_path.stat().st_size > 0

        # Load the checkpoint and verify the state dict is preserved
        loaded_data = torch.load(str(ckpt_path), map_location="cpu")
        assert "model_state_dict" in loaded_data

        for key in original_state:
            assert key in loaded_data["model_state_dict"]
            assert torch.equal(original_state[key], loaded_data["model_state_dict"][key])

    def test_corrupted_file_raises_runtime_error(self, tmp_path):
        """Test that loading a corrupted file raises RuntimeError.

        Validates: Requirements 4.4
        """
        mock_ul = self._make_mock_ultralytics()

        with patch("model.models.yolo26_wrapper.ultralytics", new=mock_ul):
            detector = YOLO26Detector({"model_size": "n", "num_classes": 5})

            # Make YOLO() raise an exception when given a corrupted file
            mock_ul.YOLO.side_effect = Exception("Invalid checkpoint format")

            # Write garbage data to a file
            corrupted_path = tmp_path / "corrupted.pt"
            corrupted_path.write_bytes(b"this is not a valid pytorch checkpoint file")

            with pytest.raises(RuntimeError, match="corrupted or not a valid checkpoint"):
                detector.load_checkpoint(corrupted_path)

    def test_parent_directory_creation_on_save(self, tmp_path):
        """Test that save_checkpoint creates parent directories as needed.

        Validates: Requirements 4.2
        """
        detector = self._make_detector_with_mock_model()

        # Save to a path with non-existent parent directories
        nested_path = tmp_path / "deep" / "nested" / "dir" / "model.pt"
        assert not nested_path.parent.exists()

        detector.save_checkpoint(nested_path)

        # Verify parent dirs were created and file exists
        assert nested_path.parent.exists()
        assert nested_path.exists()
        assert nested_path.stat().st_size > 0

    def test_load_nonexistent_path_raises_file_not_found(self, tmp_path):
        """Test that loading from a non-existent path raises FileNotFoundError.

        Validates: Requirements 4.3
        """
        mock_ul = self._make_mock_ultralytics()

        with patch("model.models.yolo26_wrapper.ultralytics", new=mock_ul):
            detector = YOLO26Detector({"model_size": "n", "num_classes": 5})

        nonexistent = tmp_path / "does_not_exist.pt"

        with pytest.raises(FileNotFoundError, match="does_not_exist.pt"):
            detector.load_checkpoint(nonexistent)

    def test_overwrite_existing_file(self, tmp_path):
        """Test that saving twice to the same path overwrites the file.

        Validates: Requirements 4.2
        """
        detector = self._make_detector_with_mock_model()
        ckpt_path = tmp_path / "model.pt"

        # First save
        detector.save_checkpoint(ckpt_path)

        # Modify the model weights to ensure different content
        with torch.no_grad():
            for param in detector._model.model.parameters():
                param.fill_(42.0)

        # Second save to same path
        detector.save_checkpoint(ckpt_path)

        # Verify file was overwritten (content changed)
        loaded = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = loaded["model_state_dict"]
        # Check that the saved weights reflect the modified values
        for key, value in state_dict.items():
            if "weight" in key or "bias" in key:
                assert torch.all(value == 42.0)


class TestYOLO26TrainingMethods:
    """Tests for training integration methods (Requirements 5.1-5.6)."""

    def _make_detector_with_real_module(self):
        """Create a YOLO26Detector with a real nn.Module for training tests."""
        mock_ul = MagicMock()
        mock_yolo_instance = MagicMock()
        # Use a real nn.Module so parameters() works correctly
        inner_model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 4),
        )
        mock_yolo_instance.model = inner_model
        mock_ul.YOLO.return_value = mock_yolo_instance

        with patch("model.models.yolo26_wrapper.ultralytics", new=mock_ul):
            detector = YOLO26Detector({"model_size": "n", "num_classes": 5})

        return detector

    def test_get_parameters_returns_list(self):
        """Test that get_parameters returns a non-empty list of Parameters.

        Validates: Requirements 5.4
        """
        detector = self._make_detector_with_real_module()
        params = detector.get_parameters()

        assert isinstance(params, list)
        assert len(params) > 0
        for p in params:
            assert isinstance(p, torch.nn.Parameter)
            assert p.requires_grad is True

    def test_get_parameters_filters_frozen(self):
        """Test that get_parameters excludes frozen parameters.

        Validates: Requirements 5.4
        """
        detector = self._make_detector_with_real_module()

        # Freeze some parameters
        all_params = list(detector._model.model.parameters())
        frozen_param = all_params[0]
        frozen_param.requires_grad = False

        params = detector.get_parameters()
        # Check that the frozen parameter is not in the returned list by identity
        assert not any(p is frozen_param for p in params)
        assert len(params) == len([p for p in all_params if p.requires_grad])

    def test_set_train_mode(self):
        """Test that set_train_mode puts model in training mode.

        Validates: Requirements 5.1
        """
        detector = self._make_detector_with_real_module()
        detector._model.model.eval()  # Start in eval mode
        assert not detector._model.model.training

        detector.set_train_mode()
        assert detector._model.model.training

    def test_set_eval_mode(self):
        """Test that set_eval_mode puts model in evaluation mode.

        Validates: Requirements 5.1
        """
        detector = self._make_detector_with_real_module()
        detector._model.model.train()  # Start in train mode
        assert detector._model.model.training

        detector.set_eval_mode()
        assert not detector._model.model.training

    def test_train_step_empty_batch_returns_zero_loss(self):
        """Test that train_step with empty images returns zero loss tensor.

        Validates: Requirements 5.6
        """
        detector = self._make_detector_with_real_module()
        result = detector.train_step([], [])

        assert "loss_tensor" in result
        assert result["loss_tensor"].item() == 0.0

    def test_train_step_empty_list_returns_zero_loss(self):
        """Test that train_step with None-like empty input returns zero loss.

        Validates: Requirements 5.6
        """
        detector = self._make_detector_with_real_module()
        result = detector.train_step([], [])

        assert isinstance(result, dict)
        assert "loss_tensor" in result
        assert result["loss_tensor"].dim() == 0  # scalar
        assert result["loss_tensor"].item() == 0.0

    def test_pretrained_weights_file_not_found(self, tmp_path):
        """Test that non-existent pretrained_weights raises FileNotFoundError.

        Validates: Requirements 5.3
        """
        config = {
            "model_size": "n",
            "num_classes": 5,
            "pretrained_weights": str(tmp_path / "nonexistent.pt"),
        }

        with patch("model.models.yolo26_wrapper.ultralytics", new=MagicMock()):
            with pytest.raises(FileNotFoundError, match="nonexistent.pt"):
                YOLO26Detector(config)

    def test_pretrained_weights_loads_from_path(self, tmp_path):
        """Test that existing pretrained_weights path is used for model loading.

        Validates: Requirements 5.2
        """
        # Create a dummy weights file
        weights_path = tmp_path / "pretrained.pt"
        weights_path.write_bytes(b"dummy")

        config = {
            "model_size": "n",
            "num_classes": 5,
            "pretrained_weights": str(weights_path),
        }

        mock_ul = MagicMock()
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.model = MagicMock()
        mock_yolo_instance.model.criterion = None
        mock_ul.YOLO.return_value = mock_yolo_instance

        with patch("model.models.yolo26_wrapper.ultralytics", new=mock_ul):
            detector = YOLO26Detector(config)

        # Verify YOLO was called with the pretrained weights path
        mock_ul.YOLO.assert_called_once_with(str(weights_path))

    def test_build_loss_fn_with_criterion(self):
        """Test that _build_loss_fn stores criterion from init_criterion().

        Validates: Requirements 5.5
        """
        mock_ul = MagicMock()
        mock_yolo_instance = MagicMock()
        mock_criterion = MagicMock()
        # init_criterion() returns our mock criterion
        mock_yolo_instance.model.init_criterion = MagicMock(return_value=mock_criterion)
        # Ensure the mock doesn't look like E2E (no one2many attribute)
        mock_criterion.one2many = None
        del mock_criterion.one2many
        mock_ul.YOLO.return_value = mock_yolo_instance

        with patch("model.models.yolo26_wrapper.ultralytics", new=mock_ul):
            detector = YOLO26Detector({"model_size": "n", "num_classes": 5})

        # Stock path: init_criterion() was called and its return used
        assert detector._loss_fn is mock_criterion

    def test_build_loss_fn_without_criterion(self):
        """Test that _build_loss_fn sets None when init_criterion fails.

        Validates: Requirements 5.5
        """
        mock_ul = MagicMock()
        mock_yolo_instance = MagicMock()
        # init_criterion raises to simulate no criterion available
        mock_yolo_instance.model.init_criterion = MagicMock(side_effect=AttributeError("no criterion"))
        mock_yolo_instance.model.criterion = None
        mock_ul.YOLO.return_value = mock_yolo_instance

        with patch("model.models.yolo26_wrapper.ultralytics", new=mock_ul):
            detector = YOLO26Detector({"model_size": "n", "num_classes": 5})

        assert detector._loss_fn is None

    def test_set_train_eval_mode_toggle(self):
        """Test toggling between train and eval modes.

        Validates: Requirements 5.1
        """
        detector = self._make_detector_with_real_module()

        # Toggle multiple times
        detector.set_train_mode()
        assert detector._model.model.training

        detector.set_eval_mode()
        assert not detector._model.model.training

        detector.set_train_mode()
        assert detector._model.model.training
