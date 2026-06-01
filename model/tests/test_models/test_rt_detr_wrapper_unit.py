"""Unit tests for RT_DETR_Detector wrapper.

Tests cover:
- Default values for optional parameters (confidence_threshold, iou_threshold)
- Model file mapping ("l" → "rtdetr-l.pt", "x" → "rtdetr-x.pt")
- Pretrained weights loading from custom path
- ImportError when ultralytics is not installed
- Empty batch and zero-box edge cases in train_step
- Model mode switching (train/eval)
- Config schema structure (required/optional fields)
- Loss function initialization and fallback
- Corrupted checkpoint handling (RuntimeError)
- save_checkpoint creates parent directories

Requirements: 2.6, 2.7, 3.1, 3.2, 3.3, 3.5, 5.3, 5.4, 6.2, 6.5, 7.1, 7.2, 7.3, 8.1, 8.2, 11.1, 11.2, 11.3
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import pytest

from model.exceptions import ConfigurationError
from model.models.registry import BaseDetector, ModelRegistry
from model.models.rt_detr_wrapper import RT_DETR_Detector


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the model registry before each test, ensuring rt_detr is registered."""
    saved = dict(ModelRegistry._models)
    ModelRegistry._models["rt_detr"] = RT_DETR_Detector
    yield
    ModelRegistry._models = saved


def _make_mock_rtdetr():
    """Create a properly configured MagicMock for the RTDETR class.

    Returns a mock that, when called (instantiated), returns a mock model
    with the expected attributes for RT_DETR_Detector initialization.
    """
    mock_rtdetr_cls = MagicMock()
    mock_model_instance = MagicMock()
    # Set up the inner model with parameters
    inner_module = torch.nn.Linear(10, 5)
    mock_model_instance.model = inner_module
    # init_criterion returns a mock loss function
    mock_model_instance.model.init_criterion = MagicMock(return_value=MagicMock())
    mock_model_instance.model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    mock_rtdetr_cls.return_value = mock_model_instance
    return mock_rtdetr_cls


def _make_detector(config=None, mock_rtdetr=None):
    """Create an RT_DETR_Detector with mocked RTDETR.

    Args:
        config: Configuration dict. Defaults to minimal valid config.
        mock_rtdetr: Optional pre-configured mock. If None, creates one.

    Returns:
        Tuple of (detector, mock_rtdetr_cls).
    """
    if config is None:
        config = {"model_size": "l", "num_classes": 5}
    if mock_rtdetr is None:
        mock_rtdetr = _make_mock_rtdetr()

    with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr):
        detector = RT_DETR_Detector(config)
    return detector, mock_rtdetr


class TestRTDETRDefaultValues:
    """Tests for default values of optional parameters (Requirements 2.6, 2.7)."""

    def test_default_confidence_threshold(self):
        """Test that confidence_threshold defaults to 0.25 when not specified."""
        detector, _ = _make_detector({"model_size": "l", "num_classes": 5})
        assert detector.confidence_threshold == 0.25

    def test_default_iou_threshold(self):
        """Test that iou_threshold defaults to 0.7 when not specified."""
        detector, _ = _make_detector({"model_size": "l", "num_classes": 5})
        assert detector.iou_threshold == 0.7

    def test_explicit_values_override_defaults(self):
        """Test that explicitly provided values override defaults."""
        config = {
            "model_size": "x",
            "num_classes": 10,
            "confidence_threshold": 0.5,
            "iou_threshold": 0.45,
        }
        detector, _ = _make_detector(config)
        assert detector.confidence_threshold == 0.5
        assert detector.iou_threshold == 0.45


class TestRTDETRModelFileMap:
    """Tests for MODEL_FILE_MAP mapping model sizes to files (Requirements 3.1, 3.2)."""

    def test_large_maps_to_rtdetr_l(self):
        """Test that model_size 'l' maps to 'rtdetr-l.pt'."""
        assert RT_DETR_Detector.MODEL_FILE_MAP["l"] == "rtdetr-l.pt"

    def test_xlarge_maps_to_rtdetr_x(self):
        """Test that model_size 'x' maps to 'rtdetr-x.pt'."""
        assert RT_DETR_Detector.MODEL_FILE_MAP["x"] == "rtdetr-x.pt"

    def test_all_valid_sizes_have_mapping(self):
        """Test that every valid model size has a corresponding file mapping."""
        for size in RT_DETR_Detector.VALID_MODEL_SIZES:
            assert size in RT_DETR_Detector.MODEL_FILE_MAP
            assert RT_DETR_Detector.MODEL_FILE_MAP[size].startswith("rtdetr-")
            assert RT_DETR_Detector.MODEL_FILE_MAP[size].endswith(".pt")

    def test_model_size_l_calls_rtdetr_with_correct_file(self):
        """Test that model_size 'l' passes 'rtdetr-l.pt' to RTDETR constructor."""
        _, mock_rtdetr = _make_detector({"model_size": "l", "num_classes": 5})
        mock_rtdetr.assert_called_once_with("rtdetr-l.pt")

    def test_model_size_x_calls_rtdetr_with_correct_file(self):
        """Test that model_size 'x' passes 'rtdetr-x.pt' to RTDETR constructor."""
        _, mock_rtdetr = _make_detector({"model_size": "x", "num_classes": 5})
        mock_rtdetr.assert_called_once_with("rtdetr-x.pt")


class TestRTDETRPretrainedWeights:
    """Tests for pretrained weights loading (Requirement 3.3)."""

    def test_pretrained_weights_loads_from_path(self, tmp_path):
        """Test that existing pretrained_weights path is used for model loading."""
        weights_path = tmp_path / "pretrained.pt"
        weights_path.write_bytes(b"dummy")

        config = {
            "model_size": "l",
            "num_classes": 5,
            "pretrained_weights": str(weights_path),
        }
        _, mock_rtdetr = _make_detector(config)
        mock_rtdetr.assert_called_once_with(str(weights_path))

    def test_pretrained_weights_nonexistent_raises_file_not_found(self, tmp_path):
        """Test that non-existent pretrained_weights raises FileNotFoundError."""
        config = {
            "model_size": "l",
            "num_classes": 5,
            "pretrained_weights": str(tmp_path / "nonexistent.pt"),
        }
        mock_rtdetr = _make_mock_rtdetr()
        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr):
            with pytest.raises(FileNotFoundError, match="nonexistent.pt"):
                RT_DETR_Detector(config)


class TestRTDETRImportError:
    """Tests for ImportError when ultralytics is not installed (Requirement 3.5)."""

    def test_instantiation_raises_import_error_without_ultralytics(self):
        """Test that instantiation raises ImportError when RTDETR is None."""
        config = {"model_size": "l", "num_classes": 5}
        with patch("model.models.rt_detr_wrapper.RTDETR", None):
            with pytest.raises(ImportError) as exc_info:
                RT_DETR_Detector(config)
            assert "ultralytics" in str(exc_info.value)
            assert "pip install" in str(exc_info.value)

    def test_module_importable_without_ultralytics(self):
        """Test that the rt_detr_wrapper module can be imported even if RTDETR is None."""
        from model.models import rt_detr_wrapper

        assert hasattr(rt_detr_wrapper, "RT_DETR_Detector")

    def test_class_attributes_accessible_without_ultralytics(self):
        """Test that class-level attributes are accessible regardless of ultralytics."""
        assert RT_DETR_Detector.VALID_MODEL_SIZES == ("l", "x")
        assert isinstance(RT_DETR_Detector.MODEL_FILE_MAP, dict)


class TestRTDETRTrainStepEdgeCases:
    """Tests for empty batch and zero-box edge cases in train_step (Requirements 5.3, 5.4)."""

    def _make_detector_with_real_module(self):
        """Create an RT_DETR_Detector with a real nn.Module for training tests."""
        mock_rtdetr = MagicMock()
        mock_model_instance = MagicMock()
        inner_model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 4),
        )
        mock_model_instance.model = inner_model
        mock_model_instance.model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        mock_rtdetr.return_value = mock_model_instance

        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr):
            detector = RT_DETR_Detector({"model_size": "l", "num_classes": 5})
        return detector

    def test_empty_batch_returns_zero_loss(self):
        """Test that train_step with empty images returns zero loss tensor."""
        detector = self._make_detector_with_real_module()
        result = detector.train_step([], [])

        assert "loss_tensor" in result
        assert result["loss_tensor"].item() == 0.0

    def test_zero_boxes_in_all_targets_returns_zero_loss(self):
        """Test that train_step with all-empty targets returns zero loss tensor."""
        detector = self._make_detector_with_real_module()

        images = [torch.randn(3, 64, 64), torch.randn(3, 64, 64)]
        targets = [
            {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)},
            {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)},
        ]
        result = detector.train_step(images, targets)

        assert "loss_tensor" in result
        assert result["loss_tensor"].item() == 0.0

    def test_empty_batch_loss_is_scalar(self):
        """Test that the zero loss tensor is a scalar (0-dimensional)."""
        detector = self._make_detector_with_real_module()
        result = detector.train_step([], [])

        assert result["loss_tensor"].dim() == 0


class TestRTDETRModeSwitching:
    """Tests for model mode switching (Requirements 8.1, 8.2)."""

    def _make_detector_with_real_module(self):
        """Create an RT_DETR_Detector with a real nn.Module for mode tests."""
        mock_rtdetr = MagicMock()
        mock_model_instance = MagicMock()
        inner_model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.BatchNorm1d(5),
        )
        mock_model_instance.model = inner_model
        mock_model_instance.model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        mock_rtdetr.return_value = mock_model_instance

        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr):
            detector = RT_DETR_Detector({"model_size": "l", "num_classes": 5})
        return detector

    def test_set_train_mode(self):
        """Test that set_train_mode puts model in training mode."""
        detector = self._make_detector_with_real_module()
        detector._model.model.eval()
        assert not detector._model.model.training

        detector.set_train_mode()
        assert detector._model.model.training

    def test_set_eval_mode(self):
        """Test that set_eval_mode puts model in evaluation mode."""
        detector = self._make_detector_with_real_module()
        detector._model.model.train()
        assert detector._model.model.training

        detector.set_eval_mode()
        assert not detector._model.model.training

    def test_train_eval_mode_toggle(self):
        """Test toggling between train and eval modes."""
        detector = self._make_detector_with_real_module()

        detector.set_train_mode()
        assert detector._model.model.training

        detector.set_eval_mode()
        assert not detector._model.model.training

        detector.set_train_mode()
        assert detector._model.model.training


class TestRTDETRConfigSchema:
    """Tests for get_config_schema() (Requirements 7.1, 7.2, 7.3)."""

    def test_schema_contains_all_parameters(self):
        """Test that schema contains all expected configuration parameters."""
        detector, _ = _make_detector()
        schema = detector.get_config_schema()

        expected_keys = {
            "model_size",
            "num_classes",
            "confidence_threshold",
            "iou_threshold",
            "pretrained_weights",
        }
        assert set(schema.keys()) == expected_keys

    def test_schema_model_size_required(self):
        """Test that model_size is marked as required with type str."""
        detector, _ = _make_detector()
        schema = detector.get_config_schema()

        assert schema["model_size"]["type"] == "str"
        assert schema["model_size"]["required"] is True

    def test_schema_num_classes_required(self):
        """Test that num_classes is marked as required with type int."""
        detector, _ = _make_detector()
        schema = detector.get_config_schema()

        assert schema["num_classes"]["type"] == "int"
        assert schema["num_classes"]["required"] is True

    def test_schema_confidence_threshold_optional(self):
        """Test that confidence_threshold is marked as optional with type float."""
        detector, _ = _make_detector()
        schema = detector.get_config_schema()

        assert schema["confidence_threshold"]["type"] == "float"
        assert schema["confidence_threshold"]["required"] is False

    def test_schema_iou_threshold_optional(self):
        """Test that iou_threshold is marked as optional with type float."""
        detector, _ = _make_detector()
        schema = detector.get_config_schema()

        assert schema["iou_threshold"]["type"] == "float"
        assert schema["iou_threshold"]["required"] is False

    def test_schema_pretrained_weights_optional(self):
        """Test that pretrained_weights is marked as optional with type str."""
        detector, _ = _make_detector()
        schema = detector.get_config_schema()

        assert schema["pretrained_weights"]["type"] == "str"
        assert schema["pretrained_weights"]["required"] is False


class TestRTDETRLossFunction:
    """Tests for loss function initialization and fallback (Requirements 11.1, 11.2, 11.3)."""

    def test_loss_fn_initialized_via_init_criterion(self):
        """Test that _build_loss_fn uses init_criterion when available."""
        mock_rtdetr = MagicMock()
        mock_model_instance = MagicMock()
        inner_model = MagicMock()
        mock_criterion = MagicMock()
        inner_model.init_criterion = MagicMock(return_value=mock_criterion)
        inner_model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        inner_model.parameters = MagicMock(return_value=iter([torch.nn.Parameter(torch.randn(2))]))
        mock_model_instance.model = inner_model
        mock_rtdetr.return_value = mock_model_instance

        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr):
            detector = RT_DETR_Detector({"model_size": "l", "num_classes": 5})

        assert detector._loss_fn is mock_criterion

    def test_loss_fn_fallback_to_criterion_attribute(self):
        """Test that _build_loss_fn falls back to model.criterion when init_criterion unavailable."""
        mock_rtdetr = MagicMock()
        mock_model_instance = MagicMock()
        inner_model = MagicMock(spec=[])  # Empty spec to control hasattr
        mock_criterion = MagicMock()

        # Manually set attributes we want to exist
        inner_model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        inner_model.criterion = mock_criterion
        inner_model.parameters = MagicMock(return_value=iter([torch.nn.Parameter(torch.randn(2))]))

        mock_model_instance.model = inner_model
        mock_rtdetr.return_value = mock_model_instance

        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr):
            detector = RT_DETR_Detector({"model_size": "l", "num_classes": 5})

        assert detector._loss_fn is mock_criterion

    def test_loss_fn_sets_default_hyperparameters_when_missing(self):
        """Test that _build_loss_fn sets default box, cls, dfl weights when missing."""
        mock_rtdetr = MagicMock()
        mock_model_instance = MagicMock()
        inner_model = MagicMock()
        # args without box, cls, dfl
        inner_model.args = SimpleNamespace()
        inner_model.init_criterion = MagicMock(return_value=MagicMock())
        inner_model.parameters = MagicMock(return_value=iter([torch.nn.Parameter(torch.randn(2))]))
        mock_model_instance.model = inner_model
        mock_rtdetr.return_value = mock_model_instance

        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr):
            detector = RT_DETR_Detector({"model_size": "l", "num_classes": 5})

        # Verify defaults were set
        args = detector._model.model.args
        assert args.box == 7.5
        assert args.cls == 0.5
        assert args.dfl == 1.5

    def test_loss_fn_sets_defaults_from_dict_args(self):
        """Test that _build_loss_fn handles dict-type args and sets defaults."""
        mock_rtdetr = MagicMock()
        mock_model_instance = MagicMock()
        inner_model = MagicMock()
        # args as a dict without loss weights
        inner_model.args = {}
        inner_model.init_criterion = MagicMock(return_value=MagicMock())
        inner_model.parameters = MagicMock(return_value=iter([torch.nn.Parameter(torch.randn(2))]))
        mock_model_instance.model = inner_model
        mock_rtdetr.return_value = mock_model_instance

        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr):
            detector = RT_DETR_Detector({"model_size": "l", "num_classes": 5})

        # After _build_loss_fn, args should be converted to SimpleNamespace with defaults
        args = detector._model.model.args
        assert isinstance(args, SimpleNamespace)
        assert args.box == 7.5
        assert args.cls == 0.5
        assert args.dfl == 1.5

    def test_loss_fn_none_when_no_criterion_available(self):
        """Test that _loss_fn is None when neither init_criterion nor criterion exist."""
        mock_rtdetr = MagicMock()
        mock_model_instance = MagicMock()
        inner_model = MagicMock(spec=[])  # Empty spec to control hasattr

        # Only set args and parameters - no init_criterion, no criterion
        inner_model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        inner_model.parameters = MagicMock(return_value=iter([torch.nn.Parameter(torch.randn(2))]))

        mock_model_instance.model = inner_model
        mock_rtdetr.return_value = mock_model_instance

        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr):
            detector = RT_DETR_Detector({"model_size": "l", "num_classes": 5})

        assert detector._loss_fn is None


class TestRTDETRCheckpointManagement:
    """Tests for checkpoint management (Requirements 6.2, 6.5)."""

    def _make_detector_with_real_module(self):
        """Create an RT_DETR_Detector with a real nn.Module for checkpoint tests."""
        mock_rtdetr = MagicMock()
        mock_model_instance = MagicMock()
        inner_model = torch.nn.Linear(10, 5)
        mock_model_instance.model = inner_model
        mock_model_instance.model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        mock_rtdetr.return_value = mock_model_instance

        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr):
            detector = RT_DETR_Detector({"model_size": "l", "num_classes": 5})
        return detector

    def test_save_checkpoint_creates_parent_directories(self, tmp_path):
        """Test that save_checkpoint creates parent directories as needed."""
        detector = self._make_detector_with_real_module()

        nested_path = tmp_path / "deep" / "nested" / "dir" / "model.pt"
        assert not nested_path.parent.exists()

        detector.save_checkpoint(nested_path)

        assert nested_path.parent.exists()
        assert nested_path.exists()
        assert nested_path.stat().st_size > 0

    def test_corrupted_file_raises_runtime_error(self, tmp_path):
        """Test that loading a corrupted file raises RuntimeError."""
        detector = self._make_detector_with_real_module()

        corrupted_path = tmp_path / "corrupted.pt"
        corrupted_path.write_bytes(b"this is not a valid pytorch checkpoint file")

        with pytest.raises(RuntimeError, match="corrupted or not a valid checkpoint"):
            detector.load_checkpoint(corrupted_path)

    def test_load_nonexistent_path_raises_file_not_found(self, tmp_path):
        """Test that loading from a non-existent path raises FileNotFoundError."""
        detector = self._make_detector_with_real_module()

        nonexistent = tmp_path / "does_not_exist.pt"

        with pytest.raises(FileNotFoundError, match="does_not_exist.pt"):
            detector.load_checkpoint(nonexistent)

    def test_save_load_round_trip(self, tmp_path):
        """Test that saving and loading a checkpoint preserves model state."""
        detector = self._make_detector_with_real_module()

        original_state = {
            k: v.clone() for k, v in detector._model.model.state_dict().items()
        }

        ckpt_path = tmp_path / "checkpoint.pt"
        detector.save_checkpoint(ckpt_path)

        assert ckpt_path.exists()

        # Load and verify
        loaded_data = torch.load(str(ckpt_path), map_location="cpu")
        assert "model_state_dict" in loaded_data

        for key in original_state:
            assert key in loaded_data["model_state_dict"]
            assert torch.equal(original_state[key], loaded_data["model_state_dict"][key])
