"""Property-based preservation tests for YOLO26Detector transfer learning.

Property 2: Preservation - Default All-Params-Trainable Behavior Unchanged

These tests lock in the current default behavior: when no freeze_backbone
option is present (or freeze_backbone is explicitly False), ALL model parameters
have requires_grad = True and get_parameters() returns the full parameter set.

These tests MUST PASS on the UNFIXED code to confirm baseline behavior.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from model.exceptions import ConfigurationError
from model.models.registry import BaseDetector, ModelRegistry
from model.models.yolo26_wrapper import YOLO26Detector


# ---------------------------------------------------------------------------
# Hypothesis strategies for YOLO26 config WITHOUT freeze options
# ---------------------------------------------------------------------------

VALID_MODEL_SIZES = st.sampled_from(["n", "s", "m", "l", "x"])
VALID_NUM_CLASSES = st.integers(min_value=1, max_value=1000)
VALID_CONFIDENCE_THRESHOLD = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)
VALID_IOU_THRESHOLD = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)


@st.composite
def valid_yolo26_config_no_freeze(draw):
    """Generate valid YOLO26 configs WITHOUT any freeze options.

    Varies model_size, num_classes, confidence_threshold, and iou_threshold
    but never includes freeze_backbone or freeze_layers keys.
    """
    config = {
        "model_size": draw(VALID_MODEL_SIZES),
        "num_classes": draw(VALID_NUM_CLASSES),
    }
    # Optionally include optional parameters (but never freeze options)
    if draw(st.booleans()):
        config["confidence_threshold"] = draw(VALID_CONFIDENCE_THRESHOLD)
    if draw(st.booleans()):
        config["iou_threshold"] = draw(VALID_IOU_THRESHOLD)
    if draw(st.booleans()):
        config["end2end"] = draw(st.booleans())
    return config


@st.composite
def valid_yolo26_config_explicit_freeze_false(draw):
    """Generate valid YOLO26 configs with explicit freeze_backbone: false.

    This tests that setting freeze_backbone to False explicitly still
    produces all-params-trainable behavior.
    """
    config = {
        "model_size": draw(VALID_MODEL_SIZES),
        "num_classes": draw(VALID_NUM_CLASSES),
        "freeze_backbone": False,
    }
    if draw(st.booleans()):
        config["confidence_threshold"] = draw(VALID_CONFIDENCE_THRESHOLD)
    if draw(st.booleans()):
        config["iou_threshold"] = draw(VALID_IOU_THRESHOLD)
    return config


# ---------------------------------------------------------------------------
# Mock model module for testing parameter behavior
# ---------------------------------------------------------------------------

class MockBackboneHead(nn.Module):
    """A simple nn.Module simulating backbone + head structure.

    Has multiple layers with parameters that can be checked for requires_grad.
    Mimics the Ultralytics model.model structure.
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        # Backbone layers (first few layers)
        self.backbone_conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.backbone_bn1 = nn.BatchNorm2d(16)
        self.backbone_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.backbone_bn2 = nn.BatchNorm2d(32)
        # Head layers
        self.head_conv = nn.Conv2d(32, 64, kernel_size=1)
        self.head_fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.backbone_conv1(x)
        x = self.backbone_bn1(x)
        x = self.backbone_conv2(x)
        x = self.backbone_bn2(x)
        x = self.head_conv(x)
        x = x.mean(dim=[2, 3])  # global avg pool
        x = self.head_fc(x)
        return x


def _make_mock_ultralytics(num_classes: int = 5):
    """Create a mock ultralytics module with a real nn.Module inside.

    Returns a mock ultralytics module whose YOLO() call returns a mock model
    with a real nn.Module as model.model, allowing proper parameter testing.
    """
    mock_ultralytics = MagicMock()

    # The inner model is a real nn.Module so parameters() works correctly
    inner_module = MockBackboneHead(num_classes=num_classes)

    # Mock the YOLO wrapper object
    mock_yolo_instance = MagicMock()
    mock_yolo_instance.model = inner_module

    # Set up args for _build_loss_fn
    inner_module.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    inner_module.init_criterion = MagicMock(return_value=MagicMock())

    mock_ultralytics.YOLO.return_value = mock_yolo_instance

    return mock_ultralytics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the model registry before each test."""
    saved = dict(ModelRegistry._models)
    ModelRegistry._models["yolo26"] = YOLO26Detector
    yield
    ModelRegistry._models = saved


# ---------------------------------------------------------------------------
# Property 2: Preservation - Default All-Params-Trainable Behavior Unchanged
# Feature: transfer-learning-freeze-layers, Property 2: Preservation
# ---------------------------------------------------------------------------


class TestPreservationAllParamsTrainable:
    """Property 2: For any valid YOLO26 config WITHOUT freeze_backbone (or with
    freeze_backbone: false), ALL model parameters SHALL have requires_grad = True
    and get_parameters() SHALL return the full parameter set.

    This locks in the current default behavior so regressions are caught.

    **Validates: Requirements 3.1, 3.2, 3.3**
    """

    @given(config=valid_yolo26_config_no_freeze())
    @settings(max_examples=50)
    def test_no_freeze_config_all_params_trainable(self, config):
        """Without freeze options, all model parameters have requires_grad=True.

        **Validates: Requirements 3.1**
        """
        mock_ul = _make_mock_ultralytics(num_classes=config["num_classes"])

        with patch("model.models.yolo26_wrapper.ultralytics", mock_ul):
            detector = YOLO26Detector(config)

        # All parameters must have requires_grad = True
        all_params = list(detector._model.model.parameters())
        assert len(all_params) > 0, "Model must have at least one parameter"

        for param in all_params:
            assert param.requires_grad is True, (
                f"Parameter has requires_grad=False but should be True "
                f"when no freeze config is present"
            )

    @given(config=valid_yolo26_config_no_freeze())
    @settings(max_examples=50)
    def test_no_freeze_config_get_parameters_returns_all(self, config):
        """Without freeze options, get_parameters() returns all model params.

        **Validates: Requirements 3.2**
        """
        mock_ul = _make_mock_ultralytics(num_classes=config["num_classes"])

        with patch("model.models.yolo26_wrapper.ultralytics", mock_ul):
            detector = YOLO26Detector(config)

        trainable_params = detector.get_parameters()
        all_params = list(detector._model.model.parameters())

        # get_parameters() should return ALL params when no freeze config
        assert len(trainable_params) == len(all_params), (
            f"get_parameters() returned {len(trainable_params)} params but "
            f"total param count is {len(all_params)}. They should be equal "
            f"when no freeze config is present."
        )

    @given(config=valid_yolo26_config_explicit_freeze_false())
    @settings(max_examples=50)
    def test_explicit_freeze_false_all_params_trainable(self, config):
        """With explicit freeze_backbone: false, all params remain trainable.

        **Validates: Requirements 3.1, 3.3**
        """
        mock_ul = _make_mock_ultralytics(num_classes=config["num_classes"])

        with patch("model.models.yolo26_wrapper.ultralytics", mock_ul):
            detector = YOLO26Detector(config)

        all_params = list(detector._model.model.parameters())
        assert len(all_params) > 0, "Model must have at least one parameter"

        for param in all_params:
            assert param.requires_grad is True, (
                f"Parameter has requires_grad=False but should be True "
                f"when freeze_backbone is explicitly False"
            )

    @given(config=valid_yolo26_config_explicit_freeze_false())
    @settings(max_examples=50)
    def test_explicit_freeze_false_get_parameters_returns_all(self, config):
        """With explicit freeze_backbone: false, get_parameters() returns all params.

        **Validates: Requirements 3.2, 3.3**
        """
        mock_ul = _make_mock_ultralytics(num_classes=config["num_classes"])

        with patch("model.models.yolo26_wrapper.ultralytics", mock_ul):
            detector = YOLO26Detector(config)

        trainable_params = detector.get_parameters()
        all_params = list(detector._model.model.parameters())

        assert len(trainable_params) == len(all_params), (
            f"get_parameters() returned {len(trainable_params)} params but "
            f"total param count is {len(all_params)}. They should be equal "
            f"when freeze_backbone is explicitly False."
        )


# ---------------------------------------------------------------------------
# Property 2 (cont.): Inference Preservation
# Feature: transfer-learning-freeze-layers, Property 2: Inference Preservation
# ---------------------------------------------------------------------------


class TestPreservationInferenceUnaffected:
    """Property 2 (Inference): For any valid config regardless of freeze settings,
    the forward() method SHALL produce predictions (list of dicts with boxes,
    labels, scores) for any valid input tensor.

    This verifies that inference works regardless of configuration.

    **Validates: Requirements 3.4**
    """

    @given(config=valid_yolo26_config_no_freeze())
    @settings(max_examples=30)
    def test_forward_produces_predictions_without_freeze(self, config):
        """Forward pass produces valid predictions without any freeze config.

        **Validates: Requirements 3.4**
        """
        mock_ul = _make_mock_ultralytics(num_classes=config["num_classes"])

        with patch("model.models.yolo26_wrapper.ultralytics", mock_ul):
            detector = YOLO26Detector(config)

        # Create a simple input tensor
        batch_size = 2
        images = torch.rand(batch_size, 3, 64, 64)

        # Mock the predict method to return mock results
        mock_boxes = MagicMock()
        mock_boxes.xyxy = torch.tensor([[10.0, 20.0, 50.0, 60.0]])
        mock_boxes.cls = torch.tensor([0.0])
        mock_boxes.conf = torch.tensor([0.9])
        mock_boxes.__len__ = lambda self: 1

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        detector._model.predict = MagicMock(return_value=[mock_result] * batch_size)

        predictions = detector.forward(images)

        # Must produce a list of exactly batch_size prediction dicts
        assert isinstance(predictions, list)
        assert len(predictions) == batch_size

        for pred in predictions:
            assert isinstance(pred, dict)
            assert "boxes" in pred
            assert "labels" in pred
            assert "scores" in pred
            # Boxes must be (N, 4) shape
            assert pred["boxes"].ndim == 2
            assert pred["boxes"].shape[1] == 4

    @given(config=valid_yolo26_config_explicit_freeze_false())
    @settings(max_examples=30)
    def test_forward_produces_predictions_with_freeze_false(self, config):
        """Forward pass produces valid predictions with freeze_backbone: false.

        **Validates: Requirements 3.4**
        """
        mock_ul = _make_mock_ultralytics(num_classes=config["num_classes"])

        with patch("model.models.yolo26_wrapper.ultralytics", mock_ul):
            detector = YOLO26Detector(config)

        batch_size = 2
        images = torch.rand(batch_size, 3, 64, 64)

        # Mock the predict method
        mock_boxes = MagicMock()
        mock_boxes.xyxy = torch.tensor([[5.0, 10.0, 40.0, 55.0]])
        mock_boxes.cls = torch.tensor([1.0])
        mock_boxes.conf = torch.tensor([0.85])
        mock_boxes.__len__ = lambda self: 1

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        detector._model.predict = MagicMock(return_value=[mock_result] * batch_size)

        predictions = detector.forward(images)

        assert isinstance(predictions, list)
        assert len(predictions) == batch_size

        for pred in predictions:
            assert isinstance(pred, dict)
            assert "boxes" in pred
            assert "labels" in pred
            assert "scores" in pred
