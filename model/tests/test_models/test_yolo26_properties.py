"""Property-based tests for YOLO26Detector configuration validation.

Tests Properties 1, 2, and 3 from the design document using Hypothesis.
"""

from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from model.exceptions import ConfigurationError
from model.models.registry import BaseDetector, ModelRegistry
from model.models.yolo26_wrapper import YOLO26Detector


# ---------------------------------------------------------------------------
# Hypothesis strategies for YOLO26 configuration generation
# ---------------------------------------------------------------------------

VALID_MODEL_SIZES = st.sampled_from(["n", "s", "m", "l", "x"])

VALID_NUM_CLASSES = st.integers(min_value=1, max_value=1000)

VALID_CONFIDENCE_THRESHOLD = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

VALID_IOU_THRESHOLD = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

VALID_END2END = st.booleans()


@st.composite
def valid_yolo26_config(draw):
    """Strategy that generates valid YOLO26 configuration dicts."""
    config = {
        "model_size": draw(VALID_MODEL_SIZES),
        "num_classes": draw(VALID_NUM_CLASSES),
    }
    # Optionally include optional parameters
    if draw(st.booleans()):
        config["end2end"] = draw(VALID_END2END)
    if draw(st.booleans()):
        config["confidence_threshold"] = draw(VALID_CONFIDENCE_THRESHOLD)
    if draw(st.booleans()):
        config["iou_threshold"] = draw(VALID_IOU_THRESHOLD)
    return config


# Strategies for invalid configurations (Property 3)

INVALID_MODEL_SIZES = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=10,
).filter(lambda s: s not in ("n", "s", "m", "l", "x"))

INVALID_NUM_CLASSES_LOW = st.integers(max_value=0)
INVALID_NUM_CLASSES_HIGH = st.integers(min_value=1001)

INVALID_THRESHOLD_LOW = st.floats(
    max_value=-0.001, allow_nan=False, allow_infinity=False
).filter(lambda x: x < 0.0)

INVALID_THRESHOLD_HIGH = st.floats(
    min_value=1.001, max_value=1e6, allow_nan=False, allow_infinity=False
).filter(lambda x: x > 1.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the model registry before each test, ensuring yolo26 is registered."""
    saved = dict(ModelRegistry._models)
    ModelRegistry._models["yolo26"] = YOLO26Detector
    yield
    ModelRegistry._models = saved


# ---------------------------------------------------------------------------
# Property 1: Valid configuration produces a BaseDetector instance
# Feature: yolo26-integration, Property 1: Valid configuration produces a BaseDetector instance
# ---------------------------------------------------------------------------


class TestProperty1ValidConfigProducesBaseDetector:
    """Property 1: For any valid config (model_size in {"n","s","m","l","x"},
    num_classes in [1,1000], optional valid thresholds),
    ModelRegistry.create("yolo26", config) returns an instance of both
    YOLO26Detector and BaseDetector.

    **Validates: Requirements 1.2**
    """

    @given(config=valid_yolo26_config())
    @settings(max_examples=100)
    def test_valid_config_creates_detector_instance(self, config):
        # Feature: yolo26-integration, Property 1: Valid configuration produces a BaseDetector instance
        """Any valid configuration produces a YOLO26Detector that is also a BaseDetector."""
        with patch("model.models.yolo26_wrapper.ultralytics", new=MagicMock()):
            instance = ModelRegistry.create("yolo26", config)

            assert isinstance(instance, YOLO26Detector)
            assert isinstance(instance, BaseDetector)


# ---------------------------------------------------------------------------
# Property 2: Missing required parameters raises ConfigurationError
# Feature: yolo26-integration, Property 2: Missing required parameters raises ConfigurationError
# ---------------------------------------------------------------------------


class TestProperty2MissingRequiredParamsRaisesError:
    """Property 2: For any config missing at least one of the required parameters
    (model_size or num_classes), ModelRegistry.create("yolo26", config) raises a
    ConfigurationError whose violations list contains the name of each missing
    required parameter.

    **Validates: Requirements 1.3**
    """

    @given(
        extra_keys=st.dictionaries(
            keys=st.text(
                alphabet=st.characters(whitelist_categories=("L", "N")),
                min_size=1,
                max_size=10,
            ).filter(lambda s: s not in ("model_size", "num_classes")),
            values=st.one_of(st.integers(), st.text(min_size=1, max_size=5), st.booleans()),
            min_size=0,
            max_size=3,
        )
    )
    @settings(max_examples=100)
    def test_missing_both_required_params_raises_error(self, extra_keys):
        # Feature: yolo26-integration, Property 2: Missing required parameters raises ConfigurationError
        """Config missing both model_size and num_classes raises ConfigurationError."""
        config = dict(extra_keys)
        # Ensure required keys are not present
        config.pop("model_size", None)
        config.pop("num_classes", None)

        with patch("model.models.yolo26_wrapper.ultralytics", new=MagicMock()):
            with pytest.raises(ConfigurationError) as exc_info:
                ModelRegistry.create("yolo26", config)

            violations = exc_info.value.violations
            assert any("model_size" in v for v in violations)
            assert any("num_classes" in v for v in violations)

    @given(num_classes=VALID_NUM_CLASSES)
    @settings(max_examples=100)
    def test_missing_model_size_raises_error(self, num_classes):
        # Feature: yolo26-integration, Property 2: Missing required parameters raises ConfigurationError
        """Config missing model_size raises ConfigurationError mentioning model_size."""
        config = {"num_classes": num_classes}

        with patch("model.models.yolo26_wrapper.ultralytics", new=MagicMock()):
            with pytest.raises(ConfigurationError) as exc_info:
                ModelRegistry.create("yolo26", config)

            violations = exc_info.value.violations
            assert any("model_size" in v for v in violations)

    @given(model_size=VALID_MODEL_SIZES)
    @settings(max_examples=100)
    def test_missing_num_classes_raises_error(self, model_size):
        # Feature: yolo26-integration, Property 2: Missing required parameters raises ConfigurationError
        """Config missing num_classes raises ConfigurationError mentioning num_classes."""
        config = {"model_size": model_size}

        with patch("model.models.yolo26_wrapper.ultralytics", new=MagicMock()):
            with pytest.raises(ConfigurationError) as exc_info:
                ModelRegistry.create("yolo26", config)

            violations = exc_info.value.violations
            assert any("num_classes" in v for v in violations)


# ---------------------------------------------------------------------------
# Property 3: Invalid configuration values raise ConfigurationError
# Feature: yolo26-integration, Property 3: Invalid configuration values raise ConfigurationError
# ---------------------------------------------------------------------------


class TestProperty3InvalidConfigValuesRaiseError:
    """Property 3: For any config with invalid model_size (not in valid set),
    num_classes outside [1,1000], or thresholds outside [0.0,1.0],
    constructing YOLO26Detector raises ConfigurationError with violations
    describing the invalid params.

    **Validates: Requirements 2.6, 2.7, 2.8**
    """

    @given(invalid_size=INVALID_MODEL_SIZES, num_classes=VALID_NUM_CLASSES)
    @settings(max_examples=100)
    def test_invalid_model_size_raises_error(self, invalid_size, num_classes):
        # Feature: yolo26-integration, Property 3: Invalid configuration values raise ConfigurationError
        """Invalid model_size raises ConfigurationError with violation describing the value."""
        config = {"model_size": invalid_size, "num_classes": num_classes}

        with patch("model.models.yolo26_wrapper.ultralytics", new=MagicMock()):
            with pytest.raises(ConfigurationError) as exc_info:
                YOLO26Detector(config)

            violations = exc_info.value.violations
            assert any("model_size" in v for v in violations)
            assert any(invalid_size in v for v in violations)

    @given(
        model_size=VALID_MODEL_SIZES,
        invalid_num_classes=st.one_of(INVALID_NUM_CLASSES_LOW, INVALID_NUM_CLASSES_HIGH),
    )
    @settings(max_examples=100)
    def test_invalid_num_classes_raises_error(self, model_size, invalid_num_classes):
        # Feature: yolo26-integration, Property 3: Invalid configuration values raise ConfigurationError
        """num_classes outside [1,1000] raises ConfigurationError with violation."""
        config = {"model_size": model_size, "num_classes": invalid_num_classes}

        with patch("model.models.yolo26_wrapper.ultralytics", new=MagicMock()):
            with pytest.raises(ConfigurationError) as exc_info:
                YOLO26Detector(config)

            violations = exc_info.value.violations
            assert any("num_classes" in v for v in violations)
            assert any(str(invalid_num_classes) in v for v in violations)

    @given(
        model_size=VALID_MODEL_SIZES,
        num_classes=VALID_NUM_CLASSES,
        invalid_conf=st.one_of(INVALID_THRESHOLD_LOW, INVALID_THRESHOLD_HIGH),
    )
    @settings(max_examples=100)
    def test_invalid_confidence_threshold_raises_error(
        self, model_size, num_classes, invalid_conf
    ):
        # Feature: yolo26-integration, Property 3: Invalid configuration values raise ConfigurationError
        """confidence_threshold outside [0.0,1.0] raises ConfigurationError."""
        config = {
            "model_size": model_size,
            "num_classes": num_classes,
            "confidence_threshold": invalid_conf,
        }

        with patch("model.models.yolo26_wrapper.ultralytics", new=MagicMock()):
            with pytest.raises(ConfigurationError) as exc_info:
                YOLO26Detector(config)

            violations = exc_info.value.violations
            assert any("confidence_threshold" in v for v in violations)

    @given(
        model_size=VALID_MODEL_SIZES,
        num_classes=VALID_NUM_CLASSES,
        invalid_iou=st.one_of(INVALID_THRESHOLD_LOW, INVALID_THRESHOLD_HIGH),
    )
    @settings(max_examples=100)
    def test_invalid_iou_threshold_raises_error(
        self, model_size, num_classes, invalid_iou
    ):
        # Feature: yolo26-integration, Property 3: Invalid configuration values raise ConfigurationError
        """iou_threshold outside [0.0,1.0] raises ConfigurationError."""
        config = {
            "model_size": model_size,
            "num_classes": num_classes,
            "iou_threshold": invalid_iou,
        }

        with patch("model.models.yolo26_wrapper.ultralytics", new=MagicMock()):
            with pytest.raises(ConfigurationError) as exc_info:
                YOLO26Detector(config)

            violations = exc_info.value.violations
            assert any("iou_threshold" in v for v in violations)


import torch
import numpy as np


# ---------------------------------------------------------------------------
# Mock classes for Ultralytics Results objects
# ---------------------------------------------------------------------------

class MockBoxes:
    """Mock for Ultralytics Results.boxes object."""

    def __init__(self, xyxy, cls, conf):
        self.xyxy = xyxy
        self.cls = cls
        self.conf = conf

    def __len__(self):
        return self.xyxy.shape[0]


class MockResults:
    """Mock for Ultralytics Results object."""

    def __init__(self, boxes):
        self.boxes = boxes


# ---------------------------------------------------------------------------
# Hypothesis strategies for forward pass testing
# ---------------------------------------------------------------------------

BATCH_SIZES = st.integers(min_value=1, max_value=4)
IMAGE_HEIGHTS = st.sampled_from([32, 64, 128, 256])
IMAGE_WIDTHS = st.sampled_from([32, 64, 128, 256])
DETECTION_COUNTS = st.integers(min_value=0, max_value=20)
CONFIDENCE_THRESHOLDS = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)


@st.composite
def mock_results_for_batch(draw, batch_size, num_classes=5, conf_range=(0.0, 1.0)):
    """Generate a list of MockResults for a batch of images.

    Args:
        batch_size: Number of images in the batch.
        num_classes: Number of classes for label generation.
        conf_range: Tuple (min, max) for confidence values.

    Returns:
        List of MockResults, one per image.
    """
    results = []
    for _ in range(batch_size):
        n_detections = draw(DETECTION_COUNTS)
        if n_detections == 0:
            xyxy = torch.zeros((0, 4))
            cls = torch.zeros((0,))
            conf = torch.zeros((0,))
        else:
            # Generate random box coordinates (x1, y1, x2, y2)
            x1 = torch.rand(n_detections) * 200
            y1 = torch.rand(n_detections) * 200
            x2 = x1 + torch.rand(n_detections) * 50 + 1
            y2 = y1 + torch.rand(n_detections) * 50 + 1
            xyxy = torch.stack([x1, y1, x2, y2], dim=1)

            # Generate random class labels
            cls = torch.randint(0, num_classes, (n_detections,)).float()

            # Generate confidence values within the specified range
            min_conf, max_conf = conf_range
            conf = torch.rand(n_detections) * (max_conf - min_conf) + min_conf

        boxes = MockBoxes(xyxy=xyxy, cls=cls, conf=conf)
        results.append(MockResults(boxes=boxes))
    return results


@st.composite
def forward_pass_inputs(draw):
    """Generate input tensor and corresponding mock results for forward pass."""
    batch_size = draw(BATCH_SIZES)
    height = draw(IMAGE_HEIGHTS)
    width = draw(IMAGE_WIDTHS)

    images = torch.rand(batch_size, 3, height, width)
    results = draw(mock_results_for_batch(batch_size))

    return images, results, batch_size


# ---------------------------------------------------------------------------
# Property 4: Forward pass output structure invariant
# Feature: yolo26-integration, Property 4: Forward pass output structure invariant
# ---------------------------------------------------------------------------


class TestProperty4ForwardPassOutputStructure:
    """Property 4: For any valid YOLO26Detector and input tensor (B, 3, H, W)
    where B >= 1, forward() returns a list of exactly B dicts, each with
    "boxes" (N,4) float32, "labels" (N,) int64, "scores" (N,) in [0.0, 1.0].

    **Validates: Requirements 3.1, 3.2**
    """

    @given(data=st.data())
    @settings(max_examples=100)
    def test_forward_returns_correct_structure(self, data):
        # Feature: yolo26-integration, Property 4: Forward pass output structure invariant
        """Forward pass returns B dicts with correct tensor shapes and dtypes."""
        batch_size = data.draw(BATCH_SIZES)
        height = data.draw(IMAGE_HEIGHTS)
        width = data.draw(IMAGE_WIDTHS)

        images = torch.rand(batch_size, 3, height, width)
        mock_results = data.draw(mock_results_for_batch(batch_size))

        config = {"model_size": "n", "num_classes": 5, "confidence_threshold": 0.0}

        with patch("model.models.yolo26_wrapper.ultralytics") as mock_ul:
            detector = YOLO26Detector(config)
            # Set up the mock model
            mock_model = MagicMock()
            mock_model.predict.return_value = mock_results
            detector._model = mock_model

            predictions = detector.forward(images)

        # Must return exactly B dicts
        assert isinstance(predictions, list)
        assert len(predictions) == batch_size

        for pred in predictions:
            assert isinstance(pred, dict)
            assert "boxes" in pred
            assert "labels" in pred
            assert "scores" in pred

            boxes = pred["boxes"]
            labels = pred["labels"]
            scores = pred["scores"]

            # boxes shape: (N, 4) with float32
            assert boxes.ndim == 2
            assert boxes.shape[1] == 4
            assert boxes.dtype == torch.float32

            # labels shape: (N,) with int64
            assert labels.ndim == 1
            assert labels.dtype == torch.int64

            # scores shape: (N,) with values in [0.0, 1.0]
            assert scores.ndim == 1
            assert scores.dtype == torch.float32
            if scores.numel() > 0:
                assert (scores >= 0.0).all()
                assert (scores <= 1.0).all()

            # All tensors must have consistent N dimension
            n = boxes.shape[0]
            assert labels.shape[0] == n
            assert scores.shape[0] == n


# ---------------------------------------------------------------------------
# Property 5: Confidence threshold filtering
# Feature: yolo26-integration, Property 5: Confidence threshold filtering
# ---------------------------------------------------------------------------


class TestProperty5ConfidenceThresholdFiltering:
    """Property 5: For any configured confidence_threshold in [0.0, 1.0],
    all scores in returned predictions satisfy score >= confidence_threshold.

    **Validates: Requirements 3.5**
    """

    @given(data=st.data())
    @settings(max_examples=100)
    def test_all_scores_above_threshold(self, data):
        # Feature: yolo26-integration, Property 5: Confidence threshold filtering
        """All returned scores are >= the configured confidence_threshold."""
        batch_size = data.draw(BATCH_SIZES)
        height = data.draw(IMAGE_HEIGHTS)
        width = data.draw(IMAGE_WIDTHS)
        threshold = data.draw(CONFIDENCE_THRESHOLDS)

        images = torch.rand(batch_size, 3, height, width)
        # Generate results with full range of confidence values
        mock_results = data.draw(mock_results_for_batch(batch_size))

        config = {
            "model_size": "n",
            "num_classes": 5,
            "confidence_threshold": threshold,
        }

        with patch("model.models.yolo26_wrapper.ultralytics") as mock_ul:
            detector = YOLO26Detector(config)
            mock_model = MagicMock()
            mock_model.predict.return_value = mock_results
            detector._model = mock_model

            predictions = detector.forward(images)

        for pred in predictions:
            scores = pred["scores"]
            if scores.numel() > 0:
                assert (scores >= threshold).all(), (
                    f"Found score {scores.min().item()} below threshold {threshold}"
                )


# ---------------------------------------------------------------------------
# Property 6: Ultralytics result format conversion preserves data
# Feature: yolo26-integration, Property 6: Ultralytics result format conversion preserves data
# ---------------------------------------------------------------------------


class TestProperty6ResultConversionPreservesData:
    """Property 6: For any valid Ultralytics Results with boxes.xyxy (N,4),
    boxes.cls (N,), boxes.conf (N,), conversion preserves data numerically
    (after filtering).

    **Validates: Requirements 3.6**
    """

    @given(data=st.data())
    @settings(max_examples=100)
    def test_conversion_preserves_data(self, data):
        # Feature: yolo26-integration, Property 6: Ultralytics result format conversion preserves data
        """Converted predictions match source data numerically after filtering."""
        n_detections = data.draw(st.integers(min_value=1, max_value=20))
        num_classes = 5

        # Generate source data with all confidences above threshold
        # so we can verify exact preservation
        threshold = 0.25
        x1 = torch.rand(n_detections) * 200
        y1 = torch.rand(n_detections) * 200
        x2 = x1 + torch.rand(n_detections) * 50 + 1
        y2 = y1 + torch.rand(n_detections) * 50 + 1
        source_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
        source_cls = torch.randint(0, num_classes, (n_detections,)).float()
        # All confidences above threshold to ensure all pass filtering
        source_conf = torch.rand(n_detections) * (1.0 - threshold) + threshold

        mock_boxes = MockBoxes(xyxy=source_xyxy, cls=source_cls, conf=source_conf)
        mock_result = MockResults(boxes=mock_boxes)

        config = {
            "model_size": "n",
            "num_classes": num_classes,
            "confidence_threshold": threshold,
        }

        with patch("model.models.yolo26_wrapper.ultralytics") as mock_ul:
            detector = YOLO26Detector(config)
            device = torch.device("cpu")

            # Call _convert_results directly
            predictions = detector._convert_results([mock_result], device)

        assert len(predictions) == 1
        pred = predictions[0]

        # All detections should pass the threshold
        assert pred["boxes"].shape[0] == n_detections
        assert pred["labels"].shape[0] == n_detections
        assert pred["scores"].shape[0] == n_detections

        # Verify numerical equivalence
        assert torch.allclose(pred["boxes"], source_xyxy.float(), atol=1e-6)
        assert torch.equal(pred["labels"], source_cls.long())
        assert torch.allclose(pred["scores"], source_conf.float(), atol=1e-6)


# ---------------------------------------------------------------------------
# Property 7: Output device consistency
# Feature: yolo26-integration, Property 7: Output device consistency
# ---------------------------------------------------------------------------


class TestProperty7OutputDeviceConsistency:
    """Property 7: For any input tensor on a specific device, all output tensors
    are on the same device.

    **Validates: Requirements 3.8**
    """

    @given(data=st.data())
    @settings(max_examples=100)
    def test_output_tensors_on_same_device_as_input(self, data):
        # Feature: yolo26-integration, Property 7: Output device consistency
        """All output tensors are on the same device as the input tensor."""
        batch_size = data.draw(BATCH_SIZES)
        height = data.draw(IMAGE_HEIGHTS)
        width = data.draw(IMAGE_WIDTHS)

        # Use CPU device (CUDA testing requires GPU availability)
        device = torch.device("cpu")
        images = torch.rand(batch_size, 3, height, width, device=device)
        mock_results = data.draw(mock_results_for_batch(batch_size))

        config = {
            "model_size": "n",
            "num_classes": 5,
            "confidence_threshold": 0.0,
        }

        with patch("model.models.yolo26_wrapper.ultralytics") as mock_ul:
            detector = YOLO26Detector(config)
            mock_model = MagicMock()
            mock_model.predict.return_value = mock_results
            detector._model = mock_model

            predictions = detector.forward(images)

        for pred in predictions:
            assert pred["boxes"].device == device, (
                f"boxes on {pred['boxes'].device}, expected {device}"
            )
            assert pred["labels"].device == device, (
                f"labels on {pred['labels'].device}, expected {device}"
            )
            assert pred["scores"].device == device, (
                f"scores on {pred['scores'].device}, expected {device}"
            )


import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Hypothesis strategies for checkpoint path testing
# ---------------------------------------------------------------------------

# Strategy for generating filesystem-safe path strings that don't exist
FILESYSTEM_SAFE_CHARS = st.characters(
    whitelist_categories=("L", "N"),
    whitelist_characters="_-.",
)

NON_EXISTENT_PATH_SEGMENTS = st.text(
    alphabet=FILESYSTEM_SAFE_CHARS,
    min_size=1,
    max_size=20,
)


@st.composite
def non_existent_file_paths(draw):
    """Strategy that generates file paths guaranteed not to exist on the filesystem.

    Generates paths with 1-4 segments joined by os.sep, prefixed with a
    non-existent root directory to ensure they cannot accidentally exist.
    """
    # Use a prefix that is extremely unlikely to exist
    prefix = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L",), whitelist_characters="_"),
        min_size=5,
        max_size=10,
    ))
    segments = draw(st.lists(NON_EXISTENT_PATH_SEGMENTS, min_size=1, max_size=3))
    # Add a .pt extension to make it look like a checkpoint path
    filename = draw(NON_EXISTENT_PATH_SEGMENTS) + ".pt"
    parts = [f"__nonexistent_{prefix}__"] + segments + [filename]
    path_str = os.sep.join(parts)
    # Ensure the path does not actually exist
    assume(not os.path.exists(path_str))
    return path_str


# ---------------------------------------------------------------------------
# Property 8: Non-existent checkpoint path raises FileNotFoundError
# Feature: yolo26-integration, Property 8: Non-existent checkpoint path raises FileNotFoundError
# ---------------------------------------------------------------------------


class TestProperty8NonExistentCheckpointRaisesFileNotFoundError:
    """Property 8: For any file path that does not exist on the filesystem,
    calling load_checkpoint(path) SHALL raise a FileNotFoundError whose message
    contains the attempted path string.

    **Validates: Requirements 4.3, 5.3**
    """

    @given(path_str=non_existent_file_paths())
    @settings(max_examples=100)
    def test_non_existent_path_raises_file_not_found_error(self, path_str):
        # Feature: yolo26-integration, Property 8: Non-existent checkpoint path raises FileNotFoundError
        """Any non-existent path passed to load_checkpoint raises FileNotFoundError with path in message."""
        config = {"model_size": "n", "num_classes": 5}

        with patch("model.models.yolo26_wrapper.ultralytics", new=MagicMock()):
            detector = YOLO26Detector(config)

            with pytest.raises(FileNotFoundError) as exc_info:
                detector.load_checkpoint(Path(path_str))

            # The error message must contain the attempted path string
            error_message = str(exc_info.value)
            assert path_str in error_message or str(Path(path_str)) in error_message, (
                f"FileNotFoundError message '{error_message}' does not contain "
                f"the attempted path '{path_str}'"
            )


import torch.nn as nn


# ---------------------------------------------------------------------------
# Helper nn.Module for training integration tests
# ---------------------------------------------------------------------------

class SimpleDetectorModule(nn.Module):
    """A minimal nn.Module that simulates a detection model for testing.

    Produces a differentiable output from input images that can be used
    to compute a loss with grad_fn.
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        # Simple conv + linear to produce a differentiable scalar output
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        """Forward pass returning feature tensor."""
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def parameters(self, recurse=True):
        """Return all parameters (all require grad by default)."""
        return super().parameters(recurse=recurse)


# ---------------------------------------------------------------------------
# Hypothesis strategies for training integration testing
# ---------------------------------------------------------------------------

TRAINING_BATCH_SIZES = st.integers(min_value=1, max_value=4)
NUM_BOXES_PER_IMAGE = st.integers(min_value=1, max_value=10)
NUM_CLASSES_TRAINING = st.integers(min_value=1, max_value=20)


@st.composite
def training_targets(draw, batch_size, num_classes, img_size=64):
    """Generate a list of target dicts for training.

    Each target has:
        - "boxes": Tensor of shape (N, 4) in xyxy format with valid coordinates
        - "labels": Tensor of shape (N,) with class indices in [0, num_classes)
    """
    targets = []
    for _ in range(batch_size):
        n_boxes = draw(NUM_BOXES_PER_IMAGE)
        # Generate valid xyxy boxes within image bounds
        x1 = torch.rand(n_boxes) * (img_size - 2)
        y1 = torch.rand(n_boxes) * (img_size - 2)
        x2 = x1 + torch.rand(n_boxes) * (img_size - x1.max().item()) + 1
        y2 = y1 + torch.rand(n_boxes) * (img_size - y1.max().item()) + 1
        # Clamp to valid range
        x2 = torch.clamp(x2, max=img_size)
        y2 = torch.clamp(y2, max=img_size)
        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        labels = torch.randint(0, num_classes, (n_boxes,))
        targets.append({"boxes": boxes, "labels": labels})
    return targets


# ---------------------------------------------------------------------------
# Property 9: get_parameters returns trainable parameters
# Feature: yolo26-integration, Property 9: get_parameters returns trainable parameters
# ---------------------------------------------------------------------------


class TestProperty9GetParametersReturnsTrainableParameters:
    """Property 9: For any valid YOLO26Detector instance (regardless of model_size),
    calling get_parameters() SHALL return a non-empty list where every element is a
    torch.nn.Parameter instance with requires_grad == True.

    **Validates: Requirements 5.4**
    """

    @given(model_size=VALID_MODEL_SIZES)
    @settings(max_examples=100)
    def test_get_parameters_returns_trainable_params(self, model_size):
        # Feature: yolo26-integration, Property 9: get_parameters returns trainable parameters
        """get_parameters() returns non-empty list of Parameters with requires_grad=True."""
        config = {"model_size": model_size, "num_classes": 5}

        # Create a real nn.Module to serve as the underlying model
        real_module = SimpleDetectorModule(num_classes=5)

        with patch("model.models.yolo26_wrapper.ultralytics") as mock_ul:
            # Mock ultralytics.YOLO to return an object with .model = real nn.Module
            mock_yolo_instance = MagicMock()
            mock_yolo_instance.model = real_module
            mock_ul.YOLO.return_value = mock_yolo_instance

            detector = YOLO26Detector(config)

        params = detector.get_parameters()

        # Must be non-empty
        assert len(params) > 0, "get_parameters() returned an empty list"

        # Every element must be a torch.nn.Parameter with requires_grad=True
        for i, p in enumerate(params):
            assert isinstance(p, torch.nn.Parameter), (
                f"Element {i} is {type(p).__name__}, expected torch.nn.Parameter"
            )
            assert p.requires_grad is True, (
                f"Parameter {i} has requires_grad=False, expected True"
            )


# ---------------------------------------------------------------------------
# Property 10: train_step returns differentiable loss tensor
# Feature: yolo26-integration, Property 10: train_step returns differentiable loss tensor
# ---------------------------------------------------------------------------


class TestProperty10TrainStepReturnsDifferentiableLoss:
    """Property 10: For any non-empty batch of image tensors and corresponding target
    dicts (each with "boxes" (N,4) and "labels" (N,)), calling train_step(images, targets)
    SHALL return a dict containing "loss_tensor" key whose value is a scalar torch.Tensor
    (0-dimensional) with a non-None grad_fn attribute.

    **Validates: Requirements 5.5**
    """

    @given(data=st.data())
    @settings(max_examples=100)
    def test_train_step_returns_scalar_loss_with_grad_fn(self, data):
        # Feature: yolo26-integration, Property 10: train_step returns differentiable loss tensor
        """train_step returns dict with scalar loss tensor that has grad_fn."""
        batch_size = data.draw(TRAINING_BATCH_SIZES)
        num_classes = data.draw(NUM_CLASSES_TRAINING)
        img_size = 64

        # Generate random images
        images = [torch.rand(3, img_size, img_size) for _ in range(batch_size)]

        # Generate random targets
        targets = data.draw(training_targets(batch_size, num_classes, img_size))

        config = {"model_size": "n", "num_classes": num_classes}

        # Create a real nn.Module as the underlying model
        real_module = SimpleDetectorModule(num_classes=num_classes)

        # Create a simple differentiable loss function that works with our module
        def simple_criterion(preds, batch_dict):
            """Simple loss: MSE between predictions and target class counts."""
            # Just compute a differentiable scalar from the predictions
            if isinstance(preds, (list, tuple)):
                preds = preds[0] if len(preds) > 0 else torch.tensor(0.0)
            if isinstance(preds, torch.Tensor):
                return preds.sum().unsqueeze(0)
            return torch.tensor(0.0, requires_grad=True)

        with patch("model.models.yolo26_wrapper.ultralytics") as mock_ul:
            # Mock ultralytics.YOLO to return an object with .model = real nn.Module
            mock_yolo_instance = MagicMock()
            mock_yolo_instance.model = real_module
            mock_ul.YOLO.return_value = mock_yolo_instance

            detector = YOLO26Detector(config)
            # Override the loss function with our simple differentiable one
            detector._loss_fn = simple_criterion

        result = detector.train_step(images, targets)

        # Must return a dict with "loss_tensor" key
        assert isinstance(result, dict), f"Expected dict, got {type(result).__name__}"
        assert "loss_tensor" in result, "Result dict missing 'loss_tensor' key"

        loss = result["loss_tensor"]

        # Must be a torch.Tensor
        assert isinstance(loss, torch.Tensor), (
            f"loss_tensor is {type(loss).__name__}, expected torch.Tensor"
        )

        # Must be scalar (0-dimensional)
        assert loss.dim() == 0, (
            f"loss_tensor has {loss.dim()} dimensions, expected 0 (scalar)"
        )

        # Must have grad_fn (differentiable)
        assert loss.grad_fn is not None, (
            "loss_tensor.grad_fn is None — loss is not differentiable"
        )


import copy
import yaml as pyyaml

from model.config.manager import ConfigManager
from model.exceptions import ValidationError


# ---------------------------------------------------------------------------
# Hypothesis strategies for YAML model.config field removal
# ---------------------------------------------------------------------------

REQUIRED_MODEL_CONFIG_FIELDS = [
    "model_size",
    "num_classes",
    "end2end",
    "confidence_threshold",
    "iou_threshold",
]

# Strategy to generate a non-empty subset of required fields to remove
FIELDS_TO_REMOVE = st.sets(
    st.sampled_from(REQUIRED_MODEL_CONFIG_FIELDS),
    min_size=1,
)

# Schema for validating model.config section with all required fields
YOLO26_MODEL_CONFIG_SCHEMA: dict = {
    "required": ["type", "config"],
    "properties": {
        "type": {"type": "str"},
        "config": {
            "type": "dict",
            "required": REQUIRED_MODEL_CONFIG_FIELDS,
            "properties": {
                "model_size": {"type": "str", "enum": ["n", "s", "m", "l", "x"]},
                "num_classes": {"type": "int", "min": 1, "max": 1000},
                "end2end": {"type": "bool"},
                "confidence_threshold": {"type": "float", "min": 0.0, "max": 1.0},
                "iou_threshold": {"type": "float", "min": 0.0, "max": 1.0},
            },
        },
    },
}

# Full experiment schema with model.config required fields for yolo26
YOLO26_EXPERIMENT_SCHEMA: dict = {
    "required": ["name", "model", "dataset", "training", "evaluation", "output"],
    "properties": {
        "name": {"type": "str"},
        "model": {
            "type": "dict",
            "required": ["type", "config"],
            "properties": {
                "type": {"type": "str"},
                "config": {
                    "type": "dict",
                    "required": REQUIRED_MODEL_CONFIG_FIELDS,
                    "properties": {
                        "model_size": {"type": "str", "enum": ["n", "s", "m", "l", "x"]},
                        "num_classes": {"type": "int", "min": 1, "max": 1000},
                        "end2end": {"type": "bool"},
                        "confidence_threshold": {"type": "float", "min": 0.0, "max": 1.0},
                        "iou_threshold": {"type": "float", "min": 0.0, "max": 1.0},
                    },
                },
            },
        },
        "dataset": {"type": "dict", "required": ["type", "path"]},
        "training": {"type": "dict", "required": ["epochs", "batch_size", "learning_rate", "optimizer"]},
        "evaluation": {"type": "dict"},
        "output": {"type": "dict"},
    },
}


def _load_valid_yolo26_config() -> dict:
    """Load the valid train_yolo26.yaml config as a dict."""
    config_path = Path(__file__).resolve().parents[2] / "configs" / "train_yolo26.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return pyyaml.safe_load(f)


# ---------------------------------------------------------------------------
# Property 11: Missing YAML model.config fields raises ValidationError
# Feature: yolo26-integration, Property 11: Missing YAML model.config fields raises ValidationError
# ---------------------------------------------------------------------------


class TestProperty11MissingYAMLModelConfigFieldsRaisesValidationError:
    """Property 11: For any experiment YAML configuration where the model.config
    section is missing at least one of the required fields (model_size, num_classes,
    end2end, confidence_threshold, iou_threshold), the ConfigManager.validate() call
    SHALL raise a ValidationError whose schema_violations list contains the name of
    each missing field.

    **Validates: Requirements 6.5**
    """

    @given(fields_to_remove=FIELDS_TO_REMOVE)
    @settings(max_examples=100)
    def test_missing_model_config_fields_raises_validation_error(self, fields_to_remove):
        # Feature: yolo26-integration, Property 11: Missing YAML model.config fields raises ValidationError
        """Removing any subset of required model.config fields raises ValidationError
        with schema_violations containing each missing field name."""
        # Load a valid config and remove the selected fields
        config = _load_valid_yolo26_config()
        assert "model" in config and "config" in config["model"], (
            "Valid YAML must have model.config section"
        )

        # Remove the selected fields from model.config
        model_config = config["model"]["config"]
        for field in fields_to_remove:
            model_config.pop(field, None)

        # Validate against the schema with required model.config fields
        manager = ConfigManager()
        with pytest.raises(ValidationError) as exc_info:
            manager.validate(config, YOLO26_EXPERIMENT_SCHEMA)

        # Each removed field must appear in schema_violations
        violations = exc_info.value.schema_violations
        for field in fields_to_remove:
            assert any(field in v for v in violations), (
                f"Missing field '{field}' not found in schema_violations: {violations}"
            )
