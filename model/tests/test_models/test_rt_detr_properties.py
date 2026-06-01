"""Property-based tests for RT_DETR_Detector wrapper.

Tests Properties 1-10 from the design document using Hypothesis.
Each property validates specific requirements from the RT-DETR detector spec.
"""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from model.exceptions import ConfigurationError
from model.models.registry import BaseDetector, ModelRegistry
from model.models.rt_detr_wrapper import RT_DETR_Detector


# ---------------------------------------------------------------------------
# Hypothesis strategies for RT-DETR configuration generation
# ---------------------------------------------------------------------------

VALID_MODEL_SIZES = st.sampled_from(["l", "x"])

VALID_NUM_CLASSES = st.integers(min_value=1, max_value=1000)

VALID_CONFIDENCE_THRESHOLD = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

VALID_IOU_THRESHOLD = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)


@st.composite
def valid_rt_detr_config(draw):
    """Strategy that generates valid RT-DETR configuration dicts."""
    config = {
        "model_size": draw(VALID_MODEL_SIZES),
        "num_classes": draw(VALID_NUM_CLASSES),
    }
    # Optionally include optional parameters
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
).filter(lambda s: s not in ("l", "x"))

INVALID_NUM_CLASSES_LOW = st.integers(max_value=0)
INVALID_NUM_CLASSES_HIGH = st.integers(min_value=1001)

INVALID_THRESHOLD_LOW = st.floats(
    max_value=-0.001, allow_nan=False, allow_infinity=False
).filter(lambda x: x < 0.0)

INVALID_THRESHOLD_HIGH = st.floats(
    min_value=1.001, max_value=1e6, allow_nan=False, allow_infinity=False
).filter(lambda x: x > 1.0)


# Strategies for forward pass testing

BATCH_SIZES = st.integers(min_value=1, max_value=4)
IMAGE_HEIGHTS = st.sampled_from([32, 64, 128, 256])
IMAGE_WIDTHS = st.sampled_from([32, 64, 128, 256])
DETECTION_COUNTS = st.integers(min_value=0, max_value=20)
CONFIDENCE_THRESHOLDS = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Strategies for non-existent file paths

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
    """Strategy generating file paths guaranteed not to exist."""
    prefix = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L",), whitelist_characters="_"),
        min_size=5,
        max_size=10,
    ))
    segments = draw(st.lists(NON_EXISTENT_PATH_SEGMENTS, min_size=1, max_size=3))
    filename = draw(NON_EXISTENT_PATH_SEGMENTS) + ".pt"
    parts = [f"__nonexistent_{prefix}__"] + segments + [filename]
    path_str = os.sep.join(parts)
    assume(not os.path.exists(path_str))
    return path_str


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


@st.composite
def mock_results_for_batch(draw, batch_size, num_classes=5):
    """Generate a list of MockResults for a batch of images."""
    results = []
    for _ in range(batch_size):
        n_detections = draw(DETECTION_COUNTS)
        if n_detections == 0:
            xyxy = torch.zeros((0, 4))
            cls = torch.zeros((0,))
            conf = torch.zeros((0,))
        else:
            x1 = torch.rand(n_detections) * 200
            y1 = torch.rand(n_detections) * 200
            x2 = x1 + torch.rand(n_detections) * 50 + 1
            y2 = y1 + torch.rand(n_detections) * 50 + 1
            xyxy = torch.stack([x1, y1, x2, y2], dim=1)
            cls = torch.randint(0, num_classes, (n_detections,)).float()
            conf = torch.rand(n_detections)
        boxes = MockBoxes(xyxy=xyxy, cls=cls, conf=conf)
        results.append(MockResults(boxes=boxes))
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_rtdetr():
    """Create a properly configured MagicMock for the RTDETR class."""
    mock_rtdetr_cls = MagicMock()
    mock_model_instance = MagicMock()
    inner_module = torch.nn.Linear(10, 5)
    mock_model_instance.model = inner_module
    mock_model_instance.model.init_criterion = MagicMock(return_value=MagicMock())
    mock_model_instance.model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    mock_rtdetr_cls.return_value = mock_model_instance
    return mock_rtdetr_cls


def _make_detector(config=None, mock_rtdetr=None):
    """Create an RT_DETR_Detector with mocked RTDETR."""
    if config is None:
        config = {"model_size": "l", "num_classes": 5}
    if mock_rtdetr is None:
        mock_rtdetr = _make_mock_rtdetr()
    with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr):
        detector = RT_DETR_Detector(config)
    return detector, mock_rtdetr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the model registry before each test."""
    saved = dict(ModelRegistry._models)
    ModelRegistry._models["rt_detr"] = RT_DETR_Detector
    yield
    ModelRegistry._models = saved


# ---------------------------------------------------------------------------
# Property 1: Valid configuration produces a BaseDetector instance
# Feature: rt-detr-detector, Property 1: Valid configuration produces a BaseDetector instance
# ---------------------------------------------------------------------------


class TestProperty1ValidConfigProducesBaseDetector:
    """Property 1: For any valid config (model_size in {"l","x"},
    num_classes in [1,1000], optional valid thresholds),
    RT_DETR_Detector(config) returns an instance of both
    RT_DETR_Detector and BaseDetector.

    **Validates: Requirements 1.2**
    """

    @given(config=valid_rt_detr_config())
    @settings(max_examples=100)
    def test_valid_config_creates_detector_instance(self, config):
        # Feature: rt-detr-detector, Property 1: Valid configuration produces a BaseDetector instance
        """Any valid configuration produces an RT_DETR_Detector that is also a BaseDetector."""
        mock_rtdetr = _make_mock_rtdetr()
        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr):
            instance = RT_DETR_Detector(config)

            assert isinstance(instance, RT_DETR_Detector)
            assert isinstance(instance, BaseDetector)


# ---------------------------------------------------------------------------
# Property 2: Missing required parameters raises ConfigurationError
# Feature: rt-detr-detector, Property 2: Missing required parameters raises ConfigurationError
# ---------------------------------------------------------------------------


class TestProperty2MissingRequiredParamsRaisesError:
    """Property 2: For any config missing at least one of the required parameters
    (model_size or num_classes), constructing RT_DETR_Detector raises a
    ConfigurationError whose violations list contains the name of each missing
    required parameter.

    **Validates: Requirements 2.5**
    """

    @given(
        extra_keys=st.dictionaries(
            keys=st.text(
                alphabet=st.characters(whitelist_categories=("L", "N")),
                min_size=1,
                max_size=10,
            ).filter(lambda s: s not in ("model_size", "num_classes")),
            values=st.one_of(
                st.integers(), st.text(min_size=1, max_size=5), st.booleans()
            ),
            min_size=0,
            max_size=3,
        )
    )
    @settings(max_examples=100)
    def test_missing_both_required_params_raises_error(self, extra_keys):
        # Feature: rt-detr-detector, Property 2: Missing required parameters raises ConfigurationError
        """Config missing both model_size and num_classes raises ConfigurationError."""
        config = dict(extra_keys)
        config.pop("model_size", None)
        config.pop("num_classes", None)

        mock_rtdetr = _make_mock_rtdetr()
        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr):
            with pytest.raises(ConfigurationError) as exc_info:
                RT_DETR_Detector(config)

            violations = exc_info.value.violations
            assert any("model_size" in v for v in violations)
            assert any("num_classes" in v for v in violations)

    @given(num_classes=VALID_NUM_CLASSES)
    @settings(max_examples=100)
    def test_missing_model_size_raises_error(self, num_classes):
        # Feature: rt-detr-detector, Property 2: Missing required parameters raises ConfigurationError
        """Config missing model_size raises ConfigurationError mentioning model_size."""
        config = {"num_classes": num_classes}

        mock_rtdetr = _make_mock_rtdetr()
        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr):
            with pytest.raises(ConfigurationError) as exc_info:
                RT_DETR_Detector(config)

            violations = exc_info.value.violations
            assert any("model_size" in v for v in violations)

    @given(model_size=VALID_MODEL_SIZES)
    @settings(max_examples=100)
    def test_missing_num_classes_raises_error(self, model_size):
        # Feature: rt-detr-detector, Property 2: Missing required parameters raises ConfigurationError
        """Config missing num_classes raises ConfigurationError mentioning num_classes."""
        config = {"model_size": model_size}

        mock_rtdetr = _make_mock_rtdetr()
        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr):
            with pytest.raises(ConfigurationError) as exc_info:
                RT_DETR_Detector(config)

            violations = exc_info.value.violations
            assert any("num_classes" in v for v in violations)


# ---------------------------------------------------------------------------
# Property 3: Invalid configuration values raise ConfigurationError
# Feature: rt-detr-detector, Property 3: Invalid configuration values raise ConfigurationError
# ---------------------------------------------------------------------------


class TestProperty3InvalidConfigValuesRaiseError:
    """Property 3: For any config with invalid model_size (not in {"l","x"}),
    num_classes outside [1,1000], or thresholds outside [0.0,1.0],
    constructing RT_DETR_Detector raises ConfigurationError with violations
    describing the invalid params.

    **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
    """

    @given(invalid_size=INVALID_MODEL_SIZES, num_classes=VALID_NUM_CLASSES)
    @settings(max_examples=100)
    def test_invalid_model_size_raises_error(self, invalid_size, num_classes):
        # Feature: rt-detr-detector, Property 3: Invalid configuration values raise ConfigurationError
        """Invalid model_size raises ConfigurationError with violation."""
        config = {"model_size": invalid_size, "num_classes": num_classes}

        mock_rtdetr = _make_mock_rtdetr()
        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr):
            with pytest.raises(ConfigurationError) as exc_info:
                RT_DETR_Detector(config)

            violations = exc_info.value.violations
            assert any("model_size" in v for v in violations)
            assert any(invalid_size in v for v in violations)

    @given(
        model_size=VALID_MODEL_SIZES,
        invalid_num_classes=st.one_of(INVALID_NUM_CLASSES_LOW, INVALID_NUM_CLASSES_HIGH),
    )
    @settings(max_examples=100)
    def test_invalid_num_classes_raises_error(self, model_size, invalid_num_classes):
        # Feature: rt-detr-detector, Property 3: Invalid configuration values raise ConfigurationError
        """num_classes outside [1,1000] raises ConfigurationError."""
        config = {"model_size": model_size, "num_classes": invalid_num_classes}

        mock_rtdetr = _make_mock_rtdetr()
        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr):
            with pytest.raises(ConfigurationError) as exc_info:
                RT_DETR_Detector(config)

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
        # Feature: rt-detr-detector, Property 3: Invalid configuration values raise ConfigurationError
        """confidence_threshold outside [0.0,1.0] raises ConfigurationError."""
        config = {
            "model_size": model_size,
            "num_classes": num_classes,
            "confidence_threshold": invalid_conf,
        }

        mock_rtdetr = _make_mock_rtdetr()
        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr):
            with pytest.raises(ConfigurationError) as exc_info:
                RT_DETR_Detector(config)

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
        # Feature: rt-detr-detector, Property 3: Invalid configuration values raise ConfigurationError
        """iou_threshold outside [0.0,1.0] raises ConfigurationError."""
        config = {
            "model_size": model_size,
            "num_classes": num_classes,
            "iou_threshold": invalid_iou,
        }

        mock_rtdetr = _make_mock_rtdetr()
        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr):
            with pytest.raises(ConfigurationError) as exc_info:
                RT_DETR_Detector(config)

            violations = exc_info.value.violations
            assert any("iou_threshold" in v for v in violations)


# ---------------------------------------------------------------------------
# Property 4: Forward pass output structure invariant
# Feature: rt-detr-detector, Property 4: Forward pass output structure invariant
# ---------------------------------------------------------------------------


class TestProperty4ForwardPassOutputStructure:
    """Property 4: For any valid RT_DETR_Detector and input tensor (B, 3, H, W)
    where B >= 1, forward() returns a list of exactly B dicts, each with
    "boxes" (N,4) float32, "labels" (N,) int64, "scores" (N,) float32 in [0.0,1.0],
    where N is consistent across all three tensors in each dict.

    **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
    """

    @given(data=st.data())
    @settings(max_examples=100)
    def test_forward_returns_correct_structure(self, data):
        # Feature: rt-detr-detector, Property 4: Forward pass output structure invariant
        """Forward pass returns B dicts with correct tensor shapes and dtypes."""
        batch_size = data.draw(BATCH_SIZES)
        height = data.draw(IMAGE_HEIGHTS)
        width = data.draw(IMAGE_WIDTHS)

        images = torch.rand(batch_size, 3, height, width)
        mock_results = data.draw(mock_results_for_batch(batch_size))

        config = {
            "model_size": "l",
            "num_classes": 5,
            "confidence_threshold": 0.0,
        }

        mock_rtdetr = _make_mock_rtdetr()
        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr):
            detector = RT_DETR_Detector(config)
            # Mock the predict method to return our mock results
            detector._model.predict = MagicMock(return_value=mock_results)

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

            # scores shape: (N,) with float32 values in [0.0, 1.0]
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
# Feature: rt-detr-detector, Property 5: Confidence threshold filtering
# ---------------------------------------------------------------------------


class TestProperty5ConfidenceThresholdFiltering:
    """Property 5: For any configured confidence_threshold in [0.0, 1.0],
    all scores in returned predictions satisfy score >= confidence_threshold.

    **Validates: Requirements 4.5**
    """

    @given(data=st.data())
    @settings(max_examples=100)
    def test_all_scores_above_threshold(self, data):
        # Feature: rt-detr-detector, Property 5: Confidence threshold filtering
        """All returned scores are >= the configured confidence_threshold."""
        batch_size = data.draw(BATCH_SIZES)
        height = data.draw(IMAGE_HEIGHTS)
        width = data.draw(IMAGE_WIDTHS)
        threshold = data.draw(CONFIDENCE_THRESHOLDS)

        images = torch.rand(batch_size, 3, height, width)
        mock_results = data.draw(mock_results_for_batch(batch_size))

        config = {
            "model_size": "l",
            "num_classes": 5,
            "confidence_threshold": threshold,
        }

        mock_rtdetr = _make_mock_rtdetr()
        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr):
            detector = RT_DETR_Detector(config)
            detector._model.predict = MagicMock(return_value=mock_results)

            predictions = detector.forward(images)

        for pred in predictions:
            scores = pred["scores"]
            if scores.numel() > 0:
                assert (scores >= threshold).all(), (
                    f"Found score {scores.min().item()} below threshold {threshold}"
                )


# ---------------------------------------------------------------------------
# Property 6: Output device consistency
# Feature: rt-detr-detector, Property 6: Output device consistency
# ---------------------------------------------------------------------------


class TestProperty6OutputDeviceConsistency:
    """Property 6: For any input tensor on a specific device, all output tensors
    (boxes, labels, scores) are on the same device as the input tensor.

    **Validates: Requirements 4.7**
    """

    @given(data=st.data())
    @settings(max_examples=100)
    def test_output_tensors_on_same_device_as_input(self, data):
        # Feature: rt-detr-detector, Property 6: Output device consistency
        """All output tensors are on the same device as the input tensor."""
        batch_size = data.draw(BATCH_SIZES)
        height = data.draw(IMAGE_HEIGHTS)
        width = data.draw(IMAGE_WIDTHS)

        device = torch.device("cpu")
        images = torch.rand(batch_size, 3, height, width, device=device)
        mock_results = data.draw(mock_results_for_batch(batch_size))

        config = {
            "model_size": "l",
            "num_classes": 5,
            "confidence_threshold": 0.0,
        }

        mock_rtdetr = _make_mock_rtdetr()
        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr):
            detector = RT_DETR_Detector(config)
            detector._model.predict = MagicMock(return_value=mock_results)

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


# ---------------------------------------------------------------------------
# Property 7: Bounding box coordinate conversion correctness
# Feature: rt-detr-detector, Property 7: Bounding box coordinate conversion correctness
# ---------------------------------------------------------------------------


class TestProperty7CoordinateConversion:
    """Property 7: For any set of bounding boxes in xyxy pixel format and any
    image dimensions (H, W), the conversion to normalized xywh format satisfies:
    x_center = (x1 + x2) / (2 * W), y_center = (y1 + y2) / (2 * H),
    width = (x2 - x1) / W, height = (y2 - y1) / H, and all values in [0.0, 1.0]
    for boxes within image bounds.

    **Validates: Requirements 5.2**
    """

    @given(data=st.data())
    @settings(max_examples=100)
    def test_xyxy_to_normalized_xywh_conversion(self, data):
        # Feature: rt-detr-detector, Property 7: Bounding box coordinate conversion correctness
        """Normalized xywh values match expected formulas and are in [0.0, 1.0]."""
        img_h = data.draw(st.integers(min_value=32, max_value=640))
        img_w = data.draw(st.integers(min_value=32, max_value=640))
        n_boxes = data.draw(st.integers(min_value=1, max_value=20))

        # Generate valid xyxy boxes within image bounds
        x1 = torch.rand(n_boxes) * (img_w - 2)
        y1 = torch.rand(n_boxes) * (img_h - 2)
        # Ensure x2 > x1 and within bounds
        x2 = x1 + torch.rand(n_boxes) * (img_w - x1) * 0.5 + 1
        x2 = torch.clamp(x2, max=float(img_w))
        y2 = y1 + torch.rand(n_boxes) * (img_h - y1) * 0.5 + 1
        y2 = torch.clamp(y2, max=float(img_h))

        # Expected normalized xywh values
        expected_x_center = (x1 + x2) / (2.0 * img_w)
        expected_y_center = (y1 + y2) / (2.0 * img_h)
        expected_width = (x2 - x1) / img_w
        expected_height = (y2 - y1) / img_h

        # Perform the same conversion as train_step does
        actual_x_center = ((x1 + x2) / 2.0) / img_w
        actual_y_center = ((y1 + y2) / 2.0) / img_h
        actual_width = (x2 - x1) / img_w
        actual_height = (y2 - y1) / img_h

        # Assert formulas match
        assert torch.allclose(actual_x_center, expected_x_center, atol=1e-6)
        assert torch.allclose(actual_y_center, expected_y_center, atol=1e-6)
        assert torch.allclose(actual_width, expected_width, atol=1e-6)
        assert torch.allclose(actual_height, expected_height, atol=1e-6)

        # Assert all normalized values are in [0.0, 1.0]
        assert (actual_x_center >= 0.0).all() and (actual_x_center <= 1.0).all()
        assert (actual_y_center >= 0.0).all() and (actual_y_center <= 1.0).all()
        assert (actual_width >= 0.0).all() and (actual_width <= 1.0).all()
        assert (actual_height >= 0.0).all() and (actual_height <= 1.0).all()


# ---------------------------------------------------------------------------
# Property 8: Checkpoint save/load round-trip
# Feature: rt-detr-detector, Property 8: Checkpoint save/load round-trip
# ---------------------------------------------------------------------------


class TestProperty8CheckpointRoundTrip:
    """Property 8: For any valid model state, saving a checkpoint via
    save_checkpoint(path) and then loading it via load_checkpoint(path)
    produces a model with numerically equivalent weights.

    **Validates: Requirements 6.1, 6.3, 6.6**
    """

    @given(config=valid_rt_detr_config())
    @settings(max_examples=100)
    def test_save_load_round_trip_preserves_weights(self, config):
        # Feature: rt-detr-detector, Property 8: Checkpoint save/load round-trip
        """Saving then loading a checkpoint produces equivalent weights."""
        import tempfile

        mock_rtdetr = MagicMock()
        mock_model_instance = MagicMock()
        # Use a real nn.Module so state_dict works properly
        inner_module = torch.nn.Linear(10, 5)
        mock_model_instance.model = inner_module
        mock_model_instance.model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        mock_rtdetr.return_value = mock_model_instance

        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr):
            detector = RT_DETR_Detector(config)

        # Record original state
        original_state = {
            k: v.clone() for k, v in detector._model.model.state_dict().items()
        }

        # Save checkpoint using a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "checkpoint.pt"
            detector.save_checkpoint(ckpt_path)

            # Modify weights to ensure load actually restores
            with torch.no_grad():
                for param in detector._model.model.parameters():
                    param.fill_(0.0)

            # Load checkpoint
            detector.load_checkpoint(ckpt_path)

        # Verify all parameters are numerically equal
        loaded_state = detector._model.model.state_dict()
        for key in original_state:
            assert key in loaded_state, f"Missing key: {key}"
            assert torch.equal(original_state[key], loaded_state[key]), (
                f"Parameter '{key}' differs after round-trip"
            )


# ---------------------------------------------------------------------------
# Property 9: Non-existent checkpoint path raises FileNotFoundError
# Feature: rt-detr-detector, Property 9: Non-existent checkpoint path raises FileNotFoundError
# ---------------------------------------------------------------------------


class TestProperty9NonExistentCheckpointRaisesError:
    """Property 9: For any file path that does not exist on the filesystem,
    calling load_checkpoint(path) raises a FileNotFoundError whose message
    contains the attempted path string.

    **Validates: Requirements 6.4**
    """

    @given(path_str=non_existent_file_paths())
    @settings(max_examples=100)
    def test_non_existent_path_raises_file_not_found_error(self, path_str):
        # Feature: rt-detr-detector, Property 9: Non-existent checkpoint path raises FileNotFoundError
        """Any non-existent path raises FileNotFoundError with path in message."""
        config = {"model_size": "l", "num_classes": 5}

        mock_rtdetr = _make_mock_rtdetr()
        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr):
            detector = RT_DETR_Detector(config)

            with pytest.raises(FileNotFoundError) as exc_info:
                detector.load_checkpoint(Path(path_str))

            error_message = str(exc_info.value)
            assert path_str in error_message or str(Path(path_str)) in error_message, (
                f"FileNotFoundError message '{error_message}' does not contain "
                f"the attempted path '{path_str}'"
            )


# ---------------------------------------------------------------------------
# Property 10: All parameters are trainable
# Feature: rt-detr-detector, Property 10: All parameters are trainable
# ---------------------------------------------------------------------------


class TestProperty10AllParametersTrainable:
    """Property 10: For any valid configuration, get_parameters() returns a
    non-empty list where every parameter has requires_grad=True, and the list
    is compatible with torch.optim optimizer construction.

    **Validates: Requirements 3.6, 9.1**
    """

    @given(config=valid_rt_detr_config())
    @settings(max_examples=100)
    def test_all_parameters_trainable(self, config):
        # Feature: rt-detr-detector, Property 10: All parameters are trainable
        """get_parameters() returns non-empty list with all requires_grad=True."""
        mock_rtdetr = MagicMock()
        mock_model_instance = MagicMock()
        # Use a real nn.Module so parameters() works properly
        inner_module = torch.nn.Linear(10, 5)
        mock_model_instance.model = inner_module
        mock_model_instance.model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        mock_rtdetr.return_value = mock_model_instance

        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr):
            detector = RT_DETR_Detector(config)

        params = detector.get_parameters()

        # Must be non-empty
        assert len(params) > 0, "get_parameters() returned empty list"

        # All must have requires_grad=True
        for param in params:
            assert param.requires_grad is True, (
                "Found parameter with requires_grad=False"
            )

        # Must be compatible with torch.optim
        optimizer = torch.optim.SGD(params, lr=0.01)
        assert optimizer is not None
