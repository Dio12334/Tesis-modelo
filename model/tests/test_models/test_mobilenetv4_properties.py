"""Property-based tests for MobileNetV4 SSD Detector.

Tests Properties 1-13 from the design document using Hypothesis.
Each property validates specific requirements from the MobileNetV4 detector spec.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from model.exceptions import ConfigurationError
from model.models.registry import BaseDetector, ModelRegistry
from model.models.mobilenetv4_ssd import (
    MobileNetV4Detector,
    MobileNetV4Backbone,
    MobileNetV4SSD,
    SSDHead,
    VALID_BACKBONE_VARIANTS,
    VARIANT_ALIASES,
)


# ---------------------------------------------------------------------------
# Helpers: Mock timm model creation
# ---------------------------------------------------------------------------


def _make_mock_timm_model(num_stages=5):
    """Create a mock timm model that returns feature maps of expected shapes.

    Simulates timm.create_model(..., features_only=True) returning a model
    whose forward pass produces feature maps at standard MobileNetV4 strides.
    For a 640x640 input: strides [2, 4, 8, 16, 32] -> sizes [320, 160, 80, 40, 20].
    """
    mock_model = MagicMock(spec=nn.Module)

    channels = [32, 64, 96, 192, 960][:num_stages]
    reductions = [2, 4, 8, 16, 32][:num_stages]

    feature_info = MagicMock()
    feature_info.channels.return_value = channels
    feature_info.reduction.return_value = reductions
    mock_model.feature_info = feature_info

    def mock_forward(x):
        B = x.shape[0]
        H, W = x.shape[2], x.shape[3]
        features = []
        for ch, red in zip(channels, reductions):
            feat_h = H // red
            feat_w = W // red
            features.append(torch.randn(B, ch, feat_h, feat_w))
        return features

    mock_model.side_effect = mock_forward
    mock_model.__call__ = mock_forward

    return mock_model, channels, reductions


def _create_detector_with_mock(**config_overrides):
    """Create a MobileNetV4Detector with timm.create_model mocked."""
    config = {
        "num_classes": 5,
        "input_size": 640,
    }
    config.update(config_overrides)

    mock_timm_model, channels, reductions = _make_mock_timm_model()

    with patch("model.models.mobilenetv4_ssd.timm.create_model") as mock_create:
        mock_create.return_value = mock_timm_model
        mock_create.return_value.feature_info = mock_timm_model.feature_info
        detector = MobileNetV4Detector(config)

    return detector


# ---------------------------------------------------------------------------
# Hypothesis strategies for MobileNetV4 configuration generation
# ---------------------------------------------------------------------------

VALID_BACKBONE_VARIANTS_STRATEGY = st.sampled_from(["small", "medium", "large"])

VALID_NUM_CLASSES = st.integers(min_value=1, max_value=1000)

VALID_CONFIDENCE_THRESHOLD = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

VALID_IOU_THRESHOLD = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)


@st.composite
def valid_mobilenetv4_config(draw):
    """Strategy that generates valid MobileNetV4 configuration dicts."""
    config = {
        "num_classes": draw(VALID_NUM_CLASSES),
        "input_size": 640,
        "backbone_variant": draw(VALID_BACKBONE_VARIANTS_STRATEGY),
    }
    # Optionally include optional parameters
    if draw(st.booleans()):
        config["confidence_threshold"] = draw(VALID_CONFIDENCE_THRESHOLD)
    if draw(st.booleans()):
        config["iou_threshold"] = draw(VALID_IOU_THRESHOLD)
    return config


# Strategies for invalid configurations (Property 3)

INVALID_BACKBONE_VARIANTS = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=10,
).filter(lambda s: s not in ("small", "medium", "large") and s not in VALID_BACKBONE_VARIANTS)

INVALID_NUM_CLASSES_LOW = st.integers(max_value=0)
INVALID_NUM_CLASSES_HIGH = st.integers(min_value=1001)

INVALID_THRESHOLD_LOW = st.floats(
    max_value=-0.001, allow_nan=False, allow_infinity=False
).filter(lambda x: x < 0.0)

INVALID_THRESHOLD_HIGH = st.floats(
    min_value=1.001, max_value=1e6, allow_nan=False, allow_infinity=False
).filter(lambda x: x > 1.0)

# Strategies for forward pass testing
BATCH_SIZES = st.integers(min_value=1, max_value=3)

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


# Strategy for invalid spatial dimensions
INVALID_HEIGHTS = st.integers(min_value=1, max_value=1024).filter(lambda h: h != 640)
INVALID_WIDTHS = st.integers(min_value=1, max_value=1024).filter(lambda w: w != 640)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the model registry before each test, ensuring mobilenetv4_ssd is registered."""
    saved = dict(ModelRegistry._models)
    ModelRegistry._models["mobilenetv4_ssd"] = MobileNetV4Detector
    yield
    ModelRegistry._models = saved


# ---------------------------------------------------------------------------
# Property 1: Valid configuration produces a BaseDetector instance
# Feature: mobilenetv4-detector, Property 1: Valid configuration produces a BaseDetector instance
# ---------------------------------------------------------------------------


class TestProperty1ValidConfigProducesBaseDetector:
    """Property 1: For any config with backbone_variant from {"small", "medium", "large"}
    and num_classes in [1, 1000], with optional confidence_threshold in [0.0, 1.0]
    and iou_threshold in [0.0, 1.0], ModelRegistry.create("mobilenetv4_ssd", config)
    shall return a BaseDetector instance.

    **Validates: Requirements 1.2**
    """

    @given(config=valid_mobilenetv4_config())
    @settings(max_examples=100, deadline=None)
    def test_valid_config_creates_detector_instance(self, config):
        # Feature: mobilenetv4-detector, Property 1: Valid configuration produces a BaseDetector instance
        """Any valid configuration produces a MobileNetV4Detector that is also a BaseDetector."""
        mock_timm_model, _, _ = _make_mock_timm_model()

        with patch("model.models.mobilenetv4_ssd.timm.create_model") as mock_create:
            mock_create.return_value = mock_timm_model
            mock_create.return_value.feature_info = mock_timm_model.feature_info
            instance = ModelRegistry.create("mobilenetv4_ssd", config)

            assert isinstance(instance, MobileNetV4Detector)
            assert isinstance(instance, BaseDetector)


# ---------------------------------------------------------------------------
# Property 2: Missing required parameters raises ConfigurationError
# Feature: mobilenetv4-detector, Property 2: Missing required parameters raises ConfigurationError
# ---------------------------------------------------------------------------


class TestProperty2MissingRequiredParamsRaisesError:
    """Property 2: For any config missing at least one of num_classes or input_size,
    construction shall raise ConfigurationError with violations listing each missing param.

    **Validates: Requirements 1.3**
    """

    @given(
        extra_keys=st.dictionaries(
            keys=st.text(
                alphabet=st.characters(whitelist_categories=("L", "N")),
                min_size=1,
                max_size=10,
            ).filter(lambda s: s not in ("num_classes", "input_size")),
            values=st.one_of(
                st.integers(), st.text(min_size=1, max_size=5), st.booleans()
            ),
            min_size=0,
            max_size=3,
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_missing_both_required_params_raises_error(self, extra_keys):
        # Feature: mobilenetv4-detector, Property 2: Missing required parameters raises ConfigurationError
        """Config missing both num_classes and input_size raises ConfigurationError."""
        config = dict(extra_keys)
        config.pop("num_classes", None)
        config.pop("input_size", None)

        mock_timm_model, _, _ = _make_mock_timm_model()

        with patch("model.models.mobilenetv4_ssd.timm.create_model") as mock_create:
            mock_create.return_value = mock_timm_model
            mock_create.return_value.feature_info = mock_timm_model.feature_info

            with pytest.raises(ConfigurationError) as exc_info:
                ModelRegistry.create("mobilenetv4_ssd", config)

            violations = exc_info.value.violations
            assert any("num_classes" in v for v in violations)
            assert any("input_size" in v for v in violations)

    @given(num_classes=VALID_NUM_CLASSES)
    @settings(max_examples=100, deadline=None)
    def test_missing_input_size_raises_error(self, num_classes):
        # Feature: mobilenetv4-detector, Property 2: Missing required parameters raises ConfigurationError
        """Config missing input_size raises ConfigurationError mentioning input_size."""
        config = {"num_classes": num_classes}

        mock_timm_model, _, _ = _make_mock_timm_model()

        with patch("model.models.mobilenetv4_ssd.timm.create_model") as mock_create:
            mock_create.return_value = mock_timm_model
            mock_create.return_value.feature_info = mock_timm_model.feature_info

            with pytest.raises(ConfigurationError) as exc_info:
                ModelRegistry.create("mobilenetv4_ssd", config)

            violations = exc_info.value.violations
            assert any("input_size" in v for v in violations)

    @given(input_size=st.just(640))
    @settings(max_examples=100, deadline=None)
    def test_missing_num_classes_raises_error(self, input_size):
        # Feature: mobilenetv4-detector, Property 2: Missing required parameters raises ConfigurationError
        """Config missing num_classes raises ConfigurationError mentioning num_classes."""
        config = {"input_size": input_size}

        mock_timm_model, _, _ = _make_mock_timm_model()

        with patch("model.models.mobilenetv4_ssd.timm.create_model") as mock_create:
            mock_create.return_value = mock_timm_model
            mock_create.return_value.feature_info = mock_timm_model.feature_info

            with pytest.raises(ConfigurationError) as exc_info:
                ModelRegistry.create("mobilenetv4_ssd", config)

            violations = exc_info.value.violations
            assert any("num_classes" in v for v in violations)


# ---------------------------------------------------------------------------
# Property 3: Invalid configuration values raise ConfigurationError
# Feature: mobilenetv4-detector, Property 3: Invalid configuration values raise ConfigurationError
# ---------------------------------------------------------------------------


class TestProperty3InvalidConfigValuesRaiseError:
    """Property 3: For any config with backbone_variant not in valid set,
    or num_classes outside [1, 1000], or confidence_threshold outside [0.0, 1.0],
    or iou_threshold outside [0.0, 1.0], construction shall raise ConfigurationError.

    **Validates: Requirements 1.4, 1.5, 2.6**
    """

    @given(
        invalid_variant=INVALID_BACKBONE_VARIANTS,
        num_classes=VALID_NUM_CLASSES,
    )
    @settings(max_examples=100, deadline=None)
    def test_invalid_backbone_variant_raises_error(self, invalid_variant, num_classes):
        # Feature: mobilenetv4-detector, Property 3: Invalid configuration values raise ConfigurationError
        """Invalid backbone_variant raises ConfigurationError."""
        config = {
            "num_classes": num_classes,
            "input_size": 640,
            "backbone_variant": invalid_variant,
        }

        mock_timm_model, _, _ = _make_mock_timm_model()

        with patch("model.models.mobilenetv4_ssd.timm.create_model") as mock_create:
            mock_create.return_value = mock_timm_model
            mock_create.return_value.feature_info = mock_timm_model.feature_info

            with pytest.raises(ConfigurationError) as exc_info:
                MobileNetV4Detector(config)

            violations = exc_info.value.violations
            assert any("backbone_variant" in v for v in violations)

    @given(
        invalid_num_classes=st.one_of(INVALID_NUM_CLASSES_LOW, INVALID_NUM_CLASSES_HIGH),
    )
    @settings(max_examples=100, deadline=None)
    def test_invalid_num_classes_raises_error(self, invalid_num_classes):
        # Feature: mobilenetv4-detector, Property 3: Invalid configuration values raise ConfigurationError
        """num_classes outside [1,1000] raises ConfigurationError."""
        config = {
            "num_classes": invalid_num_classes,
            "input_size": 640,
            "backbone_variant": "small",
        }

        mock_timm_model, _, _ = _make_mock_timm_model()

        with patch("model.models.mobilenetv4_ssd.timm.create_model") as mock_create:
            mock_create.return_value = mock_timm_model
            mock_create.return_value.feature_info = mock_timm_model.feature_info

            with pytest.raises(ConfigurationError) as exc_info:
                MobileNetV4Detector(config)

            violations = exc_info.value.violations
            assert any("num_classes" in v for v in violations)

    @given(
        invalid_conf=st.one_of(INVALID_THRESHOLD_LOW, INVALID_THRESHOLD_HIGH),
    )
    @settings(max_examples=100, deadline=None)
    def test_invalid_confidence_threshold_raises_error(self, invalid_conf):
        # Feature: mobilenetv4-detector, Property 3: Invalid configuration values raise ConfigurationError
        """confidence_threshold outside [0.0,1.0] raises ConfigurationError."""
        config = {
            "num_classes": 5,
            "input_size": 640,
            "backbone_variant": "small",
            "confidence_threshold": invalid_conf,
        }

        mock_timm_model, _, _ = _make_mock_timm_model()

        with patch("model.models.mobilenetv4_ssd.timm.create_model") as mock_create:
            mock_create.return_value = mock_timm_model
            mock_create.return_value.feature_info = mock_timm_model.feature_info

            with pytest.raises(ConfigurationError) as exc_info:
                MobileNetV4Detector(config)

            violations = exc_info.value.violations
            assert any("confidence_threshold" in v for v in violations)

    @given(
        invalid_iou=st.one_of(INVALID_THRESHOLD_LOW, INVALID_THRESHOLD_HIGH),
    )
    @settings(max_examples=100, deadline=None)
    def test_invalid_iou_threshold_raises_error(self, invalid_iou):
        # Feature: mobilenetv4-detector, Property 3: Invalid configuration values raise ConfigurationError
        """iou_threshold outside [0.0,1.0] raises ConfigurationError."""
        config = {
            "num_classes": 5,
            "input_size": 640,
            "backbone_variant": "small",
            "iou_threshold": invalid_iou,
        }

        mock_timm_model, _, _ = _make_mock_timm_model()

        with patch("model.models.mobilenetv4_ssd.timm.create_model") as mock_create:
            mock_create.return_value = mock_timm_model
            mock_create.return_value.feature_info = mock_timm_model.feature_info

            with pytest.raises(ConfigurationError) as exc_info:
                MobileNetV4Detector(config)

            violations = exc_info.value.violations
            assert any("iou_threshold" in v for v in violations)


# ---------------------------------------------------------------------------
# Property 4: Forward pass output structure invariant
# Feature: mobilenetv4-detector, Property 4: Forward pass output structure invariant
# ---------------------------------------------------------------------------


class TestProperty4ForwardPassOutputStructure:
    """Property 4: For any valid detector and input (B, 3, 640, 640) where B >= 1,
    forward() returns list of B dicts each with "boxes" (N,4 float32),
    "labels" (N, int64), "scores" (N, float32 in [0,1]), where N is consistent.

    **Validates: Requirements 3.1, 3.2**
    """

    @given(batch_size=BATCH_SIZES)
    @settings(max_examples=100, deadline=None)
    def test_forward_returns_correct_structure(self, batch_size):
        # Feature: mobilenetv4-detector, Property 4: Forward pass output structure invariant
        """Forward pass returns B dicts with correct tensor shapes and dtypes."""
        detector = _create_detector_with_mock()
        detector.set_eval_mode()

        images = torch.randn(batch_size, 3, 640, 640)

        with torch.no_grad():
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
# Feature: mobilenetv4-detector, Property 5: Confidence threshold filtering
# ---------------------------------------------------------------------------


class TestProperty5ConfidenceThresholdFiltering:
    """Property 5: For any configured confidence_threshold, all returned scores
    >= confidence_threshold.

    **Validates: Requirements 3.3**
    """

    @given(threshold=VALID_CONFIDENCE_THRESHOLD)
    @settings(max_examples=100, deadline=None)
    def test_all_scores_above_threshold(self, threshold):
        # Feature: mobilenetv4-detector, Property 5: Confidence threshold filtering
        """All returned scores are >= the configured confidence_threshold."""
        detector = _create_detector_with_mock(confidence_threshold=threshold)
        detector.set_eval_mode()

        images = torch.randn(1, 3, 640, 640)

        with torch.no_grad():
            predictions = detector.forward(images)

        for pred in predictions:
            scores = pred["scores"]
            if scores.numel() > 0:
                assert (scores >= threshold).all(), (
                    f"Found score {scores.min().item()} below threshold {threshold}"
                )


# ---------------------------------------------------------------------------
# Property 6: Maximum detection cap
# Feature: mobilenetv4-detector, Property 6: Maximum detection cap
# ---------------------------------------------------------------------------


class TestProperty6MaximumDetectionCap:
    """Property 6: For any input, number of detections N <= 200 per image.

    **Validates: Requirements 3.4**
    """

    @given(batch_size=BATCH_SIZES)
    @settings(max_examples=100, deadline=None)
    def test_detections_capped_at_200(self, batch_size):
        # Feature: mobilenetv4-detector, Property 6: Maximum detection cap
        """Number of detections per image is at most 200."""
        # Use a very low confidence threshold to maximize detections
        detector = _create_detector_with_mock(confidence_threshold=0.0)
        detector.set_eval_mode()

        images = torch.randn(batch_size, 3, 640, 640)

        with torch.no_grad():
            predictions = detector.forward(images)

        for pred in predictions:
            n = pred["boxes"].shape[0]
            assert n <= 200, f"Got {n} detections, expected <= 200"


# ---------------------------------------------------------------------------
# Property 7: Invalid spatial dimensions raise ValueError
# Feature: mobilenetv4-detector, Property 7: Invalid spatial dimensions raise ValueError
# ---------------------------------------------------------------------------


class TestProperty7InvalidSpatialDimensionsRaiseError:
    """Property 7: For any input (B, 3, H, W) where H != 640 or W != 640,
    forward() raises ValueError.

    **Validates: Requirements 3.6**
    """

    @given(
        batch_size=BATCH_SIZES,
        height=INVALID_HEIGHTS,
        width=INVALID_WIDTHS,
    )
    @settings(max_examples=100, deadline=None)
    def test_invalid_spatial_dims_raises_value_error(self, batch_size, height, width):
        # Feature: mobilenetv4-detector, Property 7: Invalid spatial dimensions raise ValueError
        """Input with H!=640 or W!=640 raises ValueError."""
        detector = _create_detector_with_mock()
        detector.set_eval_mode()

        images = torch.randn(batch_size, 3, height, width)

        with pytest.raises(ValueError):
            detector.forward(images)

    @given(batch_size=BATCH_SIZES, height=INVALID_HEIGHTS)
    @settings(max_examples=100, deadline=None)
    def test_invalid_height_only_raises_value_error(self, batch_size, height):
        # Feature: mobilenetv4-detector, Property 7: Invalid spatial dimensions raise ValueError
        """Input with H!=640 but W==640 raises ValueError."""
        detector = _create_detector_with_mock()
        detector.set_eval_mode()

        images = torch.randn(batch_size, 3, height, 640)

        with pytest.raises(ValueError):
            detector.forward(images)

    @given(batch_size=BATCH_SIZES, width=INVALID_WIDTHS)
    @settings(max_examples=100, deadline=None)
    def test_invalid_width_only_raises_value_error(self, batch_size, width):
        # Feature: mobilenetv4-detector, Property 7: Invalid spatial dimensions raise ValueError
        """Input with W!=640 but H==640 raises ValueError."""
        detector = _create_detector_with_mock()
        detector.set_eval_mode()

        images = torch.randn(batch_size, 3, 640, width)

        with pytest.raises(ValueError):
            detector.forward(images)


# ---------------------------------------------------------------------------
# Property 8: Training step produces differentiable loss
# Feature: mobilenetv4-detector, Property 8: Training step produces differentiable loss
# ---------------------------------------------------------------------------


class TestProperty8TrainingStepProducesDifferentiableLoss:
    """Property 8: For any non-empty list of images with valid targets,
    train_step() returns dict with loss_tensor (scalar with grad_fn),
    classification_loss (numeric), bbox_regression_loss (numeric).

    **Validates: Requirements 4.1, 4.2**
    """

    @given(
        num_images=st.integers(min_value=1, max_value=2),
        num_boxes=st.integers(min_value=1, max_value=5),
        num_classes=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100, deadline=None)
    def test_train_step_returns_differentiable_loss(
        self, num_images, num_boxes, num_classes
    ):
        # Feature: mobilenetv4-detector, Property 8: Training step produces differentiable loss
        """train_step returns dict with loss_tensor having grad_fn and numeric loss components."""
        detector = _create_detector_with_mock(num_classes=num_classes)
        detector.set_train_mode()

        # Create images and targets
        images = [torch.randn(3, 640, 640) for _ in range(num_images)]
        targets = []
        for _ in range(num_images):
            x1 = torch.rand(num_boxes) * 300
            y1 = torch.rand(num_boxes) * 300
            x2 = x1 + torch.rand(num_boxes) * 100 + 10
            y2 = y1 + torch.rand(num_boxes) * 100 + 10
            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            labels = torch.randint(0, num_classes, (num_boxes,))
            targets.append({"boxes": boxes, "labels": labels})

        result = detector.train_step(images, targets)

        # Verify output structure
        assert "loss_tensor" in result
        assert "classification_loss" in result
        assert "bbox_regression_loss" in result

        # loss_tensor should be a scalar tensor with grad_fn
        loss = result["loss_tensor"]
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert loss.grad_fn is not None, "loss_tensor should have grad_fn"

        # classification_loss and bbox_regression_loss should be numeric
        assert isinstance(result["classification_loss"], (int, float))
        assert isinstance(result["bbox_regression_loss"], (int, float))


# ---------------------------------------------------------------------------
# Property 9: get_parameters returns all trainable parameters
# Feature: mobilenetv4-detector, Property 9: get_parameters returns all trainable parameters
# ---------------------------------------------------------------------------


class TestProperty9GetParametersReturnsTrainable:
    """Property 9: For any instance, get_parameters() returns list where every element
    has requires_grad=True, includes params from both backbone and head.

    **Validates: Requirements 4.4**
    """

    @given(config=valid_mobilenetv4_config())
    @settings(max_examples=100, deadline=None)
    def test_get_parameters_all_require_grad(self, config):
        # Feature: mobilenetv4-detector, Property 9: get_parameters returns all trainable parameters
        """Every parameter from get_parameters() has requires_grad=True."""
        mock_timm_model, _, _ = _make_mock_timm_model()

        with patch("model.models.mobilenetv4_ssd.timm.create_model") as mock_create:
            mock_create.return_value = mock_timm_model
            mock_create.return_value.feature_info = mock_timm_model.feature_info
            detector = MobileNetV4Detector(config)

        params = detector.get_parameters()

        assert isinstance(params, list)
        assert len(params) > 0, "get_parameters() should return non-empty list"

        for param in params:
            assert param.requires_grad is True, (
                "All returned parameters must have requires_grad=True"
            )

    @given(num_classes=VALID_NUM_CLASSES)
    @settings(max_examples=100, deadline=None)
    def test_get_parameters_includes_backbone_and_head(self, num_classes):
        # Feature: mobilenetv4-detector, Property 9: get_parameters returns all trainable parameters
        """get_parameters() includes params from both backbone and head submodules."""
        detector = _create_detector_with_mock(num_classes=num_classes)

        params = detector.get_parameters()
        param_set = set(id(p) for p in params)

        # Check backbone has parameters in the returned list
        backbone_params = [
            p for p in detector._model.backbone.parameters() if p.requires_grad
        ]
        head_params = [
            p for p in detector._model.head.parameters() if p.requires_grad
        ]

        # At least some backbone params should be in get_parameters
        backbone_in_params = any(id(p) in param_set for p in backbone_params)
        head_in_params = any(id(p) in param_set for p in head_params)

        assert backbone_in_params, "get_parameters should include backbone params"
        assert head_in_params, "get_parameters should include head params"


# ---------------------------------------------------------------------------
# Property 10: Checkpoint save/load round-trip
# Feature: mobilenetv4-detector, Property 10: Checkpoint save/load round-trip
# ---------------------------------------------------------------------------


class TestProperty10CheckpointRoundTrip:
    """Property 10: Saving and loading produces same forward output within 1e-6 tolerance.

    **Validates: Requirements 5.1, 5.2**
    """

    @given(config=valid_mobilenetv4_config())
    @settings(max_examples=100, deadline=None)
    def test_save_load_round_trip_preserves_output(self, config):
        # Feature: mobilenetv4-detector, Property 10: Checkpoint save/load round-trip
        """Saving then loading a checkpoint produces equivalent forward output."""
        mock_timm_model, _, _ = _make_mock_timm_model()

        with patch("model.models.mobilenetv4_ssd.timm.create_model") as mock_create:
            mock_create.return_value = mock_timm_model
            mock_create.return_value.feature_info = mock_timm_model.feature_info
            detector = MobileNetV4Detector(config)

        detector.set_eval_mode()

        # Generate a test input
        test_input = torch.randn(1, 3, 640, 640)

        # Get original output
        with torch.no_grad():
            original_output = detector.forward(test_input)

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "checkpoint.pt"
            detector.save_checkpoint(ckpt_path)

            # Modify weights to ensure load actually restores
            with torch.no_grad():
                for param in detector._model.parameters():
                    param.fill_(0.0)

            # Load checkpoint
            detector.load_checkpoint(ckpt_path)

        detector.set_eval_mode()

        # Get output after restore
        with torch.no_grad():
            restored_output = detector.forward(test_input)

        # Compare outputs within tolerance
        assert len(original_output) == len(restored_output)
        for orig, restored in zip(original_output, restored_output):
            assert torch.allclose(
                orig["boxes"], restored["boxes"], atol=1e-6
            ), "Boxes differ after round-trip"
            assert torch.equal(
                orig["labels"], restored["labels"]
            ), "Labels differ after round-trip"
            assert torch.allclose(
                orig["scores"], restored["scores"], atol=1e-6
            ), "Scores differ after round-trip"


# ---------------------------------------------------------------------------
# Property 11: Non-existent checkpoint path raises FileNotFoundError
# Feature: mobilenetv4-detector, Property 11: Non-existent checkpoint path raises FileNotFoundError
# ---------------------------------------------------------------------------


class TestProperty11NonExistentCheckpointRaisesError:
    """Property 11: For any path that doesn't exist, load_checkpoint raises FileNotFoundError.

    **Validates: Requirements 5.3**
    """

    @given(path_str=non_existent_file_paths())
    @settings(max_examples=100, deadline=None)
    def test_non_existent_path_raises_file_not_found_error(self, path_str):
        # Feature: mobilenetv4-detector, Property 11: Non-existent checkpoint path raises FileNotFoundError
        """Any non-existent path raises FileNotFoundError."""
        detector = _create_detector_with_mock()

        with pytest.raises(FileNotFoundError) as exc_info:
            detector.load_checkpoint(Path(path_str))

        error_message = str(exc_info.value)
        assert path_str in error_message or str(Path(path_str)) in error_message, (
            f"FileNotFoundError message '{error_message}' does not contain "
            f"the attempted path '{path_str}'"
        )


# ---------------------------------------------------------------------------
# Property 12: Invalid checkpoint file raises RuntimeError
# Feature: mobilenetv4-detector, Property 12: Invalid checkpoint file raises RuntimeError
# ---------------------------------------------------------------------------


class TestProperty12InvalidCheckpointRaisesRuntimeError:
    """Property 12: For any file that can't be deserialized, load_checkpoint raises RuntimeError.

    **Validates: Requirements 5.4**
    """

    @given(
        garbage_content=st.binary(min_size=1, max_size=100),
    )
    @settings(max_examples=100, deadline=None)
    def test_invalid_checkpoint_file_raises_runtime_error(self, garbage_content):
        # Feature: mobilenetv4-detector, Property 12: Invalid checkpoint file raises RuntimeError
        """A file with garbage content raises RuntimeError on load_checkpoint."""
        detector = _create_detector_with_mock()

        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_ckpt = Path(tmp_dir) / "bad_checkpoint.pt"
            bad_ckpt.write_bytes(garbage_content)

            with pytest.raises(RuntimeError) as exc_info:
                detector.load_checkpoint(bad_ckpt)

            error_message = str(exc_info.value)
            assert str(bad_ckpt) in error_message or "bad_checkpoint" in error_message


# ---------------------------------------------------------------------------
# Property 13: Output device matches input device
# Feature: mobilenetv4-detector, Property 13: Output device matches input device
# ---------------------------------------------------------------------------


class TestProperty13OutputDeviceMatchesInput:
    """Property 13: All output tensors on same device as input tensor.

    **Validates: Requirements 6.4**
    """

    @given(batch_size=BATCH_SIZES)
    @settings(max_examples=100, deadline=None)
    def test_output_tensors_on_same_device_as_input(self, batch_size):
        # Feature: mobilenetv4-detector, Property 13: Output device matches input device
        """All output tensors are on the same device as the input tensor."""
        device = torch.device("cpu")
        detector = _create_detector_with_mock()
        detector.set_eval_mode()
        detector.to_device(device)

        images = torch.randn(batch_size, 3, 640, 640, device=device)

        with torch.no_grad():
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
