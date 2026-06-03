"""Unit tests for MobileNetV4 SSD Detector.

Tests cover:
- get_config_schema() returns correct structure (Requirement 2.2)
- Default values applied correctly for optional parameters (Requirement 2.3)
- pretrained_backbone flag is passed to timm (Requirements 2.4, 2.5)
- set_train_mode() / set_eval_mode() toggle correctly (Requirements 6.2, 6.3)
- to_device() moves all parameters (Requirement 6.1)
- Multi-scale feature extraction produces correct spatial resolutions (Requirements 7.1, 7.4)
- Anchor generator configuration (Requirement 7.3)
- Feature map exclusion for degenerate stages (Requirement 7.5)

Requirements: 2.2, 2.3, 2.4, 2.5, 6.1, 6.2, 6.3, 7.1, 7.3, 7.4, 7.5
"""

from unittest.mock import MagicMock, patch, call
from types import SimpleNamespace

import torch
import torch.nn as nn
import pytest

from model.exceptions import ConfigurationError
from model.models.registry import BaseDetector, ModelRegistry
from model.models.mobilenetv4_ssd import (
    MobileNetV4Detector,
    MobileNetV4Backbone,
    MobileNetV4SSD,
    SSDHead,
    CONFIG_SCHEMA,
    VALID_BACKBONE_VARIANTS,
    VARIANT_ALIASES,
)


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def _make_mock_timm_model(num_stages=5):
    """Create a mock timm model that returns feature maps of expected shapes.

    Simulates timm.create_model(..., features_only=True) returning a model
    whose forward pass produces feature maps at standard MobileNetV4 strides.
    For a 640x640 input: strides [2, 4, 8, 16, 32] → sizes [320, 160, 80, 40, 20].
    """
    mock_model = MagicMock(spec=nn.Module)

    # feature_info mock
    channels = [32, 64, 96, 192, 960][:num_stages]
    reductions = [2, 4, 8, 16, 32][:num_stages]

    feature_info = MagicMock()
    feature_info.channels.return_value = channels
    feature_info.reduction.return_value = reductions
    mock_model.feature_info = feature_info

    # Forward produces tensors at expected spatial sizes for 640x640 input
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


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the model registry before each test, ensuring mobilenetv4_ssd is registered."""
    saved = dict(ModelRegistry._models)
    ModelRegistry._models["mobilenetv4_ssd"] = MobileNetV4Detector
    yield
    ModelRegistry._models = saved


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

    return detector, mock_create


# ---------------------------------------------------------------------------
# Tests: get_config_schema() - Requirement 2.2
# ---------------------------------------------------------------------------


class TestConfigSchema:
    """Tests for get_config_schema() (Requirement 2.2)."""

    def test_schema_returns_dict(self):
        """get_config_schema() should return a dictionary."""
        detector, _ = _create_detector_with_mock()
        schema = detector.get_config_schema()
        assert isinstance(schema, dict)

    def test_schema_contains_all_parameters(self):
        """Schema should contain all expected configuration parameters."""
        detector, _ = _create_detector_with_mock()
        schema = detector.get_config_schema()

        expected_keys = {
            "num_classes",
            "input_size",
            "backbone_variant",
            "pretrained_backbone",
            "confidence_threshold",
            "iou_threshold",
        }
        assert set(schema.keys()) == expected_keys

    def test_schema_each_entry_has_type_and_required(self):
        """Each schema entry should have 'type' and 'required' fields."""
        detector, _ = _create_detector_with_mock()
        schema = detector.get_config_schema()

        for param_name, param_spec in schema.items():
            assert "type" in param_spec, f"Missing 'type' for {param_name}"
            assert "required" in param_spec, f"Missing 'required' for {param_name}"

    def test_schema_required_parameters(self):
        """num_classes and input_size should be marked as required."""
        detector, _ = _create_detector_with_mock()
        schema = detector.get_config_schema()

        assert schema["num_classes"]["required"] is True
        assert schema["input_size"]["required"] is True

    def test_schema_optional_parameters(self):
        """Optional parameters should be marked as not required."""
        detector, _ = _create_detector_with_mock()
        schema = detector.get_config_schema()

        assert schema["backbone_variant"]["required"] is False
        assert schema["pretrained_backbone"]["required"] is False
        assert schema["confidence_threshold"]["required"] is False
        assert schema["iou_threshold"]["required"] is False

    def test_schema_type_fields(self):
        """Schema type fields should indicate Python type names."""
        detector, _ = _create_detector_with_mock()
        schema = detector.get_config_schema()

        assert schema["num_classes"]["type"] == "int"
        assert schema["input_size"]["type"] == "int"
        assert schema["backbone_variant"]["type"] == "str"
        assert schema["pretrained_backbone"]["type"] == "bool"
        assert schema["confidence_threshold"]["type"] == "float"
        assert schema["iou_threshold"]["type"] == "float"


# ---------------------------------------------------------------------------
# Tests: Default values - Requirement 2.3
# ---------------------------------------------------------------------------


class TestDefaultValues:
    """Tests for default values applied to optional parameters (Requirement 2.3)."""

    def test_default_backbone_variant(self):
        """backbone_variant should default to mobilenetv4_conv_small.e2400_r224_in1k."""
        detector, _ = _create_detector_with_mock()
        assert detector.backbone_variant == "mobilenetv4_conv_small.e2400_r224_in1k"

    def test_default_pretrained_backbone(self):
        """pretrained_backbone should default to True."""
        detector, _ = _create_detector_with_mock()
        assert detector.pretrained_backbone is True

    def test_default_confidence_threshold(self):
        """confidence_threshold should default to 0.25."""
        detector, _ = _create_detector_with_mock()
        assert detector.confidence_threshold == 0.25

    def test_default_iou_threshold(self):
        """iou_threshold should default to 0.5."""
        detector, _ = _create_detector_with_mock()
        assert detector.iou_threshold == 0.5

    def test_explicit_values_override_defaults(self):
        """Explicitly provided values should override defaults."""
        detector, _ = _create_detector_with_mock(
            backbone_variant="small",
            pretrained_backbone=False,
            confidence_threshold=0.4,
            iou_threshold=0.7,
        )
        # "small" alias resolves to the full name
        assert detector.backbone_variant == "mobilenetv4_conv_small.e2400_r224_in1k"
        assert detector.pretrained_backbone is False
        assert detector.confidence_threshold == 0.4
        assert detector.iou_threshold == 0.7


# ---------------------------------------------------------------------------
# Tests: pretrained_backbone flag passed to timm - Requirements 2.4, 2.5
# ---------------------------------------------------------------------------


class TestPretrainedBackbone:
    """Tests for pretrained_backbone flag passed to timm (Requirements 2.4, 2.5)."""

    def test_pretrained_true_passed_to_timm(self):
        """When pretrained_backbone=True, timm.create_model gets pretrained=True."""
        config = {"num_classes": 5, "input_size": 640, "pretrained_backbone": True}

        mock_timm_model, _, _ = _make_mock_timm_model()

        with patch("model.models.mobilenetv4_ssd.timm.create_model") as mock_create:
            mock_create.return_value = mock_timm_model
            mock_create.return_value.feature_info = mock_timm_model.feature_info
            MobileNetV4Detector(config)

            # Check that timm.create_model was called with pretrained=True
            mock_create.assert_called_once()
            _, kwargs = mock_create.call_args
            assert kwargs.get("pretrained") is True

    def test_pretrained_false_passed_to_timm(self):
        """When pretrained_backbone=False, timm.create_model gets pretrained=False."""
        config = {"num_classes": 5, "input_size": 640, "pretrained_backbone": False}

        mock_timm_model, _, _ = _make_mock_timm_model()

        with patch("model.models.mobilenetv4_ssd.timm.create_model") as mock_create:
            mock_create.return_value = mock_timm_model
            mock_create.return_value.feature_info = mock_timm_model.feature_info
            MobileNetV4Detector(config)

            # Check that timm.create_model was called with pretrained=False
            mock_create.assert_called_once()
            _, kwargs = mock_create.call_args
            assert kwargs.get("pretrained") is False

    def test_features_only_always_true(self):
        """timm.create_model should always be called with features_only=True."""
        config = {"num_classes": 5, "input_size": 640}

        mock_timm_model, _, _ = _make_mock_timm_model()

        with patch("model.models.mobilenetv4_ssd.timm.create_model") as mock_create:
            mock_create.return_value = mock_timm_model
            mock_create.return_value.feature_info = mock_timm_model.feature_info
            MobileNetV4Detector(config)

            _, kwargs = mock_create.call_args
            assert kwargs.get("features_only") is True


# ---------------------------------------------------------------------------
# Tests: set_train_mode() / set_eval_mode() - Requirements 6.2, 6.3
# ---------------------------------------------------------------------------


class TestModeToggle:
    """Tests for set_train_mode() / set_eval_mode() (Requirements 6.2, 6.3)."""

    def test_set_eval_mode_all_modules_not_training(self):
        """set_eval_mode() should set all modules to training=False."""
        detector, _ = _create_detector_with_mock()
        detector.set_eval_mode()

        for module in detector._model.modules():
            assert module.training is False, (
                f"Module {module.__class__.__name__} still in training mode"
            )

    def test_set_train_mode_all_modules_training(self):
        """set_train_mode() should set all modules to training=True."""
        detector, _ = _create_detector_with_mock()
        # First go to eval then switch back to train
        detector.set_eval_mode()
        detector.set_train_mode()

        for module in detector._model.modules():
            assert module.training is True, (
                f"Module {module.__class__.__name__} not in training mode"
            )

    def test_toggle_train_eval_repeatedly(self):
        """Mode should toggle correctly on repeated switches."""
        detector, _ = _create_detector_with_mock()

        detector.set_eval_mode()
        assert detector._model.training is False

        detector.set_train_mode()
        assert detector._model.training is True

        detector.set_eval_mode()
        assert detector._model.training is False


# ---------------------------------------------------------------------------
# Tests: to_device() - Requirement 6.1
# ---------------------------------------------------------------------------


class TestToDevice:
    """Tests for to_device() (Requirement 6.1)."""

    def test_to_device_cpu_all_parameters_on_cpu(self):
        """After to_device('cpu'), all parameters should be on CPU."""
        detector, _ = _create_detector_with_mock()
        detector.to_device("cpu")

        for param in detector._model.parameters():
            assert param.device.type == "cpu"

    def test_to_device_updates_device_attribute(self):
        """to_device() should update the internal _device attribute."""
        detector, _ = _create_detector_with_mock()
        detector.to_device("cpu")
        assert detector._device == torch.device("cpu")

    def test_to_device_moves_buffers(self):
        """to_device() should move buffers (e.g., batch norm running stats) too."""
        detector, _ = _create_detector_with_mock()
        detector.to_device("cpu")

        for name, buffer in detector._model.named_buffers():
            assert buffer.device.type == "cpu", (
                f"Buffer {name} not on cpu"
            )

    def test_to_device_unavailable_device_raises_runtime_error(self):
        """to_device() with unavailable device should raise RuntimeError."""
        detector, _ = _create_detector_with_mock()

        # Only test if CUDA is not available
        if not torch.cuda.is_available():
            with pytest.raises(RuntimeError, match="not available"):
                detector.to_device("cuda:99")


# ---------------------------------------------------------------------------
# Tests: Multi-scale feature extraction - Requirements 7.1, 7.4
# ---------------------------------------------------------------------------


class TestMultiScaleFeatureExtraction:
    """Tests for multi-scale feature extraction (Requirements 7.1, 7.4)."""

    def test_backbone_produces_six_feature_maps(self):
        """Backbone should produce 6 multi-scale feature maps."""
        detector, _ = _create_detector_with_mock()
        detector.set_eval_mode()

        x = torch.randn(1, 3, 640, 640)
        features = detector._model.backbone(x)

        assert len(features) == 6, f"Expected 6 feature maps, got {len(features)}"

    def test_feature_maps_progressively_decrease_in_spatial_size(self):
        """Feature map spatial dimensions should decrease progressively."""
        detector, _ = _create_detector_with_mock()
        detector.set_eval_mode()

        x = torch.randn(1, 3, 640, 640)
        features = detector._model.backbone(x)

        spatial_sizes = [(f.shape[2], f.shape[3]) for f in features]
        for i in range(1, len(spatial_sizes)):
            assert spatial_sizes[i][0] <= spatial_sizes[i - 1][0], (
                f"Feature map {i} ({spatial_sizes[i]}) should be smaller than "
                f"feature map {i-1} ({spatial_sizes[i-1]})"
            )

    def test_largest_feature_map_at_least_40x40(self):
        """Largest feature map should be at least 40×40 (Requirement 7.4)."""
        detector, _ = _create_detector_with_mock()
        detector.set_eval_mode()

        x = torch.randn(1, 3, 640, 640)
        features = detector._model.backbone(x)

        largest = features[0]
        assert largest.shape[2] >= 40, f"Largest H={largest.shape[2]}, expected >= 40"
        assert largest.shape[3] >= 40, f"Largest W={largest.shape[3]}, expected >= 40"

    def test_smallest_feature_map_at_most_3x3(self):
        """Smallest feature map should be at most 3×3 (Requirement 7.4)."""
        detector, _ = _create_detector_with_mock()
        detector.set_eval_mode()

        x = torch.randn(1, 3, 640, 640)
        features = detector._model.backbone(x)

        smallest = features[-1]
        assert smallest.shape[2] <= 3, f"Smallest H={smallest.shape[2]}, expected <= 3"
        assert smallest.shape[3] <= 3, f"Smallest W={smallest.shape[3]}, expected <= 3"

    def test_at_least_four_backbone_stages(self):
        """Backbone should extract feature maps from at least 4 stages (Requirement 7.1)."""
        detector, _ = _create_detector_with_mock()
        detector.set_eval_mode()

        x = torch.randn(1, 3, 640, 640)
        features = detector._model.backbone(x)

        # We expect at least 4 stages from the backbone network itself
        # (the design specifies 4 backbone stages + 2 extra = 6 total)
        assert len(features) >= 4


# ---------------------------------------------------------------------------
# Tests: Anchor generator configuration - Requirement 7.3
# ---------------------------------------------------------------------------


class TestAnchorGenerator:
    """Tests for anchor generator configuration (Requirement 7.3)."""

    def test_anchor_generator_exists(self):
        """Model should have an anchor generator."""
        detector, _ = _create_detector_with_mock()
        assert hasattr(detector._model, "anchor_generator")

    def test_anchor_generator_aspect_ratios(self):
        """Anchor generator should use aspect ratios [1.0, 2.0, 0.5] per level."""
        detector, _ = _create_detector_with_mock()
        ag = detector._model.anchor_generator

        # DefaultBoxGenerator stores aspect_ratios as list of lists
        for level_ratios in ag.aspect_ratios:
            assert 1.0 in level_ratios
            assert 2.0 in level_ratios
            assert 0.5 in level_ratios

    def test_anchor_generator_six_levels(self):
        """Anchor generator should be configured for 6 feature map levels."""
        detector, _ = _create_detector_with_mock()
        ag = detector._model.anchor_generator

        assert len(ag.aspect_ratios) == 6

    def test_anchor_generator_produces_anchors(self):
        """Anchor generator should produce anchor boxes when given feature maps."""
        detector, _ = _create_detector_with_mock()
        detector.set_eval_mode()

        x = torch.randn(1, 3, 640, 640)
        features = detector._model.backbone(x)

        from torchvision.models.detection.image_list import ImageList
        image_list = ImageList(x, [(640, 640)])
        anchors = detector._model.anchor_generator(image_list, features)

        # Should produce anchors for each image in the batch
        assert len(anchors) == 1
        # Anchors should be (num_anchors, 4)
        assert anchors[0].ndim == 2
        assert anchors[0].shape[1] == 4
        # Should have many anchors across 6 levels
        assert anchors[0].shape[0] > 0

    def test_anchor_size_range_covers_20_to_500(self):
        """Anchors should cover minimum 20px and maximum 500px sizes."""
        detector, _ = _create_detector_with_mock()
        detector.set_eval_mode()

        x = torch.randn(1, 3, 640, 640)
        features = detector._model.backbone(x)

        from torchvision.models.detection.image_list import ImageList
        image_list = ImageList(x, [(640, 640)])
        anchors = detector._model.anchor_generator(image_list, features)

        anchor_boxes = anchors[0]
        widths = anchor_boxes[:, 2] - anchor_boxes[:, 0]
        heights = anchor_boxes[:, 3] - anchor_boxes[:, 1]

        # Check that minimum anchor size is <= 30px (allowing some tolerance around 20)
        min_size = torch.min(torch.min(widths), torch.min(heights)).item()
        assert min_size <= 30, f"Minimum anchor size {min_size:.1f} > 30px"

        # Check that maximum anchor size is >= 400px (allowing some tolerance around 500)
        max_size = torch.max(torch.max(widths), torch.max(heights)).item()
        assert max_size >= 400, f"Maximum anchor size {max_size:.1f} < 400px"


# ---------------------------------------------------------------------------
# Tests: Feature map exclusion for degenerate stages - Requirement 7.5
# ---------------------------------------------------------------------------


class TestDegenerateStageExclusion:
    """Tests for feature map exclusion of degenerate stages 1×1 or smaller (Requirement 7.5)."""

    def test_ssd_head_skips_1x1_features(self):
        """SSD head should skip feature maps with spatial resolution 1×1."""
        # Create a minimal SSDHead
        in_channels = [64, 128, 256]
        num_anchors_per_location = [6, 6, 6]
        num_classes = 5

        head = SSDHead(in_channels, num_anchors_per_location, num_classes)

        # Create feature maps where one is 1×1 (degenerate)
        features = [
            torch.randn(1, 64, 5, 5),   # Normal
            torch.randn(1, 128, 3, 3),   # Normal
            torch.randn(1, 256, 1, 1),   # Degenerate - should be skipped
        ]

        cls_logits, bbox_preds = head(features)

        # Only non-degenerate features contribute
        # Level 0: 5×5 → 25 locations × 6 anchors = 150
        # Level 1: 3×3 → 9 locations × 6 anchors = 54
        # Level 2: 1×1 → SKIPPED
        expected_anchors = (25 + 9) * 6
        assert cls_logits.shape == (1, expected_anchors, num_classes)
        assert bbox_preds.shape == (1, expected_anchors, 4)

    def test_ssd_head_includes_2x2_features(self):
        """SSD head should include feature maps with spatial resolution >= 2×2."""
        in_channels = [64, 128]
        num_anchors_per_location = [6, 6]
        num_classes = 5

        head = SSDHead(in_channels, num_anchors_per_location, num_classes)

        features = [
            torch.randn(1, 64, 5, 5),   # Normal
            torch.randn(1, 128, 2, 2),   # 2×2 is fine, should be included
        ]

        cls_logits, bbox_preds = head(features)

        # Level 0: 5×5 → 25 locations × 6 = 150
        # Level 1: 2×2 → 4 locations × 6 = 24
        expected_anchors = (25 + 4) * 6
        assert cls_logits.shape == (1, expected_anchors, num_classes)
        assert bbox_preds.shape == (1, expected_anchors, 4)

    def test_ssd_head_skips_smaller_than_1x1(self):
        """SSD head should skip feature maps with spatial resolution smaller than 1×1."""
        in_channels = [64, 128]
        num_anchors_per_location = [6, 6]
        num_classes = 5

        head = SSDHead(in_channels, num_anchors_per_location, num_classes)

        # Edge case: empty spatial dimensions (0×0)
        features = [
            torch.randn(1, 64, 3, 3),     # Normal
            torch.randn(1, 128, 0, 0),     # Smaller than 1×1
        ]

        cls_logits, bbox_preds = head(features)

        # Only level 0 contributes: 3×3 → 9 locations × 6 = 54
        expected_anchors = 9 * 6
        assert cls_logits.shape == (1, expected_anchors, num_classes)
        assert bbox_preds.shape == (1, expected_anchors, 4)
