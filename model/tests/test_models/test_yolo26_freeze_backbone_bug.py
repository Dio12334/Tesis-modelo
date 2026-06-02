"""Bug condition exploration property test for transfer-learning-freeze-layers bugfix.

Property 1: Bug Condition - Backbone Parameters Not Frozen When freeze_backbone Is Enabled

This test encodes the EXPECTED behavior: when freeze_backbone=True, backbone
parameters should have requires_grad=False and head parameters should have
requires_grad=True. On UNFIXED code, this test will FAIL because the
unconditional unfreezing loop sets all params to requires_grad=True.

**Validates: Requirements 1.1, 1.2, 1.3, 2.1, 2.2**
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from hypothesis import given, settings
from hypothesis import strategies as st

from model.models.yolo26_wrapper import YOLO26Detector


# ---------------------------------------------------------------------------
# Hypothesis strategies for YOLO26 freeze-backbone configurations
# ---------------------------------------------------------------------------

VALID_MODEL_SIZES = st.sampled_from(["n", "s", "m", "l", "x"])
VALID_NUM_CLASSES = st.integers(min_value=1, max_value=20)


@st.composite
def freeze_backbone_config(draw):
    """Strategy that generates valid YOLO26 configs with freeze_backbone=True.

    Varies model_size and num_classes while always setting freeze_backbone=True
    to scope the property test to the bug condition.
    """
    config = {
        "model_size": draw(VALID_MODEL_SIZES),
        "num_classes": draw(VALID_NUM_CLASSES),
        "freeze_backbone": True,
    }
    return config


# ---------------------------------------------------------------------------
# Mock model with real PyTorch parameters to test requires_grad behavior
# ---------------------------------------------------------------------------


class MockBackboneBlock(nn.Module):
    """Simulates a backbone block with conv layers (early layers in YOLO model)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)


class MockHeadBlock(nn.Module):
    """Simulates a detection head block (later layers in YOLO model)."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.conv = nn.Conv2d(16, 32, kernel_size=1)
        self.fc = nn.Linear(32, num_classes)


class MockYOLOModel(nn.Module):
    """A mock YOLO model that simulates the Ultralytics model.model structure.

    The Ultralytics YOLO model exposes layers via model.model (an nn.Sequential-like
    structure). Early indexed layers are backbone, later layers are head.
    This mock uses nn.Sequential with indexed submodules to replicate that pattern.
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        # Simulate model.model structure with backbone (layers 0-9) and head (layers 10+)
        # In real Ultralytics, model.model.model is the Sequential of layers
        self.model = nn.Sequential(
            # Backbone layers (indices 0-9)
            MockBackboneBlock(),  # 0
            MockBackboneBlock(),  # 1
            MockBackboneBlock(),  # 2
            MockBackboneBlock(),  # 3
            MockBackboneBlock(),  # 4
            MockBackboneBlock(),  # 5
            MockBackboneBlock(),  # 6
            MockBackboneBlock(),  # 7
            MockBackboneBlock(),  # 8
            MockBackboneBlock(),  # 9
            # Head layers (indices 10+)
            MockHeadBlock(num_classes),  # 10
            MockHeadBlock(num_classes),  # 11
            MockHeadBlock(num_classes),  # 12
        )
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    def init_criterion(self):
        return None


def create_mock_ultralytics(num_classes: int = 5):
    """Create a mock ultralytics module with a real PyTorch model inside.

    The mock YOLO class returns an object whose .model attribute is a real
    nn.Module with parameters that can be checked for requires_grad state.
    """
    mock_ul = MagicMock()
    yolo_model = MockYOLOModel(num_classes=num_classes)

    # Create the YOLO instance mock
    yolo_instance = MagicMock()
    yolo_instance.model = yolo_model

    # YOLO() constructor returns the instance
    mock_ul.YOLO.return_value = yolo_instance

    return mock_ul, yolo_model


# ---------------------------------------------------------------------------
# Constants: backbone vs head layer index boundary
# In the mock model, layers 0-9 are backbone, layers 10-12 are head
# ---------------------------------------------------------------------------
BACKBONE_LAYER_END = 10


def get_backbone_params(model: nn.Module):
    """Get parameters belonging to backbone layers (indices 0-9)."""
    params = []
    for name, param in model.named_parameters():
        # Parameter names in Sequential: "model.0.conv.weight", "model.1.bn.bias", etc.
        # Extract the layer index from the name
        parts = name.split(".")
        if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
            layer_idx = int(parts[1])
            if layer_idx < BACKBONE_LAYER_END:
                params.append((name, param))
    return params


def get_head_params(model: nn.Module):
    """Get parameters belonging to head layers (indices 10+)."""
    params = []
    for name, param in model.named_parameters():
        parts = name.split(".")
        if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
            layer_idx = int(parts[1])
            if layer_idx >= BACKBONE_LAYER_END:
                params.append((name, param))
    return params


# ---------------------------------------------------------------------------
# Property 1: Bug Condition - Backbone Parameters Not Frozen When
# freeze_backbone Is Enabled
#
# Feature: transfer-learning-freeze-layers
# **Validates: Requirements 1.1, 1.2, 1.3, 2.1, 2.2**
# ---------------------------------------------------------------------------


class TestProperty1BugConditionBackboneNotFrozen:
    """Property 1: Bug Condition - Backbone Parameters Not Frozen When
    freeze_backbone Is Enabled.

    For any valid YOLO26 config with freeze_backbone=True, the YOLO26Detector
    initialization SHALL:
    - Set requires_grad=False for all backbone parameters
    - Set requires_grad=True for all head parameters
    - Return only head parameters from get_parameters()

    On UNFIXED code, this test is EXPECTED TO FAIL because the unconditional
    unfreezing loop sets ALL params to requires_grad=True regardless of config.

    **Validates: Requirements 1.1, 1.2, 1.3, 2.1, 2.2**
    """

    @given(config=freeze_backbone_config())
    @settings(max_examples=50)
    def test_backbone_params_frozen_when_freeze_backbone_enabled(self, config):
        """With freeze_backbone=True, backbone params must have requires_grad=False."""
        num_classes = config["num_classes"]
        mock_ul, yolo_model = create_mock_ultralytics(num_classes=num_classes)

        with patch("model.models.yolo26_wrapper.ultralytics", new=mock_ul):
            detector = YOLO26Detector(config)

        # Get backbone and head params from the underlying model
        backbone_params = get_backbone_params(yolo_model)
        head_params = get_head_params(yolo_model)

        # Assert: ALL backbone parameters must have requires_grad = False
        for name, param in backbone_params:
            assert param.requires_grad is False, (
                f"Bug condition confirmed: With freeze_backbone=True, "
                f"backbone param '{name}' still has requires_grad=True. "
                f"Config: model_size={config['model_size']}, "
                f"num_classes={config['num_classes']}"
            )

    @given(config=freeze_backbone_config())
    @settings(max_examples=50)
    def test_head_params_trainable_when_freeze_backbone_enabled(self, config):
        """With freeze_backbone=True, head params must have requires_grad=True."""
        num_classes = config["num_classes"]
        mock_ul, yolo_model = create_mock_ultralytics(num_classes=num_classes)

        with patch("model.models.yolo26_wrapper.ultralytics", new=mock_ul):
            detector = YOLO26Detector(config)

        head_params = get_head_params(yolo_model)

        # Assert: ALL head parameters must have requires_grad = True
        for name, param in head_params:
            assert param.requires_grad is True, (
                f"Head param '{name}' should have requires_grad=True "
                f"when freeze_backbone is enabled. "
                f"Config: model_size={config['model_size']}, "
                f"num_classes={config['num_classes']}"
            )

    @given(config=freeze_backbone_config())
    @settings(max_examples=50)
    def test_get_parameters_returns_only_head_params(self, config):
        """With freeze_backbone=True, get_parameters() must return only head params."""
        num_classes = config["num_classes"]
        mock_ul, yolo_model = create_mock_ultralytics(num_classes=num_classes)

        with patch("model.models.yolo26_wrapper.ultralytics", new=mock_ul):
            detector = YOLO26Detector(config)

        # Count expected head params
        head_params = get_head_params(yolo_model)
        expected_trainable_count = len(head_params)

        # get_parameters() returns only params with requires_grad=True
        trainable_params = detector.get_parameters()
        total_params = list(yolo_model.parameters())

        # Assert: trainable params should be fewer than total (only head)
        assert len(trainable_params) == expected_trainable_count, (
            f"get_parameters() returned {len(trainable_params)} params, "
            f"expected {expected_trainable_count} (head only). "
            f"Total model params: {len(total_params)}. "
            f"Bug: all params are trainable when backbone should be frozen. "
            f"Config: model_size={config['model_size']}, "
            f"num_classes={config['num_classes']}"
        )

        # Also verify total vs trainable mismatch confirms the bug
        assert len(trainable_params) < len(total_params), (
            f"get_parameters() returned ALL {len(total_params)} params. "
            f"Expected only head params ({expected_trainable_count}) "
            f"when freeze_backbone=True. "
            f"Config: model_size={config['model_size']}, "
            f"num_classes={config['num_classes']}"
        )
