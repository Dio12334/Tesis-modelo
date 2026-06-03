"""Property-based tests for RT-DETR freeze layers feature.

Tests the correctness of the freeze_layers configuration parameter in
RT_DETR_Detector. These tests are written BEFORE implementation to
define the correctness guarantees.

Feature: rt-detr-freeze-layers
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import sys

import pytest
import torch
import torch.nn as nn
from hypothesis import given, settings
from hypothesis import strategies as st

from model.exceptions import ConfigurationError
from model.models.rt_detr_wrapper import RT_DETR_Detector


# ---------------------------------------------------------------------------
# Mock model infrastructure for RT-DETR freeze layers testing
# ---------------------------------------------------------------------------

# Total number of layers in the mock model (simulates RT-DETR-L structure)
MOCK_TOTAL_LAYERS = 22


class MockLayerBlock(nn.Module):
    """Simulates a sequential layer block in the RT-DETR model.

    Each block has a conv and bn sublayer, producing parameters with names like:
      model.<idx>.conv.weight
      model.<idx>.bn.weight
      model.<idx>.bn.bias
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)


class MockRTDETRInnerModel(nn.Module):
    """Mock of the Ultralytics RT-DETR inner model structure.

    Replicates the structure accessed via _model.model, which has a .model
    attribute that is an nn.Sequential of layer blocks. Parameters follow
    the naming pattern: model.<layer_idx>.<sublayer>.<kind>
    """

    def __init__(self, num_layers: int = MOCK_TOTAL_LAYERS):
        super().__init__()
        self.model = nn.Sequential(
            *[MockLayerBlock() for _ in range(num_layers)]
        )
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    def init_criterion(self):
        return None


def create_mock_rtdetr_cls(num_layers: int = MOCK_TOTAL_LAYERS):
    """Create a mock RTDETR class that returns a model with real PyTorch layers.

    Returns:
        Tuple of (mock_rtdetr_cls, inner_model):
        - mock_rtdetr_cls: a callable mock that acts as the RTDETR constructor
        - inner_model: the MockRTDETRInnerModel with real nn.Parameters
    """
    mock_rtdetr_cls = MagicMock()
    model_instance = MagicMock()
    inner_model = MockRTDETRInnerModel(num_layers=num_layers)
    model_instance.model = inner_model
    mock_rtdetr_cls.return_value = model_instance
    return mock_rtdetr_cls, inner_model


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

VALID_FREEZE_LAYERS = st.integers(min_value=1, max_value=MOCK_TOTAL_LAYERS)


@st.composite
def valid_config_with_freeze_layers(draw):
    """Generate a valid RT-DETR config with a random freeze_layers in [1, total_layers]."""
    config = {
        "model_size": draw(st.sampled_from(["l", "x"])),
        "num_classes": draw(st.integers(min_value=1, max_value=1000)),
        "freeze_layers": draw(VALID_FREEZE_LAYERS),
    }
    return config


# ---------------------------------------------------------------------------
# Property 1: Freeze layers correctly sets requires_grad
# Feature: rt-detr-freeze-layers, Property 1: Freeze layers correctly sets requires_grad
# ---------------------------------------------------------------------------


class TestProperty1FreezeLayersRequiresGrad:
    """Property 1: For any valid RT-DETR model configuration with freeze_layers
    set to N where 1 <= N <= total_layers, all parameters whose name starts with
    model.<idx>. where idx < N SHALL have requires_grad = False, and all parameters
    whose name starts with model.<idx>. where idx >= N SHALL have requires_grad = True.

    **Validates: Requirements 1.5, 2.1, 2.2**
    """

    @given(config=valid_config_with_freeze_layers())
    @settings(max_examples=100, deadline=None)
    def test_freeze_layers_sets_requires_grad_correctly(self, config):
        """Freezing N layers sets requires_grad=False for layers 0..N-1
        and requires_grad=True for layers N+.

        Feature: rt-detr-freeze-layers, Property 1: Freeze layers correctly sets requires_grad
        """
        freeze_layers = config["freeze_layers"]

        mock_rtdetr_cls, inner_model = create_mock_rtdetr_cls(MOCK_TOTAL_LAYERS)
        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr_cls):
            detector = RT_DETR_Detector(config)

        # Verify requires_grad for all named parameters
        for name, param in detector._model.model.named_parameters():
            parts = name.split(".")
            if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
                layer_idx = int(parts[1])
                if layer_idx < freeze_layers:
                    assert param.requires_grad is False, (
                        f"Parameter '{name}' in frozen layer {layer_idx} "
                        f"(freeze_layers={freeze_layers}) should have "
                        f"requires_grad=False but has requires_grad=True"
                    )
                else:
                    assert param.requires_grad is True, (
                        f"Parameter '{name}' in unfrozen layer {layer_idx} "
                        f"(freeze_layers={freeze_layers}) should have "
                        f"requires_grad=True but has requires_grad=False"
                    )


# ---------------------------------------------------------------------------
# Helpers for Property 6: Forward-pass-capable mock model
# ---------------------------------------------------------------------------

_FORWARD_MOCK_LAYERS = 6  # Fewer layers for faster forward pass testing


class ForwardCapableLayer(nn.Module):
    """A small layer that can execute a real forward pass on a 1-D input."""

    def __init__(self, in_features: int = 8, out_features: int = 8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ForwardCapableSequentialModel(nn.Module):
    """A model with sequential layers that can execute a real forward pass.

    Mimics the Ultralytics RT-DETR model structure (model.model is a Sequential)
    but is small enough to run forward quickly in property tests.
    """

    def __init__(self, num_layers: int = _FORWARD_MOCK_LAYERS):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(ForwardCapableLayer(in_features=8, out_features=8))
        self.model = nn.Sequential(*layers)
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def init_criterion(self):
        return None


def _make_forward_capable_rtdetr(num_layers: int = _FORWARD_MOCK_LAYERS):
    """Create a mock RTDETR class returning a model capable of real forward passes.

    Also patches the RTDETRDetectionModel import to prevent _build_loss_fn from
    changing the model's __class__, which would break the simple forward pass.

    Returns:
        Tuple of (mock_rtdetr_cls, inner_module) where inner_module is the
        ForwardCapableSequentialModel that holds the real parameters.
    """
    mock_rtdetr_cls = MagicMock()
    mock_model_instance = MagicMock()

    inner_module = ForwardCapableSequentialModel(num_layers)
    mock_model_instance.model = inner_module

    mock_rtdetr_cls.return_value = mock_model_instance
    return mock_rtdetr_cls, inner_module


# ---------------------------------------------------------------------------
# Hypothesis strategies for Property 6
# ---------------------------------------------------------------------------

VALID_FREEZE_LAYERS_FORWARD = st.integers(min_value=1, max_value=_FORWARD_MOCK_LAYERS)
INPUT_BATCH_SIZE = st.integers(min_value=1, max_value=4)


@st.composite
def forward_pass_config_and_input(draw):
    """Generate a valid config with freeze_layers and a random input tensor."""
    freeze_layers = draw(VALID_FREEZE_LAYERS_FORWARD)
    batch_size = draw(INPUT_BATCH_SIZE)
    # Generate a random input tensor with feature size 8 (matching ForwardCapableLayer)
    input_tensor = torch.randn(batch_size, 8)
    config = {
        "model_size": "l",
        "num_classes": draw(st.integers(min_value=1, max_value=1000)),
        "freeze_layers": freeze_layers,
    }
    return config, input_tensor


# ---------------------------------------------------------------------------
# Property 6: Forward pass invariance under freezing
# Feature: rt-detr-freeze-layers, Property 6: Forward pass invariance under freezing
#
# For any RT-DETR model configuration with freeze_layers set to any valid value N,
# the forward() method SHALL produce identical predictions to the same model with
# no layers frozen, given the same input tensor and weights, because frozen layers
# still execute in the forward pass.
#
# **Validates: Requirements 4.3**
# ---------------------------------------------------------------------------


class TestProperty6ForwardPassInvariance:
    """Property 6: Forward pass invariance under freezing.

    For any RT-DETR model configuration with freeze_layers set to any valid
    value N, the forward pass SHALL produce identical output tensors to the
    same model with no layers frozen, given the same input tensor and weights,
    because frozen layers still execute in the forward pass (requires_grad=False
    only disables gradient computation, not forward execution).

    **Validates: Requirements 4.3**
    """

    @given(data=forward_pass_config_and_input())
    @settings(max_examples=100, deadline=None)
    def test_forward_pass_identical_with_and_without_freezing(self, data):
        """forward() produces identical output tensors regardless of freeze config.

        Feature: rt-detr-freeze-layers, Property 6: Forward pass invariance under freezing
        """
        config_with_freeze, input_tensor = data
        freeze_layers = config_with_freeze["freeze_layers"]

        # Create config without freeze_layers (default: all trainable)
        config_without_freeze = {
            k: v for k, v in config_with_freeze.items() if k != "freeze_layers"
        }

        # Create two models with identical weights
        mock_rtdetr_frozen, inner_module_frozen = _make_forward_capable_rtdetr(
            _FORWARD_MOCK_LAYERS
        )
        mock_rtdetr_unfrozen, inner_module_unfrozen = _make_forward_capable_rtdetr(
            _FORWARD_MOCK_LAYERS
        )

        # Copy weights so both models are identical
        inner_module_unfrozen.load_state_dict(inner_module_frozen.state_dict())

        # Instantiate both detectors. Patch RTDETRDetectionModel import to
        # prevent _build_loss_fn from replacing the model's __class__, which
        # would break our simple forward-capable mock.
        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr_frozen), \
             patch.dict(sys.modules, {"ultralytics.models.rtdetr.model": None}):
            detector_frozen = RT_DETR_Detector(config_with_freeze)

        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr_unfrozen), \
             patch.dict(sys.modules, {"ultralytics.models.rtdetr.model": None}):
            detector_unfrozen = RT_DETR_Detector(config_without_freeze)

        # Run forward pass through the underlying sequential model.
        # The property being tested is that requires_grad=False does NOT
        # affect forward computation — only gradient computation is disabled.
        inner_module_frozen.eval()
        inner_module_unfrozen.eval()

        with torch.no_grad():
            output_frozen = inner_module_frozen(input_tensor)
            output_unfrozen = inner_module_unfrozen(input_tensor)

        # Assert outputs are numerically identical
        assert torch.equal(output_frozen, output_unfrozen), (
            f"Forward pass outputs differ between frozen "
            f"(freeze_layers={freeze_layers}) and unfrozen models with "
            f"the same weights.\n"
            f"Max difference: "
            f"{(output_frozen - output_unfrozen).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# Hypothesis strategies for Property 2 and Property 4
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
def valid_config_without_freeze_layers(draw):
    """Generate a valid RT-DETR config WITHOUT freeze_layers key.

    Varies model_size, num_classes, and optional thresholds but never
    includes the freeze_layers key.
    """
    config = {
        "model_size": draw(VALID_MODEL_SIZES),
        "num_classes": draw(VALID_NUM_CLASSES),
    }
    if draw(st.booleans()):
        config["confidence_threshold"] = draw(VALID_CONFIDENCE_THRESHOLD)
    if draw(st.booleans()):
        config["iou_threshold"] = draw(VALID_IOU_THRESHOLD)
    return config


@st.composite
def valid_config_with_freeze_layers_zero(draw):
    """Generate a valid RT-DETR config with freeze_layers explicitly set to 0.

    freeze_layers=0 means no layers frozen, equivalent to default behavior.
    """
    config = {
        "model_size": draw(VALID_MODEL_SIZES),
        "num_classes": draw(VALID_NUM_CLASSES),
        "freeze_layers": 0,
    }
    if draw(st.booleans()):
        config["confidence_threshold"] = draw(VALID_CONFIDENCE_THRESHOLD)
    if draw(st.booleans()):
        config["iou_threshold"] = draw(VALID_IOU_THRESHOLD)
    return config


@st.composite
def freeze_layers_exceeding_total(draw, total_layers: int = MOCK_TOTAL_LAYERS):
    """Strategy that generates freeze_layers values exceeding total layer count.

    Generates non-negative integers strictly greater than total_layers to test
    bounds validation.
    """
    value = draw(st.integers(min_value=total_layers + 1, max_value=total_layers + 1000))
    return value


# ---------------------------------------------------------------------------
# Property 4: Freeze layers bounds validation
# Feature: rt-detr-freeze-layers, Property 4: Freeze layers bounds validation
#
# For any non-negative integer value of freeze_layers that exceeds
# len(_model.model.model), the RT_DETR_Detector initialization SHALL raise a
# ConfigurationError with a violations list containing the maximum allowed value.
#
# **Validates: Requirements 1.4, 2.5**
# ---------------------------------------------------------------------------


class TestProperty4FreezeLayersBoundsValidation:
    """Property 4: Freeze layers bounds validation.

    For any non-negative integer value of freeze_layers that exceeds
    len(_model.model.model), the RT_DETR_Detector initialization SHALL
    raise a ConfigurationError with a violations list containing the
    maximum allowed value.

    **Validates: Requirements 1.4, 2.5**
    """

    @given(
        model_size=VALID_MODEL_SIZES,
        num_classes=VALID_NUM_CLASSES,
        freeze_layers=freeze_layers_exceeding_total(),
    )
    @settings(max_examples=100, deadline=None)
    def test_freeze_layers_exceeding_total_raises_configuration_error(
        self, model_size, num_classes, freeze_layers
    ):
        """freeze_layers > total layer count raises ConfigurationError with max value.

        Feature: rt-detr-freeze-layers, Property 4: Freeze layers bounds validation
        """
        config = {
            "model_size": model_size,
            "num_classes": num_classes,
            "freeze_layers": freeze_layers,
        }

        mock_rtdetr_cls, inner_model = create_mock_rtdetr_cls(
            num_layers=MOCK_TOTAL_LAYERS
        )

        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr_cls):
            with pytest.raises(ConfigurationError) as exc_info:
                RT_DETR_Detector(config)

        # The error message must contain the maximum allowed value
        violations = exc_info.value.violations

        # Assert: violations list contains maximum allowed value (total layer count)
        assert any(
            str(MOCK_TOTAL_LAYERS) in v for v in violations
        ), (
            f"ConfigurationError violations should contain the maximum "
            f"allowed value ({MOCK_TOTAL_LAYERS}), but got: {violations}. "
            f"freeze_layers={freeze_layers}"
        )

        # Assert: violations list mentions the invalid freeze_layers value
        assert any(
            str(freeze_layers) in v for v in violations
        ), (
            f"ConfigurationError violations should mention the invalid "
            f"freeze_layers value ({freeze_layers}), but got: {violations}"
        )


# ---------------------------------------------------------------------------
# Property 2: Default behavior preservation
# Feature: rt-detr-freeze-layers, Property 2: Default behavior preservation
#
# For any valid RT-DETR model configuration where freeze_layers is absent or
# set to 0, ALL model parameters SHALL have requires_grad = True, and
# get_parameters() SHALL return a list whose length equals the total number
# of model parameters.
#
# **Validates: Requirements 1.2, 3.3, 4.1, 4.4**
# ---------------------------------------------------------------------------


class TestProperty2DefaultBehaviorPreservation:
    """Property 2: Default behavior preservation.

    For any valid RT-DETR model configuration where freeze_layers is absent
    or set to 0, ALL model parameters SHALL have requires_grad = True, and
    get_parameters() SHALL return a list whose length equals the total number
    of model parameters.

    **Validates: Requirements 1.2, 3.3, 4.1, 4.4**
    """

    @given(config=valid_config_without_freeze_layers())
    @settings(max_examples=100, deadline=None)
    def test_no_freeze_layers_all_params_trainable(self, config):
        """Without freeze_layers in config, all model parameters have requires_grad=True.

        Feature: rt-detr-freeze-layers, Property 2: Default behavior preservation
        **Validates: Requirements 1.2**
        """
        mock_rtdetr_cls, inner_model = create_mock_rtdetr_cls(MOCK_TOTAL_LAYERS)
        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr_cls):
            detector = RT_DETR_Detector(config)

        # ALL parameters must have requires_grad = True
        all_params = list(detector._model.model.parameters())
        assert len(all_params) > 0, "Model must have at least one parameter"

        for param in all_params:
            assert param.requires_grad is True, (
                "Parameter has requires_grad=False but should be True "
                "when freeze_layers is not present in config"
            )

    @given(config=valid_config_without_freeze_layers())
    @settings(max_examples=100, deadline=None)
    def test_no_freeze_layers_get_parameters_returns_all(self, config):
        """Without freeze_layers, get_parameters() returns count equal to total param count.

        Feature: rt-detr-freeze-layers, Property 2: Default behavior preservation
        **Validates: Requirements 3.3, 4.4**
        """
        mock_rtdetr_cls, inner_model = create_mock_rtdetr_cls(MOCK_TOTAL_LAYERS)
        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr_cls):
            detector = RT_DETR_Detector(config)

        trainable_params = detector.get_parameters()
        all_params = list(detector._model.model.parameters())

        # get_parameters() should return ALL params when freeze_layers absent
        assert len(trainable_params) == len(all_params), (
            f"get_parameters() returned {len(trainable_params)} params but "
            f"total param count is {len(all_params)}. They should be equal "
            f"when freeze_layers is not present."
        )

    @given(config=valid_config_with_freeze_layers_zero())
    @settings(max_examples=100, deadline=None)
    def test_freeze_layers_zero_all_params_trainable(self, config):
        """With freeze_layers=0, all model parameters have requires_grad=True.

        Feature: rt-detr-freeze-layers, Property 2: Default behavior preservation
        **Validates: Requirements 4.1**
        """
        mock_rtdetr_cls, inner_model = create_mock_rtdetr_cls(MOCK_TOTAL_LAYERS)
        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr_cls):
            detector = RT_DETR_Detector(config)

        # ALL parameters must have requires_grad = True
        all_params = list(detector._model.model.parameters())
        assert len(all_params) > 0, "Model must have at least one parameter"

        for param in all_params:
            assert param.requires_grad is True, (
                "Parameter has requires_grad=False but should be True "
                "when freeze_layers is set to 0"
            )

    @given(config=valid_config_with_freeze_layers_zero())
    @settings(max_examples=100, deadline=None)
    def test_freeze_layers_zero_get_parameters_returns_all(self, config):
        """With freeze_layers=0, get_parameters() returns count equal to total param count.

        Feature: rt-detr-freeze-layers, Property 2: Default behavior preservation
        **Validates: Requirements 3.3, 4.4**
        """
        mock_rtdetr_cls, inner_model = create_mock_rtdetr_cls(MOCK_TOTAL_LAYERS)
        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr_cls):
            detector = RT_DETR_Detector(config)

        trainable_params = detector.get_parameters()
        all_params = list(detector._model.model.parameters())

        # get_parameters() should return ALL params when freeze_layers=0
        assert len(trainable_params) == len(all_params), (
            f"get_parameters() returned {len(trainable_params)} params but "
            f"total param count is {len(all_params)}. They should be equal "
            f"when freeze_layers is 0."
        )


# ---------------------------------------------------------------------------
# Hypothesis strategies for Property 3: Invalid freeze_layers values
# ---------------------------------------------------------------------------

# Invalid types: strings, floats, booleans, lists, dicts, negative ints
_INVALID_STRINGS = st.text(min_size=0, max_size=50)
_INVALID_FLOATS = st.floats(allow_nan=True, allow_infinity=True)
_INVALID_BOOLEANS = st.booleans()
_INVALID_NEGATIVE_INTS = st.integers(max_value=-1)
_INVALID_LISTS = st.lists(st.integers(), min_size=0, max_size=5)
_INVALID_DICTS = st.dictionaries(
    keys=st.text(min_size=1, max_size=10),
    values=st.integers(),
    min_size=0,
    max_size=3,
)

# Combined strategy for all invalid freeze_layers values
INVALID_FREEZE_LAYERS_VALUES = st.one_of(
    _INVALID_STRINGS,
    _INVALID_FLOATS,
    _INVALID_BOOLEANS,
    _INVALID_NEGATIVE_INTS,
    _INVALID_LISTS,
    _INVALID_DICTS,
)


@st.composite
def valid_config_with_invalid_freeze_layers(draw):
    """Generate a valid RT-DETR base config with an invalid freeze_layers value.

    The base config (model_size, num_classes) is valid so that only the
    freeze_layers validation triggers ConfigurationError.
    """
    invalid_value = draw(INVALID_FREEZE_LAYERS_VALUES)
    config = {
        "model_size": draw(VALID_MODEL_SIZES),
        "num_classes": draw(VALID_NUM_CLASSES),
        "freeze_layers": invalid_value,
    }
    return config


# ---------------------------------------------------------------------------
# Property 3: Invalid freeze_layers rejection
# Feature: rt-detr-freeze-layers, Property 3: Invalid freeze_layers rejection
#
# For any value of freeze_layers that is not an integer, is a boolean, or is a
# negative integer, the RT_DETR_Detector initialization SHALL raise a
# ConfigurationError with a violations list containing a message indicating the
# value must be a non-negative integer.
#
# **Validates: Requirements 1.3, 2.6**
# ---------------------------------------------------------------------------


class TestProperty3InvalidFreezeLayers:
    """Property 3: Invalid freeze_layers rejection.

    For any value of freeze_layers that is not an integer, is a boolean, or
    is a negative integer, the RT_DETR_Detector initialization SHALL raise a
    ConfigurationError with a violations list containing a message indicating
    the value must be a non-negative integer.

    **Validates: Requirements 1.3, 2.6**
    """

    @given(config=valid_config_with_invalid_freeze_layers())
    @settings(max_examples=100, deadline=None)
    def test_invalid_freeze_layers_raises_configuration_error(self, config):
        """Invalid freeze_layers values (strings, floats, booleans, negative ints,
        lists, dicts) raise ConfigurationError with appropriate message.

        Feature: rt-detr-freeze-layers, Property 3: Invalid freeze_layers rejection
        """
        invalid_value = config["freeze_layers"]

        mock_rtdetr_cls, inner_model = create_mock_rtdetr_cls(MOCK_TOTAL_LAYERS)

        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr_cls):
            with pytest.raises(ConfigurationError) as exc_info:
                RT_DETR_Detector(config)

        # Verify the violations list contains the expected message pattern
        error = exc_info.value
        assert hasattr(error, "violations"), (
            "ConfigurationError should have a 'violations' attribute"
        )
        assert len(error.violations) >= 1, (
            "ConfigurationError should have at least one violation"
        )

        # At least one violation should reference "non-negative integer"
        violation_text = " ".join(error.violations).lower()
        assert "non-negative integer" in violation_text, (
            f"Expected violation message about 'non-negative integer' but got: "
            f"{error.violations}"
        )

        # The violation should reference the invalid value
        full_violations = " ".join(error.violations)
        assert str(invalid_value) in full_violations, (
            f"Expected violation to mention the invalid value '{invalid_value}' "
            f"but got: {error.violations}"
        )


# ---------------------------------------------------------------------------
# Property 5: get_parameters returns correct trainable subset
# Feature: rt-detr-freeze-layers, Property 5: get_parameters returns correct trainable subset
#
# For any valid freeze_layers value N > 0, get_parameters() SHALL return
# exactly (total_parameter_count - frozen_layer_parameter_count) parameters,
# all of which have requires_grad = True and none of which belong to layers
# with index < N.
#
# **Validates: Requirements 3.1, 3.2**
# ---------------------------------------------------------------------------


class TestProperty5GetParametersTrainableSubset:
    """Property 5: get_parameters returns correct trainable subset.

    For any valid freeze_layers value N > 0, get_parameters() SHALL return
    exactly (total_parameter_count - frozen_layer_parameter_count) parameters,
    all of which have requires_grad = True and none of which belong to layers
    with index < N.

    **Validates: Requirements 3.1, 3.2**
    """

    @given(config=valid_config_with_freeze_layers())
    @settings(max_examples=100, deadline=None)
    def test_get_parameters_count_equals_total_minus_frozen(self, config):
        """get_parameters() count = total_param_count - frozen_layer_param_count.

        Feature: rt-detr-freeze-layers, Property 5: get_parameters returns correct trainable subset
        **Validates: Requirements 3.1, 3.2**
        """
        freeze_layers = config["freeze_layers"]

        mock_rtdetr_cls, inner_model = create_mock_rtdetr_cls(MOCK_TOTAL_LAYERS)
        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr_cls):
            detector = RT_DETR_Detector(config)

        # Count total parameters
        total_param_count = sum(
            1 for _ in detector._model.model.parameters()
        )

        # Count frozen layer parameters (layers with index < N)
        frozen_param_count = 0
        for name, _ in detector._model.model.named_parameters():
            parts = name.split(".")
            if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
                layer_idx = int(parts[1])
                if layer_idx < freeze_layers:
                    frozen_param_count += 1

        # get_parameters() should return total - frozen
        trainable_params = detector.get_parameters()
        expected_count = total_param_count - frozen_param_count

        assert len(trainable_params) == expected_count, (
            f"get_parameters() returned {len(trainable_params)} params, "
            f"expected {expected_count} (total={total_param_count} - "
            f"frozen={frozen_param_count}). freeze_layers={freeze_layers}"
        )

    @given(config=valid_config_with_freeze_layers())
    @settings(max_examples=100, deadline=None)
    def test_get_parameters_all_have_requires_grad_true(self, config):
        """All parameters returned by get_parameters() have requires_grad=True.

        Feature: rt-detr-freeze-layers, Property 5: get_parameters returns correct trainable subset
        **Validates: Requirements 3.1**
        """
        mock_rtdetr_cls, inner_model = create_mock_rtdetr_cls(MOCK_TOTAL_LAYERS)
        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr_cls):
            detector = RT_DETR_Detector(config)

        trainable_params = detector.get_parameters()

        for param in trainable_params:
            assert param.requires_grad is True, (
                f"get_parameters() returned a parameter with "
                f"requires_grad=False. freeze_layers={config['freeze_layers']}"
            )

    @given(config=valid_config_with_freeze_layers())
    @settings(max_examples=100, deadline=None)
    def test_get_parameters_no_frozen_layer_params(self, config):
        """No parameter returned by get_parameters() belongs to a layer with index < N.

        Feature: rt-detr-freeze-layers, Property 5: get_parameters returns correct trainable subset
        **Validates: Requirements 3.2**
        """
        freeze_layers = config["freeze_layers"]

        mock_rtdetr_cls, inner_model = create_mock_rtdetr_cls(MOCK_TOTAL_LAYERS)
        with patch("model.models.rt_detr_wrapper.RTDETR", mock_rtdetr_cls):
            detector = RT_DETR_Detector(config)

        # Collect parameter objects from frozen layers (index < N)
        frozen_param_ids = set()
        for name, param in detector._model.model.named_parameters():
            parts = name.split(".")
            if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
                layer_idx = int(parts[1])
                if layer_idx < freeze_layers:
                    frozen_param_ids.add(id(param))

        # get_parameters() should not contain any param from frozen layers
        trainable_params = detector.get_parameters()

        for param in trainable_params:
            assert id(param) not in frozen_param_ids, (
                f"get_parameters() returned a parameter belonging to a frozen "
                f"layer (index < {freeze_layers}). "
                f"freeze_layers={freeze_layers}"
            )
