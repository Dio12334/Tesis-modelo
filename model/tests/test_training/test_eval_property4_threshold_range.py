"""Property-based tests for evaluation threshold range validation.

Feature: generic-evaluation-script
Property 4: Threshold range validation

For any provided value of ``evaluation.confidence_threshold`` and
``evaluation.iou_threshold``, ``validate_config`` produces a violation for that
parameter *if and only if* the value lies outside the closed interval
``[0.0, 1.0]``.

These tests build an otherwise-valid configuration (all required parameters
present and non-null, plus a valid checkpoint exclusive-or with exactly one of
``checkpoint.path`` / ``checkpoint.run_id``) and then vary the threshold values
across in-range and out-of-range numbers, asserting the iff relationship per
parameter. ``validate_config`` is pure and performs no filesystem access, so
the configuration never needs real files on disk.

**Validates: Requirements 4.3, 4.4**
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from model.exceptions import ConfigurationError
from model.training.evaluate_detection import validate_config


# ---------------------------------------------------------------------------
# Parameter names under test
# ---------------------------------------------------------------------------

CONFIDENCE = "evaluation.confidence_threshold"
IOU = "evaluation.iou_threshold"


# ---------------------------------------------------------------------------
# Hypothesis strategies: in-range vs out-of-range numeric threshold values
# ---------------------------------------------------------------------------

# Closed interval [0.0, 1.0] -- includes the boundaries 0.0 and 1.0.
_IN_RANGE_FLOATS = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)
# Strictly below 0.0.
_BELOW_FLOATS = st.floats(
    min_value=-1e6, max_value=0.0, exclude_max=True,
    allow_nan=False, allow_infinity=False,
)
# Strictly above 1.0.
_ABOVE_FLOATS = st.floats(
    min_value=1.0, max_value=1e6, exclude_min=True,
    allow_nan=False, allow_infinity=False,
)

# A few integer values exercise the int branch of validation as well.
_IN_RANGE_INTS = st.sampled_from([0, 1])
_OUT_OF_RANGE_INTS = st.sampled_from([-5, -1, 2, 10, 100])

# Any provided threshold value: a balanced mix of in-range and out-of-range.
_THRESHOLD_VALUES = st.one_of(
    _IN_RANGE_FLOATS,
    _BELOW_FLOATS,
    _ABOVE_FLOATS,
    _IN_RANGE_INTS,
    _OUT_OF_RANGE_INTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_base_config() -> dict:
    """Return an otherwise-valid merged config with no threshold keys set.

    All required parameters are present and non-null, and exactly one of
    ``checkpoint.path`` / ``checkpoint.run_id`` is provided, so the only
    violations that can arise come from the threshold values injected by a test.
    """
    return {
        "model": {"type": "yolo26", "config": {"num_classes": 5}},
        "dataset": {"path": "model/data/rdd2022/sample"},
        "evaluation": {"split": "val"},
        "checkpoint": {"path": "checkpoints/yolo26/best_model.pt"},
    }


def _collect_violations(config: dict) -> list:
    """Run ``validate_config`` and return the list of violation strings.

    Returns an empty list when validation passes (no ``ConfigurationError``),
    otherwise the raw per-violation strings from ``ConfigurationError.violations``.
    """
    try:
        validate_config(config)
        return []
    except ConfigurationError as exc:
        return list(exc.violations)


def _has_violation_for(violations: list, param: str) -> bool:
    """True iff some violation names ``param`` (violations are ``"<param>: ..."``)."""
    return any(v.startswith(param + ":") for v in violations)


def _is_out_of_range(value) -> bool:
    """Reference predicate mirroring the closed-interval rule [0.0, 1.0]."""
    return not (0.0 <= float(value) <= 1.0)


# ---------------------------------------------------------------------------
# Property 4
# ---------------------------------------------------------------------------


class TestProperty4ThresholdRangeValidation:
    """Property 4: Threshold range validation.

    **Validates: Requirements 4.3, 4.4**
    """

    @given(value=_THRESHOLD_VALUES)
    @settings(max_examples=100)
    def test_confidence_threshold_iff_out_of_range(self, value):
        # Feature: generic-evaluation-script, Property 4: Threshold range validation
        """A confidence_threshold violation appears iff the value is outside [0,1]."""
        config = _valid_base_config()
        config["evaluation"]["confidence_threshold"] = value

        violations = _collect_violations(config)

        assert _has_violation_for(violations, CONFIDENCE) == _is_out_of_range(value)
        # The unrelated threshold (absent here) never produces a violation.
        assert not _has_violation_for(violations, IOU)

    @given(value=_THRESHOLD_VALUES)
    @settings(max_examples=100)
    def test_iou_threshold_iff_out_of_range(self, value):
        # Feature: generic-evaluation-script, Property 4: Threshold range validation
        """An iou_threshold violation appears iff the value is outside [0,1]."""
        config = _valid_base_config()
        config["evaluation"]["iou_threshold"] = value

        violations = _collect_violations(config)

        assert _has_violation_for(violations, IOU) == _is_out_of_range(value)
        # The unrelated threshold (absent here) never produces a violation.
        assert not _has_violation_for(violations, CONFIDENCE)

    @given(confidence=_THRESHOLD_VALUES, iou=_THRESHOLD_VALUES)
    @settings(max_examples=100)
    def test_both_thresholds_validated_independently(self, confidence, iou):
        # Feature: generic-evaluation-script, Property 4: Threshold range validation
        """Each threshold's violation depends only on its own value's range."""
        config = _valid_base_config()
        config["evaluation"]["confidence_threshold"] = confidence
        config["evaluation"]["iou_threshold"] = iou

        violations = _collect_violations(config)

        assert _has_violation_for(violations, CONFIDENCE) == _is_out_of_range(confidence)
        assert _has_violation_for(violations, IOU) == _is_out_of_range(iou)

    @given(
        in_value=st.one_of(_IN_RANGE_FLOATS, _IN_RANGE_INTS),
    )
    @settings(max_examples=100)
    def test_in_range_values_do_not_raise(self, in_value):
        # Feature: generic-evaluation-script, Property 4: Threshold range validation
        """In-range thresholds keep an otherwise-valid config valid (no raise)."""
        config = _valid_base_config()
        config["evaluation"]["confidence_threshold"] = in_value
        config["evaluation"]["iou_threshold"] = in_value

        # An otherwise-valid config with in-range thresholds must validate cleanly.
        validate_config(config)

    @given(
        out_value=st.one_of(_BELOW_FLOATS, _ABOVE_FLOATS, _OUT_OF_RANGE_INTS),
    )
    @settings(max_examples=100)
    def test_out_of_range_value_raises_configuration_error(self, out_value):
        # Feature: generic-evaluation-script, Property 4: Threshold range validation
        """An out-of-range threshold raises ConfigurationError naming the parameter."""
        config = _valid_base_config()
        config["evaluation"]["confidence_threshold"] = out_value

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)

        assert _has_violation_for(list(exc_info.value.violations), CONFIDENCE)


# ---------------------------------------------------------------------------
# Boundary / edge-case examples (complement the property coverage above)
# ---------------------------------------------------------------------------


class TestThresholdRangeBoundaries:
    """Explicit boundary checks for the closed interval [0.0, 1.0].

    **Validates: Requirements 4.3, 4.4**
    """

    @pytest.mark.parametrize("value", [0.0, 1.0, 0.5])
    def test_boundary_and_interior_values_accepted(self, value):
        """0.0, 1.0 (inclusive boundaries) and an interior value produce no violation."""
        config = _valid_base_config()
        config["evaluation"]["confidence_threshold"] = value
        config["evaluation"]["iou_threshold"] = value

        violations = _collect_violations(config)

        assert not _has_violation_for(violations, CONFIDENCE)
        assert not _has_violation_for(violations, IOU)

    @pytest.mark.parametrize("value", [-0.0001, 1.0001, -1.0, 2.0])
    def test_just_outside_boundary_values_flagged(self, value):
        """Values just outside [0.0, 1.0] produce a violation for that parameter."""
        config = _valid_base_config()
        config["evaluation"]["confidence_threshold"] = value

        violations = _collect_violations(config)

        assert _has_violation_for(violations, CONFIDENCE)

    def test_absent_thresholds_produce_no_violation(self):
        """When neither threshold is provided, no threshold violation is produced."""
        config = _valid_base_config()  # no threshold keys at all

        violations = _collect_violations(config)

        assert not _has_violation_for(violations, CONFIDENCE)
        assert not _has_violation_for(violations, IOU)

    def test_observed_value_and_range_in_message(self):
        """An out-of-range message includes the observed value and the expected range."""
        config = _valid_base_config()
        config["evaluation"]["confidence_threshold"] = 1.5

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)

        message = str(exc_info.value)
        assert "1.5" in message
        assert "[0.0, 1.0]" in message
