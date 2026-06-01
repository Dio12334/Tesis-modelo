"""Property-based tests for evaluation-split value validation.

Feature: generic-evaluation-script
Property 5: Split value validation

For any provided value of ``evaluation.split``, ``validate_config`` accepts it
without a split violation if and only if the value is one of ``train``,
``val``, or ``test``.

Because ``validate_config`` aggregates *all* violations into a single
``ConfigurationError`` (Req 4.1), these tests build an *otherwise-valid*
configuration and vary only ``evaluation.split``. The "split violation" under
test is the enum rule for ``evaluation.split`` (Req 4.5) - identified by a
violation string that names ``evaluation.split`` and reports the allowed
values - so that unrelated rules cannot mask or counterfeit the signal.

**Validates: Requirements 4.5**
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from model.exceptions import ConfigurationError
from model.training.evaluate_detection import validate_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALLOWED_SPLITS = {"train", "val", "test"}


def _otherwise_valid_config(split_value) -> dict:
    """Return a config that is valid in every respect except ``evaluation.split``.

    Every other required parameter is present and non-null and the checkpoint
    section satisfies the exclusive-or rule, so the *only* possible violation is
    the split enum rule. This isolates Property 5 from the other validation
    rules that ``validate_config`` evaluates in the same pass.
    """
    return {
        "model": {"type": "yolo26", "config": {"num_classes": 5}},
        "dataset": {"path": "/data/rdd2022"},
        "evaluation": {"split": split_value},
        "checkpoint": {"path": "/checkpoints/best_model.pt"},
    }


def _split_enum_violation(config: dict):
    """Return the split enum violation string if present, else ``None``.

    Runs ``validate_config`` and inspects the structured ``violations`` list of
    any raised ``ConfigurationError`` for the ``evaluation.split`` enum rule
    (the one that enumerates the allowed values). Returns ``None`` when no such
    violation is produced (including when validation passes without raising).
    """
    try:
        validate_config(config)
        return None
    except ConfigurationError as exc:
        for violation in exc.violations:
            if violation.startswith("evaluation.split:") and "expected one of" in violation:
                return violation
    return None


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# The three accepted split values.
_VALID_SPLITS = st.sampled_from(sorted(_ALLOWED_SPLITS))

# Arbitrary strings that are NOT one of the accepted split values. Includes the
# empty string and near-misses (e.g. "Train", "validation", "tests").
_INVALID_SPLIT_STRINGS = st.text(min_size=0, max_size=24).filter(
    lambda s: s not in _ALLOWED_SPLITS
)

# A mix of accepted and rejected values used to exercise the full iff.
_ANY_PROVIDED_SPLIT = st.one_of(_VALID_SPLITS, _INVALID_SPLIT_STRINGS)


# ---------------------------------------------------------------------------
# Property 5
# ---------------------------------------------------------------------------


class TestProperty5SplitValueValidation:
    """Property 5: Split value validation.

    **Validates: Requirements 4.5**
    """

    @given(split_value=_ANY_PROVIDED_SPLIT)
    @settings(max_examples=100)
    def test_split_violation_iff_value_not_allowed(self, split_value):
        # Feature: generic-evaluation-script, Property 5: Split value validation
        """A split violation is absent iff the provided value is train/val/test."""
        config = _otherwise_valid_config(split_value)

        has_violation = _split_enum_violation(config) is not None
        is_allowed = split_value in _ALLOWED_SPLITS

        # iff: accepted (no split violation) exactly when the value is allowed.
        assert has_violation == (not is_allowed)

    @given(split_value=_VALID_SPLITS)
    @settings(max_examples=100)
    def test_valid_split_passes_validation(self, split_value):
        # Feature: generic-evaluation-script, Property 5: Split value validation
        """An otherwise-valid config with an allowed split passes without raising."""
        config = _otherwise_valid_config(split_value)

        # The whole config is valid, so validate_config returns None (no raise).
        assert validate_config(config) is None

    @given(split_value=_INVALID_SPLIT_STRINGS)
    @settings(max_examples=100)
    def test_invalid_split_reports_parameter_and_allowed_values(self, split_value):
        # Feature: generic-evaluation-script, Property 5: Split value validation
        """An invalid split yields a violation naming the parameter and allowed values."""
        config = _otherwise_valid_config(split_value)

        violation = _split_enum_violation(config)

        assert violation is not None
        # Req 11.4: the violation names the parameter and the allowed values.
        assert violation.startswith("evaluation.split:")
        for allowed in _ALLOWED_SPLITS:
            assert allowed in violation

    # --- Explicit examples (documentation + fast smoke coverage) -----------

    def test_each_allowed_value_is_accepted(self):
        # Feature: generic-evaluation-script, Property 5: Split value validation
        """train, val, and test each validate cleanly."""
        for split_value in ("train", "val", "test"):
            assert validate_config(_otherwise_valid_config(split_value)) is None

    def test_representative_invalid_value_is_rejected(self):
        # Feature: generic-evaluation-script, Property 5: Split value validation
        """A common near-miss ("validation") is rejected as a split violation."""
        violation = _split_enum_violation(_otherwise_valid_config("validation"))
        assert violation is not None
