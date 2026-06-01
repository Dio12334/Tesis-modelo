"""Property-based tests for registry error listing and suggestions.

Feature: generic-evaluation-script
Property 18: Registry error listing and suggestions

Design statement of the property:

    For an unregistered ``model.type``, ``build_detector`` raises a
    ``ModelNotFoundError`` whose message lists every registered model name in
    alphabetical order, and includes a "Did you mean: <name>?" suggestion when a
    registered name has edit distance in ``[1, 2]`` from the requested value; an
    exact-match instantiation failure for non-schema reasons includes the
    underlying cause and omits the suggestion.

These tests exercise the real ``build_detector`` function in
``model/training/evaluate_detection.py``. The set of registered models is
controlled deterministically by patching ``ModelRegistry.list_models`` (so the
"available" list is exactly what the test specifies) and, for the exact-match
failure case, ``ModelRegistry.create`` (so instantiation fails for a controlled,
non-schema reason). This keeps the tests free of GPUs, real model classes, and
real checkpoints.

A small independent reference Levenshtein implementation is used to compute the
*expected* suggestion behaviour, so the assertions verify the observable
property rather than re-using the production helper tautologically.

**Validates: Requirements 1.7, 13.1, 13.2, 13.3**
"""

from unittest.mock import patch

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from model.exceptions import ConfigurationError, ModelNotFoundError
from model.models.registry import ModelRegistry
from model.training.evaluate_detection import build_detector


# ---------------------------------------------------------------------------
# Independent reference helpers (NOT the production ones)
# ---------------------------------------------------------------------------


def _ref_levenshtein(a: str, b: str) -> int:
    """Straightforward reference Levenshtein edit distance for the tests."""
    if a == b:
        return 0
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    previous = list(range(n + 1))
    for i in range(1, m + 1):
        current = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            current[j] = min(
                current[j - 1] + 1,  # insertion
                previous[j] + 1,  # deletion
                previous[j - 1] + cost,  # substitution
            )
        previous = current
    return previous[n]


def _expected_suggestion(requested, registered):
    """Alphabetically-first registered name whose distance is in [1, 2], or None.

    Mirrors the *observable* contract: a suggestion exists exactly when some
    registered name lies within edit distance [1, 2], and ties on distance are
    broken by alphabetical order (because ``build_detector`` iterates the
    alphabetically-sorted available list and keeps the first minimum).
    """
    best = None
    best_distance = None
    for name in sorted(registered):
        distance = _ref_levenshtein(requested, name)
        if 1 <= distance <= 2 and (best_distance is None or distance < best_distance):
            best = name
            best_distance = distance
    return best


def _min_distance(requested, registered):
    """Minimum edit distance from ``requested`` to any registered name."""
    return min(_ref_levenshtein(requested, name) for name in registered)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_NAME_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789_"

# Registered model names look like real registry keys.
_MODEL_NAMES = st.text(alphabet=_NAME_ALPHABET, min_size=1, max_size=12)

# A unique, non-empty set of registered names.
_REGISTERED_SETS = st.lists(_MODEL_NAMES, min_size=1, max_size=6, unique=True)

# Non-schema exception types that an exact-name instantiation might raise.
# None of these are ConfigurationError (schema failure) so they MUST be wrapped
# as a ModelNotFoundError carrying the underlying cause (Req 13.3).
_NON_SCHEMA_EXC_TYPES = st.sampled_from(
    [RuntimeError, ValueError, TypeError, KeyError, AttributeError, Exception]
)

_CAUSE_TEXT = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=1,
    max_size=50,
)


@st.composite
def _close_request(draw):
    """A single registered name plus an unregistered request 1-2 edits away.

    Using a single registered name makes the minimum distance equal to the
    distance to that name, so the request is guaranteed to fall in the [1, 2]
    suggestion band and exercise the positive branch.
    """
    base = draw(st.text(alphabet=_NAME_ALPHABET, min_size=3, max_size=10))
    num_edits = draw(st.integers(min_value=1, max_value=2))
    positions = draw(
        st.lists(
            st.integers(min_value=0, max_value=len(base) - 1),
            min_size=num_edits,
            max_size=num_edits,
            unique=True,
        )
    )
    chars = list(base)
    for pos in positions:
        replacement = draw(st.sampled_from(_NAME_ALPHABET))
        assume(replacement != chars[pos])
        chars[pos] = replacement
    request = "".join(chars)
    # Edits may coincidentally cancel out under the edit-distance metric; keep
    # only requests that genuinely land in the [1, 2] band and differ from base.
    assume(request != base)
    assume(_ref_levenshtein(request, base) in (1, 2))
    return base, request


@st.composite
def _far_request(draw):
    """Registered names (letters) plus a request (digits) guaranteed > 2 away.

    Edit distance is at least the absolute length difference. Registered names
    are <= 4 chars over letters; the request is >= 8 chars over digits, so the
    distance to every registered name exceeds 2 and no suggestion is produced.
    """
    registered = draw(
        st.lists(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=4),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    request = draw(st.text(alphabet="0123456789", min_size=8, max_size=12))
    return registered, request


# ---------------------------------------------------------------------------
# Property 18
# ---------------------------------------------------------------------------


class TestProperty18RegistryErrors:
    """Property 18: Registry error listing and suggestions.

    **Validates: Requirements 1.7, 13.1, 13.2, 13.3**
    """

    @given(registered=_REGISTERED_SETS, requested=_MODEL_NAMES)
    @settings(max_examples=100)
    def test_unregistered_raises_and_lists_models_alphabetically(
        self, registered, requested
    ):
        # Feature: generic-evaluation-script, Property 18: Registry error listing and suggestions
        """Unregistered type raises ModelNotFoundError listing names alphabetically.

        **Validates: Requirements 1.7, 13.1**
        """
        assume(requested not in registered)

        with patch.object(
            ModelRegistry, "list_models", return_value=sorted(registered)
        ), patch.object(ModelRegistry, "create") as create_mock:
            with pytest.raises(ModelNotFoundError) as exc_info:
                build_detector(requested, {})

        # The registry was never asked to instantiate an unregistered type.
        create_mock.assert_not_called()

        error = exc_info.value
        assert error.model_name == requested
        # Req 13.1: available models are exposed in alphabetical order.
        assert error.available_models == sorted(registered)

        message = str(error)
        # The rendered message lists every registered name in alphabetical order
        # (the list repr of the sorted names appears verbatim).
        assert f"Available models: {sorted(registered)}" in message
        for name in registered:
            assert name in message

    @given(registered=_REGISTERED_SETS, requested=_MODEL_NAMES)
    @settings(max_examples=100)
    def test_suggestion_present_iff_registered_name_within_edit_distance(
        self, registered, requested
    ):
        # Feature: generic-evaluation-script, Property 18: Registry error listing and suggestions
        """"Did you mean" appears iff some registered name is within distance [1, 2].

        **Validates: Requirements 13.2**
        """
        assume(requested not in registered)

        with patch.object(
            ModelRegistry, "list_models", return_value=sorted(registered)
        ), patch.object(ModelRegistry, "create"):
            with pytest.raises(ModelNotFoundError) as exc_info:
                build_detector(requested, {})

        error = exc_info.value
        expected = _expected_suggestion(requested, registered)
        message = str(error)

        if expected is None:
            # No registered name within [1, 2] -> no suggestion of any kind.
            assert error.suggestion is None
            assert "Did you mean" not in message
        else:
            # Req 13.2: the closest in-band name is suggested.
            assert error.suggestion == expected
            assert f"Did you mean: {expected}?" in message
            # Sanity: the suggested name truly lies within the [1, 2] band.
            assert 1 <= _ref_levenshtein(requested, expected) <= 2

    @given(data=_close_request())
    @settings(max_examples=100)
    def test_close_request_yields_suggestion(self, data):
        # Feature: generic-evaluation-script, Property 18: Registry error listing and suggestions
        """A request 1-2 edits from the only registered name gets a suggestion.

        **Validates: Requirements 13.2**
        """
        base, requested = data
        registered = [base]

        with patch.object(
            ModelRegistry, "list_models", return_value=sorted(registered)
        ), patch.object(ModelRegistry, "create"):
            with pytest.raises(ModelNotFoundError) as exc_info:
                build_detector(requested, {})

        error = exc_info.value
        assert error.suggestion == base
        assert f"Did you mean: {base}?" in str(error)

    @given(data=_far_request())
    @settings(max_examples=100)
    def test_far_request_yields_no_suggestion(self, data):
        # Feature: generic-evaluation-script, Property 18: Registry error listing and suggestions
        """A request farther than distance 2 from every name gets no suggestion.

        **Validates: Requirements 13.2**
        """
        registered, requested = data
        assume(requested not in registered)
        # Guaranteed by construction, asserted here for clarity.
        assert _min_distance(requested, registered) > 2

        with patch.object(
            ModelRegistry, "list_models", return_value=sorted(registered)
        ), patch.object(ModelRegistry, "create"):
            with pytest.raises(ModelNotFoundError) as exc_info:
                build_detector(requested, {})

        error = exc_info.value
        assert error.suggestion is None
        assert "Did you mean" not in str(error)

    @given(
        registered=_REGISTERED_SETS,
        exc_type=_NON_SCHEMA_EXC_TYPES,
        cause_text=_CAUSE_TEXT,
    )
    @settings(max_examples=100)
    def test_exact_match_non_schema_failure_includes_cause_and_omits_suggestion(
        self, registered, exc_type, cause_text
    ):
        # Feature: generic-evaluation-script, Property 18: Registry error listing and suggestions
        """Exact-name instantiation failure (non-schema) carries cause, no suggestion.

        When the requested name is registered (exact match) but instantiation
        fails for a non-schema reason, ``build_detector`` raises a
        ``ModelNotFoundError`` whose message includes the underlying cause text
        and omits any "Did you mean" suggestion.

        **Validates: Requirements 13.3**
        """
        requested = sorted(registered)[0]
        original = exc_type(cause_text)

        with patch.object(
            ModelRegistry, "list_models", return_value=sorted(registered)
        ), patch.object(ModelRegistry, "create", side_effect=original):
            with pytest.raises(ModelNotFoundError) as exc_info:
                build_detector(requested, {"num_classes": 4})

        error = exc_info.value
        # Req 13.3: the underlying cause text is preserved and surfaced.
        assert error.cause == str(original)
        message = str(error)
        assert str(original) in message
        assert f"Underlying error: {str(original)}" in message
        # Req 13.3: no suggestion is offered for an exact-name failure.
        assert error.suggestion is None
        assert "Did you mean" not in message
        # The original exception is preserved on the chain.
        assert error.__cause__ is original
        # Available models are still listed alphabetically.
        assert error.available_models == sorted(registered)


# ---------------------------------------------------------------------------
# Example-based unit tests complementing Property 18
# ---------------------------------------------------------------------------


class TestRegistryErrorExamples:
    """Concrete examples complementing Property 18.

    **Validates: Requirements 1.7, 13.1, 13.2, 13.3**
    """

    def test_typo_yields_did_you_mean_suggestion(self):
        """A one-edit typo of a registered name yields the right suggestion. (Req 13.2)"""
        registered = ["ssd_mobilenetv3", "yolo26", "yolov6"]

        with patch.object(
            ModelRegistry, "list_models", return_value=sorted(registered)
        ), patch.object(ModelRegistry, "create"):
            with pytest.raises(ModelNotFoundError) as exc_info:
                build_detector("yolov26", {})

        error = exc_info.value
        # "yolov26" is one edit from "yolo26" (distance 1).
        assert error.suggestion == "yolo26"
        assert "Did you mean: yolo26?" in str(error)
        assert error.available_models == ["ssd_mobilenetv3", "yolo26", "yolov6"]

    def test_distant_name_lists_models_without_suggestion(self):
        """A far-off name lists models alphabetically with no suggestion. (Req 1.7, 13.1)"""
        registered = ["yolo26", "yolov6", "ssd_mobilenetv3"]

        with patch.object(
            ModelRegistry, "list_models", return_value=sorted(registered)
        ), patch.object(ModelRegistry, "create"):
            with pytest.raises(ModelNotFoundError) as exc_info:
                build_detector("completely_different_model", {})

        error = exc_info.value
        assert error.suggestion is None
        assert "Did you mean" not in str(error)
        assert error.available_models == ["ssd_mobilenetv3", "yolo26", "yolov6"]

    def test_exact_match_runtime_failure_wraps_with_cause(self):
        """An exact-name instantiation RuntimeError is wrapped with its cause. (Req 13.3)"""
        registered = ["yolo26", "yolov6"]
        original = RuntimeError("CUDA driver mismatch")

        with patch.object(
            ModelRegistry, "list_models", return_value=sorted(registered)
        ), patch.object(ModelRegistry, "create", side_effect=original):
            with pytest.raises(ModelNotFoundError) as exc_info:
                build_detector("yolo26", {"num_classes": 4})

        error = exc_info.value
        assert error.cause == "CUDA driver mismatch"
        assert "Underlying error: CUDA driver mismatch" in str(error)
        assert error.suggestion is None
        assert "Did you mean" not in str(error)

    def test_schema_failure_propagates_as_configuration_error(self):
        """A schema ConfigurationError propagates unchanged, not wrapped. (Req 13.4)

        Complements Req 13.3 by confirming the cause-wrapping path applies only
        to non-schema failures; schema violations surface as ConfigurationError.
        """
        registered = ["yolo26"]
        schema_error = ConfigurationError(["Missing required parameter: num_classes"])

        with patch.object(
            ModelRegistry, "list_models", return_value=sorted(registered)
        ), patch.object(ModelRegistry, "create", side_effect=schema_error):
            with pytest.raises(ConfigurationError) as exc_info:
                build_detector("yolo26", {})

        assert exc_info.value is schema_error
