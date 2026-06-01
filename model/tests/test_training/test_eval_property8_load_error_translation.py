"""Property-based tests for checkpoint load-error translation.

Feature: generic-evaluation-script
Property 8: Checkpoint load-error translation

Design statement of the property:

    For any exception raised by ``Detector.load_checkpoint(path)`` that is not a
    ``FileNotFoundError``, the script re-raises a ``RuntimeError`` whose message
    contains both the resolved Checkpoint_Path and the underlying exception
    text; a ``FileNotFoundError`` from loading propagates unchanged.

Alignment with the implemented semantics
-----------------------------------------
The design (and Requirement 5.5) refines the broad statement above by splitting
non-``FileNotFoundError`` failures into two categories, and ``load_checkpoint_into``
in ``model/training/evaluate_detection.py`` implements exactly that split:

* **Open failures propagate unchanged.** ``FileNotFoundError`` and every other
  ``OSError`` subclass (``PermissionError``, ``IsADirectoryError``, low-level
  I/O errors, ...) are re-raised as-is and are **never** converted to another
  type (Req 5.5). ``FileNotFoundError`` is itself an ``OSError``.
* **Corrupt/incompatible load failures become ``RuntimeError``.** Every other
  (non-``OSError``) exception -- e.g. a ``RuntimeError`` from ``torch.load`` on
  a truncated archive, an ``EOFError``/unpickling error, or a
  ``KeyError``/``ValueError`` from an incompatible state dict -- is re-raised as
  a ``RuntimeError`` whose message names the Checkpoint_Path and includes the
  underlying exception text, chained via ``from`` (Req 5.7, 12.2).

These tests therefore assert the property against the implemented semantics:
corruption-style (non-``OSError``) exceptions are wrapped as ``RuntimeError``
naming the path + cause text; ``FileNotFoundError`` (and other ``OSError``)
propagate unchanged.

The tests exercise the real ``load_checkpoint_into`` function with a conforming
fake ``BaseDetector`` whose ``load_checkpoint`` raises a configured exception,
so they run without GPUs, real checkpoints, or real datasets.

**Validates: Requirements 5.7, 12.2**
"""

import pickle
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from model.models.registry import BaseDetector
from model.training.evaluate_detection import load_checkpoint_into


# ---------------------------------------------------------------------------
# Conforming fake BaseDetector whose load_checkpoint raises a configured error
# ---------------------------------------------------------------------------


class FakeDetector(BaseDetector):
    """Minimal conforming ``BaseDetector`` for load-error translation tests.

    ``load_checkpoint`` records each call and raises the exception instance it
    was configured with (or returns normally when configured with ``None``).
    All other abstract methods are trivial no-ops so the class is concrete.
    """

    def __init__(self, error: BaseException | None = None):
        self._error = error
        self.load_calls: list[Path] = []

    def forward(self, images):  # pragma: no cover - never invoked here
        return []

    def get_config_schema(self) -> dict:  # pragma: no cover - never invoked here
        return {}

    def load_checkpoint(self, path) -> None:
        self.load_calls.append(path)
        if self._error is not None:
            raise self._error

    def save_checkpoint(self, path) -> None:  # pragma: no cover - never invoked here
        pass


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Safe single path segments (mirrors the convention used by the Property 7
# test): non-empty, no separators or NUL, never a relative-navigation token.
_SAFE_SEGMENT_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789-_"
)

_PATH_STRINGS = st.text(
    alphabet=_SAFE_SEGMENT_ALPHABET + "/.", min_size=1, max_size=40
).filter(lambda s: s.strip() not in ("", ".", ".."))

# Human-readable underlying-cause messages (printable ASCII, non-empty).
_MESSAGE_TEXT = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=1,
    max_size=50,
)

# Corruption/incompatible-style exception TYPES that are NOT OSError subclasses.
# These represent "the file opened but its contents are corrupt/incompatible"
# and MUST be wrapped as RuntimeError (Req 5.7, 12.2).
_CORRUPTION_EXC_TYPES = st.sampled_from(
    [
        RuntimeError,
        ValueError,
        KeyError,
        EOFError,
        TypeError,
        ArithmeticError,
        AttributeError,
        pickle.UnpicklingError,
        Exception,
    ]
)

# OSError subclasses (other than FileNotFoundError) representing "open" failures
# that MUST propagate unchanged (Req 5.5).
_OTHER_OSERROR_TYPES = st.sampled_from(
    [
        PermissionError,
        IsADirectoryError,
        NotADirectoryError,
        BlockingIOError,
        ConnectionError,
        OSError,
    ]
)


# ---------------------------------------------------------------------------
# Property 8
# ---------------------------------------------------------------------------


class TestProperty8LoadErrorTranslation:
    """Property 8: Checkpoint load-error translation.

    **Validates: Requirements 5.7, 12.2**
    """

    @given(
        path_value=_PATH_STRINGS,
        exc_type=_CORRUPTION_EXC_TYPES,
        message=_MESSAGE_TEXT,
    )
    @settings(max_examples=100)
    def test_corruption_exception_wrapped_as_runtimeerror(
        self, path_value, exc_type, message
    ):
        # Feature: generic-evaluation-script, Property 8: Checkpoint load-error translation
        """Non-OSError load failures become a RuntimeError naming path + cause.

        For any corruption/incompatible-style exception (i.e. one that is not an
        ``OSError``), ``load_checkpoint_into`` re-raises a ``RuntimeError`` whose
        message contains both the resolved Checkpoint_Path and the underlying
        exception text, with the original exception preserved as ``__cause__``.

        **Validates: Requirements 5.7, 12.2**
        """
        original = exc_type(message)
        checkpoint_path = Path(path_value)
        detector = FakeDetector(error=original)

        with pytest.raises(RuntimeError) as exc_info:
            load_checkpoint_into(detector, checkpoint_path)

        raised = exc_info.value
        # The wrapper must be a plain RuntimeError, not the (RuntimeError) cause
        # leaking through, so identity must differ from the original even when
        # the corruption exception is itself a RuntimeError.
        assert raised is not original
        # Cause chain preserved (raise ... from exc).
        assert raised.__cause__ is original

        wrapped_message = str(raised)
        # Req 12.2: message contains the resolved Checkpoint_Path ...
        assert str(checkpoint_path) in wrapped_message
        # ... and the underlying exception text.
        assert str(original) in wrapped_message

        # The detector's load_checkpoint was actually invoked with the path.
        assert detector.load_calls == [checkpoint_path]

    @given(path_value=_PATH_STRINGS, message=_MESSAGE_TEXT)
    @settings(max_examples=100)
    def test_file_not_found_propagates_unchanged(self, path_value, message):
        # Feature: generic-evaluation-script, Property 8: Checkpoint load-error translation
        """A FileNotFoundError from loading propagates unchanged (same instance).

        **Validates: Requirements 5.7, 12.2**
        """
        original = FileNotFoundError(message)
        checkpoint_path = Path(path_value)
        detector = FakeDetector(error=original)

        with pytest.raises(FileNotFoundError) as exc_info:
            load_checkpoint_into(detector, checkpoint_path)

        # Same instance re-raised, never converted to RuntimeError.
        assert exc_info.value is original

    @given(
        path_value=_PATH_STRINGS,
        exc_type=_OTHER_OSERROR_TYPES,
        message=_MESSAGE_TEXT,
    )
    @settings(max_examples=100)
    def test_other_oserror_propagates_unchanged(
        self, path_value, exc_type, message
    ):
        # Feature: generic-evaluation-script, Property 8: Checkpoint load-error translation
        """Non-FileNotFoundError OSError (open failures) propagate unchanged (Req 5.5).

        The implemented semantics treat permission/IO/open errors as passthrough
        rather than wrapping them as ``RuntimeError``; only corrupt/incompatible
        (non-``OSError``) failures are wrapped.

        **Validates: Requirements 5.7, 12.2 (with Req 5.5 refinement)**
        """
        original = exc_type(message)
        checkpoint_path = Path(path_value)
        detector = FakeDetector(error=original)

        with pytest.raises(OSError) as exc_info:
            load_checkpoint_into(detector, checkpoint_path)

        # Same instance re-raised unchanged, not converted to RuntimeError.
        assert exc_info.value is original
        assert not isinstance(exc_info.value, RuntimeError)

    @given(path_value=_PATH_STRINGS)
    @settings(max_examples=100)
    def test_successful_load_returns_none(self, path_value):
        # Feature: generic-evaluation-script, Property 8: Checkpoint load-error translation
        """When load_checkpoint succeeds, load_checkpoint_into returns None.

        Loading must complete fully before the caller proceeds; a successful
        load raises nothing and returns ``None``.

        **Validates: Requirements 5.7, 12.2**
        """
        checkpoint_path = Path(path_value)
        detector = FakeDetector(error=None)

        result = load_checkpoint_into(detector, checkpoint_path)

        assert result is None
        assert detector.load_calls == [checkpoint_path]


# ---------------------------------------------------------------------------
# Example-based unit tests complementing Property 8
# ---------------------------------------------------------------------------


class TestLoadErrorTranslationExamples:
    """Concrete examples complementing Property 8.

    **Validates: Requirements 5.7, 12.2**
    """

    def test_corrupt_archive_runtimeerror_is_wrapped(self):
        """A torch.load-style RuntimeError is wrapped with path + cause. (Req 12.2)"""
        original = RuntimeError("PytorchStreamReader failed reading zip archive")
        checkpoint_path = Path("/checkpoints/yolo26/run-1/best_model.pt")
        detector = FakeDetector(error=original)

        with pytest.raises(RuntimeError) as exc_info:
            load_checkpoint_into(detector, checkpoint_path)

        message = str(exc_info.value)
        assert str(checkpoint_path) in message
        assert "PytorchStreamReader failed reading zip archive" in message
        assert exc_info.value.__cause__ is original

    def test_incompatible_state_dict_keyerror_is_wrapped(self):
        """A KeyError from an incompatible state dict is wrapped. (Req 5.7)"""
        original = KeyError("backbone.layer1.weight")
        checkpoint_path = Path("/checkpoints/ssd/run-2/last_model.pt")
        detector = FakeDetector(error=original)

        with pytest.raises(RuntimeError) as exc_info:
            load_checkpoint_into(detector, checkpoint_path)

        message = str(exc_info.value)
        assert str(checkpoint_path) in message
        # KeyError stringifies to "'backbone.layer1.weight'"; that text appears.
        assert str(original) in message
        assert exc_info.value.__cause__ is original

    def test_file_not_found_is_not_wrapped(self):
        """FileNotFoundError propagates unchanged, not converted. (Req 5.5)"""
        original = FileNotFoundError("No such file: best_model.pt")
        detector = FakeDetector(error=original)

        with pytest.raises(FileNotFoundError) as exc_info:
            load_checkpoint_into(detector, Path("/missing/best_model.pt"))

        assert exc_info.value is original

    def test_permission_error_propagates_unchanged(self):
        """PermissionError (open failure) propagates unchanged. (Req 5.5)"""
        original = PermissionError("Permission denied")
        detector = FakeDetector(error=original)

        with pytest.raises(PermissionError) as exc_info:
            load_checkpoint_into(detector, Path("/locked/best_model.pt"))

        assert exc_info.value is original
        assert not isinstance(exc_info.value, RuntimeError)
