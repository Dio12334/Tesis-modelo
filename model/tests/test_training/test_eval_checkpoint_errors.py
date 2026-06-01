"""Unit tests for non-``FileNotFoundError`` checkpoint open-error passthrough.

Feature: generic-evaluation-script, Task 3.6

These example-based tests exercise the error-classification behaviour of
``load_checkpoint_into`` in ``model/training/evaluate_detection.py`` for the
specific case where a candidate checkpoint file is present on disk but cannot
be *opened* due to a cause that is **not** ``FileNotFoundError`` -- for example
a permission error or a low-level I/O error.

Per Requirement 5.5, such an exception must be allowed to propagate **unchanged**:
it must not be converted into a ``FileNotFoundError`` (the "missing file"
category) nor into a ``RuntimeError`` (the "corrupt/incompatible contents"
category). Only ``FileNotFoundError`` and corrupt/incompatible-content failures
have special handling; every other ``OSError`` subclass raised while opening the
file flows straight through.

**Validates: Requirements 5.5**
"""

from pathlib import Path

import pytest

from model.training.evaluate_detection import load_checkpoint_into


class _RaisingDetector:
    """A minimal fake detector whose ``load_checkpoint`` raises a fixed error.

    The fake conforms only to the surface that ``load_checkpoint_into`` touches:
    a single ``load_checkpoint(path)`` method. It records the path it was called
    with so tests can assert the resolved Checkpoint_Path was forwarded verbatim.
    """

    def __init__(self, error: BaseException):
        self._error = error
        self.called_with = None

    def load_checkpoint(self, path):
        self.called_with = path
        raise self._error


class TestPermissionErrorPassthrough:
    """A ``PermissionError`` while opening an existing candidate propagates as-is."""

    def test_permission_error_propagates_unchanged(self):
        """``PermissionError`` is re-raised unchanged, not translated.

        **Validates: Requirements 5.5**
        """
        original = PermissionError(13, "Permission denied")
        detector = _RaisingDetector(original)
        checkpoint_path = Path("/some/existing/best_model.pt")

        with pytest.raises(PermissionError) as exc_info:
            load_checkpoint_into(detector, checkpoint_path)

        # The exact same exception instance propagates (re-raised unchanged).
        assert exc_info.value is original
        # The detector was asked to load the resolved Checkpoint_Path.
        assert detector.called_with == checkpoint_path

    def test_permission_error_not_converted_to_other_types(self):
        """A propagated ``PermissionError`` is neither a FileNotFoundError nor RuntimeError.

        ``PermissionError`` is an ``OSError`` subclass but is **not** a
        ``FileNotFoundError``; asserting both ``isinstance`` checks fail guards
        against accidental conversion into either special-cased category.

        **Validates: Requirements 5.5**
        """
        detector = _RaisingDetector(PermissionError(13, "Permission denied"))

        with pytest.raises(PermissionError) as exc_info:
            load_checkpoint_into(detector, Path("/some/existing/best_model.pt"))

        raised = exc_info.value
        assert not isinstance(raised, FileNotFoundError)
        assert not isinstance(raised, RuntimeError)


class TestGenericOSErrorPassthrough:
    """A generic ``OSError`` / I/O error while opening an existing candidate propagates as-is."""

    def test_generic_oserror_propagates_unchanged(self):
        """A bare ``OSError`` is re-raised unchanged, not translated.

        **Validates: Requirements 5.5**
        """
        original = OSError(5, "Input/output error")
        detector = _RaisingDetector(original)
        checkpoint_path = Path("/some/existing/last_model.pt")

        with pytest.raises(OSError) as exc_info:
            load_checkpoint_into(detector, checkpoint_path)

        raised = exc_info.value
        # Exact same instance, exact same type (not a subclass swap).
        assert raised is original
        assert type(raised) is OSError
        # Not converted into either special-cased category.
        assert not isinstance(raised, FileNotFoundError)
        assert not isinstance(raised, RuntimeError)
        assert detector.called_with == checkpoint_path

    def test_ioerror_alias_propagates_unchanged(self):
        """``IOError`` (an alias of ``OSError``) propagates unchanged.

        **Validates: Requirements 5.5**
        """
        # In Python 3, IOError is an alias for OSError; this documents that a
        # raised I/O error is not reclassified.
        original = IOError("disk read failure")
        detector = _RaisingDetector(original)

        with pytest.raises(OSError) as exc_info:
            load_checkpoint_into(detector, Path("/some/existing/best_model.pt"))

        assert exc_info.value is original
        assert not isinstance(exc_info.value, FileNotFoundError)
        assert not isinstance(exc_info.value, RuntimeError)
