"""Unit tests for dataset error translation in the evaluation script.

Feature: generic-evaluation-script, Task 6.3

These example-based tests exercise the dataset-boundary error translation
performed by ``load_split`` (and its helper ``_load_dataset``) in
``model/training/evaluate_detection.py``:

- A missing ``dataset.path`` is detected up front and surfaces a
  ``FileNotFoundError`` whose message names the offending path. The failure
  occurs *before* any dataset loading/inference begins, so the dataset's
  ``load`` method is never invoked. (Requirement 14.1)
- A malformed annotation file (the dataset layer raises ``ParseError``) is
  translated into a ``ConfigurationError`` whose message names the annotation
  file and the underlying parser error. (Requirement 14.3)

The tests do not touch a real dataset on disk: the missing-path case uses a
non-existent path, and the parse-error case uses a directory that exists while
monkeypatching ``RDD2022Dataset.load`` to raise ``ParseError``.
"""

from pathlib import Path

import pytest

from model.datasets.rdd2022 import RDD2022Dataset
from model.exceptions import ConfigurationError, ParseError
from model.training.evaluate_detection import load_split


class TestMissingDatasetPath:
    """A non-existent ``dataset.path`` must fail fast with a FileNotFoundError."""

    def test_missing_path_raises_file_not_found(self, tmp_path):
        """A non-existent ``dataset.path`` raises ``FileNotFoundError``.

        **Validates: Requirements 14.1**
        """
        missing = tmp_path / "no_such_dataset"
        config = {
            "dataset": {"path": str(missing)},
            "evaluation": {"split": "val"},
        }

        with pytest.raises(FileNotFoundError):
            load_split(config)

    def test_missing_path_message_includes_path(self, tmp_path):
        """The ``FileNotFoundError`` message identifies the offending path.

        **Validates: Requirements 14.1**
        """
        missing = tmp_path / "no_such_dataset"
        config = {
            "dataset": {"path": str(missing)},
            "evaluation": {"split": "val"},
        }

        with pytest.raises(FileNotFoundError) as exc_info:
            load_split(config)

        message = str(exc_info.value)
        # The offending path (at least its final component) appears in the message.
        assert "no_such_dataset" in message
        assert str(missing) in message

    def test_missing_path_fails_before_dataset_load(self, tmp_path, monkeypatch):
        """The missing-path check terminates before any dataset loading begins.

        Requirement 14.1 requires a missing dataset to fail *before inference
        begins*. The up-front existence check must short-circuit before the
        dataset's ``load`` method (the first I/O step toward inference) is ever
        invoked. We assert this by spying on ``RDD2022Dataset.load``.

        **Validates: Requirements 14.1**
        """
        load_calls = []

        def _spy_load(self, path):
            load_calls.append(path)

        monkeypatch.setattr(RDD2022Dataset, "load", _spy_load)

        missing = tmp_path / "no_such_dataset"
        config = {
            "dataset": {"path": str(missing)},
            "evaluation": {"split": "val"},
        }

        with pytest.raises(FileNotFoundError):
            load_split(config)

        # The dataset loader was never invoked: the failure is truly up front.
        assert load_calls == []


class TestAnnotationParseError:
    """A malformed annotation file must surface as a ConfigurationError."""

    ANNOTATION_FILE = Path("annotations/bad_image.jpg.json")
    LINE_NUMBER = 42
    DESCRIPTION = "unexpected token while parsing annotation"

    def _raise_parse_error(self, _path):
        raise ParseError(
            self.ANNOTATION_FILE,
            self.LINE_NUMBER,
            self.DESCRIPTION,
        )

    def test_parse_error_becomes_configuration_error(self, tmp_path, monkeypatch):
        """A loader ``ParseError`` is translated into a ``ConfigurationError``.

        **Validates: Requirements 14.3**
        """
        monkeypatch.setattr(RDD2022Dataset, "load", self._raise_parse_error)

        # The dataset path must exist so the up-front check passes and loading
        # proceeds to the (monkeypatched) parse step.
        config = {
            "dataset": {"path": str(tmp_path)},
            "evaluation": {"split": "test"},
        }

        with pytest.raises(ConfigurationError):
            load_split(config)

    def test_configuration_error_names_annotation_file_and_parser_error(
        self, tmp_path, monkeypatch
    ):
        """The translated error names the annotation file and the parser error.

        **Validates: Requirements 14.3**
        """
        monkeypatch.setattr(RDD2022Dataset, "load", self._raise_parse_error)

        config = {
            "dataset": {"path": str(tmp_path)},
            "evaluation": {"split": "test"},
        }

        with pytest.raises(ConfigurationError) as exc_info:
            load_split(config)

        message = str(exc_info.value)
        # The annotation file is identified.
        assert str(self.ANNOTATION_FILE) in message
        # The underlying parser error (line + description) is surfaced.
        assert str(self.LINE_NUMBER) in message
        assert self.DESCRIPTION in message

    def test_parse_error_preserved_in_violation_list(self, tmp_path, monkeypatch):
        """The translated ``ConfigurationError`` keeps a structured violation.

        **Validates: Requirements 14.3**
        """
        monkeypatch.setattr(RDD2022Dataset, "load", self._raise_parse_error)

        config = {
            "dataset": {"path": str(tmp_path)},
            "evaluation": {"split": "val"},
        }

        with pytest.raises(ConfigurationError) as exc_info:
            load_split(config)

        violations = exc_info.value.violations
        assert isinstance(violations, list)
        assert len(violations) == 1
        assert str(self.ANNOTATION_FILE) in violations[0]
        assert self.DESCRIPTION in violations[0]

    def test_parse_error_chains_original_cause(self, tmp_path, monkeypatch):
        """The ``ConfigurationError`` chains the original ``ParseError`` cause.

        **Validates: Requirements 14.3**
        """
        monkeypatch.setattr(RDD2022Dataset, "load", self._raise_parse_error)

        config = {
            "dataset": {"path": str(tmp_path)},
            "evaluation": {"split": "test"},
        }

        with pytest.raises(ConfigurationError) as exc_info:
            load_split(config)

        # The original ParseError is preserved as the chained cause.
        assert isinstance(exc_info.value.__cause__, ParseError)
