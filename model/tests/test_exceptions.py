"""Unit tests for the custom exception hierarchy."""

from pathlib import Path

import pytest

from model.exceptions import (
    ConfigurationError,
    DatasetNotFoundError,
    FrameworkError,
    ModelNotFoundError,
    ParseError,
    UnmappedClassError,
    ValidationError,
)


class TestFrameworkError:
    def test_is_base_exception(self):
        err = FrameworkError("test")
        assert isinstance(err, Exception)

    def test_message(self):
        err = FrameworkError("something went wrong")
        assert str(err) == "something went wrong"


class TestDatasetNotFoundError:
    def test_inherits_framework_error(self):
        err = DatasetNotFoundError(Path("/data/missing"))
        assert isinstance(err, FrameworkError)

    def test_stores_path(self):
        p = Path("/data/missing")
        err = DatasetNotFoundError(p)
        assert err.path == p

    def test_message_contains_path(self):
        p = Path("/data/missing")
        err = DatasetNotFoundError(p)
        assert "/data/missing" in str(err)
        assert "Dataset not found" in str(err)


class TestParseError:
    def test_inherits_framework_error(self):
        err = ParseError(Path("file.xml"), 42, "invalid tag")
        assert isinstance(err, FrameworkError)

    def test_stores_attributes(self):
        fp = Path("annotations/img.xml")
        err = ParseError(fp, 10, "missing element")
        assert err.file_path == fp
        assert err.line_number == 10
        assert err.description == "missing element"

    def test_message_contains_all_info(self):
        fp = Path("annotations/img.xml")
        err = ParseError(fp, 10, "missing element")
        msg = str(err)
        assert "annotations/img.xml" in msg
        assert "10" in msg
        assert "missing element" in msg


class TestUnmappedClassError:
    def test_inherits_framework_error(self):
        err = UnmappedClassError("D99", ["D00", "D10", "D20"])
        assert isinstance(err, FrameworkError)

    def test_stores_attributes(self):
        err = UnmappedClassError("D99", ["D00", "D10"])
        assert err.source_class == "D99"
        assert err.available_classes == ["D00", "D10"]

    def test_message_contains_class_info(self):
        err = UnmappedClassError("D99", ["D00", "D10", "D20"])
        msg = str(err)
        assert "D99" in msg
        assert "D00" in msg
        assert "D10" in msg
        assert "D20" in msg


class TestModelNotFoundError:
    def test_inherits_framework_error(self):
        err = ModelNotFoundError("resnet50", ["yolov6", "ssd_mobilenetv3"])
        assert isinstance(err, FrameworkError)

    def test_stores_attributes(self):
        err = ModelNotFoundError("resnet50", ["yolov6", "ssd_mobilenetv3"])
        assert err.model_name == "resnet50"
        assert err.available_models == ["yolov6", "ssd_mobilenetv3"]

    def test_message_contains_model_info(self):
        err = ModelNotFoundError("resnet50", ["yolov6", "ssd_mobilenetv3"])
        msg = str(err)
        assert "resnet50" in msg
        assert "yolov6" in msg
        assert "ssd_mobilenetv3" in msg


class TestConfigurationError:
    def test_inherits_framework_error(self):
        err = ConfigurationError(["missing field 'epochs'"])
        assert isinstance(err, FrameworkError)

    def test_stores_violations(self):
        violations = ["missing 'epochs'", "invalid 'lr'"]
        err = ConfigurationError(violations)
        assert err.violations == violations

    def test_message_joins_violations(self):
        violations = ["missing 'epochs'", "invalid 'lr'"]
        err = ConfigurationError(violations)
        msg = str(err)
        assert "missing 'epochs'" in msg
        assert "invalid 'lr'" in msg
        assert ";" in msg


class TestValidationError:
    def test_inherits_framework_error(self):
        err = ValidationError(["field 'name' required"])
        assert isinstance(err, FrameworkError)

    def test_stores_schema_violations(self):
        violations = ["field 'name' required", "'batch_size' must be int"]
        err = ValidationError(violations)
        assert err.schema_violations == violations

    def test_message_joins_violations(self):
        violations = ["field 'name' required", "'batch_size' must be int"]
        err = ValidationError(violations)
        msg = str(err)
        assert "field 'name' required" in msg
        assert "'batch_size' must be int" in msg
        assert ";" in msg


class TestExceptionHierarchy:
    """Test that all exceptions can be caught via FrameworkError."""

    def test_catch_all_via_base(self):
        exceptions = [
            DatasetNotFoundError(Path("/x")),
            ParseError(Path("f.xml"), 1, "bad"),
            UnmappedClassError("X", ["A", "B"]),
            ModelNotFoundError("m", ["a", "b"]),
            ConfigurationError(["v1"]),
            ValidationError(["s1"]),
        ]
        for exc in exceptions:
            with pytest.raises(FrameworkError):
                raise exc
