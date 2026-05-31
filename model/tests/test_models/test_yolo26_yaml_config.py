"""Unit tests for YOLO26 YAML configuration loading and validation.

Tests cover:
- Loading train_yolo26.yaml without errors via ConfigManager
- Invalid model_size in YAML raises ValidationError with invalid value and allowed values
- Missing required fields under model.config raises ValidationError listing each field

Requirements: 6.1, 6.5, 6.6
"""

from pathlib import Path

import pytest

from model.config import ConfigManager, EXPERIMENT_SCHEMA, YOLO26_MODEL_CONFIG_SCHEMA
from model.exceptions import ValidationError


@pytest.fixture
def config_manager():
    """Create a ConfigManager instance."""
    return ConfigManager()


@pytest.fixture
def yaml_path():
    """Path to the train_yolo26.yaml config file."""
    return Path(__file__).resolve().parents[2] / "configs" / "train_yolo26.yaml"


@pytest.fixture
def valid_config(config_manager, yaml_path):
    """Load the train_yolo26.yaml config as a dict."""
    return config_manager.load(yaml_path)


class TestYAMLConfigLoading:
    """Tests for loading train_yolo26.yaml without errors (Requirement 6.1)."""

    def test_config_manager_loads_yaml_without_errors(self, config_manager, yaml_path):
        """Test that ConfigManager loads train_yolo26.yaml without raising any exception.

        Validates: Requirement 6.1
        """
        config = config_manager.load(yaml_path)
        assert isinstance(config, dict)
        assert "model" in config
        assert config["model"]["type"] == "yolo26"

    def test_loaded_config_passes_experiment_schema_validation(
        self, config_manager, valid_config
    ):
        """Test that the loaded config passes EXPERIMENT_SCHEMA validation.

        Validates: Requirement 6.1
        """
        # Should not raise
        config_manager.validate(valid_config, EXPERIMENT_SCHEMA)

    def test_loaded_model_config_passes_yolo26_schema_validation(
        self, config_manager, valid_config
    ):
        """Test that model.config section passes YOLO26-specific schema validation.

        Validates: Requirement 6.1
        """
        model_config = valid_config["model"]["config"]
        # Should not raise
        config_manager.validate(model_config, YOLO26_MODEL_CONFIG_SCHEMA)

    def test_loaded_config_has_all_required_model_config_fields(self, valid_config):
        """Test that the loaded YAML has all required fields under model.config.

        Validates: Requirement 6.1
        """
        model_config = valid_config["model"]["config"]
        required_fields = [
            "model_size",
            "num_classes",
            "end2end",
            "confidence_threshold",
            "iou_threshold",
        ]
        for field in required_fields:
            assert field in model_config, f"Missing required field: {field}"


class TestInvalidModelSize:
    """Tests for invalid model_size raising ValidationError (Requirement 6.6)."""

    def test_invalid_model_size_raises_validation_error(self, config_manager, valid_config):
        """Test that an invalid model_size value raises ValidationError.

        Validates: Requirement 6.6
        """
        model_config = dict(valid_config["model"]["config"])
        model_config["model_size"] = "invalid_size"

        with pytest.raises(ValidationError) as exc_info:
            config_manager.validate(model_config, YOLO26_MODEL_CONFIG_SCHEMA)

        # Verify the error mentions the invalid value and allowed values
        violations = exc_info.value.schema_violations
        assert len(violations) > 0
        violation_text = " ".join(violations)
        assert "invalid_size" in violation_text
        assert "n" in violation_text
        assert "s" in violation_text
        assert "m" in violation_text
        assert "l" in violation_text
        assert "x" in violation_text

    def test_invalid_model_size_numeric_raises_validation_error(
        self, config_manager, valid_config
    ):
        """Test that a numeric model_size raises ValidationError for type mismatch.

        Validates: Requirement 6.6
        """
        model_config = dict(valid_config["model"]["config"])
        model_config["model_size"] = 123

        with pytest.raises(ValidationError) as exc_info:
            config_manager.validate(model_config, YOLO26_MODEL_CONFIG_SCHEMA)

        violations = exc_info.value.schema_violations
        assert len(violations) > 0
        # Should mention type error for model_size
        violation_text = " ".join(violations)
        assert "model_size" in violation_text


class TestMissingRequiredFields:
    """Tests for missing required fields raising ValidationError (Requirement 6.5)."""

    def test_missing_model_size_raises_validation_error(self, config_manager, valid_config):
        """Test that missing model_size raises ValidationError.

        Validates: Requirement 6.5
        """
        model_config = dict(valid_config["model"]["config"])
        del model_config["model_size"]

        with pytest.raises(ValidationError) as exc_info:
            config_manager.validate(model_config, YOLO26_MODEL_CONFIG_SCHEMA)

        violations = exc_info.value.schema_violations
        violation_text = " ".join(violations)
        assert "model_size" in violation_text

    def test_missing_num_classes_raises_validation_error(self, config_manager, valid_config):
        """Test that missing num_classes raises ValidationError.

        Validates: Requirement 6.5
        """
        model_config = dict(valid_config["model"]["config"])
        del model_config["num_classes"]

        with pytest.raises(ValidationError) as exc_info:
            config_manager.validate(model_config, YOLO26_MODEL_CONFIG_SCHEMA)

        violations = exc_info.value.schema_violations
        violation_text = " ".join(violations)
        assert "num_classes" in violation_text

    def test_missing_all_required_fields_lists_each(self, config_manager):
        """Test that missing all required fields raises ValidationError listing each one.

        Validates: Requirement 6.5
        """
        # Empty config - all required fields are missing
        model_config = {}

        with pytest.raises(ValidationError) as exc_info:
            config_manager.validate(model_config, YOLO26_MODEL_CONFIG_SCHEMA)

        violations = exc_info.value.schema_violations
        required_fields = [
            "model_size",
            "num_classes",
            "end2end",
            "confidence_threshold",
            "iou_threshold",
        ]

        # Each missing field should be mentioned in the violations
        violation_text = " ".join(violations)
        for field in required_fields:
            assert field in violation_text, (
                f"Expected '{field}' to be listed in violations, got: {violations}"
            )

    def test_missing_subset_of_fields_lists_only_missing(self, config_manager, valid_config):
        """Test that only the actually missing fields are reported.

        Validates: Requirement 6.5
        """
        model_config = dict(valid_config["model"]["config"])
        # Remove only end2end and iou_threshold
        del model_config["end2end"]
        del model_config["iou_threshold"]

        with pytest.raises(ValidationError) as exc_info:
            config_manager.validate(model_config, YOLO26_MODEL_CONFIG_SCHEMA)

        violations = exc_info.value.schema_violations
        violation_text = " ".join(violations)
        assert "end2end" in violation_text
        assert "iou_threshold" in violation_text
