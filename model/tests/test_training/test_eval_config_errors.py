"""Unit tests for configuration-file error cases in the evaluation script.

Feature: generic-evaluation-script, Task 2.10

These example-based tests exercise the error and substitution behaviour of
``load_and_merge_config`` in ``model/training/evaluate_detection.py``:

- A missing configuration path raises ``ConfigurationError`` whose message
  carries the offending path. (Requirement 3.5)
- A malformed YAML file raises ``ConfigurationError`` whose message carries the
  offending path and the underlying parser error text. (Requirement 3.5)
- ``${ENV}`` references are substituted with environment-variable values during
  loading, i.e. before any validation runs. (Requirement 3.4)
"""

import pytest

from model.exceptions import ConfigurationError
from model.training.evaluate_detection import load_and_merge_config


class TestMissingConfigPath:
    """A configuration path that does not exist must surface a ConfigurationError."""

    def test_missing_path_raises_configuration_error(self, tmp_path):
        """Loading a non-existent config file raises ConfigurationError.

        **Validates: Requirements 3.5**
        """
        missing = tmp_path / "does_not_exist.yaml"

        with pytest.raises(ConfigurationError):
            load_and_merge_config(str(missing), {})

    def test_missing_path_message_includes_path(self, tmp_path):
        """The ConfigurationError message identifies the offending path.

        **Validates: Requirements 3.5**
        """
        missing = tmp_path / "does_not_exist.yaml"

        with pytest.raises(ConfigurationError) as exc_info:
            load_and_merge_config(str(missing), {})

        message = str(exc_info.value)
        # The offending path (at least its filename) must appear in the message.
        assert "does_not_exist.yaml" in message
        # The violation should explain that the file could not be found.
        assert "not found" in message.lower()


class TestMalformedYaml:
    """A config file that cannot be parsed as YAML must surface a ConfigurationError."""

    def test_malformed_yaml_raises_configuration_error(self, tmp_path):
        """Loading an unparseable YAML file raises ConfigurationError.

        **Validates: Requirements 3.5**
        """
        bad = tmp_path / "malformed.yaml"
        # Unclosed flow sequence -> yaml.safe_load raises a YAMLError.
        bad.write_text("model:\n  type: [1, 2\n", encoding="utf-8")

        with pytest.raises(ConfigurationError):
            load_and_merge_config(str(bad), {})

    def test_malformed_yaml_message_includes_path_and_parser_text(self, tmp_path):
        """The error message carries the offending path and the parser error text.

        **Validates: Requirements 3.5**
        """
        bad = tmp_path / "malformed.yaml"
        bad.write_text("model:\n  type: [1, 2\n", encoding="utf-8")

        with pytest.raises(ConfigurationError) as exc_info:
            load_and_merge_config(str(bad), {})

        message = str(exc_info.value)
        # Offending path (filename) is present.
        assert "malformed.yaml" in message
        # Underlying parser error text is surfaced (yaml errors mention "YAML").
        assert "yaml" in message.lower()

    def test_malformed_yaml_preserves_parser_violation(self, tmp_path):
        """The translated ConfigurationError preserves the parser violation list.

        **Validates: Requirements 3.5**
        """
        bad = tmp_path / "malformed.yaml"
        bad.write_text("a: b: c\n", encoding="utf-8")

        with pytest.raises(ConfigurationError) as exc_info:
            load_and_merge_config(str(bad), {})

        # ConfigurationError exposes the structured violation list.
        violations = exc_info.value.violations
        assert isinstance(violations, list)
        assert len(violations) >= 1
        assert any("malformed.yaml" in v for v in violations)


class TestEnvVarSubstitution:
    """``${ENV}`` references are resolved during loading, before validation."""

    def test_env_var_substituted_before_validation(self, tmp_path, monkeypatch):
        """A ${ENV} reference is replaced with the environment value on load.

        **Validates: Requirements 3.4**
        """
        monkeypatch.setenv("EVAL_TEST_DATASET", "/data/rdd2022")
        monkeypatch.setenv("EVAL_TEST_MODEL", "yolo26")

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "model:\n"
            "  type: ${EVAL_TEST_MODEL}\n"
            "dataset:\n"
            "  path: ${EVAL_TEST_DATASET}/images\n",
            encoding="utf-8",
        )

        merged = load_and_merge_config(str(config_file), {})

        # Substitution happened: no ${...} pattern survives in the loaded values.
        assert merged["model"]["type"] == "yolo26"
        assert merged["dataset"]["path"] == "/data/rdd2022/images"
        assert "${" not in merged["dataset"]["path"]

    def test_env_var_substitution_precedes_override_merge(self, tmp_path, monkeypatch):
        """Resolved env values are present in the merged config alongside overrides.

        Substitution occurs on the loaded base config before overrides are
        deep-merged on top, so a resolved base value coexists with override
        values that win where they overlap.

        **Validates: Requirements 3.4**
        """
        monkeypatch.setenv("EVAL_TEST_SPLIT", "test")

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "evaluation:\n"
            "  split: ${EVAL_TEST_SPLIT}\n"
            "  confidence_threshold: 0.25\n",
            encoding="utf-8",
        )

        overrides = {"evaluation": {"confidence_threshold": 0.5}}
        merged = load_and_merge_config(str(config_file), overrides)

        # Resolved env value from the base config is preserved.
        assert merged["evaluation"]["split"] == "test"
        # Override value wins where it overlaps.
        assert merged["evaluation"]["confidence_threshold"] == 0.5
