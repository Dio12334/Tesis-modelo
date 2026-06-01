"""Unit tests for RT-DETR YAML configuration loading and validation.

Tests cover:
- File existence at model/configs/train_rt_detr.yaml
- Correct model type ("rt_detr") and model_size ("l")
- num_classes is 5 (RDD2022 dataset)
- batch_size is 4 (RTX 2070 8GB VRAM constraint)
- input_size is 640
- All required sections present (model, dataset, training, evaluation, output)
- Loadable by ConfigManager without errors

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7
"""

from pathlib import Path

import pytest

from model.config import ConfigManager, EXPERIMENT_SCHEMA


@pytest.fixture
def config_manager():
    """Create a ConfigManager instance."""
    return ConfigManager()


@pytest.fixture
def yaml_path():
    """Path to the train_rt_detr.yaml config file."""
    return Path(__file__).resolve().parents[2] / "configs" / "train_rt_detr.yaml"


@pytest.fixture
def valid_config(config_manager, yaml_path):
    """Load the train_rt_detr.yaml config as a dict."""
    return config_manager.load(yaml_path)


class TestFileExistence:
    """Tests for YAML config file existence (Requirement 10.1)."""

    def test_yaml_config_file_exists(self, yaml_path):
        """Test that train_rt_detr.yaml exists at the expected path.

        Validates: Requirement 10.1
        """
        assert yaml_path.exists(), f"Config file not found at {yaml_path}"

    def test_yaml_config_file_is_not_empty(self, yaml_path):
        """Test that train_rt_detr.yaml is not an empty file.

        Validates: Requirement 10.1
        """
        assert yaml_path.stat().st_size > 0, "Config file is empty"


class TestModelTypeAndSize:
    """Tests for correct model type and model_size (Requirement 10.2)."""

    def test_model_type_is_rt_detr(self, valid_config):
        """Test that model.type is 'rt_detr'.

        Validates: Requirement 10.2
        """
        assert valid_config["model"]["type"] == "rt_detr"

    def test_model_size_is_l(self, valid_config):
        """Test that model.config.model_size is 'l'.

        Validates: Requirement 10.2
        """
        assert valid_config["model"]["config"]["model_size"] == "l"


class TestNumClasses:
    """Tests for num_classes matching RDD2022 (Requirement 10.3)."""

    def test_num_classes_is_5(self, valid_config):
        """Test that model.config.num_classes is 5 for RDD2022 dataset.

        Validates: Requirement 10.3
        """
        assert valid_config["model"]["config"]["num_classes"] == 5


class TestBatchSize:
    """Tests for batch_size constrained by VRAM (Requirement 10.4)."""

    def test_batch_size_is_4(self, valid_config):
        """Test that training.batch_size is 4 for RTX 2070 8GB VRAM.

        Validates: Requirement 10.4
        """
        assert valid_config["training"]["batch_size"] == 4


class TestInputSize:
    """Tests for input_size matching RT-DETR default (Requirement 10.5)."""

    def test_input_size_is_640(self, valid_config):
        """Test that model.config.input_size is 640.

        Validates: Requirement 10.5
        """
        assert valid_config["model"]["config"]["input_size"] == 640


class TestRequiredSections:
    """Tests for all required sections present (Requirement 10.6)."""

    def test_model_section_present(self, valid_config):
        """Test that 'model' section is present in config.

        Validates: Requirement 10.6
        """
        assert "model" in valid_config

    def test_dataset_section_present(self, valid_config):
        """Test that 'dataset' section is present in config.

        Validates: Requirement 10.6
        """
        assert "dataset" in valid_config

    def test_training_section_present(self, valid_config):
        """Test that 'training' section is present in config.

        Validates: Requirement 10.6
        """
        assert "training" in valid_config

    def test_evaluation_section_present(self, valid_config):
        """Test that 'evaluation' section is present in config.

        Validates: Requirement 10.6
        """
        assert "evaluation" in valid_config

    def test_output_section_present(self, valid_config):
        """Test that 'output' section is present in config.

        Validates: Requirement 10.6
        """
        assert "output" in valid_config

    def test_all_required_sections_present(self, valid_config):
        """Test that all required top-level sections are present.

        Validates: Requirement 10.6
        """
        required_sections = ["model", "dataset", "training", "evaluation", "output"]
        for section in required_sections:
            assert section in valid_config, f"Missing required section: {section}"


class TestLoadableByConfigManager:
    """Tests for config loadable by ConfigManager (Requirement 10.7)."""

    def test_config_manager_loads_without_errors(self, config_manager, yaml_path):
        """Test that ConfigManager loads train_rt_detr.yaml without raising.

        Validates: Requirement 10.7
        """
        config = config_manager.load(yaml_path)
        assert isinstance(config, dict)

    def test_loaded_config_passes_experiment_schema_validation(
        self, config_manager, valid_config
    ):
        """Test that the loaded config passes EXPERIMENT_SCHEMA validation.

        Validates: Requirement 10.7
        """
        # Should not raise
        config_manager.validate(valid_config, EXPERIMENT_SCHEMA)

    def test_loaded_config_has_experiment_name(self, valid_config):
        """Test that the loaded config has a name field.

        Validates: Requirement 10.7
        """
        assert "name" in valid_config
        assert isinstance(valid_config["name"], str)
        assert len(valid_config["name"]) > 0
