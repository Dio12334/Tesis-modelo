"""Unit tests for MobileNetV4 SSD YAML configuration loading and validation.

Tests cover:
- Loading train_mobilenetv4.yaml without errors via ConfigManager
- All 6 top-level sections present (name, model, dataset, training, evaluation, output)
- model.type and config values (mobilenetv4_ssd, input_size=640, num_classes=5)
- Training hyperparameter ranges (batch_size, scheduler, warmup_epochs, learning_rate, optimizer)
- Dataset paths (dataset.path, dataset.class_mapping)
- Passes EXPERIMENT_SCHEMA validation

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
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
    """Path to the train_mobilenetv4.yaml config file."""
    return Path(__file__).resolve().parents[2] / "configs" / "train_mobilenetv4.yaml"


@pytest.fixture
def valid_config(config_manager, yaml_path):
    """Load the train_mobilenetv4.yaml config as a dict."""
    return config_manager.load(yaml_path)


class TestYAMLConfigLoading:
    """Tests for loading train_mobilenetv4.yaml without errors (Requirement 8.5)."""

    def test_yaml_file_exists(self, yaml_path):
        """Test that train_mobilenetv4.yaml exists at the expected path.

        Validates: Requirement 8.5
        """
        assert yaml_path.exists(), f"Config file not found at {yaml_path}"

    def test_config_manager_loads_without_errors(self, config_manager, yaml_path):
        """Test that ConfigManager loads train_mobilenetv4.yaml without raising.

        Validates: Requirement 8.5
        """
        config = config_manager.load(yaml_path)
        assert isinstance(config, dict)

    def test_loaded_config_passes_experiment_schema_validation(
        self, config_manager, valid_config
    ):
        """Test that the loaded config passes EXPERIMENT_SCHEMA validation.

        Validates: Requirement 8.5
        """
        config_manager.validate(valid_config, EXPERIMENT_SCHEMA)


class TestRequiredSections:
    """Tests for all 6 top-level sections present (Requirement 8.1)."""

    def test_all_six_sections_present(self, valid_config):
        """Test that all 6 required top-level sections are present.

        Validates: Requirement 8.1
        """
        required_sections = ["name", "model", "dataset", "training", "evaluation", "output"]
        for section in required_sections:
            assert section in valid_config, f"Missing required section: {section}"

    def test_name_section_is_string(self, valid_config):
        """Test that the name section is a non-empty string.

        Validates: Requirement 8.1
        """
        assert isinstance(valid_config["name"], str)
        assert len(valid_config["name"]) > 0

    def test_model_section_is_dict(self, valid_config):
        """Test that the model section is a dictionary.

        Validates: Requirement 8.1
        """
        assert isinstance(valid_config["model"], dict)

    def test_dataset_section_is_dict(self, valid_config):
        """Test that the dataset section is a dictionary.

        Validates: Requirement 8.1
        """
        assert isinstance(valid_config["dataset"], dict)

    def test_training_section_is_dict(self, valid_config):
        """Test that the training section is a dictionary.

        Validates: Requirement 8.1
        """
        assert isinstance(valid_config["training"], dict)

    def test_evaluation_section_is_dict(self, valid_config):
        """Test that the evaluation section is a dictionary.

        Validates: Requirement 8.1
        """
        assert isinstance(valid_config["evaluation"], dict)

    def test_output_section_is_dict(self, valid_config):
        """Test that the output section is a dictionary.

        Validates: Requirement 8.1
        """
        assert isinstance(valid_config["output"], dict)


class TestModelTypeAndConfig:
    """Tests for model.type and config values (Requirement 8.2)."""

    def test_model_type_is_mobilenetv4_ssd(self, valid_config):
        """Test that model.type is 'mobilenetv4_ssd'.

        Validates: Requirement 8.2
        """
        assert valid_config["model"]["type"] == "mobilenetv4_ssd"

    def test_model_config_input_size_is_640(self, valid_config):
        """Test that model.config.input_size is 640.

        Validates: Requirement 8.2
        """
        assert valid_config["model"]["config"]["input_size"] == 640

    def test_model_config_num_classes_is_5(self, valid_config):
        """Test that model.config.num_classes is 5.

        Validates: Requirement 8.2
        """
        assert valid_config["model"]["config"]["num_classes"] == 5


class TestTrainingHyperparameters:
    """Tests for training hyperparameter ranges (Requirement 8.3)."""

    def test_batch_size_in_valid_range(self, valid_config):
        """Test that training.batch_size is between 16 and 64 inclusive.

        Validates: Requirement 8.3
        """
        batch_size = valid_config["training"]["batch_size"]
        assert 16 <= batch_size <= 64, (
            f"batch_size={batch_size} not in range [16, 64]"
        )

    def test_scheduler_is_cosine(self, valid_config):
        """Test that training.scheduler is 'cosine'.

        Validates: Requirement 8.3
        """
        assert valid_config["training"]["scheduler"] == "cosine"

    def test_warmup_epochs_in_valid_range(self, valid_config):
        """Test that training.warmup_epochs is between 3 and 5 inclusive.

        Validates: Requirement 8.3
        """
        warmup_epochs = valid_config["training"]["warmup_epochs"]
        assert 3 <= warmup_epochs <= 5, (
            f"warmup_epochs={warmup_epochs} not in range [3, 5]"
        )

    def test_learning_rate_in_valid_range(self, valid_config):
        """Test that training.learning_rate is between 0.001 and 0.01 inclusive.

        Validates: Requirement 8.3
        """
        lr = valid_config["training"]["learning_rate"]
        assert 0.001 <= lr <= 0.01, (
            f"learning_rate={lr} not in range [0.001, 0.01]"
        )

    def test_optimizer_is_allowed_value(self, valid_config):
        """Test that training.optimizer is one of SGD, Adam, or AdamW.

        Validates: Requirement 8.3
        """
        optimizer = valid_config["training"]["optimizer"]
        assert optimizer in ["SGD", "Adam", "AdamW"], (
            f"optimizer='{optimizer}' not in allowed values ['SGD', 'Adam', 'AdamW']"
        )


class TestDatasetPaths:
    """Tests for dataset paths (Requirement 8.4)."""

    def test_dataset_path(self, valid_config):
        """Test that dataset.path points to the RDD2022 data directory.

        Validates: Requirement 8.4
        """
        assert valid_config["dataset"]["path"] == "model/data/rdd2022/complete"

    def test_dataset_class_mapping(self, valid_config):
        """Test that dataset.class_mapping points to rdd2022_classes.yaml.

        Validates: Requirement 8.4
        """
        assert valid_config["dataset"]["class_mapping"] == "model/configs/rdd2022_classes.yaml"
