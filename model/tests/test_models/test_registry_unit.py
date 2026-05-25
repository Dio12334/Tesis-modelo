"""Unit tests for ModelRegistry and BaseDetector."""

from pathlib import Path

import pytest

from model.exceptions import ConfigurationError, ModelNotFoundError
from model.models.registry import BaseDetector, ModelRegistry


class DummyDetector(BaseDetector):
    """A concrete detector for testing purposes."""

    def __init__(self, config: dict):
        self.config = config

    def forward(self, images):
        return [{"boxes": [], "labels": [], "scores": []}]

    def get_config_schema(self) -> dict:
        return {
            "backbone_size": {"type": "str", "required": True},
            "num_classes": {"type": "int", "required": True},
        }

    def load_checkpoint(self, path: Path) -> None:
        pass

    def save_checkpoint(self, path: Path) -> None:
        pass


class NoRequiredParamsDetector(BaseDetector):
    """A detector with no required config params."""

    def __init__(self, config: dict):
        self.config = config

    def forward(self, images):
        return []

    def get_config_schema(self) -> dict:
        return {
            "optional_param": {"type": "str", "required": False},
        }

    def load_checkpoint(self, path: Path) -> None:
        pass

    def save_checkpoint(self, path: Path) -> None:
        pass


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the model registry before each test."""
    ModelRegistry.reset()
    yield
    ModelRegistry.reset()


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_register_and_create(self):
        """Test registering a model and creating an instance."""
        ModelRegistry.register("dummy")(DummyDetector)

        config = {"backbone_size": "nano", "num_classes": 4}
        model = ModelRegistry.create("dummy", config)

        assert isinstance(model, DummyDetector)
        assert model.config == config

    def test_register_as_decorator(self):
        """Test using register as a decorator."""

        @ModelRegistry.register("decorated_model")
        class DecoratedDetector(DummyDetector):
            pass

        config = {"backbone_size": "small", "num_classes": 10}
        model = ModelRegistry.create("decorated_model", config)
        assert isinstance(model, DecoratedDetector)

    def test_create_unknown_model_raises_model_not_found_error(self):
        """Test that creating an unknown model raises ModelNotFoundError."""
        ModelRegistry.register("known_model")(DummyDetector)

        with pytest.raises(ModelNotFoundError) as exc_info:
            ModelRegistry.create("unknown_model", {})

        assert exc_info.value.model_name == "unknown_model"
        assert "known_model" in exc_info.value.available_models

    def test_create_missing_config_raises_configuration_error(self):
        """Test that missing required config params raises ConfigurationError."""
        ModelRegistry.register("dummy")(DummyDetector)

        with pytest.raises(ConfigurationError) as exc_info:
            ModelRegistry.create("dummy", {})

        violations = exc_info.value.violations
        assert any("backbone_size" in v for v in violations)
        assert any("num_classes" in v for v in violations)

    def test_create_partial_config_raises_configuration_error(self):
        """Test that partially missing config raises ConfigurationError."""
        ModelRegistry.register("dummy")(DummyDetector)

        with pytest.raises(ConfigurationError) as exc_info:
            ModelRegistry.create("dummy", {"backbone_size": "nano"})

        violations = exc_info.value.violations
        assert any("num_classes" in v for v in violations)
        assert not any("backbone_size" in v for v in violations)

    def test_list_models_empty(self):
        """Test list_models returns empty list when no models registered."""
        assert ModelRegistry.list_models() == []

    def test_list_models_sorted(self):
        """Test list_models returns sorted list of model names."""
        ModelRegistry.register("zebra")(DummyDetector)
        ModelRegistry.register("alpha")(DummyDetector)
        ModelRegistry.register("middle")(DummyDetector)

        assert ModelRegistry.list_models() == ["alpha", "middle", "zebra"]

    def test_create_with_no_required_params(self):
        """Test creating a model with no required params succeeds with empty config."""
        ModelRegistry.register("no_req")(NoRequiredParamsDetector)

        model = ModelRegistry.create("no_req", {})
        assert isinstance(model, NoRequiredParamsDetector)

    def test_model_not_found_error_lists_available(self):
        """Test ModelNotFoundError includes all available models."""
        ModelRegistry.register("model_a")(DummyDetector)
        ModelRegistry.register("model_b")(DummyDetector)

        with pytest.raises(ModelNotFoundError) as exc_info:
            ModelRegistry.create("nonexistent", {})

        assert sorted(exc_info.value.available_models) == ["model_a", "model_b"]

    def test_register_preserves_class(self):
        """Test that register decorator returns the original class."""
        result = ModelRegistry.register("test_model")(DummyDetector)
        assert result is DummyDetector


class TestBaseDetector:
    """Tests for BaseDetector ABC."""

    def test_cannot_instantiate_abstract(self):
        """Test that BaseDetector cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseDetector()

    def test_must_implement_all_methods(self):
        """Test that subclass must implement all abstract methods."""

        class IncompleteDetector(BaseDetector):
            def forward(self, images):
                return []

        with pytest.raises(TypeError):
            IncompleteDetector()

    def test_concrete_subclass_works(self):
        """Test that a fully implemented subclass can be instantiated."""
        detector = DummyDetector({"backbone_size": "nano", "num_classes": 4})
        assert detector.config["backbone_size"] == "nano"
