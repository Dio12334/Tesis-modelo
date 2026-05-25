"""Property-based tests for ModelRegistry.

# Feature: road-damage-evaluation-framework, Property 8: Model registry register and create
# Feature: road-damage-evaluation-framework, Property 9: Missing configuration parameters are reported
"""

from pathlib import Path
from typing import List

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from model.exceptions import ConfigurationError, ModelNotFoundError
from model.models.registry import BaseDetector, ModelRegistry


# --- Strategies ---

# Strategy for valid model identifiers (non-empty strings with printable chars)
model_identifiers = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
    min_size=1,
    max_size=30,
)

# Strategy for parameter names (valid Python-style identifiers)
param_names = st.text(
    alphabet=st.characters(whitelist_categories=("L",), whitelist_characters="_"),
    min_size=1,
    max_size=20,
).filter(lambda s: s[0].isalpha() or s[0] == "_")

# Strategy for sets of required parameter names (at least 1)
required_param_sets = st.lists(param_names, min_size=1, max_size=5, unique=True)

# Strategy for config values
config_values = st.one_of(
    st.text(min_size=1, max_size=20),
    st.integers(min_value=0, max_value=1000),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.booleans(),
)


# --- Helper: Dynamic concrete detector factory ---


def make_detector_class(required_params: List[str]):
    """Create a concrete BaseDetector subclass with the given required params."""

    class DynamicDetector(BaseDetector):
        def __init__(self, config: dict):
            self.config = config

        def forward(self, images):
            return [{"boxes": [], "labels": [], "scores": []}]

        def get_config_schema(self) -> dict:
            return {p: {"type": "str", "required": True} for p in required_params}

        def load_checkpoint(self, path: Path) -> None:
            pass

        def save_checkpoint(self, path: Path) -> None:
            pass

    return DynamicDetector


# --- Fixtures ---


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the model registry before each test to avoid state leakage."""
    ModelRegistry.reset()
    yield
    ModelRegistry.reset()


# --- Property 8: Model registry register and create ---


class TestProperty8RegisterAndCreate:
    """Property 8: Model registry register and create.

    For any model class registered with a unique string identifier and valid
    configuration, ModelRegistry.create(identifier, config) SHALL return an
    instance of the registered class. For any unregistered identifier, it SHALL
    raise ModelNotFoundError listing all available model identifiers.

    **Validates: Requirements 3.1, 3.2, 3.5**
    """

    @settings(max_examples=100)
    @given(
        identifier=model_identifiers,
        required_params=required_param_sets,
    )
    def test_create_returns_instance_of_registered_class(
        self, identifier, required_params
    ):
        """Registered model with valid config produces correct instance.

        # Feature: road-damage-evaluation-framework, Property 8: Model registry register and create
        """
        ModelRegistry.reset()

        detector_cls = make_detector_class(required_params)
        ModelRegistry.register(identifier)(detector_cls)

        # Build a valid config with all required params present
        config = {p: "test_value" for p in required_params}

        model = ModelRegistry.create(identifier, config)

        assert isinstance(model, detector_cls)
        assert isinstance(model, BaseDetector)
        assert model.config == config

    @settings(max_examples=100)
    @given(
        registered_ids=st.lists(model_identifiers, min_size=1, max_size=5, unique=True),
        unregistered_id=model_identifiers,
    )
    def test_unregistered_identifier_raises_model_not_found_error(
        self, registered_ids, unregistered_id
    ):
        """Unregistered identifier raises ModelNotFoundError listing available models.

        # Feature: road-damage-evaluation-framework, Property 8: Model registry register and create
        """
        # Ensure unregistered_id is truly not in registered_ids
        if unregistered_id in registered_ids:
            return  # Skip this example

        ModelRegistry.reset()

        detector_cls = make_detector_class(["dummy_param"])

        for reg_id in registered_ids:
            ModelRegistry.register(reg_id)(detector_cls)

        with pytest.raises(ModelNotFoundError) as exc_info:
            ModelRegistry.create(unregistered_id, {})

        # The error should list all available model identifiers
        assert exc_info.value.model_name == unregistered_id
        assert sorted(exc_info.value.available_models) == sorted(registered_ids)


# --- Property 9: Missing configuration parameters are reported ---


class TestProperty9MissingConfigParams:
    """Property 9: Missing configuration parameters are reported.

    For any model configuration that is missing one or more required parameters,
    instantiation SHALL raise a ConfigurationError whose message lists all
    missing parameter names.

    **Validates: Requirements 3.6, 3.7**
    """

    @settings(max_examples=100)
    @given(
        identifier=model_identifiers,
        required_params=required_param_sets,
        data=st.data(),
    )
    def test_missing_params_raises_configuration_error(
        self, identifier, required_params, data
    ):
        """Missing required params raises ConfigurationError listing them all.

        # Feature: road-damage-evaluation-framework, Property 9: Missing configuration parameters are reported
        """
        ModelRegistry.reset()

        detector_cls = make_detector_class(required_params)
        ModelRegistry.register(identifier)(detector_cls)

        # Choose a non-empty subset of params to omit
        num_to_provide = data.draw(
            st.integers(min_value=0, max_value=len(required_params) - 1)
        )
        provided_params = data.draw(
            st.sampled_from(
                sorted(
                    [
                        combo
                        for combo in _combinations(required_params, num_to_provide)
                    ]
                )
            )
            if num_to_provide > 0
            else st.just([])
        )

        config = {p: "value" for p in provided_params}
        missing_params = set(required_params) - set(provided_params)

        # There must be at least one missing param
        assert len(missing_params) > 0

        with pytest.raises(ConfigurationError) as exc_info:
            ModelRegistry.create(identifier, config)

        # All missing parameter names should be mentioned in the violations
        error_message = str(exc_info.value)
        for param in missing_params:
            assert param in error_message, (
                f"Missing param '{param}' not found in error message: {error_message}"
            )

    @settings(max_examples=100)
    @given(
        identifier=model_identifiers,
        required_params=required_param_sets,
    )
    def test_all_params_missing_reports_all(self, identifier, required_params):
        """When all required params are missing, all are reported in the error.

        # Feature: road-damage-evaluation-framework, Property 9: Missing configuration parameters are reported
        """
        ModelRegistry.reset()

        detector_cls = make_detector_class(required_params)
        ModelRegistry.register(identifier)(detector_cls)

        with pytest.raises(ConfigurationError) as exc_info:
            ModelRegistry.create(identifier, {})

        error_message = str(exc_info.value)
        for param in required_params:
            assert param in error_message, (
                f"Missing param '{param}' not found in error message: {error_message}"
            )


def _combinations(items, r):
    """Generate all combinations of r items from the list."""
    from itertools import combinations

    return [list(c) for c in combinations(items, r)]
