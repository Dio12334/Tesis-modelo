"""Property-based tests for TargetMapper.

Tests Properties 5, 6, and 7 from the design document using Hypothesis.
"""

import tempfile
from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from model.datasets.target_mapper import TargetMapper
from model.exceptions import ConfigurationError


# --- Hypothesis Strategies ---

# Strategy for generating valid class names (non-empty alphabetic strings)
class_name_strategy = st.text(
    min_size=1,
    max_size=15,
    alphabet=st.characters(whitelist_categories=("L",)),
)


@st.composite
def valid_mapping_config(draw):
    """Generate a valid mapping configuration with taxonomy and mappings.

    Ensures all target classes in mappings are present in the taxonomy.
    Supports many-to-one mappings (multiple sources -> same target).
    """
    # Generate taxonomy (at least 1 target class)
    taxonomy = draw(
        st.lists(class_name_strategy, min_size=1, max_size=8, unique=True)
    )

    # Generate source classes (distinct from each other)
    source_classes = draw(
        st.lists(class_name_strategy, min_size=1, max_size=10, unique=True)
    )

    # Map each source to a random target from the taxonomy (many-to-one allowed)
    mappings = {}
    for source in source_classes:
        target = draw(st.sampled_from(taxonomy))
        mappings[source] = target

    return {
        "taxonomy": taxonomy,
        "mappings": mappings,
        "default_class": None,
    }


@st.composite
def bijective_mapping_config(draw):
    """Generate a bijective (one-to-one) mapping configuration.

    Each source maps to a unique target, and each target has exactly one source.
    """
    # Generate N unique class names for both source and target
    n = draw(st.integers(min_value=1, max_value=8))

    all_names = draw(
        st.lists(class_name_strategy, min_size=n * 2, max_size=n * 2, unique=True)
    )

    source_classes = all_names[:n]
    target_classes = all_names[n:]

    # Create one-to-one mapping
    mappings = dict(zip(source_classes, target_classes))

    return {
        "taxonomy": target_classes,
        "mappings": mappings,
        "default_class": None,
    }


@st.composite
def invalid_target_mapping_config(draw):
    """Generate a mapping config where some target classes are NOT in the taxonomy.

    Ensures at least one mapped target is invalid (not in the provided taxonomy).
    """
    # Generate a taxonomy
    taxonomy = draw(
        st.lists(class_name_strategy, min_size=1, max_size=5, unique=True)
    )

    # Generate invalid target classes (not in taxonomy)
    invalid_targets = draw(
        st.lists(class_name_strategy, min_size=1, max_size=4, unique=True).filter(
            lambda targets: not any(t in taxonomy for t in targets)
        )
    )

    # Generate source classes
    num_sources = len(invalid_targets) + draw(st.integers(min_value=0, max_value=3))
    source_classes = draw(
        st.lists(
            class_name_strategy, min_size=num_sources, max_size=num_sources, unique=True
        )
    )

    # Build mappings: some map to valid targets, some to invalid
    mappings = {}
    # First, assign invalid targets
    for i, invalid_target in enumerate(invalid_targets):
        if i < len(source_classes):
            mappings[source_classes[i]] = invalid_target

    # Remaining sources map to valid taxonomy entries
    for i in range(len(invalid_targets), len(source_classes)):
        mappings[source_classes[i]] = draw(st.sampled_from(taxonomy))

    assume(len(mappings) > 0)

    return {
        "taxonomy": taxonomy,
        "mappings": mappings,
        "default_class": None,
        "_invalid_targets": invalid_targets,  # metadata for test assertions
    }


def write_config_to_tempfile(config: dict) -> Path:
    """Write a mapping config dict to a temporary YAML file and return the path.

    The caller is responsible for cleanup, but since these are tests,
    the OS will clean up temp files.
    """
    # Remove any metadata keys starting with _
    clean_config = {k: v for k, v in config.items() if not k.startswith("_")}
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="target_mapper_test_"
    )
    yaml.dump(clean_config, tmp)
    tmp.close()
    return Path(tmp.name)


# --- Property 5: Target class mapping correctness ---
# Feature: road-damage-evaluation-framework, Property 5: Target class mapping correctness


class TestProperty5TargetClassMappingCorrectness:
    """Property 5: Target class mapping correctness.

    For any valid mapping configuration (including many-to-one mappings) and any
    source class present in the configuration, map_class(source) SHALL return the
    configured target class. When multiple source classes map to the same target,
    all SHALL return that same target.

    Validates: Requirements 2.2, 2.4
    """

    @settings(max_examples=100)
    @given(config=valid_mapping_config())
    def test_map_class_returns_configured_target(self, config):
        """**Validates: Requirements 2.2, 2.4**"""
        # Feature: road-damage-evaluation-framework, Property 5: Target class mapping correctness
        config_path = write_config_to_tempfile(config)
        mapper = TargetMapper(config_path, strict=True)

        # For every source class in the mapping, map_class should return the configured target
        for source, expected_target in config["mappings"].items():
            result = mapper.map_class(source)
            assert result == expected_target, (
                f"map_class('{source}') returned '{result}', expected '{expected_target}'"
            )

    @settings(max_examples=100)
    @given(config=valid_mapping_config())
    def test_many_to_one_all_return_same_target(self, config):
        """**Validates: Requirements 2.2, 2.4**"""
        # Feature: road-damage-evaluation-framework, Property 5: Target class mapping correctness
        config_path = write_config_to_tempfile(config)
        mapper = TargetMapper(config_path, strict=True)

        # Group sources by their target
        target_to_sources: dict = {}
        for source, target in config["mappings"].items():
            target_to_sources.setdefault(target, []).append(source)

        # For many-to-one mappings, all sources mapping to the same target
        # should return that same target
        for target, sources in target_to_sources.items():
            for source in sources:
                result = mapper.map_class(source)
                assert result == target, (
                    f"Many-to-one: map_class('{source}') returned '{result}', "
                    f"expected '{target}' (shared with {sources})"
                )


# --- Property 6: Mapping validation detects invalid target classes ---
# Feature: road-damage-evaluation-framework, Property 6: Mapping validation detects invalid target classes


class TestProperty6MappingValidationDetectsInvalidTargetClasses:
    """Property 6: Mapping validation detects invalid target classes.

    For any mapping configuration that references target classes not present in
    the target taxonomy, validation SHALL raise a ConfigurationError whose message
    lists all invalid target classes.

    Validates: Requirements 2.5, 2.6
    """

    @settings(max_examples=100)
    @given(config=invalid_target_mapping_config())
    def test_validation_raises_for_invalid_targets(self, config):
        """**Validates: Requirements 2.5, 2.6**"""
        # Feature: road-damage-evaluation-framework, Property 6: Mapping validation detects invalid target classes
        invalid_targets = config["_invalid_targets"]
        config_path = write_config_to_tempfile(config)
        mapper = TargetMapper(config_path, strict=True)

        # Provide the config's own taxonomy as the valid target taxonomy
        valid_taxonomy = config["taxonomy"]

        with pytest.raises(ConfigurationError) as exc_info:
            mapper.validate(valid_taxonomy)

        # The error message should mention all invalid target classes
        error_message = str(exc_info.value)
        for invalid_target in invalid_targets:
            # Only check targets that are actually used in mappings
            if invalid_target in config["mappings"].values():
                assert invalid_target in error_message, (
                    f"Invalid target class '{invalid_target}' not mentioned in "
                    f"ConfigurationError message: {error_message}"
                )

    @settings(max_examples=100)
    @given(config=valid_mapping_config())
    def test_validation_passes_for_valid_config(self, config):
        """**Validates: Requirements 2.5, 2.6**"""
        # Feature: road-damage-evaluation-framework, Property 6: Mapping validation detects invalid target classes
        config_path = write_config_to_tempfile(config)
        mapper = TargetMapper(config_path, strict=True)

        # Validation should NOT raise when all targets are in the taxonomy
        # The config's taxonomy already contains all valid targets
        mapper.validate(config["taxonomy"])


# --- Property 7: Bijective mapping round-trip ---
# Feature: road-damage-evaluation-framework, Property 7: Bijective mapping round-trip


class TestProperty7BijectiveMappingRoundTrip:
    """Property 7: Bijective mapping round-trip.

    For any bijective (one-to-one) mapping configuration and any source class in
    that mapping, mapping the source class to a target and then reverse-mapping
    that target SHALL produce a set containing the original source class.

    Validates: Requirements 2.7
    """

    @settings(max_examples=100)
    @given(config=bijective_mapping_config())
    def test_map_then_reverse_map_contains_original(self, config):
        """**Validates: Requirements 2.7**"""
        # Feature: road-damage-evaluation-framework, Property 7: Bijective mapping round-trip
        config_path = write_config_to_tempfile(config)
        mapper = TargetMapper(config_path, strict=True)

        for source in config["mappings"]:
            # Map source -> target
            target = mapper.map_class(source)

            # Reverse map target -> set of sources
            reverse_sources = mapper.reverse_map(target)

            # The original source must be in the reverse-mapped set
            assert source in reverse_sources, (
                f"Round-trip failed: map_class('{source}') = '{target}', "
                f"but reverse_map('{target}') = {reverse_sources} "
                f"does not contain '{source}'"
            )

    @settings(max_examples=100)
    @given(config=bijective_mapping_config())
    def test_bijective_reverse_map_returns_exactly_one(self, config):
        """**Validates: Requirements 2.7**"""
        # Feature: road-damage-evaluation-framework, Property 7: Bijective mapping round-trip
        config_path = write_config_to_tempfile(config)
        mapper = TargetMapper(config_path, strict=True)

        # For a bijective mapping, reverse_map should return exactly one source
        for source in config["mappings"]:
            target = mapper.map_class(source)
            reverse_sources = mapper.reverse_map(target)

            assert len(reverse_sources) == 1, (
                f"Bijective mapping: reverse_map('{target}') returned "
                f"{reverse_sources} (expected exactly 1 source)"
            )
            assert reverse_sources[0] == source, (
                f"Bijective mapping: reverse_map('{target}') = {reverse_sources}, "
                f"expected ['{source}']"
            )
