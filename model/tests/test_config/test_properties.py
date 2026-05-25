"""Property-based tests for ConfigManager.

Tests Properties 14, 15, and 16 from the design document using Hypothesis.
"""

import os
from pathlib import Path

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from model.config.manager import ConfigManager


# ---------------------------------------------------------------------------
# Hypothesis strategies for YAML-safe configuration values
# ---------------------------------------------------------------------------

# Leaf values that survive YAML round-trip without ambiguity.
# Avoid strings that YAML interprets as booleans/nulls (e.g., "true", "yes", "null").
_YAML_SAFE_STRINGS = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
        blacklist_characters="\x00{}$",
    ),
    min_size=1,
    max_size=30,
).filter(lambda s: s.strip().lower() not in (
    "true", "false", "yes", "no", "on", "off", "null", "~", "none",
))

_YAML_SAFE_SCALARS = st.one_of(
    _YAML_SAFE_STRINGS,
    st.integers(min_value=-10000, max_value=10000),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.booleans(),
)

# Keys must be simple strings (valid YAML mapping keys).
_CONFIG_KEYS = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), blacklist_characters="\x00"),
    min_size=1,
    max_size=15,
).filter(lambda s: s.strip().lower() not in (
    "true", "false", "yes", "no", "on", "off", "null", "~", "none",
) and s.strip() != "")


def _yaml_safe_values(max_depth: int = 2):
    """Strategy for YAML-safe values with bounded nesting depth."""
    if max_depth <= 0:
        return _YAML_SAFE_SCALARS
    return st.one_of(
        _YAML_SAFE_SCALARS,
        st.lists(_YAML_SAFE_SCALARS, min_size=0, max_size=5),
        st.dictionaries(
            keys=_CONFIG_KEYS,
            values=_yaml_safe_values(max_depth - 1),
            min_size=0,
            max_size=5,
        ),
    )


# Strategy for flat and nested config dicts (YAML-safe).
_CONFIG_DICTS = st.dictionaries(
    keys=_CONFIG_KEYS,
    values=_yaml_safe_values(max_depth=2),
    min_size=1,
    max_size=8,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def manager():
    return ConfigManager()


# ---------------------------------------------------------------------------
# Property 14: Configuration YAML round-trip
# Feature: road-damage-evaluation-framework, Property 14: Configuration YAML round-trip
# ---------------------------------------------------------------------------


class TestProperty14YAMLRoundTrip:
    """Property 14: For any valid configuration dictionary, saving to YAML and
    loading back SHALL produce an equivalent configuration dictionary.

    **Validates: Requirements 6.1, 6.7**
    """

    @given(config=_CONFIG_DICTS)
    @settings(max_examples=100)
    def test_yaml_round_trip(self, config, tmp_path_factory):
        # Feature: road-damage-evaluation-framework, Property 14: Configuration YAML round-trip
        """Saving a config to YAML and loading it back produces an equivalent dict."""
        manager = ConfigManager()
        tmp_path = tmp_path_factory.mktemp("yaml_rt")
        file_path = tmp_path / "config.yaml"

        manager.save(config, file_path)
        loaded = manager.load(file_path)

        assert loaded == config


# ---------------------------------------------------------------------------
# Property 15: Configuration merge with child precedence
# Feature: road-damage-evaluation-framework, Property 15: Configuration merge with child precedence
# ---------------------------------------------------------------------------


# Strategy for flat config dicts (no nesting) to clearly test merge semantics.
_FLAT_CONFIG = st.dictionaries(
    keys=_CONFIG_KEYS,
    values=_YAML_SAFE_SCALARS,
    min_size=0,
    max_size=8,
)


class TestProperty15MergeChildPrecedence:
    """Property 15: For any parent and child configuration, merging SHALL produce
    a result where: (a) keys present only in parent are preserved, (b) keys present
    only in child are included, and (c) keys present in both use the child's value.

    **Validates: Requirements 6.4, 6.5**
    """

    @given(parent=_FLAT_CONFIG, child=_FLAT_CONFIG)
    @settings(max_examples=100)
    def test_merge_preserves_parent_only_keys(self, parent, child):
        # Feature: road-damage-evaluation-framework, Property 15: Configuration merge with child precedence
        """Keys present only in parent are preserved in the merged result."""
        manager = ConfigManager()
        result = manager.merge(parent, child)

        parent_only_keys = set(parent.keys()) - set(child.keys())
        for key in parent_only_keys:
            assert key in result
            assert result[key] == parent[key]

    @given(parent=_FLAT_CONFIG, child=_FLAT_CONFIG)
    @settings(max_examples=100)
    def test_merge_includes_child_only_keys(self, parent, child):
        # Feature: road-damage-evaluation-framework, Property 15: Configuration merge with child precedence
        """Keys present only in child are included in the merged result."""
        manager = ConfigManager()
        result = manager.merge(parent, child)

        child_only_keys = set(child.keys()) - set(parent.keys())
        for key in child_only_keys:
            assert key in result
            assert result[key] == child[key]

    @given(parent=_FLAT_CONFIG, child=_FLAT_CONFIG)
    @settings(max_examples=100)
    def test_merge_child_wins_on_shared_keys(self, parent, child):
        # Feature: road-damage-evaluation-framework, Property 15: Configuration merge with child precedence
        """Keys present in both parent and child use the child's value."""
        manager = ConfigManager()
        result = manager.merge(parent, child)

        shared_keys = set(parent.keys()) & set(child.keys())
        for key in shared_keys:
            assert result[key] == child[key]

    @given(parent=_CONFIG_DICTS, child=_CONFIG_DICTS)
    @settings(max_examples=100)
    def test_merge_result_contains_all_keys(self, parent, child):
        # Feature: road-damage-evaluation-framework, Property 15: Configuration merge with child precedence
        """The merged result contains all keys from both parent and child."""
        manager = ConfigManager()
        result = manager.merge(parent, child)

        all_keys = set(parent.keys()) | set(child.keys())
        assert set(result.keys()) == all_keys


# ---------------------------------------------------------------------------
# Property 16: Environment variable substitution
# Feature: road-damage-evaluation-framework, Property 16: Environment variable substitution
# ---------------------------------------------------------------------------

# Strategy for valid environment variable names.
_ENV_VAR_NAMES = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "N"), whitelist_characters="_"),
    min_size=1,
    max_size=15,
).filter(lambda s: s[0].isalpha() or s[0] == "_")

# Strategy for env var values (non-empty strings without ${} patterns).
_ENV_VAR_VALUES = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
        blacklist_characters="${}",
    ),
    min_size=1,
    max_size=30,
)


class TestProperty16EnvVarSubstitution:
    """Property 16: For any configuration containing ${VAR_NAME} patterns where
    the referenced environment variables are set, resolving SHALL replace each
    pattern with the corresponding environment variable value, and no ${...}
    patterns SHALL remain in the output.

    **Validates: Requirements 6.6**
    """

    @given(
        var_name=_ENV_VAR_NAMES,
        var_value=_ENV_VAR_VALUES,
    )
    @settings(max_examples=100)
    def test_single_env_var_substitution(self, var_name, var_value):
        # Feature: road-damage-evaluation-framework, Property 16: Environment variable substitution
        """A single ${VAR_NAME} pattern is replaced with the env var value."""
        original = os.environ.get(var_name)
        try:
            os.environ[var_name] = var_value
            manager = ConfigManager()

            config = {"key": f"${{{var_name}}}"}
            result = manager.resolve_env_vars(config)

            assert result["key"] == var_value
            assert "${" not in result["key"]
        finally:
            if original is None:
                os.environ.pop(var_name, None)
            else:
                os.environ[var_name] = original

    @given(
        data=st.data(),
        num_vars=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=100)
    def test_multiple_env_vars_all_resolved(self, data, num_vars):
        # Feature: road-damage-evaluation-framework, Property 16: Environment variable substitution
        """All ${VAR_NAME} patterns are resolved when env vars are set."""
        manager = ConfigManager()

        # Generate unique var names and values
        var_names = []
        var_values = []
        originals = {}
        for _ in range(num_vars):
            name = data.draw(_ENV_VAR_NAMES)
            value = data.draw(_ENV_VAR_VALUES)
            # Ensure unique names
            assume(name not in var_names)
            var_names.append(name)
            var_values.append(value)
            originals[name] = os.environ.get(name)
            os.environ[name] = value

        try:
            # Build a config with all vars referenced
            config = {}
            for i, name in enumerate(var_names):
                config[f"key_{i}"] = f"prefix_${{{name}}}_suffix"

            result = manager.resolve_env_vars(config)

            # Verify no ${...} patterns remain
            for key, val in result.items():
                assert "${" not in str(val), f"Unresolved pattern in {key}: {val}"

            # Verify each value was substituted correctly
            for i, (name, value) in enumerate(zip(var_names, var_values)):
                assert result[f"key_{i}"] == f"prefix_{value}_suffix"
        finally:
            for name, orig in originals.items():
                if orig is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = orig

    @given(
        var_name=_ENV_VAR_NAMES,
        var_value=_ENV_VAR_VALUES,
    )
    @settings(max_examples=100)
    def test_nested_config_env_var_resolution(self, var_name, var_value):
        # Feature: road-damage-evaluation-framework, Property 16: Environment variable substitution
        """Env vars in nested dicts and lists are resolved."""
        original = os.environ.get(var_name)
        try:
            os.environ[var_name] = var_value
            manager = ConfigManager()

            config = {
                "level1": {
                    "level2": f"${{{var_name}}}",
                    "list_val": [f"${{{var_name}}}", "static"],
                }
            }
            result = manager.resolve_env_vars(config)

            # No ${...} patterns should remain anywhere in the result
            def check_no_patterns(obj):
                if isinstance(obj, str):
                    assert "${" not in obj, f"Unresolved pattern: {obj}"
                elif isinstance(obj, dict):
                    for v in obj.values():
                        check_no_patterns(v)
                elif isinstance(obj, list):
                    for item in obj:
                        check_no_patterns(item)

            check_no_patterns(result)
            assert result["level1"]["level2"] == var_value
            assert result["level1"]["list_val"][0] == var_value
        finally:
            if original is None:
                os.environ.pop(var_name, None)
            else:
                os.environ[var_name] = original
