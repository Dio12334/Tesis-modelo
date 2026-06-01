"""Property-based tests for configuration override deep-merge precedence.

Feature: generic-evaluation-script
Property 1: Override deep-merge precedence

For any base configuration dict and any overrides dict, deep-merging the
overrides on top of the base (via ``load_and_merge_config`` which delegates to
``ConfigManager.merge``) yields a configuration in which:

* every override leaf value takes precedence over the base,
* every base leaf not mentioned in the overrides is preserved unchanged, and
* nested dicts are merged recursively rather than replaced wholesale.

These tests exercise the real load path: the base configuration is written to a
temporary YAML file and loaded through ``ConfigManager`` exactly as production
code does. Because Property 1 concerns *merge* semantics (YAML round-trip is
covered separately by Property 20), expectations are computed against the
base configuration *as actually loaded*, isolating the merge behaviour under
test.

**Validates: Requirements 3.3**
"""

from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from model.config.manager import ConfigManager
from model.training.evaluate_detection import load_and_merge_config


# ---------------------------------------------------------------------------
# Hypothesis strategies (YAML-safe scalars + bounded-depth nested dicts)
# ---------------------------------------------------------------------------

# Strings that YAML would reinterpret as another type must be excluded so that
# the loaded base reflects the dict we authored. ``$`` / ``{`` / ``}`` are
# excluded so no ``${ENV}`` substitution occurs during loading.
_YAML_RESERVED = {
    "true", "false", "yes", "no", "on", "off", "null", "~", "none", "",
}

_SAFE_STRINGS = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
        blacklist_characters="\x00${}",
    ),
    min_size=1,
    max_size=20,
).filter(lambda s: s.strip().lower() not in _YAML_RESERVED)

_SCALARS = st.one_of(
    _SAFE_STRINGS,
    st.integers(min_value=-10_000, max_value=10_000),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.booleans(),
)

# A small, shared key pool so base and overrides collide frequently. Collisions
# are what actually exercise precedence and recursive merging.
_KEYS = st.sampled_from(
    ["a", "b", "c", "d", "model", "config", "dataset", "type", "num", "size"]
)


def _values(max_depth: int):
    """YAML-safe values with bounded nesting depth."""
    if max_depth <= 0:
        return _SCALARS
    return st.one_of(
        _SCALARS,
        st.lists(_SCALARS, min_size=0, max_size=4),
        st.dictionaries(
            keys=_KEYS,
            values=_values(max_depth - 1),
            min_size=0,
            max_size=4,
        ),
    )


_CONFIG_DICTS = st.dictionaries(
    keys=_KEYS,
    values=_values(max_depth=2),
    min_size=0,
    max_size=5,
)


# ---------------------------------------------------------------------------
# Reference semantics + path helpers
# ---------------------------------------------------------------------------


def reference_deep_merge(base: dict, override: dict) -> dict:
    """Independent reference implementation of child-precedence deep-merge.

    Recurses only when *both* sides are dicts; otherwise the override value wins
    outright (mirroring ``ConfigManager.merge``).
    """
    result = dict(base)
    for key, ov in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(ov, dict):
            result[key] = reference_deep_merge(result[key], ov)
        else:
            result[key] = ov
    return result


def leaf_paths(d, prefix=()):
    """Yield ``(path_tuple, value)`` for every non-dict leaf in nested dict ``d``."""
    for key, value in d.items():
        if isinstance(value, dict):
            yield from leaf_paths(value, prefix + (key,))
        else:
            yield prefix + (key,), value


def get_path(d, path):
    """Return the value stored at ``path`` (a tuple of keys) within ``d``."""
    cur = d
    for key in path:
        cur = cur[key]
    return cur


def base_leaf_is_preserved(override, path) -> bool:
    """Return True if a base leaf at ``path`` survives the override unchanged.

    A base leaf is preserved iff the override never reaches it: at some key the
    override dict lacks that key (base subtree kept). It is NOT preserved if the
    override defines that exact path, or replaces an ancestor with a non-dict.
    """
    cur = override
    for key in path:
        if not isinstance(cur, dict):
            # The override placed a scalar/list at an ancestor: base replaced.
            return False
        if key not in cur:
            # The override never descended here: base subtree kept verbatim.
            return True
        cur = cur[key]
    # The full path exists in the override: it defines/replaces this leaf.
    return False


def _write_yaml(tmp_path: Path, config: dict) -> Path:
    file_path = tmp_path / "base.yaml"
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return file_path


def _loaded_base(path: Path) -> dict:
    """Load the base config exactly as ``load_and_merge_config`` does (pre-merge)."""
    cm = ConfigManager()
    return cm.resolve_env_vars(cm.load(path))


# ---------------------------------------------------------------------------
# Property 1
# ---------------------------------------------------------------------------


class TestProperty1OverrideMergePrecedence:
    """Property 1: Override deep-merge precedence.

    **Validates: Requirements 3.3**
    """

    @given(base=_CONFIG_DICTS, overrides=_CONFIG_DICTS)
    @settings(max_examples=100)
    def test_merge_matches_reference_deep_merge(self, base, overrides, tmp_path_factory):
        # Feature: generic-evaluation-script, Property 1: Override deep-merge precedence
        """Merging via the real load path equals an independent deep-merge reference."""
        tmp_path = tmp_path_factory.mktemp("merge_ref")
        path = _write_yaml(tmp_path, base)

        expected = reference_deep_merge(_loaded_base(path), overrides)
        result = load_and_merge_config(str(path), overrides)

        assert result == expected

    @given(base=_CONFIG_DICTS, overrides=_CONFIG_DICTS)
    @settings(max_examples=100)
    def test_every_override_leaf_takes_precedence(self, base, overrides, tmp_path_factory):
        # Feature: generic-evaluation-script, Property 1: Override deep-merge precedence
        """Every override leaf value appears at its path in the merged result."""
        tmp_path = tmp_path_factory.mktemp("merge_override")
        path = _write_yaml(tmp_path, base)

        result = load_and_merge_config(str(path), overrides)

        for leaf_path, override_value in leaf_paths(overrides):
            assert get_path(result, leaf_path) == override_value

    @given(base=_CONFIG_DICTS, overrides=_CONFIG_DICTS)
    @settings(max_examples=100)
    def test_base_leaves_not_overridden_are_preserved(self, base, overrides, tmp_path_factory):
        # Feature: generic-evaluation-script, Property 1: Override deep-merge precedence
        """Base leaves the override never reaches survive unchanged."""
        tmp_path = tmp_path_factory.mktemp("merge_preserve")
        path = _write_yaml(tmp_path, base)

        loaded_base = _loaded_base(path)
        result = load_and_merge_config(str(path), overrides)

        for leaf_path, base_value in leaf_paths(loaded_base):
            if base_leaf_is_preserved(overrides, leaf_path):
                assert get_path(result, leaf_path) == base_value

    @given(overrides=_CONFIG_DICTS)
    @settings(max_examples=100)
    def test_overrides_apply_fully_without_a_base(self, overrides):
        # Feature: generic-evaluation-script, Property 1: Override deep-merge precedence
        """With no config file the merged result equals the overrides themselves."""
        result = load_and_merge_config(None, overrides)
        assert result == overrides

    def test_nested_dicts_merged_recursively_not_replaced(self, tmp_path):
        # Feature: generic-evaluation-script, Property 1: Override deep-merge precedence
        """A nested override merges into the base subtree instead of replacing it."""
        base = {
            "model": {
                "type": "yolo26",
                "config": {"num_classes": 5, "input_size": 640},
            },
            "dataset": {"path": "data/rdd2022"},
        }
        overrides = {"model": {"config": {"input_size": 320}}}

        path = _write_yaml(tmp_path, base)
        result = load_and_merge_config(str(path), overrides)

        # Override leaf wins.
        assert result["model"]["config"]["input_size"] == 320
        # Sibling leaves in the same nested dict are preserved (not replaced).
        assert result["model"]["config"]["num_classes"] == 5
        assert result["model"]["type"] == "yolo26"
        # Untouched sibling sections are preserved.
        assert result["dataset"] == {"path": "data/rdd2022"}
