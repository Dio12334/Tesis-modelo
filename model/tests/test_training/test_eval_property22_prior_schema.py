"""Property-based tests for prior-schema configuration acceptance.

Feature: generic-evaluation-script
Property 22: Prior-schema configuration acceptance

For any prior-schema configuration that omits any subset of the newly
introduced optional fields, loading and validation succeed and each omitted
field takes its documented default value.

The implementation under test is
``model.training.evaluate_detection.load_and_merge_config``: it loads a YAML
file via ``ConfigManager`` (``yaml.safe_load``), resolves ``${ENV}`` references,
and deep-merges CLI overrides on top. It deliberately does NOT inject defaults
into the merged dict; instead, newly introduced fields are *optional*, so a
prior-schema config that omits them loads without error and each omitted field
is realized as its documented default at the point of use (``dict.get(key,
default)``). These tests therefore assert three things for every prior-schema
config:

1. Loading succeeds (no exception) -- the config is *accepted*.
2. Each omitted optional field is genuinely absent from the merged config
   (proving it is optional rather than required).
3. Resolving the merged config against the documented defaults yields the
   documented default value for each omitted field, and preserves any
   explicitly provided value unchanged.

The documented defaults below mirror the "Evaluation configuration schema"
section of the design document.

**Validates: Requirements 17.3**
"""

import yaml
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from model.training.evaluate_detection import load_and_merge_config


# ---------------------------------------------------------------------------
# Documented defaults (from design.md "Evaluation configuration schema")
# ---------------------------------------------------------------------------
# Each entry is (dotted-path-as-tuple, documented_default, value_strategy).
# ``evaluation.input_size`` is handled separately because its default is a
# *mirror* of ``model.config.input_size`` rather than a constant.

_MISSING = object()


# --- YAML-safe leaf strategies --------------------------------------------

# Simple identifier-like strings that survive a YAML round-trip unambiguously
# and never collide with YAML's implicit boolean/null tokens.
_SAFE_STRINGS = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
    min_size=1,
    max_size=20,
).filter(
    lambda s: s.strip().lower()
    not in ("true", "false", "yes", "no", "on", "off", "null", "~", "none")
    and s.strip() != ""
)

# Path-like strings (a couple of segments) that remain YAML-safe.
_PATH_STRINGS = st.lists(_SAFE_STRINGS, min_size=1, max_size=3).map("/".join)

_UNIT_FLOATS = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)
_INPUT_SIZES = st.integers(min_value=32, max_value=2048)
_NUM_CLASSES = st.integers(min_value=1, max_value=200)
_SEEDS = st.integers(min_value=0, max_value=2**31 - 1)
_SPLITS = st.sampled_from(["train", "val", "test"])
_SUBSETS = st.sampled_from(["train", "test"])


# (path, documented_default, value_strategy)
_OPTIONAL_FIELDS = [
    (("model", "config", "input_size"), 640, _INPUT_SIZES),
    (("dataset", "type"), "rdd2022", _SAFE_STRINGS),
    (("dataset", "subset"), "train", _SUBSETS),
    (("evaluation", "confidence_threshold"), 0.25, _UNIT_FLOATS),
    (("evaluation", "iou_threshold"), 0.5, _UNIT_FLOATS),
    (("evaluation", "val_split"), 0.2, _UNIT_FLOATS),
    (("evaluation", "seed"), 42, _SEEDS),
    (("evaluation", "output_dir"), None, _PATH_STRINGS),
    (("checkpoint", "checkpoint_dir"), "./checkpoints", _PATH_STRINGS),
]


# ---------------------------------------------------------------------------
# Nested-dict helpers
# ---------------------------------------------------------------------------


def _get(config, path):
    """Return the value at the dotted ``path`` tuple, or ``_MISSING`` if absent."""
    node = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return _MISSING
        node = node[key]
    return node


def _set(config, path, value):
    """Set ``value`` at the dotted ``path`` tuple, creating intermediate dicts."""
    node = config
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def _resolve_default(config, path, default):
    """Mirror the documented default-resolution at point of use."""
    val = _get(config, path)
    return default if val is _MISSING else val


def _resolve_input_size(config):
    """``evaluation.input_size`` mirrors ``model.config.input_size`` (default 640)."""
    eis = _get(config, ("evaluation", "input_size"))
    if eis is not _MISSING:
        return eis
    mcis = _get(config, ("model", "config", "input_size"))
    if mcis is not _MISSING:
        return mcis
    return 640


def _write_yaml(tmp_path_factory, config) -> str:
    """Write ``config`` to a fresh YAML file and return its path."""
    directory = tmp_path_factory.mktemp("prior_schema")
    file_path = directory / "config.yaml"
    with open(file_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, default_flow_style=False, sort_keys=False)
    return str(file_path)


# ---------------------------------------------------------------------------
# Composite strategy: a prior-schema config that omits a random subset of the
# newly introduced optional fields while always carrying the required fields.
# ---------------------------------------------------------------------------


@st.composite
def prior_schema_configs(draw):
    config = {
        "model": {
            "type": draw(_SAFE_STRINGS),
            "config": {"num_classes": draw(_NUM_CLASSES)},
        },
        "dataset": {"path": draw(_PATH_STRINGS)},
        "evaluation": {"split": draw(_SPLITS)},
    }

    # Exactly one of checkpoint.path / checkpoint.run_id (prior-schema, valid XOR).
    if draw(st.booleans()):
        config["checkpoint"] = {"path": draw(_PATH_STRINGS)}
    else:
        config["checkpoint"] = {"run_id": draw(_SAFE_STRINGS)}

    present = {}
    omitted = []
    for path, _default, strat in _OPTIONAL_FIELDS:
        if draw(st.booleans()):
            value = draw(strat)
            _set(config, path, value)
            present[path] = value
        else:
            omitted.append(path)

    return config, present, omitted


# ---------------------------------------------------------------------------
# Property 22
# ---------------------------------------------------------------------------


class TestProperty22PriorSchemaAcceptance:
    """Property 22: Prior-schema configuration acceptance.

    **Validates: Requirements 17.3**
    """

    @given(payload=prior_schema_configs())
    @settings(max_examples=100)
    def test_prior_schema_config_loads_and_omitted_fields_take_defaults(
        self, payload, tmp_path_factory
    ):
        # Feature: generic-evaluation-script, Property 22: Prior-schema configuration acceptance
        """A prior-schema config loads; omitted optional fields resolve to defaults."""
        config, present, omitted = payload
        config_path = _write_yaml(tmp_path_factory, config)

        # 1. Loading succeeds (config is accepted) and yields a dict.
        merged = load_and_merge_config(config_path, {})
        assert isinstance(merged, dict)

        # Required (prior-schema) fields are preserved unchanged.
        assert _get(merged, ("model", "type")) == config["model"]["type"]
        assert _get(merged, ("model", "config", "num_classes")) == (
            config["model"]["config"]["num_classes"]
        )
        assert _get(merged, ("dataset", "path")) == config["dataset"]["path"]
        assert _get(merged, ("evaluation", "split")) == config["evaluation"]["split"]

        # 2 + 3. Each omitted optional field is absent and resolves to its default.
        defaults = {path: default for path, default, _ in _OPTIONAL_FIELDS}
        for path in omitted:
            assert _get(merged, path) is _MISSING, (
                f"omitted field {path} should be absent from the merged config"
            )
            assert _resolve_default(merged, path, defaults[path]) == defaults[path]

        # Explicitly provided optional values survive the load unchanged.
        for path, value in present.items():
            assert _get(merged, path) == value

    @given(payload=prior_schema_configs())
    @settings(max_examples=100)
    def test_evaluation_input_size_mirrors_model_config_default(
        self, payload, tmp_path_factory
    ):
        # Feature: generic-evaluation-script, Property 22: Prior-schema configuration acceptance
        """Omitted ``evaluation.input_size`` mirrors ``model.config.input_size`` (else 640)."""
        config, _present, _omitted = payload
        # Force the mirror scenario: never provide an explicit evaluation.input_size.
        config.get("evaluation", {}).pop("input_size", None)
        config_path = _write_yaml(tmp_path_factory, config)

        merged = load_and_merge_config(config_path, {})

        assert _get(merged, ("evaluation", "input_size")) is _MISSING
        expected = _get(merged, ("model", "config", "input_size"))
        if expected is _MISSING:
            expected = 640
        assert _resolve_input_size(merged) == expected

    def test_minimal_prior_schema_config_loads_with_all_defaults(
        self, tmp_path_factory
    ):
        # Feature: generic-evaluation-script, Property 22: Prior-schema configuration acceptance
        """A minimal prior-schema config (every optional field omitted) loads cleanly."""
        config = {
            "model": {"type": "ssd_mobilenet", "config": {"num_classes": 4}},
            "dataset": {"path": "model/data/rdd2022/sample"},
            "evaluation": {"split": "val"},
            "checkpoint": {"path": "checkpoints/best_model.pt"},
        }
        config_path = _write_yaml(tmp_path_factory, config)

        merged = load_and_merge_config(config_path, {})

        for path, default, _ in _OPTIONAL_FIELDS:
            assert _get(merged, path) is _MISSING
            assert _resolve_default(merged, path, default) == default
        assert _resolve_input_size(merged) == 640
