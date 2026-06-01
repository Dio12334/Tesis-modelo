"""Property-based tests for required-parameter presence in config validation.

Feature: generic-evaluation-script, Task 2.6
Property 3: Required parameters must be present and non-null

For any configuration in which any subset of the required parameters
(``model.type``, ``model.config.num_classes``, ``dataset.path``,
``evaluation.split``) is absent or null, ``validate_config`` produces a
violation that names each absent/null required parameter, and no violation is
produced for a required parameter that is present and non-null.

These tests drive the real ``validate_config`` in
``model/training/evaluate_detection.py``. ``validate_config`` always evaluates
the checkpoint exclusive-or rule, so every generated configuration is given a
*valid* checkpoint section (exactly one of ``checkpoint.path`` /
``checkpoint.run_id``) to isolate the required-parameter behaviour under test.
Provided ``evaluation.split`` values are kept valid and thresholds are omitted,
so the only violations that can arise are the required-parameter ones this
property targets.

**Validates: Requirements 4.2**
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from model.exceptions import ConfigurationError
from model.training.evaluate_detection import validate_config


# ---------------------------------------------------------------------------
# Constants and config construction
# ---------------------------------------------------------------------------

REQUIRED_PARAMS = [
    "model.type",
    "model.config.num_classes",
    "dataset.path",
    "evaluation.split",
]

# The exact substring used by validate_config for required-parameter violations.
_REQUIRED_MARKER = "required parameter is missing or null"


def build_config(states: dict, checkpoint: dict) -> dict:
    """Build a merged configuration for the given per-parameter states.

    Args:
        states: Maps each required-parameter name to one of ``"present"``
            (include a valid non-null value), ``"absent"`` (omit it entirely),
            or ``"null"`` (include it with a ``None`` value).
        checkpoint: A *valid* checkpoint section (exactly one of ``path`` /
            ``run_id``), so the checkpoint XOR rule never contributes a
            violation.

    Returns:
        A configuration dict with sections shaped to honour ``states`` while
        respecting the nesting of ``model.type`` and ``model.config.num_classes``
        (both of which live under the shared ``model`` section).
    """
    config: dict = {}

    # --- model section (holds both model.type and model.config.num_classes) ---
    model: dict = {}
    type_state = states["model.type"]
    if type_state == "present":
        model["type"] = "yolo26"
    elif type_state == "null":
        model["type"] = None
    # "absent" -> leave model.type out entirely.

    nc_state = states["model.config.num_classes"]
    if nc_state == "present":
        model.setdefault("config", {})["num_classes"] = 5
    elif nc_state == "null":
        model.setdefault("config", {})["num_classes"] = None
    # "absent" -> do not create model.config.num_classes.

    if model:
        config["model"] = model

    # --- dataset section ---
    ds_state = states["dataset.path"]
    if ds_state == "present":
        config["dataset"] = {"path": "data/rdd2022"}
    elif ds_state == "null":
        config["dataset"] = {"path": None}

    # --- evaluation section ---
    split_state = states["evaluation.split"]
    if split_state == "present":
        config["evaluation"] = {"split": "val"}  # valid value -> no split violation
    elif split_state == "null":
        config["evaluation"] = {"split": None}

    # --- checkpoint section (always valid: exactly one of path / run_id) ---
    config["checkpoint"] = dict(checkpoint)

    return config


def required_violations(error: ConfigurationError) -> list:
    """Return only the required-parameter violations from a ConfigurationError."""
    return [v for v in error.violations if _REQUIRED_MARKER in v]


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_STATE = st.sampled_from(["present", "absent", "null"])

_STATES = st.fixed_dictionaries({name: _STATE for name in REQUIRED_PARAMS})

# A valid checkpoint section: exactly one of path / run_id, non-null.
_CHECKPOINT = st.sampled_from(["path", "run_id"]).map(lambda key: {key: "x"})


# ---------------------------------------------------------------------------
# Property 3
# ---------------------------------------------------------------------------


class TestProperty3RequiredParameterPresence:
    """Property 3: Required parameters must be present and non-null.

    **Validates: Requirements 4.2**
    """

    @given(states=_STATES, checkpoint=_CHECKPOINT)
    @settings(max_examples=100)
    def test_absent_or_null_required_params_each_named(self, states, checkpoint):
        # Feature: generic-evaluation-script, Property 3: Required parameters must be present and non-null
        """Each absent/null required param yields a violation naming it; present ones do not."""
        config = build_config(states, checkpoint)
        expected_missing = {
            param for param, state in states.items() if state in ("absent", "null")
        }

        if not expected_missing:
            # All required params present + valid checkpoint => no violations.
            # validate_config must return without raising.
            assert validate_config(config) is None
            return

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)

        req_viols = required_violations(exc_info.value)

        # Every absent/null required parameter is named by a required violation.
        for param in expected_missing:
            assert any(v.startswith(f"{param}:") for v in req_viols), (
                f"expected a required-parameter violation naming {param}; "
                f"got {req_viols}"
            )

        # No present-and-non-null required parameter produces a violation.
        present = set(REQUIRED_PARAMS) - expected_missing
        for param in present:
            assert not any(v.startswith(f"{param}:") for v in req_viols), (
                f"present parameter {param} should not produce a violation; "
                f"got {req_viols}"
            )

        # Exactly one required violation per missing parameter (no spurious ones).
        assert len(req_viols) == len(expected_missing)

    @given(checkpoint=_CHECKPOINT)
    @settings(max_examples=100)
    def test_all_present_yields_no_required_violation(self, checkpoint):
        # Feature: generic-evaluation-script, Property 3: Required parameters must be present and non-null
        """A fully-populated config (valid checkpoint) raises no ConfigurationError."""
        states = {name: "present" for name in REQUIRED_PARAMS}
        config = build_config(states, checkpoint)

        # No violations of any kind -> validate_config returns None.
        assert validate_config(config) is None

    @given(
        missing=st.lists(
            st.sampled_from(REQUIRED_PARAMS), min_size=1, max_size=4, unique=True
        ),
        checkpoint=_CHECKPOINT,
    )
    @settings(max_examples=100)
    def test_named_violation_count_matches_missing_count(self, missing, checkpoint):
        # Feature: generic-evaluation-script, Property 3: Required parameters must be present and non-null
        """The number of required violations equals the number of absent params."""
        states = {
            name: ("absent" if name in missing else "present")
            for name in REQUIRED_PARAMS
        }
        config = build_config(states, checkpoint)

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)

        req_viols = required_violations(exc_info.value)
        assert len(req_viols) == len(missing)
        for param in missing:
            assert any(v.startswith(f"{param}:") for v in req_viols)


# ---------------------------------------------------------------------------
# Targeted example checks (complement the property with concrete cases)
# ---------------------------------------------------------------------------


class TestRequiredParameterExamples:
    """Concrete examples that pin down the required-parameter behaviour.

    **Validates: Requirements 4.2**
    """

    def _valid_config(self) -> dict:
        return {
            "model": {"type": "yolo26", "config": {"num_classes": 5}},
            "dataset": {"path": "data/rdd2022"},
            "evaluation": {"split": "val"},
            "checkpoint": {"path": "best_model.pt"},
        }

    def test_fully_valid_config_passes(self):
        """A complete, valid configuration raises nothing.

        **Validates: Requirements 4.2**
        """
        assert validate_config(self._valid_config()) is None

    @pytest.mark.parametrize(
        "section_key,leaf_path",
        [
            ("model.type", ("model", "type")),
            ("model.config.num_classes", ("model", "config", "num_classes")),
            ("dataset.path", ("dataset", "path")),
            ("evaluation.split", ("evaluation", "split")),
        ],
    )
    def test_each_missing_param_is_named(self, section_key, leaf_path):
        """Removing a single required param yields a violation naming it.

        **Validates: Requirements 4.2**
        """
        config = self._valid_config()
        # Delete the targeted leaf.
        node = config
        for key in leaf_path[:-1]:
            node = node[key]
        del node[leaf_path[-1]]

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)

        req_viols = required_violations(exc_info.value)
        assert len(req_viols) == 1
        assert req_viols[0].startswith(f"{section_key}:")

    @pytest.mark.parametrize(
        "section_key,leaf_path",
        [
            ("model.type", ("model", "type")),
            ("model.config.num_classes", ("model", "config", "num_classes")),
            ("dataset.path", ("dataset", "path")),
            ("evaluation.split", ("evaluation", "split")),
        ],
    )
    def test_each_null_param_is_named(self, section_key, leaf_path):
        """Nulling a single required param yields a violation naming it.

        **Validates: Requirements 4.2**
        """
        config = self._valid_config()
        node = config
        for key in leaf_path[:-1]:
            node = node[key]
        node[leaf_path[-1]] = None

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)

        req_viols = required_violations(exc_info.value)
        assert len(req_viols) == 1
        assert req_viols[0].startswith(f"{section_key}:")

    def test_all_required_missing_names_all_four(self):
        """An empty config (valid checkpoint aside) names all four required params.

        **Validates: Requirements 4.2**
        """
        config = {"checkpoint": {"path": "best_model.pt"}}

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)

        req_viols = required_violations(exc_info.value)
        assert len(req_viols) == 4
        for param in REQUIRED_PARAMS:
            assert any(v.startswith(f"{param}:") for v in req_viols)
