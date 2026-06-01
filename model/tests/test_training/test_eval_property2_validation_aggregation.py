"""Property-based tests for aggregated configuration validation before any I/O.

Feature: generic-evaluation-script
Property 2: Validation aggregates all violations and runs before any I/O

For any merged configuration containing one or more validation violations,
``validate_config`` raises exactly one ``ConfigurationError`` whose message is a
header followed by one bullet per violation, each bullet naming the offending
parameter and its violated rule (including the observed value and expected
range/allowed values for range and enum rules), and no detector instantiation,
checkpoint load, or dataset load is performed.

The strategy below builds a configuration from a small set of independent
"slots" (the four required parameters, the two thresholds, the split, and the
checkpoint exclusive-or). Each slot is driven to either a valid state or a
single known violation, so the *exact* number and identity of the expected
violations is known up front and can be checked against the rendered message
and the exception's ``violations`` list.

The "no I/O" half of the property is exercised by patching the collaborators
that would perform detector instantiation (``ModelRegistry.create``), dataset
loading (``RDD2022Dataset.load``), and any filesystem probe
(``pathlib.Path.exists``) and asserting none of them are called while
``validate_config`` runs and fails.

**Validates: Requirements 4.1, 4.9, 11.1, 11.3, 11.4, 12.3**
"""

from pathlib import Path
from unittest import mock

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from model.datasets.rdd2022 import RDD2022Dataset
from model.exceptions import ConfigurationError
from model.models.registry import ModelRegistry
from model.training.evaluate_detection import validate_config


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Floats strictly outside the closed interval [0.0, 1.0] (range violations).
_OUT_OF_RANGE_FLOATS = st.one_of(
    st.floats(min_value=1.0001, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1e6, max_value=-0.0001, allow_nan=False, allow_infinity=False),
)

# In-range thresholds (no violation).
_IN_RANGE_FLOATS = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Values that are not numbers at all (range rule rejects them too).
_NON_NUMERIC = st.sampled_from(["abc", "high", "0.5x"])

# Split values that are present and non-null but not in the allowed set.
_BAD_SPLITS = st.sampled_from(
    ["validation", "training", "testing", "foo", "VAL", "Train", "dev"]
)

_ALLOWED_SPLITS_REPR = "['train', 'val', 'test']"


@st.composite
def _configs_with_violations(draw):
    """Build a config plus the exact set of violations it should produce.

    Returns ``(config, expected)`` where ``expected`` is a list of
    ``(param_name, [required_substrings])`` tuples - one per violation the
    implementation is expected to emit, in no particular order.
    """
    expected = []  # list of (param_name, [substrings that must be in its bullet])

    # --- model.type (required) ---
    model = {}
    mt = draw(st.sampled_from(["ok", "absent", "null"]))
    if mt == "ok":
        model["type"] = "yolo26"
    elif mt == "null":
        model["type"] = None
        expected.append(("model.type", ["required"]))
    else:  # absent
        expected.append(("model.type", ["required"]))

    # --- model.config.num_classes (required) ---
    cfg = {}
    nc = draw(st.sampled_from(["ok", "absent", "null"]))
    if nc == "ok":
        cfg["num_classes"] = 5
    elif nc == "null":
        cfg["num_classes"] = None
        expected.append(("model.config.num_classes", ["required"]))
    else:  # absent
        expected.append(("model.config.num_classes", ["required"]))
    model["config"] = cfg

    config = {"model": model}

    # --- dataset.path (required) ---
    dataset = {}
    dp = draw(st.sampled_from(["ok", "absent", "null"]))
    if dp == "ok":
        dataset["path"] = "data/rdd2022"
    elif dp == "null":
        dataset["path"] = None
        expected.append(("dataset.path", ["required"]))
    else:  # absent
        expected.append(("dataset.path", ["required"]))
    config["dataset"] = dataset

    evaluation = {}

    # --- evaluation.split (required + enum) ---
    sp = draw(st.sampled_from(["ok", "absent", "null", "invalid"]))
    if sp == "ok":
        evaluation["split"] = draw(st.sampled_from(["train", "val", "test"]))
    elif sp == "null":
        evaluation["split"] = None
        expected.append(("evaluation.split", ["required"]))
    elif sp == "invalid":
        bad = draw(_BAD_SPLITS)
        evaluation["split"] = bad
        expected.append(
            ("evaluation.split", [_ALLOWED_SPLITS_REPR, repr(bad)])
        )
    else:  # absent
        expected.append(("evaluation.split", ["required"]))

    # --- evaluation.confidence_threshold (range) ---
    ct = draw(st.sampled_from(["ok", "absent", "range", "nonnumeric"]))
    if ct == "ok":
        evaluation["confidence_threshold"] = draw(_IN_RANGE_FLOATS)
    elif ct == "range":
        val = draw(_OUT_OF_RANGE_FLOATS)
        evaluation["confidence_threshold"] = val
        expected.append(
            ("evaluation.confidence_threshold", ["[0.0, 1.0]", str(val)])
        )
    elif ct == "nonnumeric":
        val = draw(_NON_NUMERIC)
        evaluation["confidence_threshold"] = val
        expected.append(
            ("evaluation.confidence_threshold", ["[0.0, 1.0]", repr(val)])
        )
    # "absent" -> omit the key, no violation

    # --- evaluation.iou_threshold (range) ---
    it = draw(st.sampled_from(["ok", "absent", "range", "nonnumeric"]))
    if it == "ok":
        evaluation["iou_threshold"] = draw(_IN_RANGE_FLOATS)
    elif it == "range":
        val = draw(_OUT_OF_RANGE_FLOATS)
        evaluation["iou_threshold"] = val
        expected.append(("evaluation.iou_threshold", ["[0.0, 1.0]", str(val)]))
    elif it == "nonnumeric":
        val = draw(_NON_NUMERIC)
        evaluation["iou_threshold"] = val
        expected.append(("evaluation.iou_threshold", ["[0.0, 1.0]", repr(val)]))
    # "absent" -> omit the key, no violation

    config["evaluation"] = evaluation

    # --- checkpoint.path XOR checkpoint.run_id ---
    ck = draw(st.sampled_from(["path_only", "run_id_only", "both", "neither"]))
    checkpoint = {}
    if ck == "path_only":
        checkpoint["path"] = "ckpt.pt"
    elif ck == "run_id_only":
        checkpoint["run_id"] = "run-abc-123"
    elif ck == "both":
        checkpoint["path"] = "ckpt.pt"
        checkpoint["run_id"] = "run-abc-123"
        expected.append(
            ("checkpoint.path / checkpoint.run_id", ["mutually exclusive"])
        )
    else:  # neither
        expected.append(
            ("checkpoint.path / checkpoint.run_id", ["required"])
        )
    config["checkpoint"] = checkpoint

    # The property concerns configs with *one or more* violations.
    assume(len(expected) >= 1)
    return config, expected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bullets(message: str):
    """Return the bullet lines (one per violation) from a rendered message."""
    return [ln for ln in message.splitlines() if ln.startswith("  - ")]


# ---------------------------------------------------------------------------
# Property 2
# ---------------------------------------------------------------------------


class TestProperty2ValidationAggregation:
    """Property 2: Validation aggregates all violations and runs before any I/O.

    **Validates: Requirements 4.1, 4.9, 11.1, 11.3, 11.4, 12.3**
    """

    @given(case=_configs_with_violations())
    @settings(max_examples=100)
    def test_all_violations_aggregated_into_one_error(self, case):
        # Feature: generic-evaluation-script, Property 2: Validation aggregates all violations
        """A single ConfigurationError carries exactly one bullet per violation."""
        config, expected = case

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)

        exc = exc_info.value
        message = str(exc)
        lines = message.splitlines()
        bullets = _bullets(message)

        # Exactly one error per detected violation, both on the message and on
        # the structured `.violations` list (Req 4.1, 4.9, 11.1, 11.3).
        assert len(bullets) == len(expected)
        assert len(exc.violations) == len(expected)

    @given(case=_configs_with_violations())
    @settings(max_examples=100)
    def test_message_is_header_then_one_bullet_per_violation(self, case):
        # Feature: generic-evaluation-script, Property 2: Validation aggregates all violations
        """Message renders as a counted header line followed by violation bullets."""
        config, expected = case

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)

        message = str(exc_info.value)
        lines = message.splitlines()
        bullets = _bullets(message)

        # Header line first, not itself a bullet, and reporting the count (Req 11.3).
        assert not lines[0].startswith("  - ")
        assert str(len(expected)) in lines[0]
        # Every non-header line is a bullet, and there is exactly one per violation.
        assert len(lines) == len(expected) + 1
        assert len(bullets) == len(expected)

    @given(case=_configs_with_violations())
    @settings(max_examples=100)
    def test_each_bullet_names_parameter_and_rule_detail(self, case):
        # Feature: generic-evaluation-script, Property 2: Validation aggregates all violations
        """Each violation's bullet names the parameter and the violated-rule detail."""
        config, expected = case

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)

        bullets = _bullets(str(exc_info.value))

        for param, substrings in expected:
            # Each parameter name uniquely identifies exactly one bullet (Req 11.3).
            matching = [b for b in bullets if param in b]
            assert len(matching) == 1, (param, bullets)
            bullet = matching[0]
            # Range/enum rules include observed value + expected range/allowed
            # values; required rules include the rule keyword (Req 11.4, 4.9).
            for sub in substrings:
                assert sub in bullet, (param, sub, bullet)

    @given(case=_configs_with_violations())
    @settings(max_examples=100)
    def test_no_detector_checkpoint_or_dataset_io_performed(self, case):
        # Feature: generic-evaluation-script, Property 2: Validation aggregates all violations
        """Validation fails without instantiating a detector or touching disk."""
        config, _ = case

        with mock.patch.object(ModelRegistry, "create") as m_create, mock.patch.object(
            RDD2022Dataset, "load"
        ) as m_load, mock.patch.object(Path, "exists") as m_exists:
            with pytest.raises(ConfigurationError):
                validate_config(config)

            # No detector instantiation, no dataset load, no filesystem lookup
            # (Req 4.1, 11.1, 12.3).
            m_create.assert_not_called()
            m_load.assert_not_called()
            m_exists.assert_not_called()


class TestProperty2Examples:
    """Concrete examples anchoring Property 2.

    **Validates: Requirements 4.1, 4.9, 11.1, 11.3, 11.4, 12.3**
    """

    def test_many_violations_all_reported_at_once(self):
        # Feature: generic-evaluation-script, Property 2: Validation aggregates all violations
        """Seven simultaneous violations yield seven bullets in one error."""
        config = {
            "model": {"config": {}},  # model.type + num_classes missing -> 2
            "dataset": {},  # dataset.path missing -> 1
            "evaluation": {
                "split": "bogus",  # enum -> 1
                "confidence_threshold": 1.5,  # range -> 1
                "iou_threshold": -0.2,  # range -> 1
            },
            "checkpoint": {},  # neither path nor run_id -> 1
        }

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)

        exc = exc_info.value
        bullets = _bullets(str(exc))
        assert len(bullets) == 7
        assert len(exc.violations) == 7

        message = str(exc)
        for param in [
            "model.type",
            "model.config.num_classes",
            "dataset.path",
            "evaluation.split",
            "evaluation.confidence_threshold",
            "evaluation.iou_threshold",
            "checkpoint.path / checkpoint.run_id",
        ]:
            assert param in message
        # Observed values and expected ranges/allowed values are present.
        assert "1.5" in message
        assert "-0.2" in message
        assert "[0.0, 1.0]" in message
        assert _ALLOWED_SPLITS_REPR in message
        assert "'bogus'" in message

    def test_single_violation_single_bullet(self):
        # Feature: generic-evaluation-script, Property 2: Validation aggregates all violations
        """A config valid except for the checkpoint section yields one bullet."""
        config = {
            "model": {"type": "yolo26", "config": {"num_classes": 5}},
            "dataset": {"path": "data/rdd2022"},
            "evaluation": {"split": "val"},
            "checkpoint": {"path": "a.pt", "run_id": "b"},  # both -> mutually exclusive
        }

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)

        exc = exc_info.value
        bullets = _bullets(str(exc))
        assert len(bullets) == 1
        assert len(exc.violations) == 1
        assert "mutually exclusive" in bullets[0]

    def test_fully_valid_config_does_not_raise(self):
        # Feature: generic-evaluation-script, Property 2: Validation aggregates all violations
        """A configuration with no violations passes validation silently."""
        config = {
            "model": {"type": "yolo26", "config": {"num_classes": 5}},
            "dataset": {"path": "data/rdd2022"},
            "evaluation": {
                "split": "test",
                "confidence_threshold": 0.25,
                "iou_threshold": 0.5,
            },
            "checkpoint": {"run_id": "run-123"},
        }

        # Should not raise.
        assert validate_config(config) is None
