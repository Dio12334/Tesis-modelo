"""Property-based tests for the checkpoint exclusive-or validation rule.

Feature: generic-evaluation-script
Property 6: Checkpoint section is an exclusive-or

For any configuration, the checkpoint section validates cleanly if and only if
exactly one of ``checkpoint.path`` and ``checkpoint.run_id`` is present and
non-null:

* providing **both** yields a mutual-exclusivity violation,
* providing **neither** yields a "one of them is required" violation, and
* in **both** failing cases no filesystem lookup is attempted for either value.

These tests exercise the real ``validate_config`` function in
``model/training/evaluate_detection.py``. The four presence/absence/null
combinations of ``checkpoint.path`` and ``checkpoint.run_id`` are explored while
every other required parameter (``model.type``, ``model.config.num_classes``,
``dataset.path``, ``evaluation.split``) is held valid, so the only possible
violation is the checkpoint one. To prove the rule is decided purely from key
presence (Requirement 12.3), ``pathlib.Path.exists`` is patched to raise for the
duration of every ``validate_config`` call: if validation touched the
filesystem, the patched ``exists`` would raise ``AssertionError`` and the test
would fail rather than observe the expected ``ConfigurationError`` (or clean
return).

**Validates: Requirements 4.6, 4.7, 4.8**
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from model.exceptions import ConfigurationError
from model.training.evaluate_detection import validate_config


# ---------------------------------------------------------------------------
# Filesystem guard: validate_config must never probe the disk (Req 12.3)
# ---------------------------------------------------------------------------


def _fail_on_exists(*args, **kwargs):  # pragma: no cover - only runs on regression
    """Stand-in for ``Path.exists`` that fails loudly if validation calls it."""
    raise AssertionError(
        "validate_config attempted a filesystem lookup (Path.exists) while "
        "deciding the checkpoint exclusive-or; the rule must be decided purely "
        "from key presence (Requirement 12.3)."
    )


def _validate_without_filesystem(config: dict) -> None:
    """Run ``validate_config`` with ``Path.exists`` patched to raise.

    The patch is scoped to the single ``validate_config`` call so it cannot
    interfere with Hypothesis' example database or pytest's own filesystem use.
    """
    with patch.object(Path, "exists", _fail_on_exists):
        validate_config(config)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Three independent states for each checkpoint key. "null" and "absent" both
# count as "not provided" per the implementation (a present-but-None value is
# treated as missing); only "present" supplies a non-null value.
_KEY_STATES = st.sampled_from(["absent", "null", "present"])

# Any non-null value makes a key "present"; the rule keys off presence, not type.
_PRESENT_VALUES = st.one_of(
    st.text(min_size=1, max_size=20),
    st.integers(),
)


def _valid_base() -> dict:
    """A fully valid config whose ONLY potential violation is the checkpoint rule."""
    return {
        "model": {"type": "yolo26", "config": {"num_classes": 5}},
        "dataset": {"path": "data/rdd2022"},
        "evaluation": {"split": "val"},
    }


def _build_checkpoint_section(path_state, path_value, run_id_state, run_id_value):
    """Construct a ``checkpoint`` dict for the given per-key states.

    Returns ``(checkpoint_dict, has_path, has_run_id)`` where the booleans
    describe whether each key is "present and non-null".
    """
    checkpoint: dict = {}
    if path_state == "null":
        checkpoint["path"] = None
    elif path_state == "present":
        checkpoint["path"] = path_value

    if run_id_state == "null":
        checkpoint["run_id"] = None
    elif run_id_state == "present":
        checkpoint["run_id"] = run_id_value

    has_path = path_state == "present"
    has_run_id = run_id_state == "present"
    return checkpoint, has_path, has_run_id


def _checkpoint_violations(violations):
    """Return only the violation strings produced by the checkpoint XOR rule."""
    return [v for v in violations if "checkpoint.path / checkpoint.run_id" in v]


# ---------------------------------------------------------------------------
# Property 6
# ---------------------------------------------------------------------------


class TestProperty6CheckpointExclusiveOr:
    """Property 6: Checkpoint section is an exclusive-or.

    **Validates: Requirements 4.6, 4.7, 4.8**
    """

    @given(
        path_state=_KEY_STATES,
        run_id_state=_KEY_STATES,
        path_value=_PRESENT_VALUES,
        run_id_value=_PRESENT_VALUES,
    )
    @settings(max_examples=100)
    def test_xor_outcome_matches_presence(
        self, path_state, run_id_state, path_value, run_id_value
    ):
        # Feature: generic-evaluation-script, Property 6: Checkpoint exclusive-or
        """Clean iff exactly one key is present; the two failing modes otherwise.

        Also proves no filesystem lookup occurs: ``Path.exists`` is patched to
        raise for the whole ``validate_config`` call, so any disk probe would
        surface as an ``AssertionError`` rather than the expected outcome.

        **Validates: Requirements 4.6, 4.7, 4.8**
        """
        checkpoint, has_path, has_run_id = _build_checkpoint_section(
            path_state, path_value, run_id_state, run_id_value
        )
        config = _valid_base()
        config["checkpoint"] = checkpoint

        if has_path and has_run_id:
            # Req 4.6: both provided -> mutual-exclusivity violation.
            with pytest.raises(ConfigurationError) as exc_info:
                _validate_without_filesystem(config)
            ckpt_violations = _checkpoint_violations(exc_info.value.violations)
            # The base is valid, so the checkpoint rule is the sole violation.
            assert exc_info.value.violations == ckpt_violations
            assert len(ckpt_violations) == 1
            assert "mutually exclusive" in ckpt_violations[0]
            assert "one of them is required" not in ckpt_violations[0]
        elif not has_path and not has_run_id:
            # Req 4.7: neither provided -> "one of them is required" violation.
            with pytest.raises(ConfigurationError) as exc_info:
                _validate_without_filesystem(config)
            ckpt_violations = _checkpoint_violations(exc_info.value.violations)
            assert exc_info.value.violations == ckpt_violations
            assert len(ckpt_violations) == 1
            assert "one of them is required" in ckpt_violations[0]
            assert "mutually exclusive" not in ckpt_violations[0]
        else:
            # Req 4.8: exactly one provided -> validates cleanly (no raise).
            _validate_without_filesystem(config)

    @given(present_value=_PRESENT_VALUES, null_via_key=st.booleans())
    @settings(max_examples=100)
    def test_only_path_present_is_clean(self, present_value, null_via_key):
        # Feature: generic-evaluation-script, Property 6: Checkpoint exclusive-or
        """A non-null ``checkpoint.path`` alone (run_id absent or null) validates.

        **Validates: Requirements 4.8**
        """
        config = _valid_base()
        checkpoint = {"path": present_value}
        if null_via_key:
            # run_id present but null still counts as "not provided".
            checkpoint["run_id"] = None
        config["checkpoint"] = checkpoint

        # Must not raise and must not touch the filesystem.
        _validate_without_filesystem(config)

    @given(present_value=_PRESENT_VALUES, null_via_key=st.booleans())
    @settings(max_examples=100)
    def test_only_run_id_present_is_clean(self, present_value, null_via_key):
        # Feature: generic-evaluation-script, Property 6: Checkpoint exclusive-or
        """A non-null ``checkpoint.run_id`` alone (path absent or null) validates.

        **Validates: Requirements 4.8**
        """
        config = _valid_base()
        checkpoint = {"run_id": present_value}
        if null_via_key:
            checkpoint["path"] = None
        config["checkpoint"] = checkpoint

        _validate_without_filesystem(config)

    def test_missing_checkpoint_section_requires_one(self):
        # Feature: generic-evaluation-script, Property 6: Checkpoint exclusive-or
        """An entirely absent checkpoint section yields the "one is required" rule.

        **Validates: Requirements 4.7**
        """
        config = _valid_base()  # no "checkpoint" key at all

        with pytest.raises(ConfigurationError) as exc_info:
            _validate_without_filesystem(config)

        ckpt_violations = _checkpoint_violations(exc_info.value.violations)
        assert len(ckpt_violations) == 1
        assert "one of them is required" in ckpt_violations[0]

    def test_both_null_requires_one(self):
        # Feature: generic-evaluation-script, Property 6: Checkpoint exclusive-or
        """Both keys present but null counts as neither provided ("one is required").

        **Validates: Requirements 4.7**
        """
        config = _valid_base()
        config["checkpoint"] = {"path": None, "run_id": None}

        with pytest.raises(ConfigurationError) as exc_info:
            _validate_without_filesystem(config)

        ckpt_violations = _checkpoint_violations(exc_info.value.violations)
        assert len(ckpt_violations) == 1
        assert "one of them is required" in ckpt_violations[0]

    def test_empty_string_path_counts_as_present(self):
        # Feature: generic-evaluation-script, Property 6: Checkpoint exclusive-or
        """An empty-string path is non-null, so it counts as the single provided key.

        **Validates: Requirements 4.8**
        """
        config = _valid_base()
        config["checkpoint"] = {"path": ""}  # "" is non-null -> present

        # Exactly one provided -> clean.
        _validate_without_filesystem(config)
