"""Property-based tests for checkpoint resolution.

Feature: generic-evaluation-script
Property 7: Checkpoint resolution

For any checkpoint configuration:

* when ``checkpoint.path`` is set, resolution returns exactly that path (and
  performs no filesystem probe -- existence is deferred to the loading stage);
* when ``checkpoint.run_id`` is set, resolution returns
  ``<checkpoint_dir>/<model.type>/<run_id>/best_model.pt`` when that file
  exists, otherwise ``<...>/last_model.pt`` when only that exists; and
* when neither candidate exists it raises a ``FileNotFoundError`` whose message
  lists every searched candidate path in search order (``best_model.pt`` before
  ``last_model.pt``).

These tests exercise the real ``resolve_checkpoint`` function in
``model/training/evaluate_detection.py``.

For the ``run_id`` cases, real ``best_model.pt`` / ``last_model.pt`` files are
created on disk so that ``Path.exists`` reflects genuine filesystem state. Each
Hypothesis example builds an isolated temporary ``checkpoint_dir`` via
``tempfile.TemporaryDirectory`` (rather than the function-scoped ``tmp_path``
fixture, which would be shared across examples), and the four file-presence
scenarios -- ``both``, ``best_only``, ``last_only``, ``none`` -- are varied so
the full resolution/fallback/raise behaviour is covered. The example-based unit
tests at the bottom use the ``tmp_path`` fixture directly.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 12.1**
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from model.training.evaluate_detection import resolve_checkpoint


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Safe single path segments: non-empty, no path separators or NUL, and never a
# relative-navigation token, so ``<dir>/<model_type>/<run_id>`` is a single,
# deterministic directory regardless of platform.
_SAFE_SEGMENT_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789-_"
)

_SAFE_SEGMENTS = st.text(
    alphabet=_SAFE_SEGMENT_ALPHABET, min_size=1, max_size=24
).filter(lambda s: s not in (".", ".."))

# Which of the two candidate files exist for a run_id resolution.
_FILE_SCENARIOS = st.sampled_from(["both", "best_only", "last_only", "none"])

# Arbitrary direct-path strings (may or may not exist on disk).
_PATH_STRINGS = st.text(
    alphabet=_SAFE_SEGMENT_ALPHABET + "/.", min_size=1, max_size=40
).filter(lambda s: s.strip() != "")


def _fail_on_exists(*args, **kwargs):  # pragma: no cover - only runs on regression
    """Stand-in for ``Path.exists`` that fails loudly if resolution calls it."""
    raise AssertionError(
        "resolve_checkpoint attempted a filesystem lookup (Path.exists) while "
        "resolving a direct checkpoint.path; the path must be returned verbatim "
        "without probing the filesystem (Requirement 5.1)."
    )


# ---------------------------------------------------------------------------
# Property 7
# ---------------------------------------------------------------------------


class TestProperty7CheckpointResolution:
    """Property 7: Checkpoint resolution.

    **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 12.1**
    """

    @given(path_value=_PATH_STRINGS)
    @settings(max_examples=100)
    def test_direct_path_returned_verbatim(self, path_value):
        # Feature: generic-evaluation-script, Property 7: Checkpoint resolution
        """A non-null ``checkpoint.path`` is returned verbatim, with no FS probe.

        ``Path.exists`` is patched to raise for the whole call, so any filesystem
        probe would surface as an ``AssertionError`` rather than the expected
        return value.

        **Validates: Requirements 5.1**
        """
        config = {
            "model": {"type": "yolo26"},
            "checkpoint": {"path": path_value},
        }

        with patch.object(Path, "exists", _fail_on_exists):
            resolved = resolve_checkpoint(config)

        assert resolved == Path(path_value)

    @given(path_value=_PATH_STRINGS, run_id=_SAFE_SEGMENTS)
    @settings(max_examples=100)
    def test_direct_path_takes_precedence_over_run_id(self, path_value, run_id):
        # Feature: generic-evaluation-script, Property 7: Checkpoint resolution
        """When both are present, ``checkpoint.path`` wins and the FS is untouched.

        **Validates: Requirements 5.1**
        """
        config = {
            "model": {"type": "yolo26"},
            "checkpoint": {"path": path_value, "run_id": run_id},
        }

        with patch.object(Path, "exists", _fail_on_exists):
            resolved = resolve_checkpoint(config)

        assert resolved == Path(path_value)

    @given(
        model_type=_SAFE_SEGMENTS,
        run_id=_SAFE_SEGMENTS,
        scenario=_FILE_SCENARIOS,
    )
    @settings(max_examples=100)
    def test_run_id_resolution(self, model_type, run_id, scenario):
        # Feature: generic-evaluation-script, Property 7: Checkpoint resolution
        """run_id resolves to best_model.pt, falls back to last_model.pt, or raises.

        Real files are written under an isolated temporary ``checkpoint_dir`` so
        ``Path.exists`` reflects genuine filesystem state.

        **Validates: Requirements 5.2, 5.3, 5.4, 12.1**
        """
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            run_dir = Path(checkpoint_dir) / model_type / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            best = run_dir / "best_model.pt"
            last = run_dir / "last_model.pt"

            if scenario in ("both", "best_only"):
                best.write_bytes(b"best")
            if scenario in ("both", "last_only"):
                last.write_bytes(b"last")

            config = {
                "model": {"type": model_type},
                "checkpoint": {"run_id": run_id, "checkpoint_dir": checkpoint_dir},
            }

            if scenario in ("both", "best_only"):
                # Req 5.2: best_model.pt is preferred when it exists.
                assert resolve_checkpoint(config) == best
            elif scenario == "last_only":
                # Req 5.3: fall back to last_model.pt when only it exists.
                assert resolve_checkpoint(config) == last
            else:
                # Req 5.4, 12.1: neither exists -> FileNotFoundError listing both
                # searched paths in search order (best before last).
                with pytest.raises(FileNotFoundError) as exc_info:
                    resolve_checkpoint(config)

                message = str(exc_info.value)
                best_idx = message.find(str(best))
                last_idx = message.find(str(last))
                assert best_idx != -1, "best_model.pt path missing from message"
                assert last_idx != -1, "last_model.pt path missing from message"
                # Search order: best_model.pt is listed before last_model.pt.
                assert best_idx < last_idx


# ---------------------------------------------------------------------------
# Example-based unit tests (use the tmp_path fixture directly)
# ---------------------------------------------------------------------------


class TestCheckpointResolutionExamples:
    """Concrete examples complementing Property 7.

    **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 12.1**
    """

    def test_best_preferred_over_last_when_both_exist(self, tmp_path):
        """With both files present, best_model.pt is chosen. (Req 5.2)"""
        run_dir = tmp_path / "yolo26" / "run-abc"
        run_dir.mkdir(parents=True)
        best = run_dir / "best_model.pt"
        last = run_dir / "last_model.pt"
        best.write_bytes(b"best")
        last.write_bytes(b"last")

        config = {
            "model": {"type": "yolo26"},
            "checkpoint": {"run_id": "run-abc", "checkpoint_dir": str(tmp_path)},
        }

        assert resolve_checkpoint(config) == best

    def test_fallback_to_last_when_only_last_exists(self, tmp_path):
        """With only last_model.pt present, it is returned. (Req 5.3)"""
        run_dir = tmp_path / "ssd_mobilenetv3" / "run-xyz"
        run_dir.mkdir(parents=True)
        last = run_dir / "last_model.pt"
        last.write_bytes(b"last")

        config = {
            "model": {"type": "ssd_mobilenetv3"},
            "checkpoint": {"run_id": "run-xyz", "checkpoint_dir": str(tmp_path)},
        }

        assert resolve_checkpoint(config) == last

    def test_missing_both_lists_searched_paths_in_order(self, tmp_path):
        """No candidate exists -> FileNotFoundError lists both paths, best first.

        **Validates: Requirements 5.4, 12.1**
        """
        config = {
            "model": {"type": "yolo26"},
            "checkpoint": {"run_id": "run-missing", "checkpoint_dir": str(tmp_path)},
        }

        with pytest.raises(FileNotFoundError) as exc_info:
            resolve_checkpoint(config)

        message = str(exc_info.value)
        expected_best = tmp_path / "yolo26" / "run-missing" / "best_model.pt"
        expected_last = tmp_path / "yolo26" / "run-missing" / "last_model.pt"
        best_idx = message.find(str(expected_best))
        last_idx = message.find(str(expected_last))
        assert best_idx != -1
        assert last_idx != -1
        assert best_idx < last_idx

    def test_default_checkpoint_dir_is_used_when_absent(self):
        """An absent checkpoint_dir defaults to ./checkpoints in the search path.

        No files are created here; neither candidate exists, so the searched
        paths surface in the FileNotFoundError message under ``checkpoints``.

        **Validates: Requirements 5.2, 5.4**
        """
        config = {
            "model": {"type": "yolo26"},
            "checkpoint": {"run_id": "no-such-run"},
        }

        with pytest.raises(FileNotFoundError) as exc_info:
            resolve_checkpoint(config)

        message = str(exc_info.value)
        default_best = Path("./checkpoints") / "yolo26" / "no-such-run" / "best_model.pt"
        assert str(default_best) in message

    def test_direct_path_returned_even_when_absent(self):
        """A direct path that does not exist is still returned verbatim. (Req 5.1)"""
        config = {"checkpoint": {"path": "/does/not/exist/model.pt"}}
        assert resolve_checkpoint(config) == Path("/does/not/exist/model.pt")
