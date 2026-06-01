"""Property-based tests for output location selection.

Feature: generic-evaluation-script
Property 21: Output location selection

For any configuration:

* when ``evaluation.output_dir`` is non-null, outputs are written beneath that
  directory;
* when ``evaluation.output_dir`` is null, outputs are written in the checkpoint's
  parent directory.

These tests exercise the real ``write_outputs`` function in
``model/training/evaluate_detection.py``.

Each Hypothesis example builds an isolated temporary directory structure via
``tempfile.TemporaryDirectory`` so that filesystem operations are genuine and
isolated. The tests verify that the resolved output directory matches the
expected location based on the ``output_dir`` parameter.

**Validates: Requirements 16.4, 16.5**
"""

import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from model.training.evaluate_detection import write_outputs


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Safe single path segments: non-empty, no path separators or NUL, and never a
# relative-navigation token.
_SAFE_SEGMENT_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789-_"
)

_SAFE_SEGMENTS = st.text(
    alphabet=_SAFE_SEGMENT_ALPHABET, min_size=1, max_size=24
).filter(lambda s: s not in (".", ".."))

# Valid split values.
_SPLITS = st.sampled_from(["train", "val", "test"])

# Class names for the report.
_CLASS_NAMES = st.lists(
    st.text(alphabet=_SAFE_SEGMENT_ALPHABET, min_size=1, max_size=12),
    min_size=1,
    max_size=5,
    unique=True,
)


def _minimal_report(class_names: list) -> dict:
    """Build a minimal valid report dict for write_outputs."""
    return {
        "checkpoint": "/path/to/checkpoint.pt",
        "model_type": "yolo26",
        "model_config": {"num_classes": len(class_names)},
        "dataset": "/path/to/dataset",
        "split": "val",
        "num_images": 10,
        "num_classes": len(class_names),
        "class_names": class_names,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.5,
        "metrics": {
            "map_50": 0.5,
            "map_50_95": 0.4,
            "mAP@0.5": 0.5,
            "mAP@0.5:0.95": 0.4,
            "precision": 0.6,
            "recall": 0.7,
            "f1_score": 0.65,
            "per_class_ap": {name: 0.5 for name in class_names},
        },
        "confusion_matrix": [[1, 0], [0, 1]],
        "errors": {"count": 0, "items": []},
    }


def _minimal_predictions_and_gts(n_images: int = 2) -> tuple:
    """Build minimal aligned predictions and ground truths."""
    predictions = []
    ground_truths = []
    for i in range(n_images):
        image_id = f"image_{i}.jpg"
        predictions.append({
            "image_id": image_id,
            "boxes": [[0.1, 0.1, 0.2, 0.2]],
            "labels": ["crack"],
            "scores": [0.9],
        })
        ground_truths.append({
            "image_id": image_id,
            "boxes": [[0.1, 0.1, 0.2, 0.2]],
            "labels": ["crack"],
        })
    return predictions, ground_truths


# ---------------------------------------------------------------------------
# Property 21
# ---------------------------------------------------------------------------


class TestProperty21OutputLocationSelection:
    """Property 21: Output location selection.

    **Validates: Requirements 16.4, 16.5**
    """

    @given(
        output_subdir=_SAFE_SEGMENTS,
        checkpoint_subdir=_SAFE_SEGMENTS,
        split=_SPLITS,
        class_names=_CLASS_NAMES,
    )
    @settings(max_examples=100)
    def test_explicit_output_dir_is_used(
        self, output_subdir, checkpoint_subdir, split, class_names
    ):
        # Feature: generic-evaluation-script, Property 21: Output location selection
        """When ``output_dir`` is non-null, outputs are written beneath that directory.

        **Validates: Requirements 16.4**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create distinct output_dir and checkpoint parent directories.
            output_dir = tmp_path / "outputs" / output_subdir
            checkpoint_parent = tmp_path / "checkpoints" / checkpoint_subdir
            checkpoint_parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_parent / "best_model.pt"
            checkpoint_path.write_bytes(b"fake checkpoint")

            report = _minimal_report(class_names)
            predictions, ground_truths = _minimal_predictions_and_gts()

            report_path, predictions_path = write_outputs(
                report=report,
                predictions=predictions,
                ground_truths=ground_truths,
                output_dir=str(output_dir),
                split=split,
                checkpoint_path=checkpoint_path,
            )

            # Req 16.4: outputs are written beneath the explicit output_dir.
            assert report_path.parent == output_dir
            assert predictions_path.parent == output_dir

            # Verify files exist and are in the correct location.
            assert report_path.exists()
            assert predictions_path.exists()

            # Verify files are NOT in the checkpoint's parent directory.
            assert report_path.parent != checkpoint_parent
            assert predictions_path.parent != checkpoint_parent

    @given(
        checkpoint_subdir=_SAFE_SEGMENTS,
        split=_SPLITS,
        class_names=_CLASS_NAMES,
    )
    @settings(max_examples=100)
    def test_null_output_dir_uses_checkpoint_parent(
        self, checkpoint_subdir, split, class_names
    ):
        # Feature: generic-evaluation-script, Property 21: Output location selection
        """When ``output_dir`` is null, outputs are written in the checkpoint's parent.

        **Validates: Requirements 16.5**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create checkpoint parent directory.
            checkpoint_parent = tmp_path / "checkpoints" / checkpoint_subdir
            checkpoint_parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_parent / "best_model.pt"
            checkpoint_path.write_bytes(b"fake checkpoint")

            report = _minimal_report(class_names)
            predictions, ground_truths = _minimal_predictions_and_gts()

            report_path, predictions_path = write_outputs(
                report=report,
                predictions=predictions,
                ground_truths=ground_truths,
                output_dir=None,  # Null output_dir
                split=split,
                checkpoint_path=checkpoint_path,
            )

            # Req 16.5: outputs are written in the checkpoint's parent directory.
            assert report_path.parent == checkpoint_parent
            assert predictions_path.parent == checkpoint_parent

            # Verify files exist.
            assert report_path.exists()
            assert predictions_path.exists()

    @given(
        output_subdir=_SAFE_SEGMENTS,
        checkpoint_subdir=_SAFE_SEGMENTS,
        split=_SPLITS,
    )
    @settings(max_examples=100)
    def test_output_dir_takes_precedence_over_checkpoint_parent(
        self, output_subdir, checkpoint_subdir, split
    ):
        # Feature: generic-evaluation-script, Property 21: Output location selection
        """Explicit ``output_dir`` always takes precedence over checkpoint parent.

        Even when both directories exist, the explicit ``output_dir`` is used.

        **Validates: Requirements 16.4, 16.5**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create both directories.
            output_dir = tmp_path / "outputs" / output_subdir
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_parent = tmp_path / "checkpoints" / checkpoint_subdir
            checkpoint_parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_parent / "best_model.pt"
            checkpoint_path.write_bytes(b"fake checkpoint")

            report = _minimal_report(["crack", "pothole"])
            predictions, ground_truths = _minimal_predictions_and_gts()

            report_path, predictions_path = write_outputs(
                report=report,
                predictions=predictions,
                ground_truths=ground_truths,
                output_dir=str(output_dir),
                split=split,
                checkpoint_path=checkpoint_path,
            )

            # Explicit output_dir takes precedence.
            assert report_path.parent == output_dir
            assert predictions_path.parent == output_dir

    @given(split=_SPLITS)
    @settings(max_examples=100)
    def test_output_dir_created_if_not_exists(self, split):
        # Feature: generic-evaluation-script, Property 21: Output location selection
        """The output directory is created if it does not exist.

        **Validates: Requirements 16.4**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create a non-existent output directory path.
            output_dir = tmp_path / "new" / "nested" / "output"
            assert not output_dir.exists()

            checkpoint_parent = tmp_path / "checkpoints"
            checkpoint_parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_parent / "best_model.pt"
            checkpoint_path.write_bytes(b"fake checkpoint")

            report = _minimal_report(["crack"])
            predictions, ground_truths = _minimal_predictions_and_gts()

            report_path, predictions_path = write_outputs(
                report=report,
                predictions=predictions,
                ground_truths=ground_truths,
                output_dir=str(output_dir),
                split=split,
                checkpoint_path=checkpoint_path,
            )

            # The output directory should now exist.
            assert output_dir.exists()
            assert report_path.parent == output_dir
            assert predictions_path.parent == output_dir


# ---------------------------------------------------------------------------
# Example-based unit tests
# ---------------------------------------------------------------------------


class TestOutputLocationExamples:
    """Concrete examples complementing Property 21.

    **Validates: Requirements 16.4, 16.5**
    """

    def test_explicit_output_dir_used_for_report(self, tmp_path):
        """With explicit output_dir, report is written there. (Req 16.4)"""
        output_dir = tmp_path / "my_outputs"
        checkpoint_parent = tmp_path / "checkpoints" / "yolo26" / "run-123"
        checkpoint_parent.mkdir(parents=True)
        checkpoint_path = checkpoint_parent / "best_model.pt"
        checkpoint_path.write_bytes(b"fake")

        report = _minimal_report(["crack", "pothole"])
        predictions, ground_truths = _minimal_predictions_and_gts()

        report_path, predictions_path = write_outputs(
            report=report,
            predictions=predictions,
            ground_truths=ground_truths,
            output_dir=str(output_dir),
            split="val",
            checkpoint_path=checkpoint_path,
        )

        assert report_path == output_dir / "val_evaluation_report.json"
        assert predictions_path == output_dir / "val_inference.json"
        assert report_path.exists()
        assert predictions_path.exists()

    def test_null_output_dir_uses_checkpoint_parent_for_report(self, tmp_path):
        """With null output_dir, report is written in checkpoint's parent. (Req 16.5)"""
        checkpoint_parent = tmp_path / "checkpoints" / "yolo26" / "run-456"
        checkpoint_parent.mkdir(parents=True)
        checkpoint_path = checkpoint_parent / "best_model.pt"
        checkpoint_path.write_bytes(b"fake")

        report = _minimal_report(["crack", "pothole"])
        predictions, ground_truths = _minimal_predictions_and_gts()

        report_path, predictions_path = write_outputs(
            report=report,
            predictions=predictions,
            ground_truths=ground_truths,
            output_dir=None,
            split="test",
            checkpoint_path=checkpoint_path,
        )

        assert report_path == checkpoint_parent / "test_evaluation_report.json"
        assert predictions_path == checkpoint_parent / "test_inference.json"
        assert report_path.exists()
        assert predictions_path.exists()

    def test_report_content_is_valid_json(self, tmp_path):
        """Written report file contains valid JSON matching the input report."""
        output_dir = tmp_path / "outputs"
        checkpoint_path = tmp_path / "model.pt"
        checkpoint_path.write_bytes(b"fake")

        report = _minimal_report(["crack", "pothole", "spalling"])
        predictions, ground_truths = _minimal_predictions_and_gts(n_images=3)

        report_path, _ = write_outputs(
            report=report,
            predictions=predictions,
            ground_truths=ground_truths,
            output_dir=str(output_dir),
            split="train",
            checkpoint_path=checkpoint_path,
        )

        with open(report_path, "r", encoding="utf-8") as f:
            loaded_report = json.load(f)

        assert loaded_report["model_type"] == report["model_type"]
        assert loaded_report["class_names"] == report["class_names"]
        assert loaded_report["metrics"]["map_50"] == report["metrics"]["map_50"]

    def test_predictions_file_content_is_valid_json(self, tmp_path):
        """Written predictions file contains valid JSON with expected structure."""
        output_dir = tmp_path / "outputs"
        checkpoint_path = tmp_path / "model.pt"
        checkpoint_path.write_bytes(b"fake")

        report = _minimal_report(["crack", "pothole"])
        predictions, ground_truths = _minimal_predictions_and_gts(n_images=2)

        _, predictions_path = write_outputs(
            report=report,
            predictions=predictions,
            ground_truths=ground_truths,
            output_dir=str(output_dir),
            split="val",
            checkpoint_path=checkpoint_path,
        )

        with open(predictions_path, "r", encoding="utf-8") as f:
            loaded_predictions = json.load(f)

        assert loaded_predictions["model_type"] == report["model_type"]
        assert loaded_predictions["class_names"] == report["class_names"]
        assert len(loaded_predictions["images"]) == 2
        assert loaded_predictions["images"][0]["image_id"] == "image_0.jpg"

    def test_different_splits_produce_different_filenames(self, tmp_path):
        """Each split produces distinctly named files in the same directory."""
        output_dir = tmp_path / "outputs"
        checkpoint_path = tmp_path / "model.pt"
        checkpoint_path.write_bytes(b"fake")

        report = _minimal_report(["crack"])
        predictions, ground_truths = _minimal_predictions_and_gts()

        paths = {}
        for split in ["train", "val", "test"]:
            report_path, predictions_path = write_outputs(
                report=report,
                predictions=predictions,
                ground_truths=ground_truths,
                output_dir=str(output_dir),
                split=split,
                checkpoint_path=checkpoint_path,
            )
            paths[split] = (report_path, predictions_path)

        # All files should be in the same directory.
        for split, (rp, pp) in paths.items():
            assert rp.parent == output_dir
            assert pp.parent == output_dir

        # All filenames should be distinct.
        all_report_paths = [paths[s][0] for s in ["train", "val", "test"]]
        all_predictions_paths = [paths[s][1] for s in ["train", "val", "test"]]
        assert len(set(all_report_paths)) == 3
        assert len(set(all_predictions_paths)) == 3

    def test_deeply_nested_checkpoint_path(self, tmp_path):
        """Null output_dir works with deeply nested checkpoint paths. (Req 16.5)"""
        checkpoint_parent = tmp_path / "a" / "b" / "c" / "d" / "checkpoints"
        checkpoint_parent.mkdir(parents=True)
        checkpoint_path = checkpoint_parent / "model.pt"
        checkpoint_path.write_bytes(b"fake")

        report = _minimal_report(["crack"])
        predictions, ground_truths = _minimal_predictions_and_gts()

        report_path, predictions_path = write_outputs(
            report=report,
            predictions=predictions,
            ground_truths=ground_truths,
            output_dir=None,
            split="val",
            checkpoint_path=checkpoint_path,
        )

        assert report_path.parent == checkpoint_parent
        assert predictions_path.parent == checkpoint_parent
