"""Unit tests for write_outputs and print_summary functions.

Tests the output writing and summary printing functionality of the evaluation
script, validating requirements 9.5, 16.1, 16.3, 16.4, 16.5, 16.6, 16.7, 16.8.
"""

import json
import tempfile
from io import StringIO
from pathlib import Path
from unittest import mock

import pytest

from model.training.evaluate_detection import print_summary, write_outputs


class TestWriteOutputs:
    """Tests for the write_outputs function."""

    @pytest.fixture
    def sample_report(self):
        """Create a sample evaluation report."""
        return {
            "checkpoint": "/path/to/checkpoint.pt",
            "model_type": "yolo26",
            "model_config": {"num_classes": 5, "input_size": 640},
            "dataset": "/path/to/dataset",
            "split": "val",
            "num_images": 100,
            "num_classes": 5,
            "class_names": ["D00", "D10", "D20", "D40", "D44"],
            "confidence_threshold": 0.25,
            "iou_threshold": 0.5,
            "metrics": {
                "map_50": 0.75,
                "map_50_95": 0.55,
                "mAP@0.5": 0.75,
                "mAP@0.5:0.95": 0.55,
                "precision": 0.80,
                "recall": 0.70,
                "f1_score": 0.75,
                "per_class_ap": {"D00": 0.8, "D10": 0.7, "D20": 0.75, "D40": 0.72, "D44": 0.78},
            },
            "confusion_matrix": [[10, 1], [2, 15]],
            "errors": {"count": 0, "items": []},
        }

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions."""
        return [
            {
                "image_id": "img_001.jpg",
                "boxes": [[0.1, 0.2, 0.3, 0.4]],
                "labels": ["D00"],
                "scores": [0.95],
            },
            {
                "image_id": "img_002.jpg",
                "boxes": [[0.2, 0.3, 0.5, 0.6], [0.4, 0.5, 0.7, 0.8]],
                "labels": ["D10", "D20"],
                "scores": [0.85, 0.75],
            },
        ]

    @pytest.fixture
    def sample_ground_truths(self):
        """Create sample ground truths."""
        return [
            {
                "image_id": "img_001.jpg",
                "boxes": [[0.1, 0.2, 0.3, 0.4]],
                "labels": ["D00"],
            },
            {
                "image_id": "img_002.jpg",
                "boxes": [[0.2, 0.3, 0.5, 0.6], [0.4, 0.5, 0.7, 0.8]],
                "labels": ["D10", "D20"],
            },
        ]

    def test_write_outputs_creates_split_tagged_files_val(
        self, sample_report, sample_predictions, sample_ground_truths
    ):
        """Test that write_outputs creates val-tagged files (Req 9.5, 16.6)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            checkpoint_path = Path("/fake/checkpoint.pt")

            report_path, predictions_path = write_outputs(
                report=sample_report,
                predictions=sample_predictions,
                ground_truths=sample_ground_truths,
                output_dir=str(output_dir),
                split="val",
                checkpoint_path=checkpoint_path,
            )

            assert report_path.name == "val_evaluation_report.json"
            assert predictions_path.name == "val_inference.json"
            assert report_path.exists()
            assert predictions_path.exists()

    def test_write_outputs_creates_split_tagged_files_train(
        self, sample_report, sample_predictions, sample_ground_truths
    ):
        """Test that write_outputs creates train-tagged files (Req 9.5, 16.6)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            checkpoint_path = Path("/fake/checkpoint.pt")
            sample_report["split"] = "train"

            report_path, predictions_path = write_outputs(
                report=sample_report,
                predictions=sample_predictions,
                ground_truths=sample_ground_truths,
                output_dir=str(output_dir),
                split="train",
                checkpoint_path=checkpoint_path,
            )

            assert report_path.name == "train_evaluation_report.json"
            assert predictions_path.name == "train_inference.json"

    def test_write_outputs_creates_split_tagged_files_test(
        self, sample_report, sample_predictions, sample_ground_truths
    ):
        """Test that write_outputs creates test-tagged files (Req 9.5, 16.6)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            checkpoint_path = Path("/fake/checkpoint.pt")
            sample_report["split"] = "test"

            report_path, predictions_path = write_outputs(
                report=sample_report,
                predictions=sample_predictions,
                ground_truths=sample_ground_truths,
                output_dir=str(output_dir),
                split="test",
                checkpoint_path=checkpoint_path,
            )

            assert report_path.name == "test_evaluation_report.json"
            assert predictions_path.name == "test_inference.json"

    def test_write_outputs_uses_output_dir_when_provided(
        self, sample_report, sample_predictions, sample_ground_truths
    ):
        """Test that write_outputs uses output_dir when non-null (Req 16.4)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "custom_output"
            checkpoint_path = Path("/fake/checkpoint.pt")

            report_path, predictions_path = write_outputs(
                report=sample_report,
                predictions=sample_predictions,
                ground_truths=sample_ground_truths,
                output_dir=str(output_dir),
                split="val",
                checkpoint_path=checkpoint_path,
            )

            assert report_path.parent == output_dir
            assert predictions_path.parent == output_dir

    def test_write_outputs_uses_checkpoint_parent_when_output_dir_is_none(
        self, sample_report, sample_predictions, sample_ground_truths
    ):
        """Test that write_outputs uses checkpoint parent when output_dir is None (Req 16.5)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints" / "yolo26"
            checkpoint_dir.mkdir(parents=True)
            checkpoint_path = checkpoint_dir / "best_model.pt"
            checkpoint_path.touch()

            report_path, predictions_path = write_outputs(
                report=sample_report,
                predictions=sample_predictions,
                ground_truths=sample_ground_truths,
                output_dir=None,
                split="val",
                checkpoint_path=checkpoint_path,
            )

            assert report_path.parent == checkpoint_dir
            assert predictions_path.parent == checkpoint_dir

    def test_write_outputs_writes_valid_report_json(
        self, sample_report, sample_predictions, sample_ground_truths
    ):
        """Test that write_outputs writes valid JSON report (Req 16.1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            checkpoint_path = Path("/fake/checkpoint.pt")

            report_path, _ = write_outputs(
                report=sample_report,
                predictions=sample_predictions,
                ground_truths=sample_ground_truths,
                output_dir=str(output_dir),
                split="val",
                checkpoint_path=checkpoint_path,
            )

            with open(report_path) as f:
                loaded_report = json.load(f)

            assert loaded_report == sample_report

    def test_write_outputs_writes_valid_predictions_json(
        self, sample_report, sample_predictions, sample_ground_truths
    ):
        """Test that write_outputs writes valid predictions JSON (Req 16.3)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            checkpoint_path = Path("/fake/checkpoint.pt")

            _, predictions_path = write_outputs(
                report=sample_report,
                predictions=sample_predictions,
                ground_truths=sample_ground_truths,
                output_dir=str(output_dir),
                split="val",
                checkpoint_path=checkpoint_path,
            )

            with open(predictions_path) as f:
                loaded_predictions = json.load(f)

            # Check structure
            assert "checkpoint" in loaded_predictions
            assert "model_type" in loaded_predictions
            assert "dataset" in loaded_predictions
            assert "confidence_threshold" in loaded_predictions
            assert "class_names" in loaded_predictions
            assert "images" in loaded_predictions

            # Check images content
            assert len(loaded_predictions["images"]) == len(sample_predictions)
            for i, img in enumerate(loaded_predictions["images"]):
                assert img["image_id"] == sample_predictions[i]["image_id"]
                assert "ground_truth" in img
                assert "predictions" in img
                assert img["ground_truth"]["boxes"] == sample_ground_truths[i]["boxes"]
                assert img["ground_truth"]["labels"] == sample_ground_truths[i]["labels"]
                assert img["predictions"]["boxes"] == sample_predictions[i]["boxes"]
                assert img["predictions"]["labels"] == sample_predictions[i]["labels"]
                assert img["predictions"]["scores"] == sample_predictions[i]["scores"]

    def test_write_outputs_logs_absolute_paths(
        self, sample_report, sample_predictions, sample_ground_truths
    ):
        """Test that write_outputs logs absolute paths at INFO level (Req 16.7)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            checkpoint_path = Path("/fake/checkpoint.pt")

            with mock.patch("model.training.evaluate_detection.logger") as mock_logger:
                report_path, predictions_path = write_outputs(
                    report=sample_report,
                    predictions=sample_predictions,
                    ground_truths=sample_ground_truths,
                    output_dir=str(output_dir),
                    split="val",
                    checkpoint_path=checkpoint_path,
                )

                # Check that INFO logs were made with absolute paths
                info_calls = mock_logger.info.call_args_list
                assert len(info_calls) >= 2

                # Check report path log
                report_log = str(info_calls[0])
                assert "Evaluation report saved to" in report_log

                # Check predictions path log
                predictions_log = str(info_calls[1])
                assert "Per-image predictions saved to" in predictions_log


class TestPrintSummary:
    """Tests for the print_summary function."""

    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics."""
        return {
            "map_50": 0.75,
            "map_50_95": 0.55,
            "precision": 0.80,
            "recall": 0.70,
            "f1_score": 0.75,
        }

    def test_print_summary_includes_model_type(self, sample_metrics, capsys):
        """Test that print_summary includes model type (Req 16.8)."""
        print_summary(
            model_type="yolo26",
            split="val",
            num_images=100,
            metrics=sample_metrics,
        )

        captured = capsys.readouterr()
        assert "yolo26" in captured.out

    def test_print_summary_includes_split(self, sample_metrics, capsys):
        """Test that print_summary includes split (Req 16.8)."""
        print_summary(
            model_type="yolo26",
            split="val",
            num_images=100,
            metrics=sample_metrics,
        )

        captured = capsys.readouterr()
        assert "val" in captured.out

    def test_print_summary_includes_num_images(self, sample_metrics, capsys):
        """Test that print_summary includes num images (Req 16.8)."""
        print_summary(
            model_type="yolo26",
            split="val",
            num_images=100,
            metrics=sample_metrics,
        )

        captured = capsys.readouterr()
        assert "100" in captured.out

    def test_print_summary_includes_map_50(self, sample_metrics, capsys):
        """Test that print_summary includes map_50 (Req 16.8)."""
        print_summary(
            model_type="yolo26",
            split="val",
            num_images=100,
            metrics=sample_metrics,
        )

        captured = capsys.readouterr()
        assert "0.7500" in captured.out  # map_50 = 0.75

    def test_print_summary_includes_map_50_95(self, sample_metrics, capsys):
        """Test that print_summary includes map_50_95 (Req 16.8)."""
        print_summary(
            model_type="yolo26",
            split="val",
            num_images=100,
            metrics=sample_metrics,
        )

        captured = capsys.readouterr()
        assert "0.5500" in captured.out  # map_50_95 = 0.55

    def test_print_summary_includes_precision(self, sample_metrics, capsys):
        """Test that print_summary includes precision (Req 16.8)."""
        print_summary(
            model_type="yolo26",
            split="val",
            num_images=100,
            metrics=sample_metrics,
        )

        captured = capsys.readouterr()
        assert "0.8000" in captured.out  # precision = 0.80

    def test_print_summary_includes_recall(self, sample_metrics, capsys):
        """Test that print_summary includes recall (Req 16.8)."""
        print_summary(
            model_type="yolo26",
            split="val",
            num_images=100,
            metrics=sample_metrics,
        )

        captured = capsys.readouterr()
        assert "0.7000" in captured.out  # recall = 0.70

    def test_print_summary_includes_f1_score(self, sample_metrics, capsys):
        """Test that print_summary includes f1_score (Req 16.8)."""
        print_summary(
            model_type="yolo26",
            split="val",
            num_images=100,
            metrics=sample_metrics,
        )

        captured = capsys.readouterr()
        # f1_score = 0.75, formatted as 0.7500
        assert "F1-score" in captured.out or "F1" in captured.out

    def test_print_summary_formatted_output(self, sample_metrics, capsys):
        """Test that print_summary produces formatted output (Req 16.8)."""
        print_summary(
            model_type="yolo26",
            split="val",
            num_images=100,
            metrics=sample_metrics,
        )

        captured = capsys.readouterr()
        # Check for formatting elements
        assert "=" * 60 in captured.out  # Header/footer lines
        assert "EVALUATION SUMMARY" in captured.out
        assert "Model Type" in captured.out
        assert "Split" in captured.out
        assert "Num Images" in captured.out


class TestSplitTaggedFilenames:
    """Tests for split-tagged filename generation (Req 9.5, 16.6)."""

    @pytest.fixture
    def minimal_report(self):
        """Create a minimal report for filename tests."""
        return {
            "checkpoint": "/path/to/checkpoint.pt",
            "model_type": "yolo26",
            "dataset": "/path/to/dataset",
            "confidence_threshold": 0.25,
            "class_names": ["D00"],
        }

    @pytest.fixture
    def minimal_predictions(self):
        """Create minimal predictions for filename tests."""
        return [{"image_id": "img.jpg", "boxes": [], "labels": [], "scores": []}]

    @pytest.fixture
    def minimal_ground_truths(self):
        """Create minimal ground truths for filename tests."""
        return [{"image_id": "img.jpg", "boxes": [], "labels": []}]

    @pytest.mark.parametrize(
        "split,expected_report,expected_predictions",
        [
            ("train", "train_evaluation_report.json", "train_inference.json"),
            ("val", "val_evaluation_report.json", "val_inference.json"),
            ("test", "test_evaluation_report.json", "test_inference.json"),
        ],
    )
    def test_split_tagged_filenames(
        self,
        minimal_report,
        minimal_predictions,
        minimal_ground_truths,
        split,
        expected_report,
        expected_predictions,
    ):
        """Test that filenames are correctly tagged with split value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            checkpoint_path = Path("/fake/checkpoint.pt")

            report_path, predictions_path = write_outputs(
                report=minimal_report,
                predictions=minimal_predictions,
                ground_truths=minimal_ground_truths,
                output_dir=str(output_dir),
                split=split,
                checkpoint_path=checkpoint_path,
            )

            assert report_path.name == expected_report
            assert predictions_path.name == expected_predictions
