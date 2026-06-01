"""Unit tests for stdout summary contents.

Feature: generic-evaluation-script
Task 11.8: Write unit test for stdout summary contents

These are example-based (NOT property-based) unit tests for the
``print_summary`` function in ``model/training/evaluate_detection.py``.

The tests verify that the printed summary contains:
- Model type
- Split
- Number of images
- The five metrics: map_50, map_50_95, precision, recall, and f1_score

_Requirements: 16.8_
"""

import io
import sys
from contextlib import redirect_stdout

import pytest

from model.training.evaluate_detection import print_summary


class TestPrintSummaryContents:
    """Verify that print_summary outputs all required information.

    _Requirements: 16.8_
    """

    @pytest.fixture
    def sample_metrics(self):
        """Return a sample metrics dict with all required fields."""
        return {
            "map_50": 0.7523,
            "map_50_95": 0.4812,
            "precision": 0.8234,
            "recall": 0.7891,
            "f1_score": 0.8058,
        }

    def test_summary_contains_model_type(self, sample_metrics):
        """The printed summary contains the model type. (Req 16.8)"""
        model_type = "yolo26"
        captured = io.StringIO()

        with redirect_stdout(captured):
            print_summary(
                model_type=model_type,
                split="val",
                num_images=100,
                metrics=sample_metrics,
            )

        output = captured.getvalue()
        assert model_type in output, (
            f"Expected model type '{model_type}' in summary output"
        )

    def test_summary_contains_split(self, sample_metrics):
        """The printed summary contains the split. (Req 16.8)"""
        split = "test"
        captured = io.StringIO()

        with redirect_stdout(captured):
            print_summary(
                model_type="ssd_mobilenet",
                split=split,
                num_images=100,
                metrics=sample_metrics,
            )

        output = captured.getvalue()
        assert split in output, f"Expected split '{split}' in summary output"

    def test_summary_contains_num_images(self, sample_metrics):
        """The printed summary contains the number of images. (Req 16.8)"""
        num_images = 1234
        captured = io.StringIO()

        with redirect_stdout(captured):
            print_summary(
                model_type="yolo26",
                split="val",
                num_images=num_images,
                metrics=sample_metrics,
            )

        output = captured.getvalue()
        assert str(num_images) in output, (
            f"Expected num_images '{num_images}' in summary output"
        )

    def test_summary_contains_map_50(self, sample_metrics):
        """The printed summary contains the map_50 metric. (Req 16.8)"""
        captured = io.StringIO()

        with redirect_stdout(captured):
            print_summary(
                model_type="yolo26",
                split="val",
                num_images=100,
                metrics=sample_metrics,
            )

        output = captured.getvalue()
        # The metric is formatted with 4 decimal places
        assert "0.7523" in output, "Expected map_50 value '0.7523' in summary output"
        # Also check for the label
        assert "mAP@0.5" in output, "Expected 'mAP@0.5' label in summary output"

    def test_summary_contains_map_50_95(self, sample_metrics):
        """The printed summary contains the map_50_95 metric. (Req 16.8)"""
        captured = io.StringIO()

        with redirect_stdout(captured):
            print_summary(
                model_type="yolo26",
                split="val",
                num_images=100,
                metrics=sample_metrics,
            )

        output = captured.getvalue()
        # The metric is formatted with 4 decimal places
        assert "0.4812" in output, "Expected map_50_95 value '0.4812' in summary output"
        # Also check for the label
        assert "mAP@0.5:0.95" in output, (
            "Expected 'mAP@0.5:0.95' label in summary output"
        )

    def test_summary_contains_precision(self, sample_metrics):
        """The printed summary contains the precision metric. (Req 16.8)"""
        captured = io.StringIO()

        with redirect_stdout(captured):
            print_summary(
                model_type="yolo26",
                split="val",
                num_images=100,
                metrics=sample_metrics,
            )

        output = captured.getvalue()
        # The metric is formatted with 4 decimal places
        assert "0.8234" in output, "Expected precision value '0.8234' in summary output"
        # Also check for the label (case-insensitive check)
        assert "Precision" in output or "precision" in output, (
            "Expected 'Precision' label in summary output"
        )

    def test_summary_contains_recall(self, sample_metrics):
        """The printed summary contains the recall metric. (Req 16.8)"""
        captured = io.StringIO()

        with redirect_stdout(captured):
            print_summary(
                model_type="yolo26",
                split="val",
                num_images=100,
                metrics=sample_metrics,
            )

        output = captured.getvalue()
        # The metric is formatted with 4 decimal places
        assert "0.7891" in output, "Expected recall value '0.7891' in summary output"
        # Also check for the label (case-insensitive check)
        assert "Recall" in output or "recall" in output, (
            "Expected 'Recall' label in summary output"
        )

    def test_summary_contains_f1_score(self, sample_metrics):
        """The printed summary contains the f1_score metric. (Req 16.8)"""
        captured = io.StringIO()

        with redirect_stdout(captured):
            print_summary(
                model_type="yolo26",
                split="val",
                num_images=100,
                metrics=sample_metrics,
            )

        output = captured.getvalue()
        # The metric is formatted with 4 decimal places
        assert "0.8058" in output, "Expected f1_score value '0.8058' in summary output"
        # Also check for the label (case-insensitive check)
        assert "F1" in output or "f1" in output, (
            "Expected 'F1' label in summary output"
        )


class TestPrintSummaryAllFieldsPresent:
    """Verify that a single call to print_summary includes all required fields.

    _Requirements: 16.8_
    """

    def test_all_required_fields_in_single_output(self):
        """All required fields appear in a single summary output. (Req 16.8)"""
        model_type = "yolov6"
        split = "train"
        num_images = 5000
        metrics = {
            "map_50": 0.6543,
            "map_50_95": 0.3987,
            "precision": 0.7654,
            "recall": 0.6789,
            "f1_score": 0.7195,
        }

        captured = io.StringIO()

        with redirect_stdout(captured):
            print_summary(
                model_type=model_type,
                split=split,
                num_images=num_images,
                metrics=metrics,
            )

        output = captured.getvalue()

        # Verify all required fields are present
        assert model_type in output, f"Missing model_type '{model_type}'"
        assert split in output, f"Missing split '{split}'"
        assert str(num_images) in output, f"Missing num_images '{num_images}'"
        assert "0.6543" in output, "Missing map_50 value"
        assert "0.3987" in output, "Missing map_50_95 value"
        assert "0.7654" in output, "Missing precision value"
        assert "0.6789" in output, "Missing recall value"
        assert "0.7195" in output, "Missing f1_score value"


class TestPrintSummaryDifferentSplits:
    """Verify print_summary works correctly for all split values.

    _Requirements: 16.8_
    """

    @pytest.fixture
    def sample_metrics(self):
        """Return a sample metrics dict with all required fields."""
        return {
            "map_50": 0.5000,
            "map_50_95": 0.3000,
            "precision": 0.6000,
            "recall": 0.5500,
            "f1_score": 0.5738,
        }

    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_summary_with_different_splits(self, split, sample_metrics):
        """The summary correctly displays each split value. (Req 16.8)"""
        captured = io.StringIO()

        with redirect_stdout(captured):
            print_summary(
                model_type="yolo26",
                split=split,
                num_images=100,
                metrics=sample_metrics,
            )

        output = captured.getvalue()
        assert split in output, f"Expected split '{split}' in summary output"


class TestPrintSummaryEdgeCases:
    """Verify print_summary handles edge cases correctly.

    _Requirements: 16.8_
    """

    def test_summary_with_zero_metrics(self):
        """The summary correctly displays zero metric values. (Req 16.8)"""
        metrics = {
            "map_50": 0.0,
            "map_50_95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

        captured = io.StringIO()

        with redirect_stdout(captured):
            print_summary(
                model_type="yolo26",
                split="val",
                num_images=0,
                metrics=metrics,
            )

        output = captured.getvalue()
        # Zero values should be formatted as 0.0000
        assert "0.0000" in output, "Expected zero metric values formatted as '0.0000'"

    def test_summary_with_perfect_metrics(self):
        """The summary correctly displays perfect (1.0) metric values. (Req 16.8)"""
        metrics = {
            "map_50": 1.0,
            "map_50_95": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
        }

        captured = io.StringIO()

        with redirect_stdout(captured):
            print_summary(
                model_type="yolo26",
                split="val",
                num_images=100,
                metrics=metrics,
            )

        output = captured.getvalue()
        # Perfect values should be formatted as 1.0000
        assert "1.0000" in output, "Expected perfect metric values formatted as '1.0000'"

    def test_summary_with_large_num_images(self):
        """The summary correctly displays large num_images values. (Req 16.8)"""
        num_images = 1000000
        metrics = {
            "map_50": 0.5,
            "map_50_95": 0.3,
            "precision": 0.6,
            "recall": 0.5,
            "f1_score": 0.55,
        }

        captured = io.StringIO()

        with redirect_stdout(captured):
            print_summary(
                model_type="yolo26",
                split="val",
                num_images=num_images,
                metrics=metrics,
            )

        output = captured.getvalue()
        assert str(num_images) in output, (
            f"Expected large num_images '{num_images}' in summary output"
        )
