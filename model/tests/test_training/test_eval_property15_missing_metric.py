"""Property-based tests for missing-metric-field failure.

Feature: generic-evaluation-script
Property 15: Missing required metric field fails the run

For any metrics dict missing one or more of the required fields (``map_50``,
``map_50_95``, ``precision``, ``recall``, ``f1_score``, ``per_class_ap``),
``assemble_report`` raises a ``RuntimeError`` whose message names every missing
field.

These tests exercise the real ``assemble_report`` function in
``model/training/evaluate_detection.py``. The function validates that all
required metric fields are present before assembling the report. If any required
field is missing, a ``RuntimeError`` is raised that names every missing field,
and the evaluation run is considered failed (Req 8.7).

To make the property independent of GPUs, real checkpoints, and real datasets,
the test generates:

* a complete metrics dict with all required fields;
* a non-empty subset of required fields to remove;
* valid report parameters (checkpoint path, model type, etc.).

The property asserts that:

1. When any required metric field is missing, ``assemble_report`` raises a
   ``RuntimeError``;
2. The error message names every missing field;
3. When all required fields are present, no error is raised.

**Validates: Requirements 8.7**
"""

from typing import List, Set

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from model.training.evaluate_detection import assemble_report


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_METRIC_FIELDS = [
    "map_50",
    "map_50_95",
    "precision",
    "recall",
    "f1_score",
    "per_class_ap",
]


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


@st.composite
def _class_names(draw) -> List[str]:
    """Draw a list of 1–5 unique class names."""
    count = draw(st.integers(min_value=1, max_value=5))
    names = [f"class_{i}" for i in range(count)]
    return names


@st.composite
def _complete_metrics(draw, class_names: List[str]) -> dict:
    """Draw a complete metrics dict with all required fields.

    All scalar metrics are in [0.0, 1.0], and per_class_ap has an entry for
    every class name.
    """
    return {
        "map_50": draw(st.floats(min_value=0.0, max_value=1.0)),
        "map_50_95": draw(st.floats(min_value=0.0, max_value=1.0)),
        "precision": draw(st.floats(min_value=0.0, max_value=1.0)),
        "recall": draw(st.floats(min_value=0.0, max_value=1.0)),
        "f1_score": draw(st.floats(min_value=0.0, max_value=1.0)),
        "per_class_ap": {
            name: draw(st.floats(min_value=0.0, max_value=1.0))
            for name in class_names
        },
    }


@st.composite
def _non_empty_subset_of_required_fields(draw) -> Set[str]:
    """Draw a non-empty subset of required metric fields to remove.

    Returns a set of 1 to len(REQUIRED_METRIC_FIELDS) field names.
    """
    # Draw a list of booleans indicating which fields to include in the subset
    include = draw(
        st.lists(
            st.booleans(),
            min_size=len(REQUIRED_METRIC_FIELDS),
            max_size=len(REQUIRED_METRIC_FIELDS),
        )
    )
    subset = {
        field for field, inc in zip(REQUIRED_METRIC_FIELDS, include) if inc
    }
    # Ensure at least one field is in the subset
    assume(len(subset) > 0)
    return subset


@st.composite
def _report_params(draw, class_names: List[str]) -> dict:
    """Draw valid report parameters for assemble_report.

    Returns a dict with all required parameters except metrics.
    """
    num_classes = len(class_names)
    num_images = draw(st.integers(min_value=1, max_value=100))

    return {
        "checkpoint_path": draw(st.text(min_size=1, max_size=50).filter(lambda x: x.strip())),
        "model_type": draw(st.sampled_from(["ssd_mobilenetv3", "yolo26", "yolov6"])),
        "model_config": {"num_classes": num_classes},
        "dataset_path": draw(st.text(min_size=1, max_size=50).filter(lambda x: x.strip())),
        "split": draw(st.sampled_from(["train", "val", "test"])),
        "num_images": num_images,
        "num_classes": num_classes,
        "class_names": class_names,
        "confidence_threshold": draw(st.floats(min_value=0.0, max_value=1.0)),
        "iou_threshold": draw(st.floats(min_value=0.0, max_value=1.0)),
        "confusion_matrix": [[0] * num_classes for _ in range(num_classes)],
        "errors": [],
    }


@st.composite
def _metrics_with_missing_fields(draw):
    """Draw metrics with some required fields missing.

    Returns a tuple of (metrics, missing_fields, class_names, report_params)
    where:
    - metrics is a dict with some required fields removed;
    - missing_fields is the set of removed field names;
    - class_names is the list of class names used;
    - report_params is a dict with all other required parameters.
    """
    class_names = draw(_class_names())
    complete_metrics = draw(_complete_metrics(class_names))
    fields_to_remove = draw(_non_empty_subset_of_required_fields())
    report_params = draw(_report_params(class_names))

    # Remove the selected fields from the metrics dict
    incomplete_metrics = {
        k: v for k, v in complete_metrics.items() if k not in fields_to_remove
    }

    return incomplete_metrics, fields_to_remove, class_names, report_params


@st.composite
def _complete_metrics_and_params(draw):
    """Draw complete metrics and report parameters.

    Returns a tuple of (metrics, class_names, report_params) where all required
    metric fields are present.
    """
    class_names = draw(_class_names())
    complete_metrics = draw(_complete_metrics(class_names))
    report_params = draw(_report_params(class_names))

    return complete_metrics, class_names, report_params


# ---------------------------------------------------------------------------
# Property 15
# ---------------------------------------------------------------------------


class TestProperty15MissingMetricFieldFailure:
    """Property 15: Missing required metric field fails the run.

    **Validates: Requirements 8.7**
    """

    @given(data=_metrics_with_missing_fields())
    @settings(max_examples=100, deadline=None)
    def test_missing_metric_field_raises_runtime_error(self, data):
        # Feature: generic-evaluation-script, Property 15: Missing metric failure
        """Missing required metric fields raise RuntimeError.

        For any metrics dict missing one or more of the required fields,
        ``assemble_report`` raises a ``RuntimeError`` (Req 8.7).

        **Validates: Requirements 8.7**
        """
        incomplete_metrics, missing_fields, class_names, report_params = data

        with pytest.raises(RuntimeError) as exc_info:
            assemble_report(
                checkpoint_path=report_params["checkpoint_path"],
                model_type=report_params["model_type"],
                model_config=report_params["model_config"],
                dataset_path=report_params["dataset_path"],
                split=report_params["split"],
                num_images=report_params["num_images"],
                num_classes=report_params["num_classes"],
                class_names=report_params["class_names"],
                confidence_threshold=report_params["confidence_threshold"],
                iou_threshold=report_params["iou_threshold"],
                metrics=incomplete_metrics,
                confusion_matrix=report_params["confusion_matrix"],
                errors=report_params["errors"],
            )

        # Verify that the error message mentions "missing" and "metric"
        error_message = str(exc_info.value).lower()
        assert "missing" in error_message, (
            f"Error message should mention 'missing', got: {exc_info.value}"
        )

    @given(data=_metrics_with_missing_fields())
    @settings(max_examples=100, deadline=None)
    def test_error_message_names_all_missing_fields(self, data):
        # Feature: generic-evaluation-script, Property 15: Missing metric failure
        """Error message names every missing field.

        The ``RuntimeError`` message must identify every missing required metric
        field so the user knows exactly what to fix (Req 8.7).

        **Validates: Requirements 8.7**
        """
        incomplete_metrics, missing_fields, class_names, report_params = data

        with pytest.raises(RuntimeError) as exc_info:
            assemble_report(
                checkpoint_path=report_params["checkpoint_path"],
                model_type=report_params["model_type"],
                model_config=report_params["model_config"],
                dataset_path=report_params["dataset_path"],
                split=report_params["split"],
                num_images=report_params["num_images"],
                num_classes=report_params["num_classes"],
                class_names=report_params["class_names"],
                confidence_threshold=report_params["confidence_threshold"],
                iou_threshold=report_params["iou_threshold"],
                metrics=incomplete_metrics,
                confusion_matrix=report_params["confusion_matrix"],
                errors=report_params["errors"],
            )

        error_message = str(exc_info.value)

        # Every missing field must be named in the error message
        for field in missing_fields:
            assert field in error_message, (
                f"Error message should name missing field '{field}', "
                f"got: {error_message}"
            )

    @given(data=_complete_metrics_and_params())
    @settings(max_examples=100, deadline=None)
    def test_complete_metrics_does_not_raise(self, data):
        # Feature: generic-evaluation-script, Property 15: Missing metric failure
        """Complete metrics dict does not raise RuntimeError.

        When all required metric fields are present, ``assemble_report`` should
        succeed and return a valid report dict (Req 8.7).

        **Validates: Requirements 8.7**
        """
        complete_metrics, class_names, report_params = data

        # Should not raise
        report = assemble_report(
            checkpoint_path=report_params["checkpoint_path"],
            model_type=report_params["model_type"],
            model_config=report_params["model_config"],
            dataset_path=report_params["dataset_path"],
            split=report_params["split"],
            num_images=report_params["num_images"],
            num_classes=report_params["num_classes"],
            class_names=report_params["class_names"],
            confidence_threshold=report_params["confidence_threshold"],
            iou_threshold=report_params["iou_threshold"],
            metrics=complete_metrics,
            confusion_matrix=report_params["confusion_matrix"],
            errors=report_params["errors"],
        )

        # Verify the report is a dict with expected structure
        assert isinstance(report, dict), "assemble_report should return a dict"
        assert "metrics" in report, "Report should contain 'metrics' key"
        assert "checkpoint" in report, "Report should contain 'checkpoint' key"


# ---------------------------------------------------------------------------
# Example-based tests complementing Property 15
# ---------------------------------------------------------------------------


class TestMissingMetricFieldExamples:
    """Concrete examples complementing Property 15.

    **Validates: Requirements 8.7**
    """

    def _make_complete_metrics(self) -> dict:
        """Create a complete metrics dict with all required fields."""
        return {
            "map_50": 0.75,
            "map_50_95": 0.65,
            "precision": 0.80,
            "recall": 0.70,
            "f1_score": 0.74,
            "per_class_ap": {"crack": 0.8, "pothole": 0.7},
        }

    def _make_report_params(self) -> dict:
        """Create valid report parameters."""
        return {
            "checkpoint_path": "/path/to/checkpoint.pt",
            "model_type": "ssd_mobilenetv3",
            "model_config": {"num_classes": 2},
            "dataset_path": "/path/to/dataset",
            "split": "val",
            "num_images": 100,
            "num_classes": 2,
            "class_names": ["crack", "pothole"],
            "confidence_threshold": 0.5,
            "iou_threshold": 0.5,
            "confusion_matrix": [[10, 2], [3, 15]],
            "errors": [],
        }

    def test_missing_map_50_raises_error(self):
        """Missing map_50 raises RuntimeError naming the field. (Req 8.7)"""
        metrics = self._make_complete_metrics()
        del metrics["map_50"]
        params = self._make_report_params()

        with pytest.raises(RuntimeError) as exc_info:
            assemble_report(metrics=metrics, **params)

        assert "map_50" in str(exc_info.value)

    def test_missing_map_50_95_raises_error(self):
        """Missing map_50_95 raises RuntimeError naming the field. (Req 8.7)"""
        metrics = self._make_complete_metrics()
        del metrics["map_50_95"]
        params = self._make_report_params()

        with pytest.raises(RuntimeError) as exc_info:
            assemble_report(metrics=metrics, **params)

        assert "map_50_95" in str(exc_info.value)

    def test_missing_precision_raises_error(self):
        """Missing precision raises RuntimeError naming the field. (Req 8.7)"""
        metrics = self._make_complete_metrics()
        del metrics["precision"]
        params = self._make_report_params()

        with pytest.raises(RuntimeError) as exc_info:
            assemble_report(metrics=metrics, **params)

        assert "precision" in str(exc_info.value)

    def test_missing_recall_raises_error(self):
        """Missing recall raises RuntimeError naming the field. (Req 8.7)"""
        metrics = self._make_complete_metrics()
        del metrics["recall"]
        params = self._make_report_params()

        with pytest.raises(RuntimeError) as exc_info:
            assemble_report(metrics=metrics, **params)

        assert "recall" in str(exc_info.value)

    def test_missing_f1_score_raises_error(self):
        """Missing f1_score raises RuntimeError naming the field. (Req 8.7)"""
        metrics = self._make_complete_metrics()
        del metrics["f1_score"]
        params = self._make_report_params()

        with pytest.raises(RuntimeError) as exc_info:
            assemble_report(metrics=metrics, **params)

        assert "f1_score" in str(exc_info.value)

    def test_missing_per_class_ap_raises_error(self):
        """Missing per_class_ap raises RuntimeError naming the field. (Req 8.7)"""
        metrics = self._make_complete_metrics()
        del metrics["per_class_ap"]
        params = self._make_report_params()

        with pytest.raises(RuntimeError) as exc_info:
            assemble_report(metrics=metrics, **params)

        assert "per_class_ap" in str(exc_info.value)

    def test_missing_multiple_fields_names_all(self):
        """Missing multiple fields names all of them in the error. (Req 8.7)"""
        metrics = self._make_complete_metrics()
        del metrics["map_50"]
        del metrics["precision"]
        del metrics["f1_score"]
        params = self._make_report_params()

        with pytest.raises(RuntimeError) as exc_info:
            assemble_report(metrics=metrics, **params)

        error_message = str(exc_info.value)
        assert "map_50" in error_message
        assert "precision" in error_message
        assert "f1_score" in error_message

    def test_missing_all_fields_names_all(self):
        """Missing all required fields names all of them in the error. (Req 8.7)"""
        metrics = {}  # Empty metrics dict
        params = self._make_report_params()

        with pytest.raises(RuntimeError) as exc_info:
            assemble_report(metrics=metrics, **params)

        error_message = str(exc_info.value)
        for field in REQUIRED_METRIC_FIELDS:
            assert field in error_message, (
                f"Error message should name missing field '{field}'"
            )

    def test_complete_metrics_succeeds(self):
        """Complete metrics dict allows assemble_report to succeed. (Req 8.7)"""
        metrics = self._make_complete_metrics()
        params = self._make_report_params()

        # Should not raise
        report = assemble_report(metrics=metrics, **params)

        assert isinstance(report, dict)
        assert report["checkpoint"] == params["checkpoint_path"]
        assert report["model_type"] == params["model_type"]
        assert report["split"] == params["split"]

    def test_extra_fields_in_metrics_are_allowed(self):
        """Extra fields in metrics dict are allowed. (Req 8.7)"""
        metrics = self._make_complete_metrics()
        metrics["extra_field"] = 0.99
        metrics["another_extra"] = {"key": "value"}
        params = self._make_report_params()

        # Should not raise - extra fields are ignored
        report = assemble_report(metrics=metrics, **params)

        assert isinstance(report, dict)
        assert report["checkpoint"] == params["checkpoint_path"]
