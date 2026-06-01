"""Property-based tests for report completeness, metric bounds, and prior-field retention.

Feature: generic-evaluation-script
Property 14: Report completeness, metric bounds, and prior-field retention

For any successful evaluation run, the assembled report contains:

1. Every required top-level field (``checkpoint``, ``model_type``,
   ``model_config``, ``dataset``, ``split``, ``num_images``, ``num_classes``,
   ``class_names``, ``confidence_threshold``, ``iou_threshold``, ``metrics``,
   ``confusion_matrix``, ``errors``).
2. Prior display keys retained (``mAP@0.5``, ``mAP@0.5:0.95``) in the metrics
   object.
3. The metrics object contains ``map_50``, ``map_50_95``, ``precision``,
   ``recall``, ``f1_score``, ``per_class_ap``.
4. ``per_class_ap`` keys equal the set of configured class names.
5. Every scalar metric lies in ``[0.0, 1.0]``.

These tests exercise the real ``assemble_report`` function in
``model/training/evaluate_detection.py``. The function builds the complete
report structure with all required top-level fields and a ``metrics`` object
containing both new snake_case keys and retained prior display keys for backward
compatibility.

To make the property independent of GPUs, real checkpoints, and real datasets,
the test generates:

* a list of class names (1-5 classes);
* a complete metrics dict with all required fields and scalar values in
  ``[0.0, 1.0]``;
* valid report parameters (checkpoint path, model type, split, thresholds, etc.).

**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.6, 9.4, 16.2, 17.4**
"""

from typing import List

from hypothesis import given, settings
from hypothesis import strategies as st

from model.training.evaluate_detection import assemble_report


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_TOP_LEVEL_FIELDS = [
    "checkpoint",
    "model_type",
    "model_config",
    "dataset",
    "split",
    "num_images",
    "num_classes",
    "class_names",
    "confidence_threshold",
    "iou_threshold",
    "metrics",
    "confusion_matrix",
    "errors",
]

REQUIRED_METRIC_FIELDS = [
    "map_50",
    "map_50_95",
    "precision",
    "recall",
    "f1_score",
    "per_class_ap",
]

PRIOR_DISPLAY_KEYS = [
    "mAP@0.5",
    "mAP@0.5:0.95",
]

SCALAR_METRIC_KEYS = [
    "map_50",
    "map_50_95",
    "precision",
    "recall",
    "f1_score",
]


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


@st.composite
def _class_names(draw) -> List[str]:
    """Draw a list of 1-5 unique class names."""
    count = draw(st.integers(min_value=1, max_value=5))
    names = [f"class_{i}" for i in range(count)]
    return names


@st.composite
def _complete_metrics(draw, class_names: List[str]) -> dict:
    """Draw a complete metrics dict with all required fields.

    All scalar metrics are in [0.0, 1.0], and per_class_ap has an entry for
    every class name with values in [0.0, 1.0].
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
def _report_inputs(draw):
    """Draw all inputs needed for a successful assemble_report call.

    Returns a tuple of (metrics, report_params) where:
    - metrics is a complete metrics dict with all required fields;
    - report_params is a dict with all other required parameters.
    """
    class_names = draw(_class_names())
    num_classes = len(class_names)
    metrics = draw(_complete_metrics(class_names))
    num_images = draw(st.integers(min_value=1, max_value=500))

    report_params = {
        "checkpoint_path": draw(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "P")),
                min_size=1,
                max_size=50,
            ).filter(lambda x: x.strip())
        ),
        "model_type": draw(st.sampled_from(["ssd_mobilenetv3", "yolo26", "yolov6"])),
        "model_config": {"num_classes": num_classes, "input_size": 640},
        "dataset_path": draw(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "P")),
                min_size=1,
                max_size=50,
            ).filter(lambda x: x.strip())
        ),
        "split": draw(st.sampled_from(["train", "val", "test"])),
        "num_images": num_images,
        "num_classes": num_classes,
        "class_names": class_names,
        "confidence_threshold": draw(st.floats(min_value=0.0, max_value=1.0)),
        "iou_threshold": draw(st.floats(min_value=0.0, max_value=1.0)),
        "confusion_matrix": [[0] * num_classes for _ in range(num_classes)],
        "errors": draw(
            st.lists(
                st.text(min_size=1, max_size=30).map(lambda x: f"img: {x}"),
                min_size=0,
                max_size=5,
            )
        ),
    }

    return metrics, report_params


# ---------------------------------------------------------------------------
# Property 14
# ---------------------------------------------------------------------------


class TestProperty14ReportCompleteness:
    """Property 14: Report completeness, metric bounds, and prior-field retention.

    **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.6, 9.4, 16.2, 17.4**
    """

    @given(data=_report_inputs())
    @settings(max_examples=100, deadline=None)
    def test_report_contains_all_required_top_level_fields(self, data):
        # Feature: generic-evaluation-script, Property 14: Report completeness
        """The assembled report contains every required top-level field.

        For any successful evaluation run, the report must include: checkpoint,
        model_type, model_config, dataset, split, num_images, num_classes,
        class_names, confidence_threshold, iou_threshold, metrics,
        confusion_matrix, and errors (Req 16.2).

        **Validates: Requirements 16.2**
        """
        metrics, report_params = data

        report = assemble_report(metrics=metrics, **report_params)

        for field in REQUIRED_TOP_LEVEL_FIELDS:
            assert field in report, (
                f"Report is missing required top-level field '{field}'. "
                f"Present keys: {list(report.keys())}"
            )

    @given(data=_report_inputs())
    @settings(max_examples=100, deadline=None)
    def test_metrics_contains_all_required_metric_fields(self, data):
        # Feature: generic-evaluation-script, Property 14: Report completeness
        """The metrics object contains all required metric fields.

        The metrics object must contain map_50, map_50_95, precision, recall,
        f1_score, and per_class_ap (Req 8.6).

        **Validates: Requirements 8.6**
        """
        metrics, report_params = data

        report = assemble_report(metrics=metrics, **report_params)

        report_metrics = report["metrics"]
        for field in REQUIRED_METRIC_FIELDS:
            assert field in report_metrics, (
                f"Report metrics is missing required field '{field}'. "
                f"Present keys: {list(report_metrics.keys())}"
            )

    @given(data=_report_inputs())
    @settings(max_examples=100, deadline=None)
    def test_prior_display_keys_retained(self, data):
        # Feature: generic-evaluation-script, Property 14: Report completeness
        """Prior display keys (mAP@0.5, mAP@0.5:0.95) are retained in metrics.

        The report retains every field present in the prior report format, with
        new fields added only as additional keys (Req 17.4).

        **Validates: Requirements 17.4**
        """
        metrics, report_params = data

        report = assemble_report(metrics=metrics, **report_params)

        report_metrics = report["metrics"]
        for key in PRIOR_DISPLAY_KEYS:
            assert key in report_metrics, (
                f"Report metrics is missing prior display key '{key}'. "
                f"Present keys: {list(report_metrics.keys())}"
            )

        # Prior display keys should mirror the new snake_case keys
        assert report_metrics["mAP@0.5"] == report_metrics["map_50"], (
            f"mAP@0.5 ({report_metrics['mAP@0.5']}) should equal "
            f"map_50 ({report_metrics['map_50']})"
        )
        assert report_metrics["mAP@0.5:0.95"] == report_metrics["map_50_95"], (
            f"mAP@0.5:0.95 ({report_metrics['mAP@0.5:0.95']}) should equal "
            f"map_50_95 ({report_metrics['map_50_95']})"
        )

    @given(data=_report_inputs())
    @settings(max_examples=100, deadline=None)
    def test_per_class_ap_keys_equal_class_names(self, data):
        # Feature: generic-evaluation-script, Property 14: Report completeness
        """per_class_ap keys equal the set of configured class names.

        The per_class_ap dict must have exactly one entry per configured class
        name (Req 8.4, 8.6).

        **Validates: Requirements 8.4, 8.6**
        """
        metrics, report_params = data

        report = assemble_report(metrics=metrics, **report_params)

        per_class_ap = report["metrics"]["per_class_ap"]
        expected_classes = set(report_params["class_names"])
        actual_classes = set(per_class_ap.keys())

        assert actual_classes == expected_classes, (
            f"per_class_ap keys {actual_classes} should equal "
            f"configured class names {expected_classes}"
        )

    @given(data=_report_inputs())
    @settings(max_examples=100, deadline=None)
    def test_scalar_metrics_in_unit_interval(self, data):
        # Feature: generic-evaluation-script, Property 14: Report completeness
        """Every scalar metric lies in [0.0, 1.0].

        All scalar metrics (map_50, map_50_95, precision, recall, f1_score) and
        per-class AP values must be bounded within [0.0, 1.0] (Req 8.1, 8.2,
        8.3).

        **Validates: Requirements 8.1, 8.2, 8.3**
        """
        metrics, report_params = data

        report = assemble_report(metrics=metrics, **report_params)

        report_metrics = report["metrics"]

        # Check scalar metrics
        for key in SCALAR_METRIC_KEYS:
            value = report_metrics[key]
            assert 0.0 <= value <= 1.0, (
                f"Scalar metric '{key}' = {value} is outside [0.0, 1.0]"
            )

        # Check per-class AP values
        for class_name, ap_value in report_metrics["per_class_ap"].items():
            assert 0.0 <= ap_value <= 1.0, (
                f"per_class_ap['{class_name}'] = {ap_value} is outside [0.0, 1.0]"
            )

    @given(data=_report_inputs())
    @settings(max_examples=100, deadline=None)
    def test_report_includes_resolved_split_value(self, data):
        # Feature: generic-evaluation-script, Property 14: Report completeness
        """The report includes the resolved evaluation.split value.

        The split field in the report must match the configured evaluation.split
        value (Req 9.4).

        **Validates: Requirements 9.4**
        """
        metrics, report_params = data

        report = assemble_report(metrics=metrics, **report_params)

        assert report["split"] == report_params["split"], (
            f"Report split '{report['split']}' should match configured "
            f"split '{report_params['split']}'"
        )


# ---------------------------------------------------------------------------
# Example-based tests complementing Property 14
# ---------------------------------------------------------------------------


class TestReportCompletenessExamples:
    """Concrete examples complementing Property 14.

    **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.6, 9.4, 16.2, 17.4**
    """

    def _make_metrics(self, class_names: List[str]) -> dict:
        """Create a complete metrics dict with known values."""
        return {
            "map_50": 0.75,
            "map_50_95": 0.65,
            "precision": 0.80,
            "recall": 0.70,
            "f1_score": 0.74,
            "per_class_ap": {name: 0.7 + i * 0.05 for i, name in enumerate(class_names)},
        }

    def _make_report_params(self, class_names: List[str]) -> dict:
        """Create valid report parameters."""
        return {
            "checkpoint_path": "/checkpoints/ssd_mobilenetv3/run123/best_model.pt",
            "model_type": "ssd_mobilenetv3",
            "model_config": {"num_classes": len(class_names), "input_size": 640},
            "dataset_path": "/data/rdd2022",
            "split": "val",
            "num_images": 200,
            "num_classes": len(class_names),
            "class_names": class_names,
            "confidence_threshold": 0.25,
            "iou_threshold": 0.5,
            "confusion_matrix": [[10, 2], [3, 15]] if len(class_names) == 2 else [[5] * len(class_names)] * len(class_names),
            "errors": ["img_42: RuntimeError: CUDA OOM"],
        }

    def test_full_report_structure_two_classes(self):
        """Full report with two classes has all required fields. (Req 16.2)"""
        class_names = ["crack", "pothole"]
        metrics = self._make_metrics(class_names)
        params = self._make_report_params(class_names)

        report = assemble_report(metrics=metrics, **params)

        # All top-level fields present
        for field in REQUIRED_TOP_LEVEL_FIELDS:
            assert field in report, f"Missing top-level field: {field}"

        # All metric fields present
        for field in REQUIRED_METRIC_FIELDS:
            assert field in report["metrics"], f"Missing metric field: {field}"

        # Prior display keys present
        for key in PRIOR_DISPLAY_KEYS:
            assert key in report["metrics"], f"Missing prior key: {key}"

        # per_class_ap matches class names
        assert set(report["metrics"]["per_class_ap"].keys()) == set(class_names)

    def test_report_preserves_input_values(self):
        """Report preserves the input values passed to assemble_report. (Req 16.2)"""
        class_names = ["crack", "pothole", "spalling"]
        metrics = self._make_metrics(class_names)
        params = self._make_report_params(class_names)

        report = assemble_report(metrics=metrics, **params)

        assert report["checkpoint"] == params["checkpoint_path"]
        assert report["model_type"] == params["model_type"]
        assert report["model_config"] == params["model_config"]
        assert report["dataset"] == params["dataset_path"]
        assert report["split"] == params["split"]
        assert report["num_images"] == params["num_images"]
        assert report["num_classes"] == params["num_classes"]
        assert report["class_names"] == params["class_names"]
        assert report["confidence_threshold"] == params["confidence_threshold"]
        assert report["iou_threshold"] == params["iou_threshold"]

    def test_prior_keys_mirror_new_keys(self):
        """Prior display keys mirror the new snake_case keys. (Req 17.4)"""
        class_names = ["crack"]
        metrics = self._make_metrics(class_names)
        params = self._make_report_params(class_names)

        report = assemble_report(metrics=metrics, **params)

        assert report["metrics"]["mAP@0.5"] == report["metrics"]["map_50"]
        assert report["metrics"]["mAP@0.5:0.95"] == report["metrics"]["map_50_95"]

    def test_boundary_metrics_zero(self):
        """Metrics at 0.0 boundary are valid. (Req 8.1, 8.2, 8.3)"""
        class_names = ["crack", "pothole"]
        metrics = {
            "map_50": 0.0,
            "map_50_95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "per_class_ap": {name: 0.0 for name in class_names},
        }
        params = self._make_report_params(class_names)

        report = assemble_report(metrics=metrics, **params)

        for key in SCALAR_METRIC_KEYS:
            assert report["metrics"][key] == 0.0

    def test_boundary_metrics_one(self):
        """Metrics at 1.0 boundary are valid. (Req 8.1, 8.2, 8.3)"""
        class_names = ["crack", "pothole"]
        metrics = {
            "map_50": 1.0,
            "map_50_95": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
            "per_class_ap": {name: 1.0 for name in class_names},
        }
        params = self._make_report_params(class_names)

        report = assemble_report(metrics=metrics, **params)

        for key in SCALAR_METRIC_KEYS:
            assert report["metrics"][key] == 1.0

    def test_errors_structure(self):
        """Errors field has count and items. (Req 15.3)"""
        class_names = ["crack"]
        metrics = self._make_metrics(class_names)
        params = self._make_report_params(class_names)
        params["errors"] = ["img_1: ValueError: bad image", "img_5: RuntimeError: OOM"]

        report = assemble_report(metrics=metrics, **params)

        assert "errors" in report
        assert report["errors"]["count"] == 2
        assert report["errors"]["items"] == params["errors"]

    def test_empty_errors_list(self):
        """Empty errors list produces count 0. (Req 15.3)"""
        class_names = ["crack"]
        metrics = self._make_metrics(class_names)
        params = self._make_report_params(class_names)
        params["errors"] = []

        report = assemble_report(metrics=metrics, **params)

        assert report["errors"]["count"] == 0
        assert report["errors"]["items"] == []

    def test_split_value_in_report(self):
        """Report includes the resolved split value. (Req 9.4)"""
        class_names = ["crack"]
        metrics = self._make_metrics(class_names)

        for split in ["train", "val", "test"]:
            params = self._make_report_params(class_names)
            params["split"] = split

            report = assemble_report(metrics=metrics, **params)

            assert report["split"] == split
