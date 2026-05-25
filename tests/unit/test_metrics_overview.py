"""Unit tests for dashboard.components.metrics_overview module.

Tests cover:
- Rendering metrics overview with a valid evaluation report
- Handling None evaluation report with informational message
- Formatting metrics to two decimal places
- Displaying evaluation parameters (confidence threshold, IoU threshold, etc.)

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6
"""

from unittest.mock import patch, MagicMock

import pytest

from dashboard.data_loader import EvaluationReport
from dashboard.components.metrics_overview import render_metrics_overview


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_evaluation_report(
    map_50=0.037,
    map_50_95=0.0097,
    precision=0.242,
    recall=0.029,
    f1_score=0.051,
    confidence_threshold=0.5,
    iou_threshold=0.5,
    num_val_images=161,
    num_classes=5,
) -> EvaluationReport:
    """Create an EvaluationReport with configurable metrics."""
    return EvaluationReport(
        checkpoint="checkpoints/ssd_mobilenetv3/global/best_model.pt",
        dataset="model/data/rdd2022/sample",
        num_val_images=num_val_images,
        num_classes=num_classes,
        class_names=[
            "alligator crack",
            "longitudinal crack",
            "other corruption",
            "pothole",
            "transverse crack",
        ],
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        metrics={
            "mAP@0.5": map_50,
            "mAP@0.5:0.95": map_50_95,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "per_class_ap": {
                "alligator crack": 0.0,
                "longitudinal crack": 0.0078,
                "other corruption": 0.178,
                "pothole": 0.0,
                "transverse crack": 0.0,
            },
        },
        confusion_matrix=[
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 7, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    )


def _get_metric_calls_dict(mock_st):
    """Extract all st.metric calls as a label -> value dict."""
    result = {}
    for c in mock_st.metric.call_args_list:
        label = c.kwargs.get("label")
        value = c.kwargs.get("value")
        result[label] = value
    return result


# ---------------------------------------------------------------------------
# Test: None report displays informational message
# ---------------------------------------------------------------------------


class TestMetricsOverviewNoneReport:
    """Tests for render_metrics_overview when report is None."""

    @patch("dashboard.components.metrics_overview.st")
    def test_displays_info_message_when_report_is_none(self, mock_st):
        render_metrics_overview(None)

        mock_st.info.assert_called_once()
        info_msg = mock_st.info.call_args[0][0]
        assert "evaluation" in info_msg.lower()

    @patch("dashboard.components.metrics_overview.st")
    def test_does_not_display_metrics_when_report_is_none(self, mock_st):
        render_metrics_overview(None)

        mock_st.metric.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Valid report displays metrics
# ---------------------------------------------------------------------------


class TestMetricsOverviewValidReport:
    """Tests for render_metrics_overview with a valid report."""

    @patch("dashboard.components.metrics_overview.st")
    def test_does_not_display_info_message(self, mock_st):
        mock_cols = [MagicMock() for _ in range(5)]
        mock_param_cols = [MagicMock() for _ in range(4)]
        mock_st.columns.side_effect = [mock_cols, mock_param_cols]

        report = _make_evaluation_report()
        render_metrics_overview(report)

        mock_st.info.assert_not_called()

    @patch("dashboard.components.metrics_overview.st")
    def test_displays_all_primary_metrics(self, mock_st):
        mock_cols = [MagicMock() for _ in range(5)]
        mock_param_cols = [MagicMock() for _ in range(4)]
        mock_st.columns.side_effect = [mock_cols, mock_param_cols]

        report = _make_evaluation_report(
            map_50=0.037, map_50_95=0.0097, precision=0.242, recall=0.029, f1_score=0.051
        )
        render_metrics_overview(report)

        calls = _get_metric_calls_dict(mock_st)
        assert calls["mAP@0.5"] == "0.04"
        assert calls["mAP@0.5:0.95"] == "0.01"
        assert calls["Precision"] == "0.24"
        assert calls["Recall"] == "0.03"
        assert calls["F1-Score"] == "0.05"

    @patch("dashboard.components.metrics_overview.st")
    def test_formats_metrics_to_two_decimal_places(self, mock_st):
        mock_cols = [MagicMock() for _ in range(5)]
        mock_param_cols = [MagicMock() for _ in range(4)]
        mock_st.columns.side_effect = [mock_cols, mock_param_cols]

        report = _make_evaluation_report(
            map_50=0.12345, precision=0.9, recall=1.0, f1_score=0.333333
        )
        render_metrics_overview(report)

        calls = _get_metric_calls_dict(mock_st)
        assert calls["mAP@0.5"] == "0.12"
        assert calls["Precision"] == "0.90"
        assert calls["Recall"] == "1.00"
        assert calls["F1-Score"] == "0.33"

    @patch("dashboard.components.metrics_overview.st")
    def test_handles_missing_metric_keys_gracefully(self, mock_st):
        """If a metric key is missing from the dict, defaults to 0.0."""
        mock_cols = [MagicMock() for _ in range(5)]
        mock_param_cols = [MagicMock() for _ in range(4)]
        mock_st.columns.side_effect = [mock_cols, mock_param_cols]

        report = _make_evaluation_report()
        del report.metrics["precision"]
        render_metrics_overview(report)

        calls = _get_metric_calls_dict(mock_st)
        assert calls["Precision"] == "0.00"


# ---------------------------------------------------------------------------
# Test: Evaluation parameters display
# ---------------------------------------------------------------------------


class TestMetricsOverviewParameters:
    """Tests for evaluation parameter display."""

    @patch("dashboard.components.metrics_overview.st")
    def test_displays_all_evaluation_parameters(self, mock_st):
        mock_cols = [MagicMock() for _ in range(5)]
        mock_param_cols = [MagicMock() for _ in range(4)]
        mock_st.columns.side_effect = [mock_cols, mock_param_cols]

        report = _make_evaluation_report(
            confidence_threshold=0.25,
            iou_threshold=0.75,
            num_val_images=200,
            num_classes=5,
        )
        render_metrics_overview(report)

        calls = _get_metric_calls_dict(mock_st)
        assert calls["Confidence Threshold"] == "0.25"
        assert calls["IoU Threshold"] == "0.75"
        assert calls["Validation Images"] == "200"
        assert calls["Number of Classes"] == "5"

    @patch("dashboard.components.metrics_overview.st")
    def test_formats_thresholds_to_two_decimal_places(self, mock_st):
        mock_cols = [MagicMock() for _ in range(5)]
        mock_param_cols = [MagicMock() for _ in range(4)]
        mock_st.columns.side_effect = [mock_cols, mock_param_cols]

        report = _make_evaluation_report(
            confidence_threshold=0.3333, iou_threshold=0.6667
        )
        render_metrics_overview(report)

        calls = _get_metric_calls_dict(mock_st)
        assert calls["Confidence Threshold"] == "0.33"
        assert calls["IoU Threshold"] == "0.67"
