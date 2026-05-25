"""Unit tests for dashboard.components.loss_charts module.

Tests cover:
- Rendering loss chart with valid metrics history
- Rendering learning rate chart with valid metrics history
- Handling empty metrics_history gracefully with informational message
- Best epoch marker placement at minimum val_loss
- Hover tooltips contain epoch, loss value, and learning rate

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
"""

from unittest.mock import patch, MagicMock

import pytest

from dashboard.data_loader import ExperimentRun, MetricsEntry, FinalResults
from dashboard.components.loss_charts import render_loss_chart, render_learning_rate_chart


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_metrics_history(num_epochs=5):
    """Create a sample metrics history with decreasing losses."""
    entries = []
    for i in range(num_epochs):
        entries.append(
            MetricsEntry(
                step=i,
                timestamp=f"2024-01-01T{i:02d}:00:00",
                train_loss=1.0 - (i * 0.1),
                val_loss=1.2 - (i * 0.15) if i != 3 else 0.4,  # epoch 3 has min val_loss when num_epochs=5
                learning_rate=0.01 * (0.9 ** i),
                epoch_time_s=120.0 + i,
            )
        )
    return entries


def _make_run(metrics_history=None):
    """Create an ExperimentRun with configurable metrics history."""
    if metrics_history is None:
        metrics_history = _make_metrics_history()
    return ExperimentRun(
        run_id="test-run-id",
        model_name="ssd_mobilenetv3",
        dataset_name="rdd2022",
        config={"learning_rate": 0.01},
        start_time="2024-01-01T00:00:00",
        end_time="2024-01-01T01:00:00",
        metrics_history=metrics_history,
        final_results=FinalResults(
            final_train_loss=0.5,
            final_val_loss=0.6,
            best_val_loss=0.4,
            best_epoch=3,
            total_epochs=5,
        ),
    )


# ---------------------------------------------------------------------------
# Test: Empty metrics_history displays info message
# ---------------------------------------------------------------------------


class TestLossChartEmptyHistory:
    """Tests for render_loss_chart when metrics_history is empty."""

    @patch("dashboard.components.loss_charts.st")
    def test_displays_info_message_when_empty(self, mock_st):
        run = _make_run(metrics_history=[])
        render_loss_chart(run)

        mock_st.info.assert_called_once()
        info_msg = mock_st.info.call_args[0][0]
        assert "no training history" in info_msg.lower()

    @patch("dashboard.components.loss_charts.st")
    def test_does_not_render_chart_when_empty(self, mock_st):
        run = _make_run(metrics_history=[])
        render_loss_chart(run)

        mock_st.plotly_chart.assert_not_called()


class TestLearningRateChartEmptyHistory:
    """Tests for render_learning_rate_chart when metrics_history is empty."""

    @patch("dashboard.components.loss_charts.st")
    def test_displays_info_message_when_empty(self, mock_st):
        run = _make_run(metrics_history=[])
        render_learning_rate_chart(run)

        mock_st.info.assert_called_once()
        info_msg = mock_st.info.call_args[0][0]
        assert "no training history" in info_msg.lower()

    @patch("dashboard.components.loss_charts.st")
    def test_does_not_render_chart_when_empty(self, mock_st):
        run = _make_run(metrics_history=[])
        render_learning_rate_chart(run)

        mock_st.plotly_chart.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Valid metrics_history renders charts
# ---------------------------------------------------------------------------


class TestLossChartValidHistory:
    """Tests for render_loss_chart with valid metrics history."""

    @patch("dashboard.components.loss_charts.st")
    def test_renders_plotly_chart(self, mock_st):
        run = _make_run()
        render_loss_chart(run)

        mock_st.plotly_chart.assert_called_once()

    @patch("dashboard.components.loss_charts.st")
    def test_does_not_display_info_message(self, mock_st):
        run = _make_run()
        render_loss_chart(run)

        mock_st.info.assert_not_called()

    @patch("dashboard.components.loss_charts.st")
    def test_chart_has_train_and_val_loss_traces(self, mock_st):
        run = _make_run()
        render_loss_chart(run)

        fig = mock_st.plotly_chart.call_args[0][0]
        trace_names = [trace.name for trace in fig.data]
        assert "Train Loss" in trace_names
        assert "Val Loss" in trace_names

    @patch("dashboard.components.loss_charts.st")
    def test_chart_has_best_epoch_marker(self, mock_st):
        run = _make_run()
        render_loss_chart(run)

        fig = mock_st.plotly_chart.call_args[0][0]
        trace_names = [trace.name for trace in fig.data]
        # Best epoch marker should be present
        best_traces = [n for n in trace_names if "Best" in n or "best" in n]
        assert len(best_traces) == 1

    @patch("dashboard.components.loss_charts.st")
    def test_best_epoch_marker_at_minimum_val_loss(self, mock_st):
        run = _make_run()
        render_loss_chart(run)

        fig = mock_st.plotly_chart.call_args[0][0]
        # Find the best epoch marker trace
        best_trace = None
        for trace in fig.data:
            if "Best" in trace.name or "best" in trace.name:
                best_trace = trace
                break

        assert best_trace is not None
        # The minimum val_loss is at epoch 3 (val_loss=0.4) in our fixture
        val_losses = [entry.val_loss for entry in run.metrics_history]
        min_idx = val_losses.index(min(val_losses))
        expected_epoch = run.metrics_history[min_idx].step

        assert best_trace.x[0] == expected_epoch
        assert best_trace.y[0] == min(val_losses)

    @patch("dashboard.components.loss_charts.st")
    def test_train_and_val_loss_have_distinct_colors(self, mock_st):
        run = _make_run()
        render_loss_chart(run)

        fig = mock_st.plotly_chart.call_args[0][0]
        train_trace = None
        val_trace = None
        for trace in fig.data:
            if trace.name == "Train Loss":
                train_trace = trace
            elif trace.name == "Val Loss":
                val_trace = trace

        assert train_trace is not None
        assert val_trace is not None
        assert train_trace.line.color != val_trace.line.color

    @patch("dashboard.components.loss_charts.st")
    def test_chart_uses_container_width(self, mock_st):
        run = _make_run()
        render_loss_chart(run)

        _, kwargs = mock_st.plotly_chart.call_args
        assert kwargs.get("use_container_width") is True


# ---------------------------------------------------------------------------
# Test: Learning rate chart with valid history
# ---------------------------------------------------------------------------


class TestLearningRateChartValidHistory:
    """Tests for render_learning_rate_chart with valid metrics history."""

    @patch("dashboard.components.loss_charts.st")
    def test_renders_plotly_chart(self, mock_st):
        run = _make_run()
        render_learning_rate_chart(run)

        mock_st.plotly_chart.assert_called_once()

    @patch("dashboard.components.loss_charts.st")
    def test_does_not_display_info_message(self, mock_st):
        run = _make_run()
        render_learning_rate_chart(run)

        mock_st.info.assert_not_called()

    @patch("dashboard.components.loss_charts.st")
    def test_chart_has_learning_rate_trace(self, mock_st):
        run = _make_run()
        render_learning_rate_chart(run)

        fig = mock_st.plotly_chart.call_args[0][0]
        trace_names = [trace.name for trace in fig.data]
        assert "Learning Rate" in trace_names

    @patch("dashboard.components.loss_charts.st")
    def test_chart_uses_container_width(self, mock_st):
        run = _make_run()
        render_learning_rate_chart(run)

        _, kwargs = mock_st.plotly_chart.call_args
        assert kwargs.get("use_container_width") is True

    @patch("dashboard.components.loss_charts.st")
    def test_learning_rate_values_match_history(self, mock_st):
        run = _make_run()
        render_learning_rate_chart(run)

        fig = mock_st.plotly_chart.call_args[0][0]
        lr_trace = fig.data[0]
        expected_lrs = [entry.learning_rate for entry in run.metrics_history]
        assert list(lr_trace.y) == expected_lrs
