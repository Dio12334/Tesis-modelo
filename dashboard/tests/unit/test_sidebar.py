"""Unit tests for dashboard.components.sidebar module.

Tests cover:
- Rendering sidebar with empty runs displays info message
- Model filter extracts unique model names and includes "All" option
- Filtering runs by model_name returns only matching runs
- Run list displays key info (run_id, model_name, start_time, final_val_loss, total_epochs)
- Returns selected ExperimentRun or None
- Displays warning count when loading errors exist

Requirements: 2.1, 2.3, 2.4, 2.5, 2.6, 9.1
"""

from unittest.mock import patch, MagicMock

import pytest

from dashboard.data_loader import DashboardData, ExperimentRun, FinalResults, MetricsEntry
from dashboard.components.sidebar import render_sidebar, render_model_filter, render_run_list


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_run(
    run_id: str = "test-run-id-1234",
    model_name: str = "ssd_mobilenetv3",
    start_time: str = "2026-05-24T18:48:35.604147+00:00",
    final_val_loss: float = 6.5,
    total_epochs: int = 21,
    has_final_results: bool = True,
) -> ExperimentRun:
    """Create an ExperimentRun with configurable fields."""
    final_results = None
    if has_final_results:
        final_results = FinalResults(
            final_train_loss=0.9,
            final_val_loss=final_val_loss,
            best_val_loss=final_val_loss - 0.5,
            best_epoch=5,
            total_epochs=total_epochs,
        )
    return ExperimentRun(
        run_id=run_id,
        model_name=model_name,
        dataset_name="rdd2022",
        config={},
        start_time=start_time,
        end_time="2026-05-24T19:00:00+00:00",
        metrics_history=[],
        final_results=final_results,
    )


def _make_dashboard_data(runs=None, errors=None) -> DashboardData:
    """Create DashboardData with configurable runs and errors."""
    return DashboardData(
        runs=runs or [],
        evaluation_report=None,
        best_model_metadata=None,
        errors=errors or [],
    )


# ---------------------------------------------------------------------------
# Test: Empty runs displays info message
# ---------------------------------------------------------------------------


class TestSidebarEmptyRuns:
    """Tests for render_sidebar when no runs are available."""

    @patch("dashboard.components.sidebar.st")
    def test_displays_info_when_no_runs(self, mock_st):
        mock_st.sidebar.__enter__ = MagicMock(return_value=mock_st)
        mock_st.sidebar.__exit__ = MagicMock(return_value=False)

        data = _make_dashboard_data(runs=[])
        result = render_sidebar(data)

        assert result is None
        mock_st.info.assert_called_once()
        info_msg = mock_st.info.call_args[0][0]
        assert "no" in info_msg.lower() or "available" in info_msg.lower()


# ---------------------------------------------------------------------------
# Test: Model filter extracts unique model names
# ---------------------------------------------------------------------------


class TestModelFilter:
    """Tests for render_model_filter."""

    @patch("dashboard.components.sidebar.st")
    def test_extracts_unique_model_names(self, mock_st):
        mock_st.selectbox.return_value = "All"

        runs = [
            _make_run(run_id="r1", model_name="ssd_mobilenetv3"),
            _make_run(run_id="r2", model_name="ssd_mobilenetv3"),
            _make_run(run_id="r3", model_name="faster_rcnn"),
        ]
        data = _make_dashboard_data(runs=runs)
        result = render_model_filter(data)

        # Check that selectbox was called with correct options
        call_args = mock_st.selectbox.call_args
        options = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("options")
        assert "All" in options
        assert "ssd_mobilenetv3" in options
        assert "faster_rcnn" in options
        # "All" should be first
        assert options[0] == "All"

    @patch("dashboard.components.sidebar.st")
    def test_returns_selected_model(self, mock_st):
        mock_st.selectbox.return_value = "faster_rcnn"

        runs = [
            _make_run(run_id="r1", model_name="ssd_mobilenetv3"),
            _make_run(run_id="r2", model_name="faster_rcnn"),
        ]
        data = _make_dashboard_data(runs=runs)
        result = render_model_filter(data)

        assert result == "faster_rcnn"

    @patch("dashboard.components.sidebar.st")
    def test_model_names_are_sorted(self, mock_st):
        mock_st.selectbox.return_value = "All"

        runs = [
            _make_run(run_id="r1", model_name="yolo_v5"),
            _make_run(run_id="r2", model_name="faster_rcnn"),
            _make_run(run_id="r3", model_name="ssd_mobilenetv3"),
        ]
        data = _make_dashboard_data(runs=runs)
        render_model_filter(data)

        call_args = mock_st.selectbox.call_args
        options = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("options")
        # After "All", model names should be sorted
        model_options = options[1:]
        assert model_options == sorted(model_options)


# ---------------------------------------------------------------------------
# Test: Run list displays key info
# ---------------------------------------------------------------------------


class TestRunList:
    """Tests for render_run_list."""

    @patch("dashboard.components.sidebar.st")
    def test_returns_selected_run_id(self, mock_st):
        mock_st.radio.return_value = 0  # Select first run

        runs = [
            _make_run(run_id="run-abc-123"),
            _make_run(run_id="run-def-456"),
        ]
        result = render_run_list(runs)

        assert result == "run-abc-123"

    @patch("dashboard.components.sidebar.st")
    def test_returns_second_run_when_selected(self, mock_st):
        mock_st.radio.return_value = 1  # Select second run

        runs = [
            _make_run(run_id="run-abc-123"),
            _make_run(run_id="run-def-456"),
        ]
        result = render_run_list(runs)

        assert result == "run-def-456"

    @patch("dashboard.components.sidebar.st")
    def test_returns_none_for_empty_list(self, mock_st):
        result = render_run_list([])
        assert result is None

    @patch("dashboard.components.sidebar.st")
    def test_format_func_includes_key_info(self, mock_st):
        mock_st.radio.return_value = 0

        runs = [
            _make_run(
                run_id="abcdef12-3456-7890-abcd-ef1234567890",
                model_name="ssd_mobilenetv3",
                start_time="2026-05-24T18:48:35.604147+00:00",
                final_val_loss=6.5,
                total_epochs=21,
            ),
        ]
        render_run_list(runs)

        # Get the format_func from the radio call
        call_kwargs = mock_st.radio.call_args[1] if mock_st.radio.call_args[1] else {}
        format_func = call_kwargs.get("format_func")

        if format_func:
            label = format_func(0)
            # Should contain truncated run_id
            assert "abcdef12" in label
            # Should contain model_name
            assert "ssd_mobilenetv3" in label
            # Should contain start_time
            assert "2026-05-24" in label
            # Should contain val_loss
            assert "6.5" in label
            # Should contain epochs
            assert "21" in label

    @patch("dashboard.components.sidebar.st")
    def test_handles_run_without_final_results(self, mock_st):
        mock_st.radio.return_value = 0

        runs = [_make_run(has_final_results=False)]
        render_run_list(runs)

        # Get the format_func from the radio call
        call_kwargs = mock_st.radio.call_args[1] if mock_st.radio.call_args[1] else {}
        format_func = call_kwargs.get("format_func")

        if format_func:
            label = format_func(0)
            assert "N/A" in label


# ---------------------------------------------------------------------------
# Test: Sidebar filtering integration
# ---------------------------------------------------------------------------


class TestSidebarFiltering:
    """Tests for the full render_sidebar filtering flow."""

    @patch("dashboard.components.sidebar.st")
    def test_returns_selected_run_with_all_filter(self, mock_st):
        mock_st.sidebar.__enter__ = MagicMock(return_value=mock_st)
        mock_st.sidebar.__exit__ = MagicMock(return_value=False)
        mock_st.selectbox.return_value = "All"
        mock_st.radio.return_value = 0

        runs = [
            _make_run(run_id="run-1", model_name="ssd_mobilenetv3"),
            _make_run(run_id="run-2", model_name="faster_rcnn"),
        ]
        data = _make_dashboard_data(runs=runs)
        result = render_sidebar(data)

        assert result is not None
        assert result.run_id == "run-1"

    @patch("dashboard.components.sidebar.st")
    def test_filters_runs_by_model_name(self, mock_st):
        mock_st.sidebar.__enter__ = MagicMock(return_value=mock_st)
        mock_st.sidebar.__exit__ = MagicMock(return_value=False)
        mock_st.selectbox.return_value = "faster_rcnn"
        mock_st.radio.return_value = 0

        runs = [
            _make_run(run_id="run-1", model_name="ssd_mobilenetv3"),
            _make_run(run_id="run-2", model_name="faster_rcnn"),
            _make_run(run_id="run-3", model_name="ssd_mobilenetv3"),
        ]
        data = _make_dashboard_data(runs=runs)
        result = render_sidebar(data)

        assert result is not None
        assert result.run_id == "run-2"
        assert result.model_name == "faster_rcnn"

    @patch("dashboard.components.sidebar.st")
    def test_returns_none_when_no_runs_match_filter(self, mock_st):
        mock_st.sidebar.__enter__ = MagicMock(return_value=mock_st)
        mock_st.sidebar.__exit__ = MagicMock(return_value=False)
        mock_st.selectbox.return_value = "nonexistent_model"
        mock_st.radio.return_value = 0

        runs = [
            _make_run(run_id="run-1", model_name="ssd_mobilenetv3"),
        ]
        data = _make_dashboard_data(runs=runs)
        result = render_sidebar(data)

        assert result is None
        mock_st.info.assert_called()

    @patch("dashboard.components.sidebar.st")
    def test_displays_warning_when_errors_exist(self, mock_st):
        mock_st.sidebar.__enter__ = MagicMock(return_value=mock_st)
        mock_st.sidebar.__exit__ = MagicMock(return_value=False)
        mock_st.selectbox.return_value = "All"
        mock_st.radio.return_value = 0

        runs = [_make_run(run_id="run-1")]
        errors = ["Failed to load file1.json", "Failed to load file2.json"]
        data = _make_dashboard_data(runs=runs, errors=errors)
        render_sidebar(data)

        mock_st.warning.assert_called_once()
        warning_msg = mock_st.warning.call_args[0][0]
        assert "2" in warning_msg
