"""Integration tests for dashboard startup.

Tests cover:
- Streamlit app initializes without errors
- File discovery with temp directories containing JSON fixtures
- End-to-end flow: create temp dirs with JSON fixtures, load data, verify runs are found

Requirements: 10.1, 1.1, 1.2
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from dashboard.data_loader import (
    DashboardData,
    ExperimentRun,
    EvaluationReport,
    load_all_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_EXPERIMENT_RUN = {
    "run_id": "aaaaaaaa-1111-2222-3333-444444444444",
    "model_name": "ssd_mobilenetv3",
    "dataset_name": "rdd2022_804imgs",
    "config": {
        "name": "rdd2022_ssd_mobilenetv3",
        "model": {"type": "ssd_mobilenetv3", "config": {"input_size": 320, "num_classes": 5}},
        "training": {"epochs": 50, "batch_size": 16, "learning_rate": 0.01},
    },
    "start_time": "2026-06-01T10:00:00+00:00",
    "end_time": "2026-06-01T11:00:00+00:00",
    "metrics_history": [
        {
            "step": 0,
            "timestamp": "2026-06-01T10:01:00+00:00",
            "train_loss": 8.5,
            "val_loss": 9.0,
            "learning_rate": 0.003,
            "epoch_time_s": 25.0,
        },
        {
            "step": 1,
            "timestamp": "2026-06-01T10:02:00+00:00",
            "train_loss": 6.2,
            "val_loss": 7.1,
            "learning_rate": 0.006,
            "epoch_time_s": 24.5,
        },
    ],
    "final_results": {
        "final_train_loss": 1.0,
        "final_val_loss": 2.5,
        "best_val_loss": 2.0,
        "best_epoch": 8,
        "total_epochs": 50,
    },
}

VALID_EXPERIMENT_RUN_2 = {
    "run_id": "bbbbbbbb-5555-6666-7777-888888888888",
    "model_name": "ssd_mobilenetv3",
    "dataset_name": "rdd2022_804imgs",
    "config": {
        "name": "rdd2022_ssd_mobilenetv3",
        "model": {"type": "ssd_mobilenetv3", "config": {"input_size": 320, "num_classes": 5}},
        "training": {"epochs": 100, "batch_size": 8, "learning_rate": 0.005},
    },
    "start_time": "2026-07-15T14:00:00+00:00",
    "end_time": "2026-07-15T16:00:00+00:00",
    "metrics_history": [
        {
            "step": 0,
            "timestamp": "2026-07-15T14:01:00+00:00",
            "train_loss": 9.0,
            "val_loss": 9.5,
            "learning_rate": 0.001,
            "epoch_time_s": 30.0,
        },
    ],
    "final_results": {
        "final_train_loss": 0.8,
        "final_val_loss": 1.9,
        "best_val_loss": 1.5,
        "best_epoch": 12,
        "total_epochs": 100,
    },
}

VALID_EVALUATION_REPORT = {
    "checkpoint": "checkpoints/ssd_mobilenetv3/global/best_model.pt",
    "dataset": "model/data/rdd2022/sample",
    "num_val_images": 161,
    "num_classes": 5,
    "class_names": [
        "alligator crack",
        "longitudinal crack",
        "other corruption",
        "pothole",
        "transverse crack",
    ],
    "confidence_threshold": 0.5,
    "iou_threshold": 0.5,
    "metrics": {
        "mAP@0.5": 0.037,
        "mAP@0.5:0.95": 0.0097,
        "precision": 0.242,
        "recall": 0.029,
        "f1_score": 0.051,
        "per_class_ap": {
            "alligator crack": 0.0,
            "longitudinal crack": 0.0078,
            "other corruption": 0.178,
            "pothole": 0.0,
            "transverse crack": 0.0,
        },
    },
    "confusion_matrix": [
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 7, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
}


def _write_json(directory: Path, filename: str, data: dict) -> Path:
    """Helper to write a JSON file in a directory."""
    filepath = directory / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return filepath


# ---------------------------------------------------------------------------
# Test: Streamlit app initializes without errors
# ---------------------------------------------------------------------------


class TestDashboardAppInitialization:
    """Verify that importing dashboard.app and calling main() doesn't crash."""

    def test_import_dashboard_app_succeeds(self):
        """Importing dashboard.app should not raise any errors."""
        import dashboard.app  # noqa: F401

    def test_main_function_exists_and_is_callable(self):
        """The main() function should be importable and callable."""
        from dashboard.app import main

        assert callable(main)

    def test_main_runs_without_crash_with_mocked_streamlit(self, tmp_path: Path):
        """Calling main() with mocked Streamlit should not raise exceptions.

        We mock Streamlit's API to avoid needing a running Streamlit server,
        and point to empty temp directories so data loading works cleanly.
        """
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()

        # Write one valid run so the app has data to work with
        _write_json(results_dir, "run.json", VALID_EXPERIMENT_RUN)

        mock_st = MagicMock()
        mock_st.query_params = {
            "results_dir": str(results_dir),
            "checkpoints_dir": str(checkpoints_dir),
        }
        # render_sidebar returns None (no run selected) -> welcome message path
        mock_st.set_page_config = MagicMock()

        with patch("dashboard.app.st", mock_st):
            # Also patch render_sidebar to return None (no run selected)
            with patch("dashboard.app.render_sidebar", return_value=None):
                from dashboard.app import main

                # Should not raise
                main()

        # Verify set_page_config was called
        mock_st.set_page_config.assert_called_once()

    def test_main_with_selected_run_renders_tabs(self, tmp_path: Path):
        """When a run is selected, main() should render tabs without crashing."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()

        _write_json(results_dir, "run.json", VALID_EXPERIMENT_RUN)

        mock_st = MagicMock()
        mock_st.query_params = {
            "results_dir": str(results_dir),
            "checkpoints_dir": str(checkpoints_dir),
        }
        # Mock tabs to return context managers
        mock_tab = MagicMock()
        mock_tab.__enter__ = MagicMock(return_value=None)
        mock_tab.__exit__ = MagicMock(return_value=False)
        mock_st.tabs.return_value = [mock_tab] * 6

        # Load data to get a real ExperimentRun for the sidebar to return
        data = load_all_data(results_dir, checkpoints_dir)
        selected_run = data.runs[0]

        with patch("dashboard.app.st", mock_st):
            with patch("dashboard.app.render_sidebar", return_value=selected_run):
                with patch("dashboard.app.render_metrics_overview"):
                    with patch("dashboard.app.render_config"):
                        with patch("dashboard.app.render_loss_chart"):
                            with patch("dashboard.app.render_learning_rate_chart"):
                                with patch("dashboard.app.render_class_performance"):
                                    with patch("dashboard.app.render_confusion_matrix"):
                                        with patch("dashboard.app.render_run_comparison"):
                                            with patch("dashboard.app.render_image_prediction_viewer"):
                                                from dashboard.app import main

                                                main()

        # Verify tabs were created
        mock_st.tabs.assert_called_once()


# ---------------------------------------------------------------------------
# Test: File discovery with temp directories containing JSON fixtures
# ---------------------------------------------------------------------------


class TestFileDiscoveryIntegration:
    """Verify load_all_data correctly discovers JSON files in temp directories."""

    def test_discovers_single_run_in_flat_directory(self, tmp_path: Path):
        """A single JSON file in results/ should be discovered and loaded."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()

        _write_json(results_dir, "experiment.json", VALID_EXPERIMENT_RUN)

        data = load_all_data(results_dir, checkpoints_dir)

        assert len(data.runs) == 1
        assert data.runs[0].run_id == "aaaaaaaa-1111-2222-3333-444444444444"
        assert data.runs[0].model_name == "ssd_mobilenetv3"
        assert data.errors == []

    def test_discovers_multiple_runs_in_nested_directories(self, tmp_path: Path):
        """JSON files in nested subdirectories should all be discovered."""
        results_dir = tmp_path / "results"
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()

        # Create nested model directories (mimics real structure)
        model_dir = results_dir / "ssd_mobilenetv3"
        model_dir.mkdir(parents=True)

        _write_json(model_dir, "run1.json", VALID_EXPERIMENT_RUN)
        _write_json(model_dir, "run2.json", VALID_EXPERIMENT_RUN_2)

        data = load_all_data(results_dir, checkpoints_dir)

        assert len(data.runs) == 2
        run_ids = {r.run_id for r in data.runs}
        assert "aaaaaaaa-1111-2222-3333-444444444444" in run_ids
        assert "bbbbbbbb-5555-6666-7777-888888888888" in run_ids

    def test_discovers_evaluation_report_in_checkpoints(self, tmp_path: Path):
        """Evaluation report in checkpoints/ tree should be discovered."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        checkpoints_dir = tmp_path / "checkpoints"
        global_dir = checkpoints_dir / "ssd_mobilenetv3" / "global"
        global_dir.mkdir(parents=True)

        _write_json(global_dir, "evaluation_report.json", VALID_EVALUATION_REPORT)

        data = load_all_data(results_dir, checkpoints_dir)

        assert data.evaluation_report is not None
        assert isinstance(data.evaluation_report, EvaluationReport)
        assert data.evaluation_report.num_val_images == 161
        assert data.evaluation_report.num_classes == 5

    def test_skips_malformed_files_and_loads_valid_ones(self, tmp_path: Path):
        """Malformed JSON files should be skipped; valid ones still loaded."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()

        _write_json(results_dir, "good.json", VALID_EXPERIMENT_RUN)
        # Write malformed JSON
        bad_file = results_dir / "bad.json"
        bad_file.write_text("{this is not valid json", encoding="utf-8")

        data = load_all_data(results_dir, checkpoints_dir)

        assert len(data.runs) == 1
        assert data.runs[0].run_id == "aaaaaaaa-1111-2222-3333-444444444444"
        assert len(data.errors) >= 1
        assert "bad.json" in data.errors[0]

    def test_empty_directories_return_empty_data(self, tmp_path: Path):
        """Empty results and checkpoints directories should return empty data."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()

        data = load_all_data(results_dir, checkpoints_dir)

        assert data.runs == []
        assert data.evaluation_report is None
        assert data.best_model_metadata is None


# ---------------------------------------------------------------------------
# Test: End-to-end flow
# ---------------------------------------------------------------------------


class TestEndToEndFlow:
    """End-to-end: create temp dirs with JSON fixtures, load data, verify runs."""

    def test_full_data_loading_pipeline(self, tmp_path: Path):
        """Complete pipeline: results + checkpoints -> DashboardData with all fields."""
        # Set up results directory with multiple runs
        results_dir = tmp_path / "results" / "ssd_mobilenetv3"
        results_dir.mkdir(parents=True)

        _write_json(results_dir, "run1.json", VALID_EXPERIMENT_RUN)
        _write_json(results_dir, "run2.json", VALID_EXPERIMENT_RUN_2)

        # Set up checkpoints directory with evaluation report
        checkpoints_dir = tmp_path / "checkpoints"
        global_dir = checkpoints_dir / "ssd_mobilenetv3" / "global"
        global_dir.mkdir(parents=True)

        _write_json(global_dir, "evaluation_report.json", VALID_EVALUATION_REPORT)
        _write_json(global_dir, "best.json", {
            "run_id": "aaaaaaaa-1111-2222-3333-444444444444",
            "best_val_loss": 2.0,
            "best_epoch": 8,
            "config": {"name": "test"},
        })

        # Load all data
        data = load_all_data(tmp_path / "results", checkpoints_dir)

        # Verify runs are loaded and sorted by start_time descending
        assert len(data.runs) == 2
        assert data.runs[0].run_id == "bbbbbbbb-5555-6666-7777-888888888888"  # later start_time
        assert data.runs[1].run_id == "aaaaaaaa-1111-2222-3333-444444444444"  # earlier start_time

        # Verify evaluation report is loaded
        assert data.evaluation_report is not None
        assert data.evaluation_report.metrics["mAP@0.5"] == pytest.approx(0.037)

        # Verify best model metadata is loaded
        assert data.best_model_metadata is not None
        assert data.best_model_metadata["run_id"] == "aaaaaaaa-1111-2222-3333-444444444444"

        # Verify no errors
        assert data.errors == []

    def test_runs_have_correct_structure(self, tmp_path: Path):
        """Loaded runs should have all expected fields populated."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()

        _write_json(results_dir, "run.json", VALID_EXPERIMENT_RUN)

        data = load_all_data(results_dir, checkpoints_dir)

        run = data.runs[0]
        assert isinstance(run, ExperimentRun)
        assert run.run_id == "aaaaaaaa-1111-2222-3333-444444444444"
        assert run.model_name == "ssd_mobilenetv3"
        assert run.dataset_name == "rdd2022_804imgs"
        assert run.start_time == "2026-06-01T10:00:00+00:00"
        assert run.end_time == "2026-06-01T11:00:00+00:00"
        assert len(run.metrics_history) == 2
        assert run.final_results is not None
        assert run.final_results.best_val_loss == pytest.approx(2.0)
        assert run.final_results.total_epochs == 50
        assert run.config["training"]["learning_rate"] == 0.01

    def test_nonexistent_directories_handled_gracefully(self, tmp_path: Path):
        """Non-existent directories should not crash, just record errors."""
        results_dir = tmp_path / "does_not_exist_results"
        checkpoints_dir = tmp_path / "does_not_exist_checkpoints"

        data = load_all_data(results_dir, checkpoints_dir)

        assert isinstance(data, DashboardData)
        assert data.runs == []
        assert data.evaluation_report is None
        assert len(data.errors) >= 2  # One for each missing directory

    def test_mixed_valid_and_invalid_files_in_realistic_structure(self, tmp_path: Path):
        """Realistic directory structure with some bad files still loads valid data."""
        # Mimic real project structure
        results_dir = tmp_path / "results" / "ssd_mobilenetv3"
        results_dir.mkdir(parents=True)

        # Valid runs
        _write_json(results_dir, "aaaaaaaa-1111-2222-3333-444444444444.json", VALID_EXPERIMENT_RUN)
        _write_json(results_dir, "bbbbbbbb-5555-6666-7777-888888888888.json", VALID_EXPERIMENT_RUN_2)

        # Corrupted file
        corrupted = results_dir / "corrupted.json"
        corrupted.write_text("{'single quotes': 'not valid json'}", encoding="utf-8")

        # File with wrong schema (valid JSON but not an experiment run)
        _write_json(results_dir, "not_a_run.json", {"some_key": "some_value"})

        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()

        data = load_all_data(tmp_path / "results", checkpoints_dir)

        # Should load the 2 valid runs
        assert len(data.runs) == 2
        # Should record errors for the 2 bad files
        assert len(data.errors) == 2
