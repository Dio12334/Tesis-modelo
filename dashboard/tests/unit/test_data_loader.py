"""Unit tests for dashboard.data_loader module.

Tests cover:
- Parsing valid experiment run JSON
- Parsing valid evaluation report JSON
- Parsing best model metadata JSON
- Handling empty/non-existent directories
- Handling malformed JSON
- Handling missing required fields
- Handling permission errors
- Directory scanning finds all JSON files recursively
- Runs sorted by start_time descending

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest

from dashboard.data_loader import (
    DashboardData,
    EvaluationReport,
    ExperimentRun,
    FinalResults,
    MetricsEntry,
    load_all_data,
    load_best_model_metadata,
    load_evaluation_report,
    load_experiment_run,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_EXPERIMENT_RUN = {
    "run_id": "0284294e-08f0-46d7-bce7-a2e4391cc12e",
    "model_name": "ssd_mobilenetv3",
    "dataset_name": "rdd2022_804imgs",
    "config": {
        "name": "rdd2022_ssd_mobilenetv3",
        "model": {"type": "ssd_mobilenetv3", "config": {"input_size": 320, "num_classes": 5}},
        "training": {"epochs": 100, "batch_size": 16, "learning_rate": 0.01},
    },
    "start_time": "2026-05-24T18:48:35.604147+00:00",
    "end_time": "2026-05-24T18:59:16.428986+00:00",
    "metrics_history": [
        {
            "step": 0,
            "timestamp": "2026-05-24T18:49:03.968049+00:00",
            "train_loss": 10.558699905872345,
            "val_loss": 10.363025856018066,
            "learning_rate": 0.003333,
            "epoch_time_s": 28.36,
        },
        {
            "step": 1,
            "timestamp": "2026-05-24T18:49:29.273437+00:00",
            "train_loss": 7.480253080042397,
            "val_loss": 9.241947364807128,
            "learning_rate": 0.006667,
            "epoch_time_s": 25.08,
        },
    ],
    "final_results": {
        "final_train_loss": 0.927,
        "final_val_loss": 9.59,
        "best_val_loss": 6.626,
        "best_epoch": 5,
        "total_epochs": 21,
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

VALID_BEST_MODEL_METADATA = {
    "run_id": "b3206eba-5f2a-4367-8b08-09199d1c6e77",
    "best_val_loss": 6.528306865692139,
    "best_epoch": 5,
    "config": {
        "name": "rdd2022_ssd_mobilenetv3",
        "model": {"type": "ssd_mobilenetv3", "config": {"input_size": 320, "num_classes": 5}},
    },
}


def _write_json(directory: Path, filename: str, data: dict) -> Path:
    """Helper to write a JSON file in a directory."""
    filepath = directory / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return filepath


# ---------------------------------------------------------------------------
# Test: Parsing valid experiment run JSON
# ---------------------------------------------------------------------------


class TestLoadExperimentRun:
    """Tests for load_experiment_run with valid data."""

    def test_parses_valid_run_fields(self, tmp_path: Path):
        filepath = _write_json(tmp_path, "run.json", VALID_EXPERIMENT_RUN)
        run = load_experiment_run(filepath)

        assert isinstance(run, ExperimentRun)
        assert run.run_id == "0284294e-08f0-46d7-bce7-a2e4391cc12e"
        assert run.model_name == "ssd_mobilenetv3"
        assert run.dataset_name == "rdd2022_804imgs"
        assert run.start_time == "2026-05-24T18:48:35.604147+00:00"
        assert run.end_time == "2026-05-24T18:59:16.428986+00:00"
        assert run.config["name"] == "rdd2022_ssd_mobilenetv3"

    def test_parses_metrics_history(self, tmp_path: Path):
        filepath = _write_json(tmp_path, "run.json", VALID_EXPERIMENT_RUN)
        run = load_experiment_run(filepath)

        assert len(run.metrics_history) == 2
        first = run.metrics_history[0]
        assert isinstance(first, MetricsEntry)
        assert first.step == 0
        assert first.train_loss == pytest.approx(10.558699905872345)
        assert first.val_loss == pytest.approx(10.363025856018066)
        assert first.learning_rate == pytest.approx(0.003333)
        assert first.epoch_time_s == pytest.approx(28.36)

    def test_parses_final_results(self, tmp_path: Path):
        filepath = _write_json(tmp_path, "run.json", VALID_EXPERIMENT_RUN)
        run = load_experiment_run(filepath)

        assert isinstance(run.final_results, FinalResults)
        assert run.final_results.final_train_loss == pytest.approx(0.927)
        assert run.final_results.final_val_loss == pytest.approx(9.59)
        assert run.final_results.best_val_loss == pytest.approx(6.626)
        assert run.final_results.best_epoch == 5
        assert run.final_results.total_epochs == 21

    def test_handles_missing_end_time(self, tmp_path: Path):
        data = {**VALID_EXPERIMENT_RUN, "end_time": None}
        filepath = _write_json(tmp_path, "run.json", data)
        run = load_experiment_run(filepath)

        assert run.end_time is None

    def test_handles_missing_final_results(self, tmp_path: Path):
        data = {**VALID_EXPERIMENT_RUN}
        del data["final_results"]
        filepath = _write_json(tmp_path, "run.json", data)
        run = load_experiment_run(filepath)

        # When final_results is missing but metrics_history is non-empty,
        # the loader computes final_results from history as a fallback.
        assert run.final_results is not None
        assert run.final_results.total_epochs == 2
        assert run.final_results.final_train_loss == 7.480253080042397
        assert run.final_results.final_val_loss == 9.241947364807128
        assert run.final_results.best_val_loss == 9.241947364807128
        assert run.final_results.best_epoch == 2  # 1-indexed

    def test_handles_missing_final_results_and_empty_history(self, tmp_path: Path):
        data = {**VALID_EXPERIMENT_RUN}
        del data["final_results"]
        data["metrics_history"] = []
        filepath = _write_json(tmp_path, "run.json", data)
        run = load_experiment_run(filepath)

        # No history to compute from — final_results stays None.
        assert run.final_results is None

    def test_handles_total_epochs_trained_field_name(self, tmp_path: Path):
        """The loader supports both 'total_epochs' and 'total_epochs_trained'."""
        data = {**VALID_EXPERIMENT_RUN}
        data["final_results"] = {
            "final_train_loss": 0.5,
            "final_val_loss": 1.0,
            "best_val_loss": 0.8,
            "best_epoch": 3,
            "total_epochs_trained": 10,
        }
        filepath = _write_json(tmp_path, "run.json", data)
        run = load_experiment_run(filepath)

        assert run.final_results.total_epochs == 10


# ---------------------------------------------------------------------------
# Test: Parsing valid evaluation report JSON
# ---------------------------------------------------------------------------


class TestLoadEvaluationReport:
    """Tests for load_evaluation_report with valid data."""

    def test_parses_valid_report_fields(self, tmp_path: Path):
        filepath = _write_json(tmp_path, "eval.json", VALID_EVALUATION_REPORT)
        report = load_evaluation_report(filepath)

        assert isinstance(report, EvaluationReport)
        assert report.checkpoint == "checkpoints/ssd_mobilenetv3/global/best_model.pt"
        assert report.dataset == "model/data/rdd2022/sample"
        assert report.num_val_images == 161
        assert report.num_classes == 5
        assert report.confidence_threshold == 0.5
        assert report.iou_threshold == 0.5

    def test_parses_class_names(self, tmp_path: Path):
        filepath = _write_json(tmp_path, "eval.json", VALID_EVALUATION_REPORT)
        report = load_evaluation_report(filepath)

        assert report.class_names == [
            "alligator crack",
            "longitudinal crack",
            "other corruption",
            "pothole",
            "transverse crack",
        ]

    def test_parses_metrics(self, tmp_path: Path):
        filepath = _write_json(tmp_path, "eval.json", VALID_EVALUATION_REPORT)
        report = load_evaluation_report(filepath)

        assert report.metrics["mAP@0.5"] == pytest.approx(0.037)
        assert report.metrics["precision"] == pytest.approx(0.242)
        assert "per_class_ap" in report.metrics

    def test_parses_confusion_matrix(self, tmp_path: Path):
        filepath = _write_json(tmp_path, "eval.json", VALID_EVALUATION_REPORT)
        report = load_evaluation_report(filepath)

        assert len(report.confusion_matrix) == 5
        assert report.confusion_matrix[1][1] == 1
        assert report.confusion_matrix[2][2] == 7


# ---------------------------------------------------------------------------
# Test: Parsing best model metadata JSON
# ---------------------------------------------------------------------------


class TestLoadBestModelMetadata:
    """Tests for load_best_model_metadata."""

    def test_parses_valid_metadata(self, tmp_path: Path):
        filepath = _write_json(tmp_path, "best.json", VALID_BEST_MODEL_METADATA)
        metadata = load_best_model_metadata(filepath)

        assert isinstance(metadata, dict)
        assert metadata["run_id"] == "b3206eba-5f2a-4367-8b08-09199d1c6e77"
        assert metadata["best_val_loss"] == pytest.approx(6.528306865692139)
        assert metadata["best_epoch"] == 5
        assert "config" in metadata


# ---------------------------------------------------------------------------
# Test: load_all_data with empty/non-existent directories
# ---------------------------------------------------------------------------


class TestLoadAllDataEmptyAndMissing:
    """Tests for load_all_data with empty or non-existent directories."""

    def test_empty_results_directory_returns_empty_runs_with_error(self, tmp_path: Path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()

        data = load_all_data(results_dir, checkpoints_dir)

        assert isinstance(data, DashboardData)
        assert data.runs == []
        # No error for empty dir — it just has no JSON files

    def test_nonexistent_results_directory_returns_empty_runs_with_error(self, tmp_path: Path):
        results_dir = tmp_path / "nonexistent_results"
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()

        data = load_all_data(results_dir, checkpoints_dir)

        assert data.runs == []
        assert len(data.errors) >= 1
        assert "not found" in data.errors[0].lower() or "nonexistent_results" in data.errors[0]

    def test_nonexistent_checkpoints_directory_records_error(self, tmp_path: Path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        checkpoints_dir = tmp_path / "nonexistent_checkpoints"

        data = load_all_data(results_dir, checkpoints_dir)

        assert data.evaluation_report is None
        assert data.best_model_metadata is None
        assert any("nonexistent_checkpoints" in err for err in data.errors)


# ---------------------------------------------------------------------------
# Test: Malformed JSON handling
# ---------------------------------------------------------------------------


class TestMalformedJsonHandling:
    """Tests for graceful handling of malformed JSON files."""

    def test_malformed_json_is_skipped_and_error_recorded(self, tmp_path: Path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()

        # Write invalid JSON
        bad_file = results_dir / "bad.json"
        bad_file.write_text("{invalid json content", encoding="utf-8")

        data = load_all_data(results_dir, checkpoints_dir)

        assert data.runs == []
        assert len(data.errors) >= 1
        assert "bad.json" in data.errors[0]

    def test_valid_and_malformed_json_mixed(self, tmp_path: Path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()

        # Write one valid and one invalid file
        _write_json(results_dir, "good.json", VALID_EXPERIMENT_RUN)
        bad_file = results_dir / "bad.json"
        bad_file.write_text("not json at all", encoding="utf-8")

        data = load_all_data(results_dir, checkpoints_dir)

        assert len(data.runs) == 1
        assert data.runs[0].run_id == "0284294e-08f0-46d7-bce7-a2e4391cc12e"
        assert len(data.errors) >= 1


# ---------------------------------------------------------------------------
# Test: Missing required fields
# ---------------------------------------------------------------------------


class TestMissingRequiredFields:
    """Tests for handling JSON with missing required fields."""

    def test_missing_run_id_is_handled_gracefully(self, tmp_path: Path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()

        data = {**VALID_EXPERIMENT_RUN}
        del data["run_id"]
        _write_json(results_dir, "incomplete.json", data)

        result = load_all_data(results_dir, checkpoints_dir)

        assert result.runs == []
        assert len(result.errors) >= 1

    def test_missing_metrics_history_is_handled_gracefully(self, tmp_path: Path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()

        data = {**VALID_EXPERIMENT_RUN}
        del data["metrics_history"]
        _write_json(results_dir, "no_metrics.json", data)

        result = load_all_data(results_dir, checkpoints_dir)

        assert result.runs == []
        assert len(result.errors) >= 1

    def test_missing_fields_in_metrics_entry_is_handled(self, tmp_path: Path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()

        data = {**VALID_EXPERIMENT_RUN}
        data["metrics_history"] = [{"step": 0}]  # Missing required fields
        _write_json(results_dir, "bad_metrics.json", data)

        result = load_all_data(results_dir, checkpoints_dir)

        assert result.runs == []
        assert len(result.errors) >= 1


# ---------------------------------------------------------------------------
# Test: Directory scanning finds all JSON files recursively
# ---------------------------------------------------------------------------


class TestDirectoryScanning:
    """Tests for recursive directory scanning."""

    def test_finds_json_files_in_subdirectories(self, tmp_path: Path):
        results_dir = tmp_path / "results"
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()

        # Create nested structure
        sub1 = results_dir / "model_a"
        sub2 = results_dir / "model_b"
        sub1.mkdir(parents=True)
        sub2.mkdir(parents=True)

        run1 = {**VALID_EXPERIMENT_RUN, "run_id": "run-1", "start_time": "2026-01-01T00:00:00+00:00"}
        run2 = {**VALID_EXPERIMENT_RUN, "run_id": "run-2", "start_time": "2026-01-02T00:00:00+00:00"}

        _write_json(sub1, "run1.json", run1)
        _write_json(sub2, "run2.json", run2)

        data = load_all_data(results_dir, checkpoints_dir)

        assert len(data.runs) == 2
        run_ids = {r.run_id for r in data.runs}
        assert run_ids == {"run-1", "run-2"}

    def test_ignores_non_json_files(self, tmp_path: Path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()

        _write_json(results_dir, "valid.json", VALID_EXPERIMENT_RUN)
        (results_dir / "readme.txt").write_text("not a json file", encoding="utf-8")
        (results_dir / "data.csv").write_text("a,b,c\n1,2,3", encoding="utf-8")

        data = load_all_data(results_dir, checkpoints_dir)

        assert len(data.runs) == 1

    def test_loads_evaluation_report_from_checkpoints(self, tmp_path: Path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        checkpoints_dir = tmp_path / "checkpoints"
        global_dir = checkpoints_dir / "ssd_mobilenetv3" / "global"
        global_dir.mkdir(parents=True)

        _write_json(global_dir, "evaluation_report.json", VALID_EVALUATION_REPORT)
        _write_json(global_dir, "best.json", VALID_BEST_MODEL_METADATA)

        data = load_all_data(results_dir, checkpoints_dir)

        assert data.evaluation_report is not None
        assert data.evaluation_report.num_val_images == 161
        assert data.best_model_metadata is not None
        assert data.best_model_metadata["run_id"] == "b3206eba-5f2a-4367-8b08-09199d1c6e77"


# ---------------------------------------------------------------------------
# Test: Runs sorted by start_time descending
# ---------------------------------------------------------------------------


class TestRunsSortedDescending:
    """Tests that runs are sorted by start_time descending."""

    def test_runs_sorted_most_recent_first(self, tmp_path: Path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()

        # Create runs with different start times
        run_early = {**VALID_EXPERIMENT_RUN, "run_id": "early", "start_time": "2026-01-01T00:00:00+00:00"}
        run_mid = {**VALID_EXPERIMENT_RUN, "run_id": "mid", "start_time": "2026-06-15T12:00:00+00:00"}
        run_late = {**VALID_EXPERIMENT_RUN, "run_id": "late", "start_time": "2026-12-31T23:59:59+00:00"}

        # Write in non-sorted order
        _write_json(results_dir, "mid.json", run_mid)
        _write_json(results_dir, "early.json", run_early)
        _write_json(results_dir, "late.json", run_late)

        data = load_all_data(results_dir, checkpoints_dir)

        assert len(data.runs) == 3
        assert data.runs[0].run_id == "late"
        assert data.runs[1].run_id == "mid"
        assert data.runs[2].run_id == "early"

    def test_single_run_is_returned_as_is(self, tmp_path: Path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()

        _write_json(results_dir, "only.json", VALID_EXPERIMENT_RUN)

        data = load_all_data(results_dir, checkpoints_dir)

        assert len(data.runs) == 1
        assert data.runs[0].run_id == "0284294e-08f0-46d7-bce7-a2e4391cc12e"


# ---------------------------------------------------------------------------
# Test: Permission errors
# ---------------------------------------------------------------------------


class TestPermissionErrors:
    """Tests for handling permission errors when reading files."""

    @pytest.mark.skipif(sys.platform == "win32", reason="chmod not effective on Windows")
    def test_permission_error_on_run_file_is_handled(self, tmp_path: Path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()

        filepath = _write_json(results_dir, "locked.json", VALID_EXPERIMENT_RUN)
        os.chmod(filepath, 0o000)

        try:
            data = load_all_data(results_dir, checkpoints_dir)
            assert data.runs == []
            assert len(data.errors) >= 1
            assert "permission" in data.errors[0].lower() or "locked.json" in data.errors[0]
        finally:
            os.chmod(filepath, 0o644)

    @pytest.mark.skipif(sys.platform == "win32", reason="chmod not effective on Windows")
    def test_permission_error_on_evaluation_report_is_handled(self, tmp_path: Path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        checkpoints_dir = tmp_path / "checkpoints"
        global_dir = checkpoints_dir / "ssd_mobilenetv3" / "global"
        global_dir.mkdir(parents=True)

        filepath = _write_json(global_dir, "evaluation_report.json", VALID_EVALUATION_REPORT)
        os.chmod(filepath, 0o000)

        try:
            data = load_all_data(results_dir, checkpoints_dir)
            assert data.evaluation_report is None
            assert len(data.errors) >= 1
            assert "permission" in data.errors[0].lower()
        finally:
            os.chmod(filepath, 0o644)

    def test_permission_error_via_mock_on_run_file(self, tmp_path: Path):
        """Test permission error handling using mock (works on all platforms)."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()

        # Write a valid file so it gets discovered
        _write_json(results_dir, "run.json", VALID_EXPERIMENT_RUN)

        # Mock open to raise PermissionError
        original_open = open

        def mock_open_permission_error(path, *args, **kwargs):
            if "run.json" in str(path):
                raise PermissionError(f"Permission denied: {path}")
            return original_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=mock_open_permission_error):
            data = load_all_data(results_dir, checkpoints_dir)

        assert data.runs == []
        assert len(data.errors) >= 1
        assert "permission" in data.errors[0].lower() or "Permission" in data.errors[0]


# ---------------------------------------------------------------------------
# Test: Direct function error raising
# ---------------------------------------------------------------------------


class TestDirectFunctionErrors:
    """Tests that individual load functions raise appropriate exceptions."""

    def test_load_experiment_run_raises_on_invalid_json(self, tmp_path: Path):
        filepath = tmp_path / "bad.json"
        filepath.write_text("{not valid json", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            load_experiment_run(filepath)

    def test_load_experiment_run_raises_on_missing_field(self, tmp_path: Path):
        data = {"model_name": "test"}  # Missing run_id, metrics_history, etc.
        filepath = _write_json(tmp_path, "incomplete.json", data)

        with pytest.raises(KeyError):
            load_experiment_run(filepath)

    def test_load_evaluation_report_raises_on_invalid_json(self, tmp_path: Path):
        filepath = tmp_path / "bad_eval.json"
        filepath.write_text("{{{{", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            load_evaluation_report(filepath)

    def test_load_evaluation_report_raises_on_missing_field(self, tmp_path: Path):
        data = {"checkpoint": "test"}  # Missing most required fields
        filepath = _write_json(tmp_path, "incomplete_eval.json", data)

        with pytest.raises(KeyError):
            load_evaluation_report(filepath)

    def test_load_best_model_metadata_raises_on_invalid_json(self, tmp_path: Path):
        filepath = tmp_path / "bad_best.json"
        filepath.write_text("not json", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            load_best_model_metadata(filepath)
