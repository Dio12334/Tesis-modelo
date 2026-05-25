"""Unit tests for ExperimentTracker."""

import csv
import json
from pathlib import Path

import pytest

from model.tracking.tracker import ExperimentTracker


@pytest.fixture
def tracker(tmp_path):
    """Create an ExperimentTracker with a temporary output directory."""
    return ExperimentTracker(output_dir=tmp_path / "experiments")


@pytest.fixture
def sample_config():
    """Sample experiment configuration."""
    return {
        "model": {"type": "yolov6", "backbone_size": "nano"},
        "training": {"epochs": 10, "batch_size": 16, "learning_rate": 0.01},
    }


class TestInit:
    def test_creates_output_dir(self, tmp_path):
        output_dir = tmp_path / "new_dir" / "experiments"
        assert not output_dir.exists()
        tracker = ExperimentTracker(output_dir=output_dir)
        assert output_dir.exists()
        assert tracker.output_dir == output_dir

    def test_existing_dir_no_error(self, tmp_path):
        output_dir = tmp_path / "existing"
        output_dir.mkdir()
        tracker = ExperimentTracker(output_dir=output_dir)
        assert tracker.output_dir == output_dir


class TestStartRun:
    def test_returns_unique_id(self, tracker, sample_config):
        id1 = tracker.start_run(sample_config, "yolov6", "rdd2022")
        id2 = tracker.start_run(sample_config, "yolov6", "rdd2022")
        assert id1 != id2

    def test_creates_json_file(self, tracker, sample_config):
        run_id = tracker.start_run(sample_config, "yolov6", "rdd2022")
        json_path = tracker.output_dir / f"{run_id}.json"
        assert json_path.exists()

    def test_json_contains_expected_fields(self, tracker, sample_config):
        run_id = tracker.start_run(sample_config, "yolov6", "rdd2022")
        json_path = tracker.output_dir / f"{run_id}.json"
        with open(json_path) as f:
            data = json.load(f)

        assert data["run_id"] == run_id
        assert data["model_name"] == "yolov6"
        assert data["dataset_name"] == "rdd2022"
        assert data["config"] == sample_config
        assert data["start_time"] is not None
        assert data["end_time"] is None
        assert data["metrics_history"] == []
        assert data["final_results"] is None

    def test_run_id_is_valid_uuid(self, tracker, sample_config):
        import uuid

        run_id = tracker.start_run(sample_config, "yolov6", "rdd2022")
        # Should not raise
        uuid.UUID(run_id)


class TestLogMetrics:
    def test_appends_metrics_to_history(self, tracker, sample_config):
        run_id = tracker.start_run(sample_config, "yolov6", "rdd2022")
        tracker.log_metrics(run_id, step=1, metrics={"loss": 0.5, "lr": 0.01})
        tracker.log_metrics(run_id, step=2, metrics={"loss": 0.4, "lr": 0.009})

        json_path = tracker.output_dir / f"{run_id}.json"
        with open(json_path) as f:
            data = json.load(f)

        assert len(data["metrics_history"]) == 2
        assert data["metrics_history"][0]["step"] == 1
        assert data["metrics_history"][0]["loss"] == 0.5
        assert data["metrics_history"][1]["step"] == 2
        assert data["metrics_history"][1]["loss"] == 0.4

    def test_metrics_include_timestamp(self, tracker, sample_config):
        run_id = tracker.start_run(sample_config, "yolov6", "rdd2022")
        tracker.log_metrics(run_id, step=0, metrics={"loss": 1.0})

        json_path = tracker.output_dir / f"{run_id}.json"
        with open(json_path) as f:
            data = json.load(f)

        assert "timestamp" in data["metrics_history"][0]


class TestEndRun:
    def test_sets_end_time_and_results(self, tracker, sample_config):
        run_id = tracker.start_run(sample_config, "yolov6", "rdd2022")
        results = {"map_50": 0.75, "map_50_95": 0.55}
        tracker.end_run(run_id, results)

        json_path = tracker.output_dir / f"{run_id}.json"
        with open(json_path) as f:
            data = json.load(f)

        assert data["end_time"] is not None
        assert data["final_results"] == results

    def test_preserves_metrics_history(self, tracker, sample_config):
        run_id = tracker.start_run(sample_config, "yolov6", "rdd2022")
        tracker.log_metrics(run_id, step=1, metrics={"loss": 0.5})
        tracker.end_run(run_id, {"map_50": 0.8})

        json_path = tracker.output_dir / f"{run_id}.json"
        with open(json_path) as f:
            data = json.load(f)

        assert len(data["metrics_history"]) == 1
        assert data["metrics_history"][0]["loss"] == 0.5


class TestExportCsv:
    def test_exports_all_runs(self, tracker, sample_config):
        id1 = tracker.start_run(sample_config, "yolov6", "rdd2022")
        tracker.end_run(id1, {"map_50": 0.75})
        id2 = tracker.start_run(sample_config, "ssd", "rdd2022")
        tracker.end_run(id2, {"map_50": 0.65})

        csv_path = tracker.export_csv()
        assert csv_path.exists()

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        run_ids = {row["run_id"] for row in rows}
        assert id1 in run_ids
        assert id2 in run_ids

    def test_exports_specific_runs(self, tracker, sample_config):
        id1 = tracker.start_run(sample_config, "yolov6", "rdd2022")
        tracker.end_run(id1, {"map_50": 0.75})
        id2 = tracker.start_run(sample_config, "ssd", "rdd2022")
        tracker.end_run(id2, {"map_50": 0.65})

        csv_path = tracker.export_csv(run_ids=[id1])

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["run_id"] == id1

    def test_csv_contains_metric_columns(self, tracker, sample_config):
        run_id = tracker.start_run(sample_config, "yolov6", "rdd2022")
        tracker.end_run(run_id, {"map_50": 0.75, "f1": 0.8})

        csv_path = tracker.export_csv()

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows[0]["map_50"] == "0.75"
        assert rows[0]["f1"] == "0.8"

    def test_custom_output_path(self, tracker, tmp_path, sample_config):
        run_id = tracker.start_run(sample_config, "yolov6", "rdd2022")
        tracker.end_run(run_id, {"map_50": 0.75})

        custom_path = tmp_path / "custom_export.csv"
        result_path = tracker.export_csv(output_path=custom_path)
        assert result_path == custom_path
        assert custom_path.exists()

    def test_default_output_path(self, tracker, sample_config):
        run_id = tracker.start_run(sample_config, "yolov6", "rdd2022")
        tracker.end_run(run_id, {"map_50": 0.75})

        csv_path = tracker.export_csv()
        assert csv_path == tracker.output_dir / "experiments.csv"


class TestQuery:
    def test_query_by_model_type(self, tracker, sample_config):
        id1 = tracker.start_run(sample_config, "yolov6", "rdd2022")
        id2 = tracker.start_run(sample_config, "ssd", "rdd2022")

        results = tracker.query(model_type="yolov6")
        assert len(results) == 1
        assert results[0]["run_id"] == id1

    def test_query_by_date_range(self, tracker, sample_config):
        run_id = tracker.start_run(sample_config, "yolov6", "rdd2022")

        # Query with a range that includes now
        results = tracker.query(date_range=("2020-01-01", "2099-12-31"))
        assert len(results) == 1
        assert results[0]["run_id"] == run_id

        # Query with a range in the past
        results = tracker.query(date_range=("2000-01-01", "2001-01-01"))
        assert len(results) == 0

    def test_query_no_filters_returns_all(self, tracker, sample_config):
        id1 = tracker.start_run(sample_config, "yolov6", "rdd2022")
        id2 = tracker.start_run(sample_config, "ssd", "rdd2022")

        results = tracker.query()
        assert len(results) == 2

    def test_query_combined_filters(self, tracker, sample_config):
        id1 = tracker.start_run(sample_config, "yolov6", "rdd2022")
        id2 = tracker.start_run(sample_config, "ssd", "rdd2022")

        results = tracker.query(
            model_type="yolov6", date_range=("2020-01-01", "2099-12-31")
        )
        assert len(results) == 1
        assert results[0]["run_id"] == id1

    def test_query_no_matches(self, tracker, sample_config):
        tracker.start_run(sample_config, "yolov6", "rdd2022")
        results = tracker.query(model_type="nonexistent")
        assert len(results) == 0
