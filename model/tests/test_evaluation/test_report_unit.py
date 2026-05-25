"""Unit tests for EvaluationReport serialization/deserialization."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from model.evaluation.report import EvaluationReport


@pytest.fixture
def sample_report():
    """Create a sample EvaluationReport for testing."""
    return EvaluationReport(
        model_id="yolov6-nano-v1",
        timestamp="2024-01-15T10:30:00",
        map_50=0.75,
        map_50_95=0.52,
        per_class_ap={"bache": 0.80, "fisura_longitudinal": 0.70},
        precision=0.85,
        recall=0.78,
        f1_score=0.81,
        confusion_matrix=np.array([[10, 2], [3, 15]]),
        class_names=["bache", "fisura_longitudinal"],
        config={"confidence_threshold": 0.5, "iou_threshold": 0.5},
    )


class TestToJson:
    """Tests for EvaluationReport.to_json()."""

    def test_returns_valid_json(self, sample_report):
        json_str = sample_report.to_json()
        data = json.loads(json_str)
        assert isinstance(data, dict)

    def test_contains_all_fields(self, sample_report):
        json_str = sample_report.to_json()
        data = json.loads(json_str)
        expected_keys = {
            "model_id", "timestamp", "map_50", "map_50_95",
            "per_class_ap", "precision", "recall", "f1_score",
            "confusion_matrix", "class_names", "config",
        }
        assert set(data.keys()) == expected_keys

    def test_model_id_preserved(self, sample_report):
        json_str = sample_report.to_json()
        data = json.loads(json_str)
        assert data["model_id"] == "yolov6-nano-v1"

    def test_timestamp_preserved(self, sample_report):
        json_str = sample_report.to_json()
        data = json.loads(json_str)
        assert data["timestamp"] == "2024-01-15T10:30:00"

    def test_confusion_matrix_as_nested_list(self, sample_report):
        json_str = sample_report.to_json()
        data = json.loads(json_str)
        assert data["confusion_matrix"] == [[10, 2], [3, 15]]

    def test_metrics_preserved(self, sample_report):
        json_str = sample_report.to_json()
        data = json.loads(json_str)
        assert data["map_50"] == 0.75
        assert data["map_50_95"] == 0.52
        assert data["precision"] == 0.85
        assert data["recall"] == 0.78
        assert data["f1_score"] == 0.81

    def test_per_class_ap_preserved(self, sample_report):
        json_str = sample_report.to_json()
        data = json.loads(json_str)
        assert data["per_class_ap"] == {"bache": 0.80, "fisura_longitudinal": 0.70}

    def test_empty_config(self):
        report = EvaluationReport(
            model_id="test",
            timestamp="2024-01-01T00:00:00",
            map_50=0.5,
            map_50_95=0.3,
            per_class_ap={},
            precision=0.5,
            recall=0.5,
            f1_score=0.5,
            confusion_matrix=np.array([[5]]),
            class_names=["bache"],
            config={},
        )
        json_str = report.to_json()
        data = json.loads(json_str)
        assert data["config"] == {}


class TestFromJson:
    """Tests for EvaluationReport.from_json()."""

    def test_round_trip_model_id(self, sample_report):
        json_str = sample_report.to_json()
        loaded = EvaluationReport.from_json(json_str)
        assert loaded.model_id == sample_report.model_id

    def test_round_trip_timestamp(self, sample_report):
        json_str = sample_report.to_json()
        loaded = EvaluationReport.from_json(json_str)
        assert loaded.timestamp == sample_report.timestamp

    def test_round_trip_metrics(self, sample_report):
        json_str = sample_report.to_json()
        loaded = EvaluationReport.from_json(json_str)
        assert loaded.map_50 == sample_report.map_50
        assert loaded.map_50_95 == sample_report.map_50_95
        assert loaded.precision == sample_report.precision
        assert loaded.recall == sample_report.recall
        assert loaded.f1_score == sample_report.f1_score

    def test_round_trip_confusion_matrix(self, sample_report):
        json_str = sample_report.to_json()
        loaded = EvaluationReport.from_json(json_str)
        np.testing.assert_array_equal(loaded.confusion_matrix, sample_report.confusion_matrix)

    def test_confusion_matrix_is_numpy_array(self, sample_report):
        json_str = sample_report.to_json()
        loaded = EvaluationReport.from_json(json_str)
        assert isinstance(loaded.confusion_matrix, np.ndarray)

    def test_round_trip_per_class_ap(self, sample_report):
        json_str = sample_report.to_json()
        loaded = EvaluationReport.from_json(json_str)
        assert loaded.per_class_ap == sample_report.per_class_ap

    def test_round_trip_class_names(self, sample_report):
        json_str = sample_report.to_json()
        loaded = EvaluationReport.from_json(json_str)
        assert loaded.class_names == sample_report.class_names

    def test_round_trip_config(self, sample_report):
        json_str = sample_report.to_json()
        loaded = EvaluationReport.from_json(json_str)
        assert loaded.config == sample_report.config


class TestSaveAndLoad:
    """Tests for EvaluationReport.save() and EvaluationReport.load()."""

    def test_save_creates_file(self, sample_report):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            sample_report.save(path)
            assert path.exists()

    def test_save_creates_parent_directories(self, sample_report):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "report.json"
            sample_report.save(path)
            assert path.exists()

    def test_load_round_trip(self, sample_report):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            sample_report.save(path)
            loaded = EvaluationReport.load(path)
            assert loaded.model_id == sample_report.model_id
            assert loaded.timestamp == sample_report.timestamp
            assert loaded.map_50 == sample_report.map_50
            assert loaded.map_50_95 == sample_report.map_50_95
            assert loaded.precision == sample_report.precision
            assert loaded.recall == sample_report.recall
            assert loaded.f1_score == sample_report.f1_score
            assert loaded.per_class_ap == sample_report.per_class_ap
            assert loaded.class_names == sample_report.class_names
            assert loaded.config == sample_report.config
            np.testing.assert_array_equal(loaded.confusion_matrix, sample_report.confusion_matrix)

    def test_saved_file_is_valid_json(self, sample_report):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            sample_report.save(path)
            content = path.read_text(encoding="utf-8")
            data = json.loads(content)
            assert data["model_id"] == "yolov6-nano-v1"

    def test_large_confusion_matrix(self):
        """Test with a larger confusion matrix (4 classes)."""
        matrix = np.array([
            [50, 5, 2, 1],
            [3, 45, 4, 0],
            [1, 2, 48, 3],
            [0, 1, 2, 40],
        ])
        report = EvaluationReport(
            model_id="ssd-mobilenetv3",
            timestamp="2024-06-01T12:00:00",
            map_50=0.82,
            map_50_95=0.61,
            per_class_ap={
                "bache": 0.85,
                "fisura_longitudinal": 0.80,
                "fisura_transversal": 0.82,
                "piel_de_cocodrilo": 0.78,
            },
            precision=0.88,
            recall=0.83,
            f1_score=0.85,
            confusion_matrix=matrix,
            class_names=["bache", "fisura_longitudinal", "fisura_transversal", "piel_de_cocodrilo"],
            config={"model_type": "ssd_mobilenetv3", "input_size": 320},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            report.save(path)
            loaded = EvaluationReport.load(path)
            np.testing.assert_array_equal(loaded.confusion_matrix, matrix)
            assert loaded.class_names == report.class_names
