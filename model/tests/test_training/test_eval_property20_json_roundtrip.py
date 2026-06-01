"""Property-based tests for report JSON round-trip preservation.

Feature: generic-evaluation-script
Property 20: Report JSON round-trip preserves content

For any assembled report and predictions structure, writing it to JSON and
reading it back yields a structure equal to the original (numbers, strings,
lists, and nested objects preserved), and the per-image predictions file
contains exactly one image object per image in the split, each with
``image_id``, ``boxes``, ``labels``, and ``scores``.

These tests exercise the JSON serialization/deserialization round-trip to
ensure that the Evaluation_Report and per-image predictions file can be
written and read back without data loss. This is critical for:

* Requirement 16.1: Write Evaluation_Report as JSON document
* Requirement 16.3: Write per-image predictions JSON file

The property tests generate:

* Valid Evaluation_Report dicts with all required fields
* Per-image predictions structures with varying numbers of images and detections
* Edge cases like empty predictions, special float values, and Unicode strings

**Validates: Requirements 16.1, 16.3**
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Hypothesis strategies for generating valid report structures
# ---------------------------------------------------------------------------

# Safe strings that won't cause JSON issues
_SAFE_STRINGS = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
        blacklist_characters="\x00",
    ),
    min_size=1,
    max_size=50,
)

# Class names (simple identifiers)
_CLASS_NAMES = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), blacklist_characters="\x00"),
    min_size=1,
    max_size=20,
).filter(lambda s: s.strip() and s[0].isalpha())


@st.composite
def _class_name_list(draw) -> List[str]:
    """Draw a list of 1-10 unique class names."""
    count = draw(st.integers(min_value=1, max_value=10))
    names = [f"class_{i}" for i in range(count)]
    return names


@st.composite
def _metric_value(draw) -> float:
    """Draw a valid metric value in [0.0, 1.0]."""
    return draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))


@st.composite
def _per_class_ap(draw, class_names: List[str]) -> Dict[str, float]:
    """Draw per-class AP values for the given class names."""
    return {name: draw(_metric_value()) for name in class_names}


@st.composite
def _metrics_dict(draw, class_names: List[str]) -> dict:
    """Draw a valid metrics dict with all required fields."""
    per_class = draw(_per_class_ap(class_names))
    map_50 = draw(_metric_value())
    map_50_95 = draw(_metric_value())
    precision = draw(_metric_value())
    recall = draw(_metric_value())
    f1_score = draw(_metric_value())
    
    return {
        "map_50": map_50,
        "map_50_95": map_50_95,
        "mAP@0.5": map_50,  # Prior key retained
        "mAP@0.5:0.95": map_50_95,  # Prior key retained
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "per_class_ap": per_class,
    }


@st.composite
def _confusion_matrix(draw, num_classes: int) -> List[List[int]]:
    """Draw a confusion matrix of shape (num_classes, num_classes)."""
    return [
        [draw(st.integers(min_value=0, max_value=100)) for _ in range(num_classes)]
        for _ in range(num_classes)
    ]


@st.composite
def _error_list(draw) -> List[str]:
    """Draw a list of error strings."""
    count = draw(st.integers(min_value=0, max_value=5))
    errors = []
    for i in range(count):
        image_id = f"image_{i}"
        error_text = draw(st.text(min_size=1, max_size=50).filter(lambda s: s.strip()))
        errors.append(f"{image_id}: {error_text}")
    return errors


@st.composite
def _model_config(draw) -> dict:
    """Draw a valid model configuration dict."""
    return {
        "num_classes": draw(st.integers(min_value=1, max_value=100)),
        "input_size": draw(st.integers(min_value=32, max_value=1024)),
        "confidence_threshold": draw(_metric_value()),
        "iou_threshold": draw(_metric_value()),
    }


@st.composite
def _evaluation_report(draw) -> dict:
    """Draw a complete, valid Evaluation_Report dict."""
    class_names = draw(_class_name_list())
    num_classes = len(class_names)
    num_images = draw(st.integers(min_value=1, max_value=100))
    errors = draw(_error_list())
    
    return {
        "checkpoint": draw(st.text(min_size=1, max_size=100).filter(lambda s: s.strip())),
        "model_type": draw(st.sampled_from(["yolo26", "ssd_mobilenet", "yolov6"])),
        "model_config": draw(_model_config()),
        "dataset": draw(st.text(min_size=1, max_size=100).filter(lambda s: s.strip())),
        "split": draw(st.sampled_from(["train", "val", "test"])),
        "num_images": num_images,
        "num_classes": num_classes,
        "class_names": class_names,
        "confidence_threshold": draw(_metric_value()),
        "iou_threshold": draw(_metric_value()),
        "metrics": draw(_metrics_dict(class_names)),
        "confusion_matrix": draw(_confusion_matrix(num_classes)),
        "errors": {
            "count": len(errors),
            "items": errors,
        },
    }


# ---------------------------------------------------------------------------
# Strategies for per-image predictions file
# ---------------------------------------------------------------------------


@st.composite
def _normalized_box(draw) -> List[float]:
    """Draw a valid, non-degenerate box in normalized [0, 1] coordinates."""
    x_min = draw(st.floats(min_value=0.0, max_value=0.4, allow_nan=False, allow_infinity=False))
    y_min = draw(st.floats(min_value=0.0, max_value=0.4, allow_nan=False, allow_infinity=False))
    x_max = draw(st.floats(min_value=x_min + 0.1, max_value=1.0, allow_nan=False, allow_infinity=False))
    y_max = draw(st.floats(min_value=y_min + 0.1, max_value=1.0, allow_nan=False, allow_infinity=False))
    return [x_min, y_min, x_max, y_max]


@st.composite
def _image_prediction(draw, class_names: List[str]) -> dict:
    """Draw a single image's prediction entry."""
    num_detections = draw(st.integers(min_value=0, max_value=5))
    boxes = [draw(_normalized_box()) for _ in range(num_detections)]
    labels = [draw(st.sampled_from(class_names)) for _ in range(num_detections)]
    scores = [draw(_metric_value()) for _ in range(num_detections)]
    return {"boxes": boxes, "labels": labels, "scores": scores}


@st.composite
def _image_ground_truth(draw, class_names: List[str]) -> dict:
    """Draw a single image's ground truth entry."""
    num_boxes = draw(st.integers(min_value=0, max_value=5))
    boxes = [draw(_normalized_box()) for _ in range(num_boxes)]
    labels = [draw(st.sampled_from(class_names)) for _ in range(num_boxes)]
    return {"boxes": boxes, "labels": labels}


@st.composite
def _image_entry(draw, class_names: List[str], image_id: str) -> dict:
    """Draw a complete image entry for the predictions file."""
    return {
        "image_id": image_id,
        "ground_truth": draw(_image_ground_truth(class_names)),
        "predictions": draw(_image_prediction(class_names)),
    }


@st.composite
def _predictions_file(draw) -> dict:
    """Draw a complete per-image predictions file structure."""
    class_names = draw(_class_name_list())
    num_images = draw(st.integers(min_value=1, max_value=20))
    
    images = [
        draw(_image_entry(class_names, f"image_{i}"))
        for i in range(num_images)
    ]
    
    return {
        "checkpoint": draw(st.text(min_size=1, max_size=100).filter(lambda s: s.strip())),
        "model_type": draw(st.sampled_from(["yolo26", "ssd_mobilenet", "yolov6"])),
        "dataset": draw(st.text(min_size=1, max_size=100).filter(lambda s: s.strip())),
        "confidence_threshold": draw(_metric_value()),
        "class_names": class_names,
        "images": images,
    }


# ---------------------------------------------------------------------------
# Helper functions for JSON round-trip
# ---------------------------------------------------------------------------


def _json_roundtrip(data: Any) -> Any:
    """Serialize to JSON and deserialize back."""
    json_str = json.dumps(data, indent=2)
    return json.loads(json_str)


def _floats_equal(a: float, b: float, rel_tol: float = 1e-9) -> bool:
    """Compare floats with tolerance for JSON round-trip precision."""
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isinf(a) and math.isinf(b):
        return a == b  # Same sign infinity
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-15)


def _deep_equal(a: Any, b: Any) -> bool:
    """Deep equality check with float tolerance."""
    if type(a) != type(b):
        # Special case: int vs float (JSON may convert)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return _floats_equal(float(a), float(b))
        return False
    
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_deep_equal(a[k], b[k]) for k in a.keys())
    
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(_deep_equal(ai, bi) for ai, bi in zip(a, b))
    
    if isinstance(a, float):
        return _floats_equal(a, b)
    
    return a == b


# ---------------------------------------------------------------------------
# Property 20: Report JSON round-trip preserves content
# ---------------------------------------------------------------------------


class TestProperty20JsonRoundtrip:
    """Property 20: Report JSON round-trip preserves content.

    **Validates: Requirements 16.1, 16.3**
    """

    @given(report=_evaluation_report())
    @settings(max_examples=100, deadline=None)
    def test_evaluation_report_roundtrip_preserves_content(self, report):
        # Feature: generic-evaluation-script, Property 20: Report JSON round-trip
        """Writing an Evaluation_Report to JSON and reading it back preserves all content.

        For any valid Evaluation_Report dict, the JSON round-trip produces an
        identical structure (numbers, strings, lists, and nested objects preserved).

        **Validates: Requirements 16.1**
        """
        roundtripped = _json_roundtrip(report)
        
        assert _deep_equal(report, roundtripped), (
            f"Report content changed after JSON round-trip.\n"
            f"Original: {report}\n"
            f"Roundtripped: {roundtripped}"
        )

    @given(predictions=_predictions_file())
    @settings(max_examples=100, deadline=None)
    def test_predictions_file_roundtrip_preserves_content(self, predictions):
        # Feature: generic-evaluation-script, Property 20: Report JSON round-trip
        """Writing a predictions file to JSON and reading it back preserves all content.

        For any valid per-image predictions structure, the JSON round-trip
        produces an identical structure.

        **Validates: Requirements 16.3**
        """
        roundtripped = _json_roundtrip(predictions)
        
        assert _deep_equal(predictions, roundtripped), (
            f"Predictions content changed after JSON round-trip.\n"
            f"Original: {predictions}\n"
            f"Roundtripped: {roundtripped}"
        )

    @given(predictions=_predictions_file())
    @settings(max_examples=100, deadline=None)
    def test_predictions_file_has_one_entry_per_image(self, predictions):
        # Feature: generic-evaluation-script, Property 20: Report JSON round-trip
        """The predictions file contains exactly one image object per image.

        Each image entry must have ``image_id``, ``boxes``, ``labels``, and
        ``scores`` in its predictions sub-object.

        **Validates: Requirements 16.3**
        """
        roundtripped = _json_roundtrip(predictions)
        
        images = roundtripped["images"]
        image_ids = [img["image_id"] for img in images]
        
        # Each image has a unique image_id
        assert len(image_ids) == len(set(image_ids)), (
            f"Duplicate image_ids found: {image_ids}"
        )
        
        # Each image entry has the required fields
        for img in images:
            assert "image_id" in img, "Missing image_id in image entry"
            assert "predictions" in img, "Missing predictions in image entry"
            assert "ground_truth" in img, "Missing ground_truth in image entry"
            
            pred = img["predictions"]
            assert "boxes" in pred, "Missing boxes in predictions"
            assert "labels" in pred, "Missing labels in predictions"
            assert "scores" in pred, "Missing scores in predictions"

    @given(report=_evaluation_report())
    @settings(max_examples=100, deadline=None)
    def test_report_file_write_and_read(self, report, tmp_path_factory):
        # Feature: generic-evaluation-script, Property 20: Report JSON round-trip
        """Writing a report to a file and reading it back preserves content.

        This tests the actual file I/O path, not just in-memory serialization.

        **Validates: Requirements 16.1**
        """
        tmp_path = tmp_path_factory.mktemp("report_roundtrip")
        report_path = tmp_path / "test_report.json"
        
        # Write to file
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        
        # Read back
        with open(report_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        
        assert _deep_equal(report, loaded), (
            f"Report content changed after file write/read.\n"
            f"Original: {report}\n"
            f"Loaded: {loaded}"
        )

    @given(predictions=_predictions_file())
    @settings(max_examples=100, deadline=None)
    def test_predictions_file_write_and_read(self, predictions, tmp_path_factory):
        # Feature: generic-evaluation-script, Property 20: Report JSON round-trip
        """Writing predictions to a file and reading it back preserves content.

        This tests the actual file I/O path for the predictions file.

        **Validates: Requirements 16.3**
        """
        tmp_path = tmp_path_factory.mktemp("predictions_roundtrip")
        predictions_path = tmp_path / "test_predictions.json"
        
        # Write to file
        with open(predictions_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2)
        
        # Read back
        with open(predictions_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        
        assert _deep_equal(predictions, loaded), (
            f"Predictions content changed after file write/read.\n"
            f"Original: {predictions}\n"
            f"Loaded: {loaded}"
        )


# ---------------------------------------------------------------------------
# Example-based tests complementing Property 20
# ---------------------------------------------------------------------------


class TestJsonRoundtripExamples:
    """Concrete examples complementing Property 20.

    **Validates: Requirements 16.1, 16.3**
    """

    def test_report_with_all_required_fields_roundtrips(self):
        """A complete report with all required fields survives JSON round-trip."""
        report = {
            "checkpoint": "/path/to/checkpoint.pt",
            "model_type": "yolo26",
            "model_config": {"num_classes": 5, "input_size": 640},
            "dataset": "/path/to/dataset",
            "split": "val",
            "num_images": 100,
            "num_classes": 5,
            "class_names": ["crack", "pothole", "spalling", "rutting", "other"],
            "confidence_threshold": 0.25,
            "iou_threshold": 0.5,
            "metrics": {
                "map_50": 0.75,
                "map_50_95": 0.65,
                "mAP@0.5": 0.75,
                "mAP@0.5:0.95": 0.65,
                "precision": 0.8,
                "recall": 0.7,
                "f1_score": 0.746,
                "per_class_ap": {
                    "crack": 0.8,
                    "pothole": 0.7,
                    "spalling": 0.75,
                    "rutting": 0.72,
                    "other": 0.78,
                },
            },
            "confusion_matrix": [[10, 2, 1, 0, 0], [1, 15, 0, 1, 0], [0, 1, 12, 0, 1], [0, 0, 1, 8, 0], [1, 0, 0, 0, 5]],
            "errors": {"count": 2, "items": ["img_1: decode error", "img_5: timeout"]},
        }
        
        roundtripped = _json_roundtrip(report)
        assert _deep_equal(report, roundtripped)

    def test_report_with_empty_errors_roundtrips(self):
        """A report with no errors survives JSON round-trip."""
        report = {
            "checkpoint": "checkpoint.pt",
            "model_type": "ssd_mobilenet",
            "model_config": {"num_classes": 3},
            "dataset": "data/rdd2022",
            "split": "test",
            "num_images": 50,
            "num_classes": 3,
            "class_names": ["a", "b", "c"],
            "confidence_threshold": 0.5,
            "iou_threshold": 0.5,
            "metrics": {
                "map_50": 0.9,
                "map_50_95": 0.85,
                "mAP@0.5": 0.9,
                "mAP@0.5:0.95": 0.85,
                "precision": 0.88,
                "recall": 0.92,
                "f1_score": 0.9,
                "per_class_ap": {"a": 0.9, "b": 0.88, "c": 0.92},
            },
            "confusion_matrix": [[5, 0, 0], [0, 6, 1], [0, 0, 4]],
            "errors": {"count": 0, "items": []},
        }
        
        roundtripped = _json_roundtrip(report)
        assert _deep_equal(report, roundtripped)

    def test_predictions_file_with_empty_predictions_roundtrips(self):
        """A predictions file with empty predictions survives JSON round-trip."""
        predictions = {
            "checkpoint": "checkpoint.pt",
            "model_type": "yolo26",
            "dataset": "data/rdd2022",
            "confidence_threshold": 0.25,
            "class_names": ["crack", "pothole"],
            "images": [
                {
                    "image_id": "img_0",
                    "ground_truth": {"boxes": [[0.1, 0.1, 0.5, 0.5]], "labels": ["crack"]},
                    "predictions": {"boxes": [], "labels": [], "scores": []},
                },
                {
                    "image_id": "img_1",
                    "ground_truth": {"boxes": [], "labels": []},
                    "predictions": {"boxes": [[0.2, 0.2, 0.6, 0.6]], "labels": ["pothole"], "scores": [0.9]},
                },
            ],
        }
        
        roundtripped = _json_roundtrip(predictions)
        assert _deep_equal(predictions, roundtripped)

    def test_predictions_file_with_multiple_detections_roundtrips(self):
        """A predictions file with multiple detections per image survives JSON round-trip."""
        predictions = {
            "checkpoint": "checkpoint.pt",
            "model_type": "yolov6",
            "dataset": "data/rdd2022",
            "confidence_threshold": 0.3,
            "class_names": ["crack", "pothole", "spalling"],
            "images": [
                {
                    "image_id": "img_0",
                    "ground_truth": {
                        "boxes": [[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.8, 0.8]],
                        "labels": ["crack", "pothole"],
                    },
                    "predictions": {
                        "boxes": [[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.8, 0.8], [0.2, 0.6, 0.4, 0.9]],
                        "labels": ["crack", "pothole", "spalling"],
                        "scores": [0.95, 0.88, 0.72],
                    },
                },
            ],
        }
        
        roundtripped = _json_roundtrip(predictions)
        assert _deep_equal(predictions, roundtripped)

    def test_report_with_zero_metrics_roundtrips(self):
        """A report with all-zero metrics survives JSON round-trip."""
        report = {
            "checkpoint": "checkpoint.pt",
            "model_type": "yolo26",
            "model_config": {"num_classes": 2},
            "dataset": "data/rdd2022",
            "split": "val",
            "num_images": 10,
            "num_classes": 2,
            "class_names": ["a", "b"],
            "confidence_threshold": 0.5,
            "iou_threshold": 0.5,
            "metrics": {
                "map_50": 0.0,
                "map_50_95": 0.0,
                "mAP@0.5": 0.0,
                "mAP@0.5:0.95": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "per_class_ap": {"a": 0.0, "b": 0.0},
            },
            "confusion_matrix": [[0, 0], [0, 0]],
            "errors": {"count": 10, "items": [f"img_{i}: error" for i in range(10)]},
        }
        
        roundtripped = _json_roundtrip(report)
        assert _deep_equal(report, roundtripped)

    def test_report_with_unicode_strings_roundtrips(self):
        """A report with Unicode strings survives JSON round-trip."""
        report = {
            "checkpoint": "/path/to/模型.pt",
            "model_type": "yolo26",
            "model_config": {"num_classes": 2, "description": "Détection de défauts"},
            "dataset": "/données/rdd2022",
            "split": "val",
            "num_images": 5,
            "num_classes": 2,
            "class_names": ["fisura", "bache"],
            "confidence_threshold": 0.25,
            "iou_threshold": 0.5,
            "metrics": {
                "map_50": 0.8,
                "map_50_95": 0.7,
                "mAP@0.5": 0.8,
                "mAP@0.5:0.95": 0.7,
                "precision": 0.85,
                "recall": 0.75,
                "f1_score": 0.8,
                "per_class_ap": {"fisura": 0.82, "bache": 0.78},
            },
            "confusion_matrix": [[3, 1], [0, 4]],
            "errors": {"count": 1, "items": ["img_日本: エラー"]},
        }
        
        roundtripped = _json_roundtrip(report)
        assert _deep_equal(report, roundtripped)

    def test_report_with_float_precision_roundtrips(self):
        """A report with high-precision floats survives JSON round-trip."""
        report = {
            "checkpoint": "checkpoint.pt",
            "model_type": "yolo26",
            "model_config": {"num_classes": 1},
            "dataset": "data",
            "split": "val",
            "num_images": 1,
            "num_classes": 1,
            "class_names": ["a"],
            "confidence_threshold": 0.123456789,
            "iou_threshold": 0.987654321,
            "metrics": {
                "map_50": 0.111111111,
                "map_50_95": 0.222222222,
                "mAP@0.5": 0.111111111,
                "mAP@0.5:0.95": 0.222222222,
                "precision": 0.333333333,
                "recall": 0.444444444,
                "f1_score": 0.555555555,
                "per_class_ap": {"a": 0.666666666},
            },
            "confusion_matrix": [[1]],
            "errors": {"count": 0, "items": []},
        }
        
        roundtripped = _json_roundtrip(report)
        assert _deep_equal(report, roundtripped)

    def test_single_image_predictions_file_roundtrips(self):
        """A predictions file with a single image survives JSON round-trip."""
        predictions = {
            "checkpoint": "checkpoint.pt",
            "model_type": "yolo26",
            "dataset": "data",
            "confidence_threshold": 0.5,
            "class_names": ["crack"],
            "images": [
                {
                    "image_id": "single_image",
                    "ground_truth": {"boxes": [[0.1, 0.1, 0.5, 0.5]], "labels": ["crack"]},
                    "predictions": {"boxes": [[0.1, 0.1, 0.5, 0.5]], "labels": ["crack"], "scores": [0.99]},
                },
            ],
        }
        
        roundtripped = _json_roundtrip(predictions)
        assert _deep_equal(predictions, roundtripped)
        assert len(roundtripped["images"]) == 1
        assert roundtripped["images"][0]["image_id"] == "single_image"

    def test_predictions_file_image_entries_have_required_fields(self):
        """Each image entry in predictions file has all required fields."""
        predictions = {
            "checkpoint": "checkpoint.pt",
            "model_type": "yolo26",
            "dataset": "data",
            "confidence_threshold": 0.5,
            "class_names": ["a", "b"],
            "images": [
                {
                    "image_id": f"img_{i}",
                    "ground_truth": {"boxes": [], "labels": []},
                    "predictions": {"boxes": [], "labels": [], "scores": []},
                }
                for i in range(5)
            ],
        }
        
        roundtripped = _json_roundtrip(predictions)
        
        for img in roundtripped["images"]:
            assert "image_id" in img
            assert "ground_truth" in img
            assert "predictions" in img
            assert "boxes" in img["predictions"]
            assert "labels" in img["predictions"]
            assert "scores" in img["predictions"]
