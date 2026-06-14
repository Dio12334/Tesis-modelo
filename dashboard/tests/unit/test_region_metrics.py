"""Unit tests for :mod:`dashboard.region_metrics`.

Covers the slim helper surface the dashboard actually uses:

* :func:`extract_region` matches each known RDD2022 prefix (incl.
  multi-segment prefixes like ``China_Drone`` / ``United_States``).
* :func:`discover_regions` returns sorted unique regions present in an
  inference dump.
* :func:`filter_predictions_by_region` filters the ``images`` list while
  preserving top-level metadata, returns the input untouched for the
  ``All`` sentinel, and propagates ``None``.
* :func:`regions_from_report` reads the report's ``per_region`` field.
* :func:`region_report_view` swaps ``metrics`` / ``confusion_matrix`` /
  ``num_val_images`` for the requested region without mutating the input.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from dashboard.data_loader import EvaluationReport
from dashboard.region_metrics import (
    ALL_REGIONS_LABEL,
    discover_regions,
    extract_region,
    filter_predictions_by_region,
    region_report_view,
    regions_from_report,
)


CLASS_NAMES = [
    "alligator crack",
    "longitudinal crack",
    "other corruption",
    "pothole",
    "transverse crack",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _img(image_id: str, gt: dict, pred: dict) -> dict:
    return {
        "image_id": image_id,
        "ground_truth": gt,
        "predictions": pred,
    }


@pytest.fixture
def sample_predictions_data() -> dict:
    return {
        "checkpoint": "checkpoints/dummy/best.pt",
        "model_type": "dummy",
        "dataset": "model/data/rdd2022/complete",
        "confidence_threshold": 0.25,
        "class_names": CLASS_NAMES,
        "display_class_names": CLASS_NAMES,
        "images": [
            _img(
                "model/data/rdd2022/complete/train/img/Japan_000001.jpg",
                gt={"boxes": [[0.1, 0.1, 0.3, 0.3]], "labels": ["pothole"]},
                pred={
                    "boxes": [[0.11, 0.11, 0.29, 0.29]],
                    "labels": ["pothole"],
                    "scores": [0.91],
                },
            ),
            _img(
                "model/data/rdd2022/complete/train/img/Japan_000002.jpg",
                gt={"boxes": [[0.5, 0.5, 0.7, 0.7]], "labels": ["alligator crack"]},
                pred={
                    "boxes": [[0.0, 0.0, 0.05, 0.05]],
                    "labels": ["pothole"],
                    "scores": [0.6],
                },
            ),
            _img(
                "model/data/rdd2022/complete/train/img/United_States_000010.jpg",
                gt={"boxes": [[0.2, 0.2, 0.4, 0.4]], "labels": ["longitudinal crack"]},
                pred={
                    "boxes": [[0.21, 0.21, 0.39, 0.39]],
                    "labels": ["longitudinal crack"],
                    "scores": [0.85],
                },
            ),
        ],
    }


def _per_region_block(num_images: int, num_gt_boxes: int, *, scale: float = 1.0) -> dict:
    """Build a per-region metric block with the dashboard-required keys."""
    size = len(CLASS_NAMES) + 1
    return {
        "num_images": num_images,
        "num_gt_boxes": num_gt_boxes,
        "metrics": {
            "map_50": 0.4 * scale,
            "map_50_95": 0.25 * scale,
            "per_class_ap": {name: 0.4 * scale for name in CLASS_NAMES},
            "mAP@0.5": 0.4 * scale,
            "mAP@0.5:0.95": 0.25 * scale,
            "precision": 0.5 * scale,
            "recall": 0.5 * scale,
            "f1_score": 0.5 * scale,
        },
        "confusion_matrix": [[0] * size for _ in range(size)],
    }


@pytest.fixture
def report_with_per_region() -> EvaluationReport:
    base_cm = [[0] * (len(CLASS_NAMES) + 1) for _ in range(len(CLASS_NAMES) + 1)]
    return EvaluationReport(
        checkpoint="checkpoints/dummy/best.pt",
        dataset="model/data/rdd2022/complete",
        num_val_images=100,
        num_classes=len(CLASS_NAMES),
        class_names=CLASS_NAMES,
        confidence_threshold=0.25,
        iou_threshold=0.5,
        metrics={
            "map_50": 0.6,
            "map_50_95": 0.4,
            "per_class_ap": {name: 0.6 for name in CLASS_NAMES},
            "mAP@0.5": 0.6,
            "mAP@0.5:0.95": 0.4,
            "precision": 0.7,
            "recall": 0.6,
            "f1_score": 0.65,
        },
        confusion_matrix=base_cm,
        display_class_names=CLASS_NAMES,
        per_region={
            "Japan": _per_region_block(40, 100, scale=1.0),
            "Czech": _per_region_block(20, 50, scale=0.5),
        },
    )


@pytest.fixture
def report_without_per_region() -> EvaluationReport:
    base_cm = [[0] * (len(CLASS_NAMES) + 1) for _ in range(len(CLASS_NAMES) + 1)]
    return EvaluationReport(
        checkpoint="checkpoints/dummy/best.pt",
        dataset="model/data/rdd2022/complete",
        num_val_images=100,
        num_classes=len(CLASS_NAMES),
        class_names=CLASS_NAMES,
        confidence_threshold=0.25,
        iou_threshold=0.5,
        metrics={"map_50": 0.6, "f1_score": 0.65},
        confusion_matrix=base_cm,
        display_class_names=CLASS_NAMES,
    )


# ---------------------------------------------------------------------------
# extract_region
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "image_id,expected",
    [
        ("Japan_001234.jpg", "Japan"),
        ("Czech_000001.jpg", "Czech"),
        ("India_009999.png", "India"),
        ("Norway_000007.jpeg", "Norway"),
        ("United_States_002354.jpg", "United_States"),
        ("China_Drone_000000.jpg", "China_Drone"),
        ("China_MotorBike_000123.jpg", "China_MotorBike"),
        (
            "model/data/rdd2022/complete/train/img/United_States_002354.jpg",
            "United_States",
        ),
    ],
)
def test_extract_region_matches_known_prefixes(image_id: str, expected: str) -> None:
    assert extract_region(image_id) == expected


@pytest.mark.parametrize(
    "image_id",
    [
        "",
        "no_digits.jpg",
        "trailing_underscore_.jpg",
        "noextension_001234",
        "weird.name.jpg",
    ],
)
def test_extract_region_returns_none_for_unrecognised(image_id: str) -> None:
    assert extract_region(image_id) is None


# ---------------------------------------------------------------------------
# discover_regions
# ---------------------------------------------------------------------------


def test_discover_regions_sorted_unique(sample_predictions_data: dict) -> None:
    assert discover_regions(sample_predictions_data) == ["Japan", "United_States"]


def test_discover_regions_handles_none() -> None:
    assert discover_regions(None) == []
    assert discover_regions({}) == []
    assert discover_regions({"images": []}) == []


def test_discover_regions_skips_unrecognised_filenames() -> None:
    data = {
        "images": [
            {"image_id": "Japan_000001.jpg"},
            {"image_id": "no_digits.jpg"},
            {"image_id": "Czech_000002.jpg"},
        ]
    }
    assert discover_regions(data) == ["Czech", "Japan"]


# ---------------------------------------------------------------------------
# filter_predictions_by_region
# ---------------------------------------------------------------------------


def test_filter_keeps_only_matching_region(sample_predictions_data: dict) -> None:
    filtered = filter_predictions_by_region(sample_predictions_data, "Japan")
    assert filtered is not None
    assert len(filtered["images"]) == 2
    assert all(
        extract_region(img["image_id"]) == "Japan" for img in filtered["images"]
    )


def test_filter_preserves_top_level_metadata(sample_predictions_data: dict) -> None:
    filtered = filter_predictions_by_region(sample_predictions_data, "Japan")
    assert filtered["class_names"] == sample_predictions_data["class_names"]
    assert filtered["confidence_threshold"] == 0.25
    assert filtered["dataset"] == sample_predictions_data["dataset"]


def test_filter_does_not_mutate_input(sample_predictions_data: dict) -> None:
    original = deepcopy(sample_predictions_data)
    _ = filter_predictions_by_region(sample_predictions_data, "Japan")
    assert sample_predictions_data == original


def test_filter_all_returns_input_unchanged(sample_predictions_data: dict) -> None:
    assert (
        filter_predictions_by_region(sample_predictions_data, ALL_REGIONS_LABEL)
        is sample_predictions_data
    )


def test_filter_unknown_region_returns_empty_images(
    sample_predictions_data: dict,
) -> None:
    filtered = filter_predictions_by_region(sample_predictions_data, "Mars")
    assert filtered["images"] == []


def test_filter_propagates_none() -> None:
    assert filter_predictions_by_region(None, "Japan") is None


# ---------------------------------------------------------------------------
# regions_from_report
# ---------------------------------------------------------------------------


def test_regions_from_report_returns_sorted_keys(
    report_with_per_region: EvaluationReport,
) -> None:
    assert regions_from_report(report_with_per_region) == ["Czech", "Japan"]


def test_regions_from_report_handles_legacy_report(
    report_without_per_region: EvaluationReport,
) -> None:
    assert regions_from_report(report_without_per_region) == []


def test_regions_from_report_handles_none() -> None:
    assert regions_from_report(None) == []


# ---------------------------------------------------------------------------
# region_report_view
# ---------------------------------------------------------------------------


def test_region_report_view_swaps_metrics_for_selected_region(
    report_with_per_region: EvaluationReport,
) -> None:
    view = region_report_view(report_with_per_region, "Japan")
    assert view is not report_with_per_region
    assert view.metrics["f1_score"] == pytest.approx(0.5)
    assert view.num_val_images == 40


def test_region_report_view_returns_input_for_all(
    report_with_per_region: EvaluationReport,
) -> None:
    assert (
        region_report_view(report_with_per_region, ALL_REGIONS_LABEL)
        is report_with_per_region
    )


def test_region_report_view_returns_input_for_missing_region(
    report_with_per_region: EvaluationReport,
) -> None:
    """Unknown region falls back to the global view (no mutation)."""
    assert (
        region_report_view(report_with_per_region, "Mars")
        is report_with_per_region
    )


def test_region_report_view_does_not_mutate_input(
    report_with_per_region: EvaluationReport,
) -> None:
    original_metrics = deepcopy(report_with_per_region.metrics)
    original_cm = deepcopy(report_with_per_region.confusion_matrix)
    original_num = report_with_per_region.num_val_images
    _ = region_report_view(report_with_per_region, "Czech")
    assert report_with_per_region.metrics == original_metrics
    assert report_with_per_region.confusion_matrix == original_cm
    assert report_with_per_region.num_val_images == original_num


def test_region_report_view_swaps_confusion_matrix(
    report_with_per_region: EvaluationReport,
) -> None:
    view = region_report_view(report_with_per_region, "Japan")
    expected = report_with_per_region.per_region["Japan"]["confusion_matrix"]
    assert view.confusion_matrix == expected
