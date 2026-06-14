"""Regression tests for the per-region evaluator wiring.

Covers the integration points added when ``model/evaluation/region.py`` was
hooked into ``model/training/evaluate_detection.py``:

* :func:`assemble_report` accepts a ``per_region`` kwarg and embeds the
  region-keyed metric blocks under a top-level ``per_region`` field.
* When ``per_region`` is omitted (or ``None``), the field is **not**
  present in the assembled report — preserving the prior schema for
  legacy callers and non-RDD2022 runs.
* :func:`print_summary` accepts a ``per_region`` kwarg and emits a
  per-region table after the global summary, with one row per region.
* The summary is suppressed when ``per_region`` is ``None``/empty.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from model.training.evaluate_detection import assemble_report, print_summary


CLASS_NAMES = ["alligator crack", "longitudinal crack", "pothole"]


def _baseline_metrics() -> dict:
    return {
        "map_50": 0.5,
        "map_50_95": 0.3,
        "per_class_ap": {name: 0.5 for name in CLASS_NAMES},
        "precision": 0.7,
        "recall": 0.6,
        "f1_score": 0.65,
    }


def _baseline_kwargs(metrics: dict) -> dict:
    size = len(CLASS_NAMES)
    return dict(
        checkpoint_path="checkpoints/foo/best.pt",
        model_type="yolo26",
        model_config={"input_size": 640},
        dataset_path="model/data/rdd2022/complete",
        split="val",
        num_images=10,
        num_classes=size,
        class_names=list(CLASS_NAMES),
        confidence_threshold=0.25,
        iou_threshold=0.5,
        metrics=metrics,
        confusion_matrix=[[0] * size for _ in range(size)],
        errors=[],
    )


def _per_region_block(num_images: int, num_gt_boxes: int) -> dict:
    size = len(CLASS_NAMES) + 1
    return {
        "num_images": num_images,
        "num_gt_boxes": num_gt_boxes,
        "metrics": {
            "map_50": 0.4,
            "map_50_95": 0.25,
            "per_class_ap": {name: 0.4 for name in CLASS_NAMES},
            "mAP@0.5": 0.4,
            "mAP@0.5:0.95": 0.25,
            "precision": 0.5,
            "recall": 0.5,
            "f1_score": 0.5,
        },
        "confusion_matrix": [[0] * size for _ in range(size)],
    }


# ---------------------------------------------------------------------------
# assemble_report
# ---------------------------------------------------------------------------


def test_assemble_report_omits_per_region_field_by_default() -> None:
    report = assemble_report(**_baseline_kwargs(_baseline_metrics()))
    assert "per_region" not in report


def test_assemble_report_omits_per_region_when_none_explicit() -> None:
    report = assemble_report(
        **_baseline_kwargs(_baseline_metrics()),
        per_region=None,
    )
    assert "per_region" not in report


def test_assemble_report_embeds_per_region_when_supplied() -> None:
    per_region = {
        "Japan": _per_region_block(num_images=5, num_gt_boxes=12),
        "Czech": _per_region_block(num_images=3, num_gt_boxes=6),
    }
    report = assemble_report(
        **_baseline_kwargs(_baseline_metrics()),
        per_region=per_region,
    )
    assert "per_region" in report
    assert set(report["per_region"].keys()) == {"Japan", "Czech"}
    assert report["per_region"]["Japan"]["num_images"] == 5
    assert report["per_region"]["Japan"]["num_gt_boxes"] == 12
    assert "metrics" in report["per_region"]["Czech"]


def test_assemble_report_per_region_shape_matches_global() -> None:
    """Each region block must carry the metric keys the dashboard reads."""
    per_region = {"Japan": _per_region_block(num_images=2, num_gt_boxes=3)}
    report = assemble_report(
        **_baseline_kwargs(_baseline_metrics()),
        per_region=per_region,
    )
    block = report["per_region"]["Japan"]
    metrics = block["metrics"]
    for key in (
        "map_50",
        "map_50_95",
        "precision",
        "recall",
        "f1_score",
        "per_class_ap",
    ):
        assert key in metrics, key
    assert "confusion_matrix" in block


# ---------------------------------------------------------------------------
# print_summary
# ---------------------------------------------------------------------------


def _capture(metrics: dict, **kwargs) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_summary(
            model_type="yolo26",
            split="val",
            num_images=10,
            metrics=metrics,
            **kwargs,
        )
    return buf.getvalue()


def test_print_summary_without_per_region_omits_section() -> None:
    out = _capture(_baseline_metrics())
    assert "EVALUATION SUMMARY" in out
    assert "PER-REGION SUMMARY" not in out


def test_print_summary_with_empty_per_region_dict_omits_section() -> None:
    out = _capture(_baseline_metrics(), per_region={})
    assert "PER-REGION SUMMARY" not in out


def test_print_summary_with_per_region_emits_table() -> None:
    per_region = {
        "Japan": _per_region_block(num_images=5, num_gt_boxes=12),
        "Czech": _per_region_block(num_images=3, num_gt_boxes=6),
    }
    out = _capture(_baseline_metrics(), per_region=per_region)
    assert "PER-REGION SUMMARY" in out
    # Both regions appear in the table.
    assert "Japan" in out
    assert "Czech" in out
    # Sorted ordering: Czech precedes Japan.
    assert out.index("Czech") < out.index("Japan")


def test_print_summary_per_region_row_includes_metric_values() -> None:
    per_region = {"Japan": _per_region_block(num_images=5, num_gt_boxes=12)}
    out = _capture(_baseline_metrics(), per_region=per_region)
    # Per-region block uses precision/recall/F1 = 0.5 — verify they appear.
    japan_line = [
        line for line in out.splitlines() if line.lstrip().startswith("Japan")
    ]
    assert len(japan_line) == 1
    line = japan_line[0]
    assert "0.5000" in line  # P / R / F1
    assert "0.4000" in line  # mAP@0.5
