"""Integration tests for the F1-sweep wiring in evaluate_detection.

These tests cover the Phase-2 evaluation-architecture additions inside
``model/training/evaluate_detection.py``:

* :func:`compute_all_metrics` accepts an optional
  ``confidence_thresholds_sweep`` and, when supplied, returns the four sweep
  fields (``f1_sweep``, ``best_f1``, ``per_class_best_f1``,
  ``default_confidence_threshold``).
* :func:`compute_all_metrics` preserves the prior contract (no sweep fields)
  when the sweep argument is omitted.
* :func:`assemble_report` propagates the sweep fields into the report's
  ``metrics`` block when they are present, and tolerates their absence so
  legacy callers continue to work.

Tests are example-based (no Hypothesis), with hand-built predictions and
ground truths whose precision/recall outcomes are computable by hand.
"""

from __future__ import annotations

from typing import List

import pytest

from model.training.evaluate_detection import (
    _INFERENCE_CONFIDENCE_FLOOR,
    assemble_report,
    compute_all_metrics,
)


_BOX_A = [0.10, 0.10, 0.30, 0.30]
_BOX_B = [0.50, 0.50, 0.70, 0.70]


def _pred(image_id: str, items: List[tuple]) -> dict:
    return {
        "image_id": image_id,
        "boxes": [it[0] for it in items],
        "labels": [it[1] for it in items],
        "scores": [it[2] for it in items],
    }


def _gt(image_id: str, items: List[tuple]) -> dict:
    return {
        "image_id": image_id,
        "boxes": [it[0] for it in items],
        "labels": [it[1] for it in items],
    }


def _baseline_report_params() -> dict:
    """Minimal parameter dict for assemble_report excluding ``metrics``."""
    return {
        "checkpoint_path": "/tmp/best.pt",
        "model_type": "yolo26",
        "model_config": {"num_classes": 1},
        "dataset_path": "/tmp/rdd2022",
        "split": "val",
        "num_images": 1,
        "num_classes": 1,
        "class_names": ["bache"],
        "confidence_threshold": 0.25,
        "iou_threshold": 0.5,
        "confusion_matrix": [[1]],
        "errors": [],
    }


def _baseline_metrics() -> dict:
    """Minimal metrics dict containing all six required scalar fields."""
    return {
        "map_50": 0.5,
        "map_50_95": 0.3,
        "per_class_ap": {"bache": 0.5},
        "precision": 0.5,
        "recall": 0.5,
        "f1_score": 0.5,
        "confusion_matrix": [[1]],
    }


# ---------------------------------------------------------------------------
# _INFERENCE_CONFIDENCE_FLOOR sanity
# ---------------------------------------------------------------------------


class TestInferenceConfidenceFloor:
    """The inference-floor constant must match the Phase-2 design (0.001)."""

    def test_floor_is_low_enough_for_full_pr_curve(self):
        # 0.001 corresponds to the corrected-mAP eval used in Phase-1 re-evals.
        assert _INFERENCE_CONFIDENCE_FLOOR == 0.001


# ---------------------------------------------------------------------------
# compute_all_metrics — sweep optional
# ---------------------------------------------------------------------------


class TestComputeAllMetricsSweep:
    """``compute_all_metrics`` with/without ``confidence_thresholds_sweep``."""

    def test_without_sweep_returns_legacy_shape(self):
        predictions = [_pred("img1", [(_BOX_A, "bache", 0.9)])]
        ground_truths = [_gt("img1", [(_BOX_A, "bache")])]

        metrics = compute_all_metrics(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=["bache"],
            confidence_threshold=0.25,
            iou_threshold=0.5,
        )

        # Legacy keys present.
        for key in (
            "map_50",
            "map_50_95",
            "per_class_ap",
            "precision",
            "recall",
            "f1_score",
            "confusion_matrix",
        ):
            assert key in metrics

        # Sweep keys absent (legacy contract).
        for key in (
            "f1_sweep",
            "best_f1",
            "per_class_best_f1",
            "default_confidence_threshold",
        ):
            assert key not in metrics

    def test_with_sweep_emits_four_sweep_fields(self):
        predictions = [
            _pred(
                "img1",
                [
                    (_BOX_A, "bache", 0.9),
                    (_BOX_B, "bache", 0.4),
                ],
            ),
        ]
        ground_truths = [_gt("img1", [(_BOX_A, "bache"), (_BOX_B, "bache")])]
        thresholds = [0.05, 0.25, 0.5]

        metrics = compute_all_metrics(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=["bache"],
            confidence_threshold=0.25,
            iou_threshold=0.5,
            confidence_thresholds_sweep=thresholds,
        )

        # All four sweep keys present.
        assert "f1_sweep" in metrics
        assert "best_f1" in metrics
        assert "per_class_best_f1" in metrics
        assert "default_confidence_threshold" in metrics

        # f1_sweep length matches threshold count.
        assert len(metrics["f1_sweep"]) == len(thresholds)

        # default_confidence_threshold echoes the input confidence_threshold.
        assert metrics["default_confidence_threshold"] == pytest.approx(0.25)

        # per_class_best_f1 has one entry per class.
        assert set(metrics["per_class_best_f1"].keys()) == {"bache"}

        # best_f1 carries the four standard sweep keys.
        assert set(metrics["best_f1"].keys()) == {
            "confidence",
            "precision",
            "recall",
            "f1",
        }

    def test_with_sweep_best_f1_attains_one_for_perfect_predictions(self):
        # 2 perfect predictions matching 2 GTs at scores >= 0.4 → there exists
        # a sweep threshold in [0.05, 0.25, 0.4] where F1 == 1.0.
        predictions = [
            _pred(
                "img1",
                [(_BOX_A, "bache", 0.9), (_BOX_B, "bache", 0.4)],
            ),
        ]
        ground_truths = [_gt("img1", [(_BOX_A, "bache"), (_BOX_B, "bache")])]

        metrics = compute_all_metrics(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=["bache"],
            confidence_threshold=0.25,
            iou_threshold=0.5,
            confidence_thresholds_sweep=[0.05, 0.25, 0.4, 0.5],
        )

        assert metrics["best_f1"]["f1"] == pytest.approx(1.0)
        assert metrics["per_class_best_f1"]["bache"]["f1"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# assemble_report — sweep propagation
# ---------------------------------------------------------------------------


class TestAssembleReportSweepPropagation:
    """``assemble_report`` propagates sweep fields when present."""

    def test_propagates_sweep_fields_into_metrics_block(self):
        metrics = _baseline_metrics()
        metrics["f1_sweep"] = [
            {"confidence": 0.1, "precision": 0.5, "recall": 0.8, "f1": 0.62},
            {"confidence": 0.5, "precision": 0.9, "recall": 0.5, "f1": 0.64},
        ]
        metrics["best_f1"] = {
            "confidence": 0.5,
            "precision": 0.9,
            "recall": 0.5,
            "f1": 0.64,
        }
        metrics["per_class_best_f1"] = {
            "bache": {
                "confidence": 0.5,
                "precision": 0.9,
                "recall": 0.5,
                "f1": 0.64,
            }
        }
        metrics["default_confidence_threshold"] = 0.25

        report = assemble_report(metrics=metrics, **_baseline_report_params())

        report_metrics = report["metrics"]
        assert report_metrics["f1_sweep"] == metrics["f1_sweep"]
        assert report_metrics["best_f1"] == metrics["best_f1"]
        assert report_metrics["per_class_best_f1"] == metrics["per_class_best_f1"]
        assert report_metrics["default_confidence_threshold"] == 0.25

    def test_tolerates_missing_sweep_fields(self):
        # Legacy callers that don't compute the sweep must still get a valid
        # report — sweep fields simply absent.
        metrics = _baseline_metrics()

        report = assemble_report(metrics=metrics, **_baseline_report_params())

        report_metrics = report["metrics"]
        # Standard fields preserved.
        assert report_metrics["map_50"] == 0.5
        assert report_metrics["f1_score"] == 0.5
        # Sweep fields not synthesized.
        for key in (
            "f1_sweep",
            "best_f1",
            "per_class_best_f1",
            "default_confidence_threshold",
        ):
            assert key not in report_metrics
