"""Example-based tests for confidence-sweep metric helpers.

Covers the Phase-2 evaluation-architecture additions:

* :func:`compute_precision_recall_f1_sweep` — overall P/R/F1 across a list of
  confidence thresholds.
* :func:`find_best_f1` — picks the sweep entry with the highest F1, breaking
  ties by preferring the higher confidence (more conservative deployment
  point).
* :func:`compute_per_class_f1_sweep` — per-class best-F1 search across the
  configured confidence sweep.

Tests are example-based (no Hypothesis) and exercise hand-built prediction /
ground-truth pairs whose precision-recall outcomes are computed by hand, so any
regression in the sweep semantics fails loudly.
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from model.evaluation.metrics import (
    compute_per_class_f1_sweep,
    compute_precision_recall_f1,
    compute_precision_recall_f1_sweep,
    find_best_f1,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_BOX_A = [0.10, 0.10, 0.30, 0.30]
_BOX_B = [0.50, 0.50, 0.70, 0.70]
_BOX_FAR = [0.80, 0.80, 0.95, 0.95]


def _pred(image_id: str, items: List[tuple]) -> dict:
    """Build a single-image prediction dict.

    ``items`` is a list of ``(box, label, score)`` tuples.
    """
    return {
        "image_id": image_id,
        "boxes": [it[0] for it in items],
        "labels": [it[1] for it in items],
        "scores": [it[2] for it in items],
    }


def _gt(image_id: str, items: List[tuple]) -> dict:
    """Build a single-image ground-truth dict.

    ``items`` is a list of ``(box, label)`` tuples.
    """
    return {
        "image_id": image_id,
        "boxes": [it[0] for it in items],
        "labels": [it[1] for it in items],
    }


# ---------------------------------------------------------------------------
# compute_precision_recall_f1_sweep
# ---------------------------------------------------------------------------


class TestComputePrecisionRecallF1Sweep:
    """Contract for the overall P/R/F1 confidence sweep."""

    def test_returns_one_entry_per_threshold_sorted_ascending(self):
        predictions = [_pred("img1", [(_BOX_A, "bache", 0.9)])]
        ground_truths = [_gt("img1", [(_BOX_A, "bache")])]

        # Pass thresholds out of order to exercise the sort guarantee.
        thresholds = [0.5, 0.1, 0.9]
        sweep = compute_precision_recall_f1_sweep(
            predictions=predictions,
            ground_truths=ground_truths,
            confidence_thresholds=thresholds,
            iou_threshold=0.5,
        )

        assert len(sweep) == len(thresholds)
        confidences = [entry["confidence"] for entry in sweep]
        assert confidences == sorted(confidences)
        assert confidences == [0.1, 0.5, 0.9]

    def test_each_entry_has_required_keys(self):
        predictions = [_pred("img1", [(_BOX_A, "bache", 0.9)])]
        ground_truths = [_gt("img1", [(_BOX_A, "bache")])]

        sweep = compute_precision_recall_f1_sweep(
            predictions=predictions,
            ground_truths=ground_truths,
            confidence_thresholds=[0.1, 0.5],
            iou_threshold=0.5,
        )

        for entry in sweep:
            assert set(entry.keys()) == {"confidence", "precision", "recall", "f1"}
            for key in ("confidence", "precision", "recall", "f1"):
                assert isinstance(entry[key], float)

    def test_entry_matches_compute_precision_recall_f1_at_same_threshold(self):
        # Mix: 1 TP at score 0.9, 1 FP at score 0.8, 1 missed GT.
        predictions = [
            _pred(
                "img1",
                [
                    (_BOX_A, "bache", 0.9),  # matches GT A → TP
                    (_BOX_FAR, "bache", 0.8),  # no overlapping GT → FP
                ],
            )
        ]
        ground_truths = [
            _gt("img1", [(_BOX_A, "bache"), (_BOX_B, "bache")]),
        ]

        thresholds = [0.0, 0.5, 0.85, 0.95]
        sweep = compute_precision_recall_f1_sweep(
            predictions=predictions,
            ground_truths=ground_truths,
            confidence_thresholds=thresholds,
            iou_threshold=0.5,
        )

        for entry in sweep:
            reference = compute_precision_recall_f1(
                predictions=predictions,
                ground_truths=ground_truths,
                confidence_threshold=entry["confidence"],
                iou_threshold=0.5,
            )
            assert entry["precision"] == pytest.approx(reference["precision"])
            assert entry["recall"] == pytest.approx(reference["recall"])
            assert entry["f1"] == pytest.approx(reference["f1"])

    def test_recall_is_non_increasing_with_confidence(self):
        # Two TPs at distinct scores, no FPs → recall drops from 1.0 → 0.5 → 0.0
        predictions = [
            _pred(
                "img1",
                [
                    (_BOX_A, "bache", 0.9),
                    (_BOX_B, "bache", 0.3),
                ],
            )
        ]
        ground_truths = [_gt("img1", [(_BOX_A, "bache"), (_BOX_B, "bache")])]

        sweep = compute_precision_recall_f1_sweep(
            predictions=predictions,
            ground_truths=ground_truths,
            confidence_thresholds=[0.1, 0.5, 0.95],
            iou_threshold=0.5,
        )

        recalls = [entry["recall"] for entry in sweep]
        assert recalls == sorted(recalls, reverse=True)
        assert recalls[0] == pytest.approx(1.0)
        assert recalls[1] == pytest.approx(0.5)
        assert recalls[2] == pytest.approx(0.0)

    def test_empty_predictions_yields_zero_metrics_at_every_threshold(self):
        predictions = [_pred("img1", [])]
        ground_truths = [_gt("img1", [(_BOX_A, "bache")])]

        sweep = compute_precision_recall_f1_sweep(
            predictions=predictions,
            ground_truths=ground_truths,
            confidence_thresholds=[0.1, 0.5, 0.9],
            iou_threshold=0.5,
        )

        for entry in sweep:
            assert entry["precision"] == 0.0
            assert entry["recall"] == 0.0
            assert entry["f1"] == 0.0

    def test_raises_on_empty_thresholds(self):
        with pytest.raises(ValueError, match="confidence_thresholds"):
            compute_precision_recall_f1_sweep(
                predictions=[],
                ground_truths=[],
                confidence_thresholds=[],
                iou_threshold=0.5,
            )

    def test_raises_on_threshold_outside_unit_interval(self):
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            compute_precision_recall_f1_sweep(
                predictions=[],
                ground_truths=[],
                confidence_thresholds=[0.5, 1.5],
                iou_threshold=0.5,
            )


# ---------------------------------------------------------------------------
# find_best_f1
# ---------------------------------------------------------------------------


class TestFindBestF1:
    """Contract for the best-F1 picker (tie-break: higher confidence wins)."""

    def test_picks_entry_with_max_f1(self):
        sweep = [
            {"confidence": 0.10, "precision": 0.20, "recall": 1.00, "f1": 0.33},
            {"confidence": 0.50, "precision": 0.80, "recall": 0.80, "f1": 0.80},
            {"confidence": 0.90, "precision": 1.00, "recall": 0.20, "f1": 0.33},
        ]

        best = find_best_f1(sweep)

        assert best == sweep[1]

    def test_tie_breaks_by_higher_confidence(self):
        # Two entries share the same F1 — picker must prefer the higher conf
        # (more conservative deployment point).
        sweep = [
            {"confidence": 0.20, "precision": 0.50, "recall": 0.50, "f1": 0.50},
            {"confidence": 0.40, "precision": 0.50, "recall": 0.50, "f1": 0.50},
            {"confidence": 0.60, "precision": 0.50, "recall": 0.50, "f1": 0.50},
        ]

        best = find_best_f1(sweep)

        assert best["confidence"] == pytest.approx(0.60)

    def test_raises_on_empty_sweep(self):
        with pytest.raises(ValueError, match="empty"):
            find_best_f1([])

    def test_raises_on_entry_missing_required_key(self):
        sweep = [{"confidence": 0.5, "precision": 0.5, "recall": 0.5}]
        with pytest.raises(ValueError, match="f1"):
            find_best_f1(sweep)


# ---------------------------------------------------------------------------
# compute_per_class_f1_sweep
# ---------------------------------------------------------------------------


class TestComputePerClassF1Sweep:
    """Contract for the per-class best-F1 sweep."""

    def test_returns_one_entry_per_class(self):
        class_names = ["bache", "fisura"]
        predictions = [
            _pred("img1", [(_BOX_A, "bache", 0.9), (_BOX_B, "fisura", 0.7)]),
        ]
        ground_truths = [
            _gt("img1", [(_BOX_A, "bache"), (_BOX_B, "fisura")]),
        ]

        result = compute_per_class_f1_sweep(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=class_names,
            confidence_thresholds=[0.1, 0.5, 0.9],
            iou_threshold=0.5,
        )

        assert set(result.keys()) == set(class_names)
        for cls in class_names:
            entry = result[cls]
            assert set(entry.keys()) == {"confidence", "precision", "recall", "f1"}

    def test_perfect_per_class_returns_f1_one_at_low_threshold(self):
        class_names = ["bache"]
        predictions = [_pred("img1", [(_BOX_A, "bache", 0.9)])]
        ground_truths = [_gt("img1", [(_BOX_A, "bache")])]

        result = compute_per_class_f1_sweep(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=class_names,
            confidence_thresholds=[0.1, 0.5, 0.95],
            iou_threshold=0.5,
        )

        assert result["bache"]["f1"] == pytest.approx(1.0)
        # A confidence of 0.5 still includes the score-0.9 prediction → best
        # entry should be the highest such conf, here 0.5.
        assert result["bache"]["confidence"] == pytest.approx(0.5)

    def test_class_absent_from_predictions_yields_zero_f1(self):
        # GT has the class but predictions don't → recall=0 at every conf
        # → best F1 = 0.0.
        class_names = ["bache", "fisura"]
        predictions = [_pred("img1", [(_BOX_A, "bache", 0.9)])]
        ground_truths = [
            _gt("img1", [(_BOX_A, "bache"), (_BOX_B, "fisura")]),
        ]

        result = compute_per_class_f1_sweep(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=class_names,
            confidence_thresholds=[0.1, 0.5, 0.9],
            iou_threshold=0.5,
        )

        assert result["fisura"]["f1"] == 0.0
        assert result["fisura"]["recall"] == 0.0

    def test_class_only_in_predictions_yields_zero_f1(self):
        # Predictions exist for a class with no GT → precision=0 at every conf
        # (no possible matches) → best F1 = 0.0.
        class_names = ["bache", "fisura"]
        predictions = [
            _pred("img1", [(_BOX_A, "bache", 0.9), (_BOX_B, "fisura", 0.9)]),
        ]
        ground_truths = [_gt("img1", [(_BOX_A, "bache")])]

        result = compute_per_class_f1_sweep(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=class_names,
            confidence_thresholds=[0.1, 0.5, 0.9],
            iou_threshold=0.5,
        )

        assert result["fisura"]["f1"] == 0.0

    def test_per_class_entry_matches_class_specific_compute_prf1(self):
        # End-to-end consistency: per-class best entry should equal the
        # class-specific compute_precision_recall_f1 evaluated at the
        # selected confidence.
        class_names = ["bache"]
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

        result = compute_per_class_f1_sweep(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=class_names,
            confidence_thresholds=[0.1, 0.5, 0.95],
            iou_threshold=0.5,
        )

        best = result["bache"]
        reference = compute_precision_recall_f1(
            predictions=predictions,
            ground_truths=ground_truths,
            confidence_threshold=best["confidence"],
            iou_threshold=0.5,
        )
        assert best["precision"] == pytest.approx(reference["precision"])
        assert best["recall"] == pytest.approx(reference["recall"])
        assert best["f1"] == pytest.approx(reference["f1"])
