"""Per-region (per-country) detection metrics for RDD2022 evaluation.

The RDD2022 dataset stores images named ``<Region>_<digits>.<ext>``, where
``<Region>`` is one of seven country/source labels:

* ``China_Drone``
* ``China_MotorBike``
* ``Czech``
* ``India``
* ``Japan``
* ``Norway``
* ``United_States``

This module slices the (already-aligned) ``predictions`` / ``ground_truths``
lists produced during evaluation by region, then delegates to the
:mod:`model.evaluation.metrics` collaborators to compute a full metric block
per region. The result is a dict keyed by region label whose entries match the
global ``metrics`` shape produced by :func:`compute_all_metrics`, with two
extra bookkeeping fields (``num_images``, ``num_gt_boxes``).

Design notes:

* The region-extraction regex is anchored on ``_<digits>.<ext>$`` and uses a
  non-greedy ``(.+?)`` for the region prefix, so multi-segment regions like
  ``China_Drone`` and ``United_States`` are kept intact.
* This module is **pure**: no Streamlit imports, no I/O. The evaluator wires it
  in between :func:`compute_all_metrics` and :func:`assemble_report`.
* When a region's slice contains zero ground-truth boxes the metric block is
  populated with zeroed scalars (mirroring the empty-subset behaviour of
  :mod:`model.evaluation.metrics`) so the dashboard can still render the tab.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from model.evaluation.metrics import (
    compute_confusion_matrix,
    compute_map,
    compute_per_class_f1_sweep,
    compute_precision_recall_f1,
    compute_precision_recall_f1_sweep,
    find_best_f1,
)

# ---------------------------------------------------------------------------
# Region extraction
# ---------------------------------------------------------------------------

# Match ``<region>_<digits>.<ext>`` where ``<region>`` may itself contain
# underscores (e.g. ``China_Drone``, ``United_States``). The non-greedy
# ``.+?`` combined with the strict ``_\d+\.[A-Za-z0-9]+$`` suffix anchors
# the split at the *last* ``_<digits>.<ext>`` group while preserving the
# region prefix verbatim.
REGION_RE = re.compile(r"^(.+?)_\d+\.[A-Za-z0-9]+$")


def extract_region(image_id: str) -> Optional[str]:
    """Return the region prefix for an RDD2022 image identifier.

    Args:
        image_id: Image identifier as it appears in the inference dump.
            May be a path or a bare filename.

    Returns:
        The region label (e.g. ``"Japan"``, ``"China_Drone"``) when the
        filename matches ``<region>_<digits>.<ext>``; ``None`` otherwise.
    """
    if not image_id:
        return None
    name = Path(image_id).name
    match = REGION_RE.match(name)
    if match is None:
        return None
    return match.group(1)


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def group_by_region(
    predictions: List[dict],
    ground_truths: List[dict],
) -> Dict[str, Tuple[List[dict], List[dict]]]:
    """Slice aligned prediction/ground-truth lists by region.

    Args:
        predictions: Per-image prediction dicts as produced by the evaluator.
            Must carry an ``image_id`` field.
        ground_truths: Per-image ground-truth dicts aligned 1:1 with
            ``predictions``.

    Returns:
        Dict keyed by region label, each value a ``(predictions,
        ground_truths)`` tuple containing only the entries belonging to that
        region. Images whose filename does not match the region pattern are
        silently skipped (they cannot be attributed to a region).
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            "predictions and ground_truths must be aligned 1:1; got "
            f"{len(predictions)} vs {len(ground_truths)}"
        )

    groups: Dict[str, Tuple[List[dict], List[dict]]] = {}
    for pred, gt in zip(predictions, ground_truths):
        image_id = pred.get("image_id", "")
        region = extract_region(image_id)
        if region is None:
            continue
        bucket = groups.setdefault(region, ([], []))
        bucket[0].append(pred)
        bucket[1].append(gt)
    return groups


# ---------------------------------------------------------------------------
# Per-region metric computation
# ---------------------------------------------------------------------------


def _empty_metrics_block(class_names: List[str]) -> dict:
    """Return a zeroed metrics block for an empty region slice.

    Mirrors the keys produced by the populated path so downstream consumers
    (dashboard, comparison reports) can use a single accessor regardless of
    whether the region had any annotations.
    """
    return {
        "map_50": 0.0,
        "map_50_95": 0.0,
        "per_class_ap": {name: 0.0 for name in class_names},
        "mAP@0.5": 0.0,
        "mAP@0.5:0.95": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
    }


def _count_gt_boxes(ground_truths: Iterable[dict]) -> int:
    return sum(len(gt.get("boxes") or []) for gt in ground_truths)


def compute_region_metrics(
    predictions: List[dict],
    ground_truths: List[dict],
    class_names: List[str],
    confidence_threshold: float,
    iou_threshold: float,
    confidence_thresholds_sweep: Optional[List[float]] = None,
) -> dict:
    """Compute the full metric block for a single region's slice.

    The returned dict mirrors the shape produced by the global
    :func:`model.training.evaluate_detection.compute_all_metrics` (minus the
    per-image error accounting), plus two bookkeeping fields:

    * ``num_images``: the number of images in the slice.
    * ``num_gt_boxes``: the total number of ground-truth boxes in the slice.

    Args:
        predictions: Per-image prediction dicts (already filtered to one
            region).
        ground_truths: Aligned ground-truth dicts.
        class_names: Ordered class names defining confusion-matrix axes.
        confidence_threshold: Operating-point confidence for the default
            P/R/F1 entry and the confusion matrix.
        iou_threshold: IoU threshold for matching predictions to ground truth.
        confidence_thresholds_sweep: Optional list of confidence thresholds
            to sweep for ``f1_sweep`` / ``best_f1`` / ``per_class_best_f1``.
            When ``None``, the sweep fields are omitted.

    Returns:
        A dict with ``num_images``, ``num_gt_boxes``, ``metrics`` (full
        metrics block) and ``confusion_matrix`` (nested-list ``(C+1, C+1)``).
    """
    num_images = len(predictions)
    num_gt_boxes = _count_gt_boxes(ground_truths)

    if num_images == 0 or num_gt_boxes == 0 or not class_names:
        size = len(class_names) + 1
        cm = [[0] * size for _ in range(size)]
        metrics = _empty_metrics_block(class_names)
        if confidence_thresholds_sweep is not None:
            metrics["f1_sweep"] = [
                {
                    "confidence": float(c),
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                }
                for c in sorted(float(c) for c in confidence_thresholds_sweep)
            ]
            metrics["best_f1"] = {
                "confidence": float(confidence_threshold),
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }
            metrics["per_class_best_f1"] = {
                name: {
                    "confidence": float(confidence_threshold),
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                }
                for name in class_names
            }
            metrics["default_confidence_threshold"] = float(confidence_threshold)
        return {
            "num_images": num_images,
            "num_gt_boxes": num_gt_boxes,
            "metrics": metrics,
            "confusion_matrix": cm,
        }

    map_results = compute_map(
        predictions=predictions,
        ground_truths=ground_truths,
        class_names=class_names,
    )
    prf1 = compute_precision_recall_f1(
        predictions=predictions,
        ground_truths=ground_truths,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )
    cm = compute_confusion_matrix(
        predictions=predictions,
        ground_truths=ground_truths,
        class_names=class_names,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
        include_background=True,
    )

    metrics = {
        "map_50": map_results["map_50"],
        "map_50_95": map_results["map_50_95"],
        "per_class_ap": map_results["per_class_ap"],
        # Retained display keys for backward compatibility with the
        # dashboard's prior accessors.
        "mAP@0.5": map_results["map_50"],
        "mAP@0.5:0.95": map_results["map_50_95"],
        "precision": prf1["precision"],
        "recall": prf1["recall"],
        "f1_score": prf1["f1"],
    }

    if confidence_thresholds_sweep is not None:
        f1_sweep = compute_precision_recall_f1_sweep(
            predictions=predictions,
            ground_truths=ground_truths,
            confidence_thresholds=[float(c) for c in confidence_thresholds_sweep],
            iou_threshold=iou_threshold,
        )
        metrics["f1_sweep"] = f1_sweep
        metrics["best_f1"] = find_best_f1(f1_sweep)
        metrics["per_class_best_f1"] = compute_per_class_f1_sweep(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=class_names,
            confidence_thresholds=[float(c) for c in confidence_thresholds_sweep],
            iou_threshold=iou_threshold,
        )
        metrics["default_confidence_threshold"] = float(confidence_threshold)

    return {
        "num_images": num_images,
        "num_gt_boxes": num_gt_boxes,
        "metrics": metrics,
        "confusion_matrix": cm.tolist() if isinstance(cm, np.ndarray) else cm,
    }


def compute_per_region_metrics(
    predictions: List[dict],
    ground_truths: List[dict],
    class_names: List[str],
    confidence_threshold: float,
    iou_threshold: float,
    confidence_thresholds_sweep: Optional[List[float]] = None,
) -> Dict[str, dict]:
    """Compute the per-region metric blocks for an entire evaluation run.

    Groups the aligned ``predictions`` / ``ground_truths`` lists by region
    using :func:`group_by_region`, then delegates each slice to
    :func:`compute_region_metrics`.

    Images whose filename does not match the RDD2022 region pattern are
    skipped (they cannot be attributed to a region).

    Args:
        predictions: Per-image prediction dicts (the full evaluation run).
        ground_truths: Aligned ground-truth dicts.
        class_names: Ordered class names.
        confidence_threshold: Operating-point confidence used for the
            default P/R/F1 entry and confusion matrix in each region.
        iou_threshold: IoU threshold for matching.
        confidence_thresholds_sweep: Optional confidence thresholds to sweep
            (forwarded to :func:`compute_region_metrics`).

    Returns:
        Dict keyed by region label, each entry as returned by
        :func:`compute_region_metrics`. Regions are returned in sorted order
        so the output is deterministic across runs.
    """
    groups = group_by_region(predictions, ground_truths)
    out: Dict[str, dict] = {}
    for region in sorted(groups):
        region_preds, region_gts = groups[region]
        out[region] = compute_region_metrics(
            predictions=region_preds,
            ground_truths=region_gts,
            class_names=class_names,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            confidence_thresholds_sweep=confidence_thresholds_sweep,
        )
    return out
