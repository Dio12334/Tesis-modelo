"""Metrics computation for object detection evaluation.

Provides functions for computing mAP, precision, recall, F1-score,
and confusion matrices for object detection models.

Input format:
- predictions: List of dicts with keys:
    image_id, boxes (list of [x_min, y_min, x_max, y_max]),
    labels (list of str), scores (list of float)
- ground_truths: List of dicts with keys:
    image_id, boxes (list of [x_min, y_min, x_max, y_max]),
    labels (list of str)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union between two bounding boxes.

    Args:
        box1: [x_min, y_min, x_max, y_max] in normalized coordinates.
        box2: [x_min, y_min, x_max, y_max] in normalized coordinates.

    Returns:
        IoU value in [0, 1].
    """
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    inter_width = max(0.0, x_max_inter - x_min_inter)
    inter_height = max(0.0, y_max_inter - y_min_inter)
    intersection = inter_width * inter_height

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    iou = intersection / union
    return float(np.clip(iou, 0.0, 1.0))


def compute_ap(
    predictions: List[dict],
    ground_truths: List[dict],
    iou_threshold: float,
    class_name: str,
) -> float:
    """Compute Average Precision for a single class at a given IoU threshold.

    Uses the all-point interpolation method (area under the precision-recall curve).

    Args:
        predictions: List of prediction dicts per image.
        ground_truths: List of ground truth dicts per image.
        iou_threshold: IoU threshold for matching predictions to ground truths.
        class_name: The class to compute AP for.

    Returns:
        Average Precision value in [0, 1].
    """
    # Collect all predictions for this class across all images
    all_preds = []
    for pred in predictions:
        image_id = pred["image_id"]
        for i, label in enumerate(pred["labels"]):
            if label == class_name:
                all_preds.append(
                    {
                        "image_id": image_id,
                        "box": pred["boxes"][i],
                        "score": pred["scores"][i],
                    }
                )

    # Collect all ground truths for this class, indexed by image_id
    gt_by_image: Dict[str, List[dict]] = {}
    total_gt = 0
    for gt in ground_truths:
        image_id = gt["image_id"]
        if image_id not in gt_by_image:
            gt_by_image[image_id] = []
        for i, label in enumerate(gt["labels"]):
            if label == class_name:
                gt_by_image[image_id].append(
                    {"box": gt["boxes"][i], "matched": False}
                )
                total_gt += 1

    # If no ground truths exist for this class, AP is 0
    if total_gt == 0:
        return 0.0

    # Sort predictions by confidence (descending)
    all_preds.sort(key=lambda x: x["score"], reverse=True)

    # Compute TP/FP for each prediction
    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))

    for idx, pred in enumerate(all_preds):
        image_id = pred["image_id"]
        pred_box = pred["box"]

        # Get ground truths for this image and class
        image_gts = gt_by_image.get(image_id, [])

        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_item in enumerate(image_gts):
            iou = compute_iou(pred_box, gt_item["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            if not image_gts[best_gt_idx]["matched"]:
                tp[idx] = 1
                image_gts[best_gt_idx]["matched"] = True
            else:
                fp[idx] = 1
        else:
            fp[idx] = 1

    # Compute cumulative TP and FP
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    # Compute precision and recall at each threshold
    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / total_gt

    # All-point interpolation: compute area under the PR curve
    # Prepend (0, 1) for precision and (0, 0) for recall
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))

    # Make precision monotonically decreasing (from right to left)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Compute area under the curve using trapezoidal integration
    ap = 0.0
    for i in range(1, len(recall)):
        ap += (recall[i] - recall[i - 1]) * precision[i]

    return float(np.clip(ap, 0.0, 1.0))


def compute_map(
    predictions: List[dict],
    ground_truths: List[dict],
    iou_thresholds: Optional[List[float]] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute mean Average Precision across all classes and IoU thresholds.

    Args:
        predictions: List of prediction dicts per image.
        ground_truths: List of ground truth dicts per image.
        iou_thresholds: IoU thresholds to evaluate. If None, uses
            [0.5] for mAP@0.5 and [0.5, 0.55, ..., 0.95] for mAP@0.5:0.95.
        class_names: List of class names to evaluate. If None, inferred
            from ground truths.

    Returns:
        Dict with keys:
            - "map_50": mAP at IoU=0.5
            - "map_50_95": mAP at IoU=0.5:0.95
            - "per_class_ap": dict mapping class_name -> AP at IoU=0.5
    """
    # Infer class names from ground truths if not provided
    if class_names is None:
        class_set = set()
        for gt in ground_truths:
            for label in gt["labels"]:
                class_set.add(label)
        class_names = sorted(class_set)

    # Define IoU thresholds for mAP@0.5:0.95
    iou_thresholds_50_95 = [0.5 + 0.05 * i for i in range(10)]

    # Compute per-class AP at IoU=0.5
    per_class_ap_50: Dict[str, float] = {}
    for cls in class_names:
        ap = compute_ap(predictions, ground_truths, 0.5, cls)
        per_class_ap_50[cls] = ap

    # mAP@0.5 is the mean of per-class APs at IoU=0.5
    if len(per_class_ap_50) > 0:
        map_50 = float(np.mean(list(per_class_ap_50.values())))
    else:
        map_50 = 0.0

    # Compute mAP@0.5:0.95 (mean over all IoU thresholds and classes)
    all_aps = []
    for iou_thresh in iou_thresholds_50_95:
        for cls in class_names:
            ap = compute_ap(predictions, ground_truths, iou_thresh, cls)
            all_aps.append(ap)

    if len(all_aps) > 0:
        map_50_95 = float(np.mean(all_aps))
    else:
        map_50_95 = 0.0

    return {
        "map_50": float(np.clip(map_50, 0.0, 1.0)),
        "map_50_95": float(np.clip(map_50_95, 0.0, 1.0)),
        "per_class_ap": per_class_ap_50,
    }


def compute_precision_recall_f1(
    predictions: List[dict],
    ground_truths: List[dict],
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute precision, recall, and F1-score at a given confidence threshold.

    Args:
        predictions: List of prediction dicts per image.
        ground_truths: List of ground truth dicts per image.
        confidence_threshold: Minimum confidence score for predictions.
        iou_threshold: IoU threshold for matching predictions to ground truths.

    Returns:
        Dict with keys "precision", "recall", "f1", all in [0, 1].
    """
    tp = 0
    fp = 0
    total_gt = 0

    for gt in ground_truths:
        total_gt += len(gt["boxes"])

    # Build a lookup of ground truths by image_id
    gt_by_image: Dict[str, List[dict]] = {}
    for gt in ground_truths:
        image_id = gt["image_id"]
        gt_by_image[image_id] = [
            {"box": gt["boxes"][i], "label": gt["labels"][i], "matched": False}
            for i in range(len(gt["boxes"]))
        ]

    # Process predictions filtered by confidence threshold
    for pred in predictions:
        image_id = pred["image_id"]
        image_gts = gt_by_image.get(image_id, [])

        for i in range(len(pred["boxes"])):
            if pred["scores"][i] < confidence_threshold:
                continue

            pred_box = pred["boxes"][i]
            pred_label = pred["labels"][i]

            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt_item in enumerate(image_gts):
                if gt_item["matched"]:
                    continue
                if gt_item["label"] != pred_label:
                    continue
                iou = compute_iou(pred_box, gt_item["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                image_gts[best_gt_idx]["matched"] = True
            else:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "precision": float(np.clip(precision, 0.0, 1.0)),
        "recall": float(np.clip(recall, 0.0, 1.0)),
        "f1": float(np.clip(f1, 0.0, 1.0)),
    }


def compute_confusion_matrix(
    predictions: List[dict],
    ground_truths: List[dict],
    class_names: List[str],
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.5,
) -> np.ndarray:
    """Generate a confusion matrix showing predicted vs actual class distributions.

    The matrix has dimensions C×C where C is the number of classes.
    Entry [i, j] represents ground truth class i predicted as class j.

    Args:
        predictions: List of prediction dicts per image.
        ground_truths: List of ground truth dicts per image.
        class_names: Ordered list of class names (defines matrix indices).
        iou_threshold: IoU threshold for matching predictions to ground truths.
        confidence_threshold: Minimum confidence for predictions.

    Returns:
        Confusion matrix as numpy array of shape (C, C) with non-negative integers.
    """
    num_classes = len(class_names)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    # Build ground truth lookup by image_id
    gt_by_image: Dict[str, List[dict]] = {}
    for gt in ground_truths:
        image_id = gt["image_id"]
        gt_by_image[image_id] = [
            {"box": gt["boxes"][i], "label": gt["labels"][i], "matched": False}
            for i in range(len(gt["boxes"]))
            if gt["labels"][i] in class_to_idx
        ]

    # Match predictions to ground truths
    for pred in predictions:
        image_id = pred["image_id"]
        image_gts = gt_by_image.get(image_id, [])

        for i in range(len(pred["boxes"])):
            if pred["scores"][i] < confidence_threshold:
                continue

            pred_label = pred["labels"][i]
            if pred_label not in class_to_idx:
                continue

            pred_box = pred["boxes"][i]
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt_item in enumerate(image_gts):
                if gt_item["matched"]:
                    continue
                iou = compute_iou(pred_box, gt_item["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                gt_label = image_gts[best_gt_idx]["label"]
                gt_class_idx = class_to_idx[gt_label]
                pred_class_idx = class_to_idx[pred_label]
                matrix[gt_class_idx, pred_class_idx] += 1
                image_gts[best_gt_idx]["matched"] = True

    return matrix
