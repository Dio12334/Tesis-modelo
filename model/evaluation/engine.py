"""Evaluation engine for running model evaluation on datasets.

Provides the EvaluationEngine class that orchestrates model inference,
metric computation, and report generation.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from model.evaluation.metrics import (
    compute_confusion_matrix,
    compute_map,
    compute_precision_recall_f1,
)
from model.evaluation.report import EvaluationReport

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """Computes detection metrics for trained models.

    The engine runs model inference on a dataset, collects predictions
    and ground truths, applies optional class filtering, and computes
    comprehensive evaluation metrics.

    Supports two modes of operation:
    1. Standard mode: runs model.forward() on dataset images (requires torch).
    2. Pre-computed mode: accepts pre-computed predictions directly via the
       `evaluate` method's `precomputed_predictions` parameter (for testing
       without torch).
    """

    def evaluate(
        self,
        model=None,
        dataset=None,
        iou_thresholds: Optional[List[float]] = None,
        confidence_threshold: float = 0.5,
        target_classes: Optional[List[str]] = None,
        precomputed_predictions: Optional[List[dict]] = None,
        precomputed_ground_truths: Optional[List[dict]] = None,
        model_id: Optional[str] = None,
    ) -> EvaluationReport:
        """Run evaluation and return comprehensive metrics report.

        Args:
            model: Trained detection model (BaseDetector). Can be None if
                precomputed_predictions is provided.
            dataset: Test dataset (BaseDataset). Can be None if
                precomputed_ground_truths is provided.
            iou_thresholds: IoU thresholds for mAP computation. If None,
                uses default [0.5, 0.55, ..., 0.95].
            confidence_threshold: Minimum confidence for predictions when
                computing precision/recall/F1 and confusion matrix.
            target_classes: If provided, only evaluate these classes.
                Predictions and ground truths for other classes are filtered out.
            precomputed_predictions: Pre-computed predictions in metrics format.
                Each dict has keys: image_id, boxes, labels, scores.
                If provided, model inference is skipped.
            precomputed_ground_truths: Pre-computed ground truths in metrics format.
                Each dict has keys: image_id, boxes, labels.
                If provided, dataset iteration is skipped.
            model_id: Optional model identifier for the report. If None,
                uses the model's class name or "unknown".

        Returns:
            EvaluationReport with all computed metrics.

        Raises:
            ValueError: If neither model nor precomputed_predictions is provided,
                or if neither dataset nor precomputed_ground_truths is provided.
        """
        if model is None and precomputed_predictions is None:
            raise ValueError(
                "Either 'model' or 'precomputed_predictions' must be provided."
            )
        if dataset is None and precomputed_ground_truths is None:
            raise ValueError(
                "Either 'dataset' or 'precomputed_ground_truths' must be provided."
            )

        # Determine model_id for the report
        if model_id is None:
            if model is not None:
                model_id = type(model).__name__
            else:
                model_id = "unknown"

        # Collect ground truths
        if precomputed_ground_truths is not None:
            ground_truths = precomputed_ground_truths
        else:
            ground_truths = self._collect_ground_truths(dataset)

        # Collect predictions
        if precomputed_predictions is not None:
            predictions = precomputed_predictions
        else:
            predictions = self._run_inference(model, dataset)

        # Apply class filtering if target_classes is specified
        if target_classes is not None:
            predictions = self._filter_by_classes(predictions, target_classes)
            ground_truths = self._filter_by_classes(ground_truths, target_classes)

        # Determine class names for evaluation
        if target_classes is not None:
            class_names = sorted(target_classes)
        else:
            # Infer from ground truths
            class_set = set()
            for gt in ground_truths:
                for label in gt["labels"]:
                    class_set.add(label)
            class_names = sorted(class_set)

        # Compute mAP metrics
        map_results = compute_map(
            predictions=predictions,
            ground_truths=ground_truths,
            iou_thresholds=iou_thresholds,
            class_names=class_names,
        )

        # Compute precision, recall, F1
        prf1_results = compute_precision_recall_f1(
            predictions=predictions,
            ground_truths=ground_truths,
            confidence_threshold=confidence_threshold,
            iou_threshold=0.5,
        )

        # Compute confusion matrix
        confusion_mat = compute_confusion_matrix(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=class_names,
            iou_threshold=0.5,
            confidence_threshold=confidence_threshold,
        )

        # Build config dict for the report
        config = {
            "iou_thresholds": iou_thresholds,
            "confidence_threshold": confidence_threshold,
            "target_classes": target_classes,
        }

        # Build and return the evaluation report
        report = EvaluationReport(
            model_id=model_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            map_50=map_results["map_50"],
            map_50_95=map_results["map_50_95"],
            per_class_ap=map_results["per_class_ap"],
            precision=prf1_results["precision"],
            recall=prf1_results["recall"],
            f1_score=prf1_results["f1"],
            confusion_matrix=confusion_mat,
            class_names=class_names,
            config=config,
        )

        return report

    def _collect_ground_truths(self, dataset) -> List[dict]:
        """Collect ground truth annotations from a dataset.

        Converts dataset annotations into the format expected by metrics functions.

        Args:
            dataset: A BaseDataset instance.

        Returns:
            List of ground truth dicts with keys: image_id, boxes, labels.
        """
        ground_truths = []
        for idx, annotation in enumerate(dataset):
            image_id = str(annotation.image_path)
            boxes = []
            labels = []
            for bbox in annotation.bounding_boxes:
                boxes.append([bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max])
                labels.append(bbox.class_label)
            ground_truths.append(
                {"image_id": image_id, "boxes": boxes, "labels": labels}
            )
        return ground_truths

    def _run_inference(self, model, dataset) -> List[dict]:
        """Run model inference on all dataset images.

        Calls model.forward() on each image and converts the output
        to the format expected by metrics functions.

        Args:
            model: A BaseDetector instance.
            dataset: A BaseDataset instance.

        Returns:
            List of prediction dicts with keys: image_id, boxes, labels, scores.
        """
        predictions = []

        # Get class names from dataset for label index mapping
        class_names = dataset.get_class_names()

        for annotation in dataset:
            image_id = str(annotation.image_path)

            try:
                # Attempt to load and run inference on the image
                import torch
                from PIL import Image
                from torchvision import transforms

                img = Image.open(annotation.image_path).convert("RGB")
                img_w, img_h = img.size  # PIL: (width, height)
                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                )
                img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

                # Run forward pass
                results = model.forward(img_tensor)

                if results and len(results) > 0:
                    result = results[0]  # First (and only) image in batch
                    boxes = []
                    labels = []
                    scores = []

                    # Convert tensor outputs to lists
                    pred_boxes = result.get("boxes", [])
                    pred_labels = result.get("labels", [])
                    pred_scores = result.get("scores", [])

                    for j in range(len(pred_boxes)):
                        box = pred_boxes[j]
                        if hasattr(box, "tolist"):
                            box = box.tolist()

                        # Normalize prediction box from pixel-space to [0, 1]
                        # to match the project's convention (BoundingBox stores
                        # normalized coords; metrics.compute_iou expects normalized).
                        # Detector wrappers return xyxy in pixels of the original
                        # image; we divide by (W, H, W, H).
                        if (
                            isinstance(box, (list, tuple))
                            and len(box) == 4
                            and img_w > 0
                            and img_h > 0
                        ):
                            x1, y1, x2, y2 = box
                            box = [
                                float(x1) / img_w,
                                float(y1) / img_h,
                                float(x2) / img_w,
                                float(y2) / img_h,
                            ]

                        label = pred_labels[j]
                        if hasattr(label, "item"):
                            label = label.item()

                        # Defensive filter: drop predictions whose integer label
                        # index is outside [0, len(class_names) - 1]. This protects
                        # against detector wrappers that return raw pretrained-nc
                        # indices (e.g., 0..79 from a COCO checkpoint) when their
                        # head was not reshaped to num_classes. After fixing the
                        # head reshape upstream this branch should never fire,
                        # but the warning here surfaces such bugs loudly instead
                        # of silently producing zero mAP.
                        if isinstance(label, int):
                            if label < 0 or label >= len(class_names):
                                logger.warning(
                                    "Dropping prediction with out-of-range label "
                                    "index %d (class_names has %d entries) for "
                                    "image %s. This typically means the detector's "
                                    "classification head was not reshaped to match "
                                    "num_classes.",
                                    label,
                                    len(class_names),
                                    image_id,
                                )
                                continue
                            label = class_names[label]
                        labels.append(str(label))
                        boxes.append(box)

                        score = pred_scores[j]
                        if hasattr(score, "item"):
                            score = score.item()
                        scores.append(float(score))

                    predictions.append(
                        {
                            "image_id": image_id,
                            "boxes": boxes,
                            "labels": labels,
                            "scores": scores,
                        }
                    )
                else:
                    predictions.append(
                        {
                            "image_id": image_id,
                            "boxes": [],
                            "labels": [],
                            "scores": [],
                        }
                    )
            except (ImportError, FileNotFoundError, OSError):
                # If torch/PIL not available or image can't be loaded,
                # return empty predictions for this image
                predictions.append(
                    {
                        "image_id": image_id,
                        "boxes": [],
                        "labels": [],
                        "scores": [],
                    }
                )

        return predictions

    def _filter_by_classes(
        self, data: List[dict], target_classes: List[str]
    ) -> List[dict]:
        """Filter predictions or ground truths to only include target classes.

        Args:
            data: List of prediction or ground truth dicts.
            target_classes: List of class names to keep.

        Returns:
            Filtered list where each dict only contains entries for target classes.
        """
        target_set = set(target_classes)
        filtered = []

        for item in data:
            labels = item["labels"]
            boxes = item["boxes"]
            has_scores = "scores" in item

            new_boxes = []
            new_labels = []
            new_scores = [] if has_scores else None

            for i, label in enumerate(labels):
                if label in target_set:
                    new_boxes.append(boxes[i])
                    new_labels.append(label)
                    if has_scores:
                        new_scores.append(item["scores"][i])

            new_item = {
                "image_id": item["image_id"],
                "boxes": new_boxes,
                "labels": new_labels,
            }
            if has_scores:
                new_item["scores"] = new_scores

            filtered.append(new_item)

        return filtered
