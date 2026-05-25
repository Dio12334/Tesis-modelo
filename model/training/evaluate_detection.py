"""Evaluate a trained SSD MobileNetV3 model on the RDD2022 dataset.

Computes mAP, precision, recall, F1, and confusion matrix on the validation
or test set using a saved checkpoint.

Usage:
    python -m model.training.evaluate_detection --run-id <uuid>
    python -m model.training.evaluate_detection --checkpoint <path_to_pt_file>
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
from torchvision import transforms as T
from PIL import Image

from model.config.manager import ConfigManager
from model.datasets.rdd2022 import RDD2022Dataset
from model.evaluation.metrics import (
    compute_confusion_matrix,
    compute_map,
    compute_precision_recall_f1,
)
from model.models.ssd_mobilenet import SSDMobileNetV3

logger = logging.getLogger(__name__)


def evaluate(
    checkpoint_path: str,
    dataset_path: str = "model/data/rdd2022/sample",
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    input_size: int = 320,
    num_classes: int = 5,
    val_split: float = 0.2,
    output_dir: str = None,
    split: str = "val",
    verbose: bool = False,
) -> dict:
    """Run evaluation on a trained model checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        dataset_path: Path to the RDD2022 dataset.
        confidence_threshold: Min confidence for predictions.
        iou_threshold: IoU threshold for matching.
        input_size: Model input resolution.
        num_classes: Number of classes.
        val_split: Fraction used for validation (to recreate the same split).
        output_dir: Where to save the evaluation report. If None, saves next to checkpoint.
        split: Which split to evaluate on: "val" or "train".
        verbose: Enable debug logging.

    Returns:
        Dict with evaluation metrics.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load dataset and recreate the same val split
    logger.info("Loading dataset from %s", dataset_path)
    dataset = RDD2022Dataset()
    dataset.load(Path(dataset_path))
    logger.info("Dataset: %d images, classes: %s", len(dataset), dataset.get_class_names())

    train_ratio = 1.0 - val_split
    train_ds, val_ds, _ = dataset.split(train_ratio, val_split, 0.0, seed=42)

    if split == "train":
        eval_ds = train_ds
        logger.info("Evaluating on TRAINING set: %d images", len(eval_ds))
        predictions_filename = "train_inference.json"
        report_filename = "train_evaluation_report.json"
    else:
        eval_ds = val_ds
        logger.info("Evaluating on VALIDATION set: %d images", len(eval_ds))
        predictions_filename = "validation_inference.json"
        report_filename = "evaluation_report.json"

    class_names = dataset.get_class_names()
    class_to_idx = {name: idx + 1 for idx, name in enumerate(class_names)}
    idx_to_class = {idx + 1: name for idx, name in enumerate(class_names)}

    # Load model
    logger.info("Loading model from %s", checkpoint_path)
    model_config = {"input_size": input_size, "num_classes": num_classes}
    model = SSDMobileNetV3(model_config)
    model.load_checkpoint(checkpoint_path)
    model.set_eval_mode()

    # Run inference on validation set
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
    ])

    predictions = []
    ground_truths = []

    logger.info("Running inference on %d images...", len(eval_ds))
    annotations = eval_ds.get_annotations()

    for i, annotation in enumerate(annotations):
        image_id = str(annotation.image_path)

        # Build ground truth
        gt_boxes = []
        gt_labels = []
        for bbox in annotation.bounding_boxes:
            gt_boxes.append([bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max])
            gt_labels.append(bbox.class_label)

        ground_truths.append({
            "image_id": image_id,
            "boxes": gt_boxes,
            "labels": gt_labels,
        })

        # Run inference
        try:
            image = Image.open(annotation.image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model._model([image_tensor.squeeze(0)])

            output = outputs[0]
            pred_boxes = []
            pred_labels = []
            pred_scores = []

            h, w = input_size, input_size
            for j in range(len(output["boxes"])):
                score = output["scores"][j].item()
                if score < confidence_threshold:
                    continue

                box = output["boxes"][j]
                # Normalize to [0, 1]
                x1 = box[0].item() / w
                y1 = box[1].item() / h
                x2 = box[2].item() / w
                y2 = box[3].item() / h

                label_idx = output["labels"][j].item()
                label = idx_to_class.get(label_idx, f"class_{label_idx}")

                pred_boxes.append([x1, y1, x2, y2])
                pred_labels.append(label)
                pred_scores.append(score)

            predictions.append({
                "image_id": image_id,
                "boxes": pred_boxes,
                "labels": pred_labels,
                "scores": pred_scores,
            })

        except (FileNotFoundError, OSError) as e:
            logger.warning("Could not process %s: %s", annotation.image_path, e)
            predictions.append({
                "image_id": image_id,
                "boxes": [],
                "labels": [],
                "scores": [],
            })

        if (i + 1) % 50 == 0:
            logger.info("  Processed %d/%d images", i + 1, len(annotations))

    # Compute metrics
    logger.info("Computing metrics...")

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

    confusion_mat = compute_confusion_matrix(
        predictions=predictions,
        ground_truths=ground_truths,
        class_names=class_names,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
    )

    # Build report
    report = {
        "checkpoint": str(checkpoint_path),
        "dataset": dataset_path,
        "split": split,
        "num_images": len(eval_ds),
        "num_classes": len(class_names),
        "class_names": class_names,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "metrics": {
            "mAP@0.5": map_results["map_50"],
            "mAP@0.5:0.95": map_results["map_50_95"],
            "precision": prf1["precision"],
            "recall": prf1["recall"],
            "f1_score": prf1["f1"],
            "per_class_ap": map_results["per_class_ap"],
        },
        "confusion_matrix": confusion_mat.tolist(),
    }

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Checkpoint:    {checkpoint_path.name}")
    print(f"  Split:         {split}")
    print(f"  Images:        {len(eval_ds)}")
    print(f"  Confidence:    {confidence_threshold}")
    print(f"  IoU threshold: {iou_threshold}")
    print("-" * 60)
    print(f"  mAP@0.5:       {map_results['map_50']:.4f}")
    print(f"  mAP@0.5:0.95:  {map_results['map_50_95']:.4f}")
    print(f"  Precision:      {prf1['precision']:.4f}")
    print(f"  Recall:         {prf1['recall']:.4f}")
    print(f"  F1-score:       {prf1['f1']:.4f}")
    print("-" * 60)
    print("  Per-class AP@0.5:")
    for cls_name, ap in map_results["per_class_ap"].items():
        print(f"    {cls_name:25s} {ap:.4f}")
    print("-" * 60)
    print("  Confusion Matrix:")
    header = "  " + " " * 18 + "  ".join(f"{c[:8]:>8s}" for c in class_names)
    print(header)
    for i, row in enumerate(confusion_mat):
        row_str = "  ".join(f"{int(v):>8d}" for v in row)
        print(f"  {class_names[i]:16s} {row_str}")
    print("=" * 60)

    # Save report
    if output_dir is None:
        output_dir = checkpoint_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / report_filename
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Evaluation report saved to: %s", report_path)

    # Save per-image predictions and ground truths for the dashboard viewer
    predictions_output = {
        "checkpoint": str(checkpoint_path),
        "dataset": dataset_path,
        "confidence_threshold": confidence_threshold,
        "class_names": class_names,
        "images": [],
    }
    for pred, gt in zip(predictions, ground_truths):
        predictions_output["images"].append({
            "image_id": pred["image_id"],
            "ground_truth": {
                "boxes": gt["boxes"],
                "labels": gt["labels"],
            },
            "predictions": {
                "boxes": pred["boxes"],
                "labels": pred["labels"],
                "scores": pred["scores"],
            },
        })

    predictions_path = output_dir / predictions_filename
    with open(predictions_path, "w") as f:
        json.dump(predictions_output, f, indent=2)
    logger.info("Per-image predictions saved to: %s", predictions_path)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SSD MobileNetV3 on RDD2022")
    parser.add_argument("--run-id", type=str, help="UUID of the training run to evaluate")
    parser.add_argument("--checkpoint", type=str, help="Direct path to .pt checkpoint file")
    parser.add_argument("--dataset", type=str, default="model/data/rdd2022/sample")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/ssd_mobilenetv3")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save evaluation report and predictions")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Which split to evaluate: train or val")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Resolve checkpoint path
    if args.checkpoint:
        ckpt_path = args.checkpoint
    elif args.run_id:
        ckpt_path = str(Path(args.checkpoint_dir) / args.run_id / "best_model.pt")
    else:
        # Default to global best
        ckpt_path = str(Path(args.checkpoint_dir) / "global" / "best_model.pt")

    evaluate(
        checkpoint_path=ckpt_path,
        dataset_path=args.dataset,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        input_size=args.input_size,
        num_classes=args.num_classes,
        val_split=args.val_split,
        output_dir=args.output_dir,
        split=args.split,
        verbose=args.verbose,
    )
