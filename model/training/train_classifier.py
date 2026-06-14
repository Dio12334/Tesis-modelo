"""Training script for BinaryDamageClassifier.

Trains a binary image classifier (background vs. road damage) on RDD2022 images.
Binary labels are derived automatically: images with ≥1 annotation → damage (1),
images with zero annotations → background (0).

Usage:
    python -m model.training.train_classifier model/configs/train_binary_classifier.yaml
    python -m model train-classifier --config model/configs/train_binary_classifier.yaml
"""

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from PIL import Image
from torchvision import transforms as T

from model.config.manager import ConfigManager
from model.datasets.base import Annotation
from model.datasets.rdd2022 import RDD2022Dataset
from model.models.binary_classifier import BinaryDamageClassifier
from model.tracking.tracker import ExperimentTracker
from model.training.augmentation import build_augmentation_pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BinaryClassificationDataset(torch.utils.data.Dataset):
    """PyTorch Dataset that wraps RDD2022 annotations as binary (damage/no-damage).

    Label assignment:
        - 1 (damage)     — annotation has at least one bounding box
        - 0 (background) — annotation has zero bounding boxes

    Args:
        annotations: List of :class:`~model.datasets.base.Annotation` objects.
        input_size: Images are resized to ``(input_size, input_size)``.
        augmentation: Optional per-image spatial augmentation pipeline from
            :func:`~model.training.augmentation.build_augmentation_pipeline`.
            Only image-level transforms are applied (bounding-box outputs discarded).
        normalize: If ``True`` (default), apply ImageNet mean/std normalization
            after converting to float.  Set to ``False`` for debug / visualization.
    """

    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        annotations: List[Annotation],
        input_size: int = 224,
        augmentation=None,
        normalize: bool = True,
    ) -> None:
        self._annotations = annotations
        self._input_size = input_size
        self._augmentation = augmentation
        self._to_tensor = T.ToTensor()
        self._normalize = T.Normalize(mean=self._MEAN, std=self._STD) if normalize else None

    def __len__(self) -> int:
        return len(self._annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        ann = self._annotations[idx]
        label = int(len(ann.bounding_boxes) > 0)

        # Load image
        try:
            img = Image.open(ann.image_path).convert("RGB")
        except (FileNotFoundError, OSError):
            logger.warning("Missing image: %s — using blank", ann.image_path)
            img = Image.new("RGB", (self._input_size, self._input_size), color=128)

        img = img.resize((self._input_size, self._input_size), Image.BILINEAR)

        # Apply per-image spatial augmentation (bboxes not needed for classification)
        if self._augmentation is not None:
            img_np = np.array(img)
            img_np, _ = self._augmentation(img_np, [])
            img = Image.fromarray(img_np.astype("uint8"))

        img_t = self._to_tensor(img)  # (3, H, W) in [0, 1]
        if self._normalize is not None:
            img_t = self._normalize(img_t)
        return img_t, label


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _compute_auc_roc(labels: List[int], probs: List[float]) -> float:
    """Compute AUC-ROC via the trapezoidal rule (no sklearn dependency)."""
    pairs = sorted(zip(probs, labels), key=lambda x: -x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = fp = 0
    auc = 0.0
    prev_fp = 0
    for _, lbl in pairs:
        if lbl == 1:
            tp += 1
        else:
            fp += 1
            auc += tp  # trapezoid width = 1 FP step
    return auc / (n_pos * n_neg)


def _compute_accuracy(labels: List[int], predictions: List[int]) -> float:
    correct = sum(l == p for l, p in zip(labels, predictions))
    return correct / len(labels) if labels else 0.0


# ---------------------------------------------------------------------------
# Optimizer / scheduler helpers (mirror train_detection.py style)
# ---------------------------------------------------------------------------


def _build_optimizer(model: BinaryDamageClassifier, config: dict) -> torch.optim.Optimizer:
    optimizer_name = config.get("optimizer", "AdamW").upper()
    lr = float(config.get("learning_rate", 1e-4))
    weight_decay = float(config.get("weight_decay", 1e-4))
    momentum = float(config.get("momentum", 0.937))
    params = model.get_parameters()
    if optimizer_name == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if optimizer_name == "ADAM":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _build_scheduler(optimizer: torch.optim.Optimizer, config: dict, total_steps: int):
    scheduler_name = config.get("scheduler", "cosine").lower()
    warmup_epochs = int(config.get("warmup_epochs", 3))
    epochs = int(config.get("epochs", 30))

    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, scheduler], milestones=[warmup_epochs]
        )
    return scheduler


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_classifier(config_path: str, verbose: bool = False) -> dict:
    """Train a BinaryDamageClassifier on RDD2022.

    Args:
        config_path: Path to the YAML training configuration.
        verbose: Enable DEBUG logging.

    Returns:
        Dict with final training metrics (best_val_auc, best_epoch, etc.).
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ------------------------------------------------------------------ #
    # Config                                                               #
    # ------------------------------------------------------------------ #
    config_manager = ConfigManager()
    config = config_manager.load(Path(config_path))
    config = config_manager.resolve_env_vars(config)

    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    eval_cfg = config.get("evaluation", {})

    input_size = int(model_cfg.get("input_size", 224))
    dataset_path = dataset_cfg.get("path", "model/data/rdd2022/")
    country_filter = dataset_cfg.get("country_filter")

    epochs = int(training_cfg.get("epochs", 30))
    batch_size = int(training_cfg.get("batch_size", 32))
    val_split = float(training_cfg.get("val_split", 0.2))
    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "./checkpoints/binary_classifier"))
    seed = int(training_cfg.get("seed", 42))
    early_stopping_patience = int(training_cfg.get("early_stopping_patience", 10))
    num_workers = int(training_cfg.get("num_workers", 4))
    use_amp = bool(training_cfg.get("use_amp", True))
    log_interval = int(training_cfg.get("log_interval", 10))
    augmentation_cfg = training_cfg.get("augmentation", {})

    # ------------------------------------------------------------------ #
    # Reproducibility                                                      #
    # ------------------------------------------------------------------ #
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Seed: %d", seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------ #
    # Dataset                                                              #
    # ------------------------------------------------------------------ #
    logger.info("Loading dataset from %s …", dataset_path)
    rdd_dataset = RDD2022Dataset(country_filter=country_filter)
    rdd_dataset.load(Path(dataset_path))
    all_annotations = rdd_dataset.get_annotations()
    logger.info("Loaded %d annotations", len(all_annotations))

    n_damage = sum(1 for a in all_annotations if a.bounding_boxes)
    n_background = len(all_annotations) - n_damage
    logger.info("Labels — damage: %d, background: %d", n_damage, n_background)

    # Train / val split (deterministic)
    rng = random.Random(seed)
    indices = list(range(len(all_annotations)))
    rng.shuffle(indices)
    split_at = int(len(indices) * (1.0 - val_split))
    train_indices = indices[:split_at]
    val_indices = indices[split_at:]

    train_annotations = [all_annotations[i] for i in train_indices]
    val_annotations = [all_annotations[i] for i in val_indices]
    logger.info("Train: %d, Val: %d", len(train_annotations), len(val_annotations))

    # Augmentation (image-level only; boxes are ignored for classification)
    augmentation = build_augmentation_pipeline(augmentation_cfg) if augmentation_cfg else None

    train_ds = BinaryClassificationDataset(
        train_annotations, input_size=input_size, augmentation=augmentation
    )
    val_ds = BinaryClassificationDataset(
        val_annotations, input_size=input_size, augmentation=None
    )

    # Class-weighted sampler to handle ~67% damage / ~33% background imbalance
    train_labels = [int(len(a.bounding_boxes) > 0) for a in train_annotations]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    weight_pos = 1.0 / max(n_pos, 1)
    weight_neg = 1.0 / max(n_neg, 1)
    sample_weights = [weight_pos if l == 1 else weight_neg for l in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_labels), replacement=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    model_init_cfg = dict(model_cfg)
    model_init_cfg["pretrained"] = bool(model_init_cfg.get("pretrained", True))
    classifier = BinaryDamageClassifier(model_init_cfg)
    classifier.to_device(device)

    # ------------------------------------------------------------------ #
    # Loss — BCEWithLogitsLoss with configurable pos_weight               #
    # ------------------------------------------------------------------ #
    # WeightedRandomSampler already balances the dataset 50/50, so the
    # natural pos_weight is 1.0.  Setting pos_weight > 1 biases the loss
    # toward recall (fewer missed-damage FNs), which is desirable when
    # this classifier is used as a pre-filter in TwoStageDetector.
    pos_weight_val = float(training_cfg.get("pos_weight", 2.0))
    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info("BCEWithLogitsLoss pos_weight=%.4f (n_neg=%d, n_pos=%d)", pos_weight_val, n_neg, n_pos)

    # ------------------------------------------------------------------ #
    # Optimizer & scheduler                                                #
    # ------------------------------------------------------------------ #
    optimizer = _build_optimizer(classifier, training_cfg)
    scheduler = _build_scheduler(optimizer, training_cfg, total_steps=len(train_loader))

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    # ------------------------------------------------------------------ #
    # ExperimentTracker + Run ID                                          #
    # ------------------------------------------------------------------ #
    # start_run() generates the UUID — reuse it for the checkpoint folder
    # so the JSON file and the folder share the same UUID (required by dashboard).
    tracker = ExperimentTracker(output_dir=checkpoint_dir)
    run_id = tracker.start_run(
        config=config,
        model_name="binary_classifier",
        dataset_name="rdd2022",
    )
    run_dir = checkpoint_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = run_dir / "best_model.pt"
    last_model_path = run_dir / "last_model.pt"
    logger.info("Run ID: %s", run_id)
    logger.info("Checkpoint dir: %s", run_dir)

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    best_val_auc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()

        # --- Train ---
        classifier.set_train_mode()
        train_loss_sum = 0.0
        train_batches = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                logits = classifier.forward(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_params = [p for g in optimizer.param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            train_batches += 1

            if (batch_idx + 1) % log_interval == 0:
                avg = train_loss_sum / train_batches
                logger.info(
                    "Epoch %d/%d  step %d/%d  train_loss=%.4f",
                    epoch + 1, epochs, batch_idx + 1, len(train_loader), avg,
                )

        scheduler.step()
        avg_train_loss = train_loss_sum / max(train_batches, 1)

        # --- Validate ---
        classifier.set_eval_mode()
        val_loss_sum = 0.0
        val_batches = 0
        all_labels: List[int] = []
        all_probs: List[float] = []
        all_preds: List[int] = []
        threshold = float(model_init_cfg.get("threshold", 0.5))

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels_dev = labels.to(device, non_blocking=True).float()

                with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                    logits = classifier.forward(images)
                    loss = criterion(logits, labels_dev)

                val_loss_sum += loss.item()
                val_batches += 1

                probs = torch.sigmoid(logits).cpu().tolist()
                all_probs.extend(probs)
                all_labels.extend(labels.tolist())
                all_preds.extend([int(p >= threshold) for p in probs])

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        val_auc = _compute_auc_roc(all_labels, all_probs)
        val_acc = _compute_accuracy(all_labels, all_preds)

        logger.info(
            "Epoch %d/%d — train_loss=%.4f  val_loss=%.4f  val_auc=%.4f  val_acc=%.4f",
            epoch + 1, epochs, avg_train_loss, avg_val_loss, val_auc, val_acc,
        )

        tracker.log_metrics(run_id, step=epoch, metrics={
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_auc": val_auc,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time_s": round(time.time() - epoch_start, 2),
        })

        # Save last checkpoint every epoch
        classifier.save_checkpoint(last_model_path)

        # Early stopping by val AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            classifier.save_checkpoint(best_model_path)
            logger.info("  → New best val AUC=%.4f — checkpoint saved", best_val_auc)
        else:
            epochs_without_improvement += 1
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                logger.info(
                    "Early stopping: no improvement for %d epochs", early_stopping_patience
                )
                break

    logger.info(
        "Training complete — best_val_auc=%.4f at epoch %d", best_val_auc, best_epoch
    )
    logger.info("Best model: %s", best_model_path)

    results = {
        # Fields required by dashboard FinalResults dataclass
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
        "best_val_loss": best_val_auc,   # dashboard uses this for "best" selection; repurpose as best AUC
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        # Extra classifier-specific fields
        "best_val_auc": best_val_auc,
        "best_model_path": str(best_model_path),
        "run_id": run_id,
    }
    tracker.end_run(run_id, results)
    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _compute_binary_metrics_at_threshold(
    labels: List[int], probs: List[float], threshold: float
) -> Dict:
    preds = [int(p >= threshold) for p in probs]
    tp = sum(l == 1 and p == 1 for l, p in zip(labels, preds))
    fp = sum(l == 0 and p == 1 for l, p in zip(labels, preds))
    fn = sum(l == 1 and p == 0 for l, p in zip(labels, preds))
    tn = sum(l == 0 and p == 0 for l, p in zip(labels, preds))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "confidence": round(threshold, 2),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def evaluate_classifier(config_path: str, checkpoint_path: str, split: str = "val") -> dict:
    """Evaluate a trained BinaryDamageClassifier and write a dashboard-compatible report.

    Loads the checkpoint, runs inference on the val split of RDD2022, sweeps
    decision thresholds, and writes ``{split}_evaluation_report.json`` to the
    checkpoint's parent directory (i.e. the run directory).

    Args:
        config_path: Path to the training config YAML (same file used for training).
        checkpoint_path: Path to the ``.pt`` checkpoint file (e.g. best_model.pt).
        split: Dataset split label written into the report filename, default ``"val"``.

    Returns:
        Dict with evaluation metrics.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config_manager = ConfigManager()
    config = config_manager.load(Path(config_path))
    config = config_manager.resolve_env_vars(config)

    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})

    input_size = int(model_cfg.get("input_size", 224))
    dataset_path = dataset_cfg.get("path", "model/data/rdd2022/")
    val_split = float(training_cfg.get("val_split", 0.2))
    seed = int(training_cfg.get("seed", 42))
    num_workers = int(training_cfg.get("num_workers", 4))
    batch_size = int(training_cfg.get("batch_size", 32))
    threshold = float(model_cfg.get("threshold", 0.5))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluate on device: %s", device)

    # Dataset
    logger.info("Loading dataset from %s …", dataset_path)
    rdd_dataset = RDD2022Dataset(country_filter=dataset_cfg.get("country_filter"))
    rdd_dataset.load(Path(dataset_path))
    all_annotations = rdd_dataset.get_annotations()

    rng = random.Random(seed)
    indices = list(range(len(all_annotations)))
    rng.shuffle(indices)
    split_at = int(len(indices) * (1.0 - val_split))
    if split == "val":
        eval_annotations = [all_annotations[i] for i in indices[split_at:]]
    else:
        eval_annotations = [all_annotations[i] for i in indices[:split_at]]
    logger.info("Evaluating on %d images (%s split)", len(eval_annotations), split)

    eval_ds = BinaryClassificationDataset(eval_annotations, input_size=input_size, augmentation=None)
    eval_loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # Model
    classifier_cfg = {
        "architecture": model_cfg.get("architecture", "efficientnet_b0"),
        "pretrained": False,
        "threshold": threshold,
    }
    classifier = BinaryDamageClassifier(classifier_cfg)
    classifier.load_checkpoint(Path(checkpoint_path))
    classifier.to_device(device)
    classifier.set_eval_mode()

    # Inference
    all_labels: List[int] = []
    all_probs: List[float] = []
    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device, non_blocking=True)
            logits = classifier.forward(images)
            probs = torch.sigmoid(logits).cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.tolist())

    # Metrics
    auc = _compute_auc_roc(all_labels, all_probs)
    thresholds = [round(t, 2) for t in [i / 20 for i in range(1, 20)]]  # 0.05 … 0.95
    sweep = [_compute_binary_metrics_at_threshold(all_labels, all_probs, t) for t in thresholds]
    best_f1_entry = max(sweep, key=lambda x: x["f1"])

    main = _compute_binary_metrics_at_threshold(all_labels, all_probs, threshold)
    tp, fp, fn, tn = main["tp"], main["fp"], main["fn"], main["tn"]
    confusion = [[tn, fp], [fn, tp]]

    n_pos = sum(all_labels)
    n_neg = len(all_labels) - n_pos
    sensitivity = tp / max(n_pos, 1)   # recall for damage class
    specificity = tn / max(n_neg, 1)   # recall for background class

    logger.info(
        "Eval — AUC=%.4f  F1=%.4f  precision=%.4f  recall=%.4f  acc=%.4f",
        auc, main["f1"], main["precision"], main["recall"],
        _compute_accuracy(all_labels, [int(p >= threshold) for p in all_probs]),
    )

    # Build report in dashboard-compatible format
    f1_sweep_clean = [
        {"confidence": e["confidence"], "precision": e["precision"],
         "recall": e["recall"], "f1": e["f1"]}
        for e in sweep
    ]
    best_f1_clean = {
        "confidence": best_f1_entry["confidence"],
        "precision": best_f1_entry["precision"],
        "recall": best_f1_entry["recall"],
        "f1": best_f1_entry["f1"],
    }

    report = {
        "checkpoint": str(checkpoint_path),
        "model_type": "binary_classifier",
        "dataset": str(dataset_path),
        "split": split,
        "confidence_threshold": threshold,
        "iou_threshold": 0.0,           # N/A for classification; required by dashboard
        "class_names": ["background", "damage"],
        "num_classes": 2,
        "num_images": len(all_labels),
        "num_val_images": len(all_labels),
        "metrics": {
            "map_50": round(auc, 4),        # AUC-ROC as primary metric
            "map_50_95": round(auc, 4),
            "per_class_ap": {
                "background": round(specificity, 4),
                "damage": round(sensitivity, 4),
            },
            "precision": main["precision"],
            "recall": main["recall"],
            "f1_score": main["f1"],
            "f1_sweep": f1_sweep_clean,
            "best_f1": best_f1_clean,
            "per_class_best_f1": {
                "background": {"specificity": round(specificity, 4)},
                "damage": {"sensitivity": round(sensitivity, 4)},
            },
            "default_confidence_threshold": threshold,
            "auc_roc": round(auc, 4),
        },
        "confusion_matrix": confusion,
        "errors": {"count": 0, "list": []},
    }

    # Write to checkpoint parent directory (the run dir)
    ckpt_path = Path(checkpoint_path)
    out_dir = ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{split}_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Evaluation report → %s", report_path)

    return {
        "auc_roc": auc,
        "precision": main["precision"],
        "recall": main["recall"],
        "f1": main["f1"],
        "best_f1": best_f1_clean["f1"],
        "best_threshold": best_f1_clean["confidence"],
        "report_path": str(report_path),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train BinaryDamageClassifier on RDD2022."
    )
    parser.add_argument("config", help="Path to the training config YAML.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    results = train_classifier(args.config, verbose=args.verbose)
    print(f"\nDone. best_val_auc={results['best_val_auc']:.4f} at epoch {results['best_epoch']}")
