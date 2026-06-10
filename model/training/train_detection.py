"""Unified training loop for detection models.

This module provides a single, model-agnostic training function that operates
exclusively through the BaseDetector interface. All registered model types are
trained using the same epoch loop, optimizer construction, data loading, and
loss computation path via `model.train_step()`.

Usage:
    python -m model.training.train_detection --config model/configs/train_ssd_mobilenet.yaml
"""

import argparse
import inspect
import logging
import math
import platform
import random
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple


import cv2
import torch
import torch.utils.data
from torchvision import transforms as T
from PIL import Image
import numpy as np

from model.config.manager import ConfigManager
from model.training.augmentation import build_augmentation_pipeline
from model.datasets.rdd2022 import RDD2022Dataset
from model.models import ModelRegistry
from model.exceptions import ModelNotFoundError, ConfigurationError
from model.tracking.tracker import ExperimentTracker

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# PyTorch Dataset adapter
# -------------------------------------------------------------------------


class RDD2022TorchDataset(torch.utils.data.Dataset):
    """PyTorch Dataset adapter for RDD2022Dataset.

    Converts the framework's Annotation objects into the format expected
    by torchvision detection models: (image_tensor, target_dict).

    Supports multi-image augmentations (Mosaic, MixUp) at the dataset level
    in addition to per-image transforms from augmentation.py.
    """

    def __init__(
        self,
        dataset: RDD2022Dataset,
        input_size: int = 320,
        augmentation=None,
        mosaic: float = 0.0,
        mixup: float = 0.0,
    ):
        """Initialize the dataset adapter.

        Args:
            dataset: Underlying RDD2022Dataset instance.
            input_size: Target image size (square).
            augmentation: Per-image augmentation Compose pipeline or None.
            mosaic: Probability of applying Mosaic augmentation (0=off, 1=always).
            mixup: Probability of applying MixUp after Mosaic (0=off).
        """
        self._annotations = dataset.get_annotations()
        self._input_size = input_size
        self._class_names = dataset.get_class_names()
        self._augmentation = augmentation  # augmentation.Compose pipeline or None
        self._mosaic_p = mosaic
        self._mixup_p = mixup
        self._mosaic_enabled = mosaic > 0  # Can be toggled off for final epochs
        # Map class names to 0-indexed labels (YOLO models don't use a
        # background class; valid indices are 0..num_classes-1)
        self._class_to_idx = {
            name: idx for idx, name in enumerate(self._class_names)
        }
        self._transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
        ])

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @property
    def num_classes(self) -> int:
        return len(self._class_names)

    def set_mosaic_enabled(self, enabled: bool) -> None:
        """Enable or disable Mosaic augmentation (for mosaic_off_epochs)."""
        self._mosaic_enabled = enabled

    def compute_sample_weights(self) -> List[float]:
        """Compute per-image sampling weights for class-balanced training.

        Uses inverse-frequency class weighting: rare classes get higher weight.
        Each image's weight is the MAX weight among the classes it contains,
        so an image with a rare class (e.g. pothole) is oversampled even if it
        also contains common classes.

        Returns:
            List of float weights, one per annotation, suitable for
            torch.utils.data.WeightedRandomSampler.
        """
        # Count occurrences per class label
        class_counts: dict = {name: 0 for name in self._class_names}
        for ann in self._annotations:
            present = set()
            for bbox in ann.bounding_boxes:
                if bbox.class_label in class_counts:
                    present.add(bbox.class_label)
            for label in present:
                class_counts[label] += 1

        # Inverse-frequency class weight (avoid div-by-zero)
        total = sum(class_counts.values()) or 1
        class_weight = {
            name: (total / (count + 1)) for name, count in class_counts.items()
        }

        # Per-image weight = max class weight among present classes
        weights: List[float] = []
        for ann in self._annotations:
            present = [
                bbox.class_label for bbox in ann.bounding_boxes
                if bbox.class_label in class_weight
            ]
            if present:
                weights.append(max(class_weight[c] for c in present))
            else:
                # Background-only / empty image: lowest weight
                weights.append(1.0)
        return weights

    def __len__(self) -> int:
        return len(self._annotations)

    def _load_image_and_bboxes(self, idx: int) -> Tuple[np.ndarray, List[List]]:
        """Load a single image and its bounding boxes as numpy + normalized coords.

        The image is immediately resized to (input_size x input_size) to cap
        per-image memory at ~1.2MB regardless of source resolution. Bounding
        boxes remain in normalized [0,1] coordinates (unaffected by resize).

        Args:
            idx: Index into annotations list.

        Returns:
            Tuple of (image_np [input_size, input_size, 3 uint8],
                      bboxes [[x1,y1,x2,y2,class_label], ...]).
        """
        annotation = self._annotations[idx]
        try:
            image = Image.open(annotation.image_path).convert("RGB")
            image_np = np.array(image)
            image.close()  # Release file descriptor and PIL buffer
        except (FileNotFoundError, OSError) as e:
            logger.warning("Could not load image %s: %s", annotation.image_path, e)
            return np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8), []

        # Early resize: cap memory at input_size×input_size×3 (~1.2MB for 640)
        # regardless of source resolution (e.g. 3264×2448 = ~24MB).
        # Bounding boxes are normalized so they are unaffected by resize.
        h, w = image_np.shape[:2]
        if h != self._input_size or w != self._input_size:
            image_np = cv2.resize(
                image_np, (self._input_size, self._input_size),
                interpolation=cv2.INTER_LINEAR,
            )

        bboxes = []
        for bbox in annotation.bounding_boxes:
            bboxes.append([bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max, bbox.class_label])
        return image_np, bboxes

    def _build_mosaic(self, idx: int) -> Tuple[np.ndarray, List[List]]:
        """Build a 4-image mosaic centered at a random point.

        Samples 3 additional random images and places all 4 into a 2x2 grid
        with a random center point in [0.25*size, 0.75*size] for both axes
        (Ultralytics default).

        Args:
            idx: Primary image index.

        Returns:
            Tuple of (mosaic_image [input_size, input_size, 3], merged_bboxes).
        """
        s = self._input_size
        # Random mosaic center
        cx = int(random.uniform(0.25 * s, 0.75 * s))
        cy = int(random.uniform(0.25 * s, 0.75 * s))

        # Sample 3 additional indices
        n = len(self._annotations)
        indices = [idx] + [random.randint(0, n - 1) for _ in range(3)]

        # Canvas (gray fill like Ultralytics)
        mosaic_img = np.full((s, s, 3), 114, dtype=np.uint8)
        merged_bboxes: List[List] = []

        for i, img_idx in enumerate(indices):
            img_np, bboxes = self._load_image_and_bboxes(img_idx)

            # Determine placement region for each quadrant
            if i == 0:  # top-left
                # Image region that maps to canvas [0:cy, 0:cx]
                x1c, y1c, x2c, y2c = 0, 0, cx, cy
            elif i == 1:  # top-right
                x1c, y1c, x2c, y2c = cx, 0, s, cy
            elif i == 2:  # bottom-left
                x1c, y1c, x2c, y2c = 0, cy, cx, s
            else:  # bottom-right
                x1c, y1c, x2c, y2c = cx, cy, s, s

            # Target region dimensions on canvas
            tw = x2c - x1c
            th = y2c - y1c
            if tw <= 0 or th <= 0:
                del img_np  # Free early
                continue

            # Resize image to fit into its quadrant
            mosaic_img[y1c:y2c, x1c:x2c] = cv2.resize(
                img_np, (tw, th), interpolation=cv2.INTER_LINEAR
            )
            del img_np  # Free source image immediately after placement

            # Transform bboxes: map from normalized [0,1] of original image
            # to normalized [0,1] of the mosaic canvas
            for bbox in bboxes:
                # Map bbox from original image space to canvas pixel space
                bx1 = x1c + bbox[0] * tw
                by1 = y1c + bbox[1] * th
                bx2 = x1c + bbox[2] * tw
                by2 = y1c + bbox[3] * th
                # Normalize to mosaic canvas [0, 1]
                nx1 = bx1 / s
                ny1 = by1 / s
                nx2 = bx2 / s
                ny2 = by2 / s
                merged_bboxes.append([nx1, ny1, nx2, ny2, bbox[4]])

        # Clip and filter degenerate boxes
        from model.training.augmentation import _clip_and_filter_bboxes
        merged_bboxes = _clip_and_filter_bboxes(merged_bboxes)

        return mosaic_img, merged_bboxes

    def _apply_mixup(self, img1: np.ndarray, bboxes1: List[List]) -> Tuple[np.ndarray, List[List]]:
        """Apply MixUp by blending with a random second image.

        Uses beta distribution (alpha=1.5, beta=1.5) for blend ratio,
        yielding a ratio typically in [0.3, 0.7].

        Args:
            img1: Primary image (numpy array, same size as input_size).
            bboxes1: Bounding boxes for primary image.

        Returns:
            Tuple of (blended_image, merged_bboxes).
        """
        # Pick a random second image
        idx2 = random.randint(0, len(self._annotations) - 1)

        # If mosaic is enabled, second image is also a mosaic
        if self._mosaic_enabled and self._mosaic_p > 0:
            img2, bboxes2 = self._build_mosaic(idx2)
        else:
            img2, bboxes2 = self._load_image_and_bboxes(idx2)
            # Resize img2 to input_size
            img2 = cv2.resize(img2, (self._input_size, self._input_size), interpolation=cv2.INTER_LINEAR)
            # bboxes2 are already normalized

        # Blend ratio from beta distribution
        ratio = np.random.beta(1.5, 1.5)
        ratio = max(0.3, min(0.7, ratio))  # Clamp to avoid near-identity

        # Blend images using cv2.addWeighted (avoids allocating float32 copies)
        blended = cv2.addWeighted(img1, ratio, img2, 1.0 - ratio, 0.0)

        # Merge bboxes from both images (both are already normalized)
        merged = list(bboxes1) + list(bboxes2)
        return blended, merged

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        # --- Multi-image augmentations (Mosaic + MixUp) ---
        # MixUp is independent of Mosaic: it can trigger on its own (blending
        # two single images) or after a mosaic. Previously MixUp was gated by
        # Mosaic, so with mosaic=0 MixUp never fired regardless of its prob.
        use_mosaic = self._mosaic_enabled and random.random() < self._mosaic_p
        use_mixup = self._mixup_p > 0 and random.random() < self._mixup_p

        if use_mosaic or use_mixup:
            if use_mosaic:
                image_np, bboxes = self._build_mosaic(idx)
            else:
                # MixUp-only path: load the primary image as the base
                image_np, bboxes = self._load_image_and_bboxes(idx)
                image_np = cv2.resize(
                    image_np, (self._input_size, self._input_size),
                    interpolation=cv2.INTER_LINEAR,
                )

            if use_mixup:
                image_np, bboxes = self._apply_mixup(image_np, bboxes)

            # Apply per-image augmentation pipeline after mosaic/mixup
            if self._augmentation is not None:
                image_np, bboxes = self._augmentation(image_np, bboxes)

            # Convert to tensor directly from numpy (avoids PIL intermediate allocation)
            # image_np is already (input_size, input_size, 3) uint8
            image_tensor = torch.from_numpy(
                image_np.transpose(2, 0, 1).copy()  # HWC -> CHW
            ).float().div_(255.0)

            # Convert bboxes to pixel coords at input_size
            boxes = []
            labels = []
            for bbox in bboxes:
                x1 = bbox[0] * self._input_size
                y1 = bbox[1] * self._input_size
                x2 = bbox[2] * self._input_size
                y2 = bbox[3] * self._input_size
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    class_idx = self._class_to_idx.get(bbox[4], 0)
                    labels.append(class_idx)
        else:
            # --- Single-image path (original or with per-image augmentation) ---
            annotation = self._annotations[idx]

            # Load image
            try:
                image = Image.open(annotation.image_path).convert("RGB")
            except (FileNotFoundError, OSError) as e:
                logger.warning("Could not load image %s: %s", annotation.image_path, e)
                blank = Image.new("RGB", (self._input_size, self._input_size))
                target = {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                }
                tensor = self._transform(blank)
                blank.close()
                return tensor, target

            orig_w, orig_h = image.size

            # Apply augmentation if configured (operates on numpy array + normalized bboxes)
            if self._augmentation is not None:
                image_np = np.array(image)
                image.close()  # Release PIL buffer
                aug_bboxes = []
                for bbox in annotation.bounding_boxes:
                    aug_bboxes.append([bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max, bbox.class_label])

                image_np, aug_bboxes = self._augmentation(image_np, aug_bboxes)

                # Resize to input_size and convert to tensor (avoid PIL intermediate)
                h, w = image_np.shape[:2]
                if h != self._input_size or w != self._input_size:
                    image_np = cv2.resize(
                        image_np, (self._input_size, self._input_size),
                        interpolation=cv2.INTER_LINEAR,
                    )
                image_tensor = torch.from_numpy(
                    image_np.transpose(2, 0, 1).copy()
                ).float().div_(255.0)
                boxes = []
                labels = []
                for bbox in aug_bboxes:
                    x1 = bbox[0] * self._input_size
                    y1 = bbox[1] * self._input_size
                    x2 = bbox[2] * self._input_size
                    y2 = bbox[3] * self._input_size
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        class_idx = self._class_to_idx.get(bbox[4], 0)
                        labels.append(class_idx)
            else:
                image_tensor = self._transform(image)
                image.close()  # Release PIL buffer
                boxes = []
                labels = []
                for bbox in annotation.bounding_boxes:
                    x1 = bbox.x_min * self._input_size
                    y1 = bbox.y_min * self._input_size
                    x2 = bbox.x_max * self._input_size
                    y2 = bbox.y_max * self._input_size
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        class_idx = self._class_to_idx.get(bbox.class_label, 0)
                        labels.append(class_idx)

        if boxes:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
            }
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            }

        return image_tensor, target


def collate_fn(batch):
    """Custom collate function for detection (variable number of boxes per image)."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


# -------------------------------------------------------------------------
# mAP validation (Group A)
# -------------------------------------------------------------------------


@torch.no_grad()
def evaluate_map(model, val_loader, class_names, input_size, device,
                 confidence_threshold=0.25, max_batches=None):
    """Compute mAP@0.5 on the validation set using model.forward().

    Runs the detector in eval mode over the val_loader (images already resized
    to input_size, boxes in pixel coords of input_size). Predictions and ground
    truths are normalized to [0,1] and labels mapped to class-name strings, then
    fed to compute_map().

    Args:
        model: BaseDetector with set_eval_mode() and forward().
        val_loader: DataLoader yielding (images, targets) like the training loop.
        class_names: Ordered list of class-name strings (index -> name).
        input_size: Square input size used to normalize boxes.
        device: torch device.
        confidence_threshold: Drop predictions below this score.
        max_batches: If set, evaluate only the first N batches (subset eval).

    Returns:
        Dict from compute_map: {"map_50", "map_50_95", "per_class_ap"}.
    """
    from model.evaluation.metrics import compute_map

    model.set_eval_mode()
    predictions: List[dict] = []
    ground_truths: List[dict] = []

    img_counter = 0
    for batch_idx, (images, targets) in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = torch.stack([img.to(device) for img in images])
        results = model.forward(batch)

        for i, (result, target) in enumerate(zip(results, targets)):
            image_id = f"img_{img_counter}"
            img_counter += 1

            # --- Predictions ---
            p_boxes, p_labels, p_scores = [], [], []
            r_boxes = result.get("boxes")
            r_labels = result.get("labels")
            r_scores = result.get("scores")
            n = len(r_boxes) if r_boxes is not None else 0
            for j in range(n):
                score = float(r_scores[j].item() if hasattr(r_scores[j], "item") else r_scores[j])
                if score < confidence_threshold:
                    continue
                box = r_boxes[j].tolist() if hasattr(r_boxes[j], "tolist") else list(r_boxes[j])
                box = [box[0] / input_size, box[1] / input_size,
                       box[2] / input_size, box[3] / input_size]
                lbl = int(r_labels[j].item() if hasattr(r_labels[j], "item") else r_labels[j])
                if lbl < 0 or lbl >= len(class_names):
                    continue
                p_boxes.append(box)
                p_labels.append(class_names[lbl])
                p_scores.append(score)
            predictions.append({
                "image_id": image_id, "boxes": p_boxes,
                "labels": p_labels, "scores": p_scores,
            })

            # --- Ground truths (target boxes are pixel coords at input_size) ---
            g_boxes, g_labels = [], []
            t_boxes = target["boxes"]
            t_labels = target["labels"]
            for j in range(len(t_boxes)):
                box = t_boxes[j].tolist()
                box = [box[0] / input_size, box[1] / input_size,
                       box[2] / input_size, box[3] / input_size]
                lbl = int(t_labels[j].item())
                if lbl < 0 or lbl >= len(class_names):
                    continue
                g_boxes.append(box)
                g_labels.append(class_names[lbl])
            ground_truths.append({
                "image_id": image_id, "boxes": g_boxes, "labels": g_labels,
            })

    return compute_map(
        predictions=predictions,
        ground_truths=ground_truths,
        iou_thresholds=[0.5],
        class_names=list(class_names),
    )


# -------------------------------------------------------------------------
# Resume helpers
# -------------------------------------------------------------------------


def _get_model_state_dict(model):
    """Extract state dict from various model wrapper patterns."""
    if hasattr(model, "_model") and hasattr(model._model, "model"):
        return model._model.model.state_dict()
    elif hasattr(model, "_model"):
        return model._model.state_dict()
    elif hasattr(model, "model"):
        return model.model.state_dict()
    return model.state_dict()


def _set_model_state_dict(model, state_dict):
    """Load state dict into various model wrapper patterns."""
    if hasattr(model, "_model") and hasattr(model._model, "model"):
        model._model.model.load_state_dict(state_dict)
    elif hasattr(model, "_model"):
        model._model.load_state_dict(state_dict)
    elif hasattr(model, "model"):
        model.model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


def _save_training_state(path, model, optimizer, scheduler, scaler, epoch,
                         best_val_loss, best_epoch, epochs_without_improvement,
                         run_id, config_used, best_map=0.0):
    """Save full training state (model + optimizer + scheduler + metadata) for resume."""
    state = {
        "model_state_dict": _get_model_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "best_map": best_map,
        "run_id": run_id,
        "config_used": config_used,
    }
    torch.save(state, str(path))


def _load_training_state(path, model, optimizer, scheduler, scaler, device):
    """Load training state; restore model/optimizer/scheduler. Returns metadata dict.

    Supports both new-format checkpoints (with optimizer/scheduler/scaler state) and
    old-format checkpoints (model weights only). When optimizer state is missing,
    only model weights are loaded and a warning is logged.
    """
    state = torch.load(str(path), map_location=device)
    _set_model_state_dict(model, state["model_state_dict"])

    if "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler and state.get("scheduler_state_dict"):
            scheduler.load_state_dict(state["scheduler_state_dict"])
        if scaler and state.get("scaler_state_dict"):
            scaler.load_state_dict(state["scaler_state_dict"])
    else:
        logger.warning(
            "Checkpoint '%s' has no optimizer state (old format). "
            "Only model weights loaded; optimizer will start fresh.",
            path,
        )
        # Fill in missing metadata with defaults for old-format checkpoints
        state.setdefault("epoch", -1)
        state.setdefault("best_val_loss", float("inf"))
        state.setdefault("best_epoch", 0)
        state.setdefault("epochs_without_improvement", 0)
        state.setdefault("run_id", None)

    return state


def _resolve_resume_path(resume_from, checkpoint_dir, model_type):
    """Resolve a --resume argument to an actual .pt file path.

    Supports:
    - Direct path to a .pt file
    - Run ID (looks up standard checkpoint locations)
    """
    path = Path(resume_from)
    if path.suffix == ".pt":
        if path.exists():
            return path
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")

    # Treat as run ID: search under <checkpoint_dir>/<run_id>/
    # (checkpoint_dir already includes the model-type subdirectory)
    run_dir = Path(checkpoint_dir) / resume_from
    candidates = [
        run_dir / "training_state.pt",
        run_dir / "recovery.pt",
        run_dir / "final_model.pt",
        run_dir / "best_model.pt",
    ]
    for c in candidates:
        if c.exists():
            return c

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"No checkpoint found for run ID '{resume_from}'.\nSearched:\n  {searched}"
    )


# -------------------------------------------------------------------------
# Unified training function
# -------------------------------------------------------------------------


def train(config_path: str, verbose: bool = False, resume_from: Optional[str] = None) -> dict:
    """Run the unified training loop for any registered detection model.

    This function provides a single entry point that trains any model type
    through the BaseDetector interface without model-type branching. All models
    use the same data pipeline, optimizer construction, and epoch loop.

    Args:
        config_path: Path to the YAML training configuration.
        verbose: Enable debug logging.

    Returns:
        Dict with final training metrics.
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config
    config_manager = ConfigManager()
    config = config_manager.load(Path(config_path))
    config = config_manager.resolve_env_vars(config)

    # Extract settings
    model_config = config.get("model", {}).get("config", {})
    model_type = config.get("model", {}).get("type", "ssd_mobilenetv3")
    dataset_config = config.get("dataset", {})
    training_config = config.get("training", {})

    input_size = model_config.get("input_size", 320)
    num_classes = model_config.get("num_classes", 5)
    dataset_path = dataset_config.get("path", "model/data/rdd2022/sample")
    country_filter = dataset_config.get("country_filter")

    epochs = training_config.get("epochs", 100)
    batch_size = training_config.get("batch_size", 16)
    learning_rate = training_config.get("learning_rate", 0.01)
    optimizer_name = training_config.get("optimizer", "SGD")
    weight_decay = training_config.get("weight_decay", 0.0005)
    momentum = training_config.get("momentum", 0.937)
    warmup_epochs = training_config.get("warmup_epochs", 3)
    val_split = training_config.get("val_split", 0.2)
    checkpoint_dir = Path(training_config.get("checkpoint_dir", "./checkpoints"))
    log_interval = training_config.get("log_interval", 10)
    use_amp = training_config.get("use_amp", True)
    num_workers = training_config.get("num_workers", 4)
    early_stopping_patience = training_config.get("early_stopping_patience", 15)
    resume_from = resume_from or training_config.get("resume_from")

    # --- mAP validation settings (Group A) ---
    # early_stopping_metric: "loss" (default, legacy) or "map".
    # When "map", best_model.pt is selected by highest mAP@0.5 and early
    # stopping monitors mAP. eval_interval controls how often mAP is computed.
    early_stopping_metric = str(training_config.get("early_stopping_metric", "loss")).lower()
    eval_interval = int(training_config.get("eval_interval", 5))
    eval_subset_batches = training_config.get("eval_subset_batches")  # None = full val
    eval_confidence = float(
        config.get("evaluation", {}).get("confidence_threshold", 0.25)
    )
    use_map_selection = early_stopping_metric == "map"

    # Reproducibility seed
    seed = training_config.get("seed", 42)
    import random as _random
    import numpy as _np
    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    logger.info("Random seed set to %d", seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # --- Instantiate model via ModelRegistry (no model-type branching) ---
    model_cfg = dict(model_config)
    model_cfg["num_classes"] = num_classes
    # Pass loss configuration to the model (label smoothing, focal loss)
    loss_config = training_config.get("loss", {})
    if loss_config:
        model_cfg["loss"] = loss_config
    logger.info("Building model '%s' (num_classes=%d)", model_type, num_classes)

    try:
        model = ModelRegistry.create(model_type, model_cfg)
    except ModelNotFoundError as e:
        logger.error("Failed to create model: %s", e)
        return {}
    except ConfigurationError as e:
        logger.error("Model configuration error: %s", e)
        return {}

    # Move model to device — check for underlying nn.Module via common wrapper patterns
    if hasattr(model, "_model") and hasattr(model._model, "model"):
        # Ultralytics-style wrapper (e.g., YOLO26Detector, RT_DETR_Detector)
        model._model.model.to(device)
        # Re-initialize criterion on the correct device (must happen after .to(device))
        # Note: RT-DETR's init_criterion() may fail because RTDETRDecoder lacks
        # the 'stride' attribute expected by v8DetectionLoss. In that case, fall
        # back to the existing criterion or use model.loss() directly.
        if hasattr(model._model.model, "init_criterion"):
            try:
                model._model.model.criterion = model._model.model.init_criterion()
                if hasattr(model, "_loss_fn"):
                    model._loss_fn = model._model.model.criterion
            except (AttributeError, TypeError) as e:
                logger.debug(
                    "init_criterion() not supported for this model, "
                    "using existing loss function: %s", e
                )
    elif hasattr(model, "_model") and hasattr(model._model, "to"):
        model._model.to(device)
    elif hasattr(model, "model") and hasattr(model.model, "to"):
        model.model.to(device)
    elif hasattr(model, "to"):
        model.to(device)
    logger.info("Model moved to %s", device)

    # Compile model for faster CUDA execution (torch.compile, requires PyTorch 2.0+)
    #
    # IMPORTANT: Ultralytics-based wrappers (rt_detr, mmr_detr, yolo26) drive
    # inference through self._model.predict(), which internally references
    # self._model.model. Wrapping that module in torch.compile() yields an
    # OptimizedModule that breaks the Ultralytics predict() pipeline, causing
    # forward()/evaluate_map() to silently return empty detections (mAP=0.0).
    # We therefore SKIP compilation for these wrappers and only compile plain
    # nn.Module models that run inference directly through model.forward().
    is_ultralytics_wrapper = (
        hasattr(model, "_model")
        and hasattr(model._model, "predict")
        and hasattr(model._model, "model")
    )
    if hasattr(torch, "compile") and device.type == "cuda" and not is_ultralytics_wrapper:
        try:
            if hasattr(model, "_model"):
                model._model = torch.compile(model._model, mode="reduce-overhead")
            elif hasattr(model, "model"):
                model.model = torch.compile(model.model, mode="reduce-overhead")
            logger.info("Model compiled with torch.compile (mode=reduce-overhead)")
        except Exception as e:
            logger.warning("torch.compile failed, continuing without compilation: %s", e)
    elif is_ultralytics_wrapper:
        logger.info(
            "Skipping torch.compile for Ultralytics-based wrapper "
            "(compilation breaks predict()/mAP evaluation)"
        )


    logger.info("Loading dataset from %s", dataset_path)
    rdd_dataset = RDD2022Dataset(country_filter=country_filter)
    rdd_dataset.load(Path(dataset_path))
    logger.info("Dataset loaded: %d images, classes: %s", len(rdd_dataset), rdd_dataset.get_class_names())

    # Split into train/val
    train_ratio = 1.0 - val_split
    train_ds, val_ds, _ = rdd_dataset.split(train_ratio, val_split, 0.0, seed=seed)
    logger.info("Train: %d, Val: %d", len(train_ds), len(val_ds))

    # Build augmentation pipeline from config (only applied to training set)
    aug_config = training_config.get("augmentation", {})
    augmentation_pipeline = build_augmentation_pipeline(aug_config) if aug_config else None
    logger.info("Augmentation pipeline: %s", augmentation_pipeline)

    # Multi-image augmentation params (handled at dataset level)
    mosaic_p = float(aug_config.get("mosaic", 0.0))
    mixup_p = float(aug_config.get("mixup", 0.0))
    mosaic_off_epochs = int(aug_config.get("mosaic_off_epochs", 0))
    logger.info("Mosaic p=%.2f, MixUp p=%.2f, mosaic_off_epochs=%d", mosaic_p, mixup_p, mosaic_off_epochs)

    # Create PyTorch datasets
    train_torch = RDD2022TorchDataset(
        train_ds, input_size=input_size, augmentation=augmentation_pipeline,
        mosaic=mosaic_p, mixup=mixup_p,
    )
    val_torch = RDD2022TorchDataset(val_ds, input_size=input_size)  # No augmentation for validation

    # Ordered class names for mAP label-index mapping (Group A)
    class_names_list = train_torch.class_names

    # On Windows, DataLoader workers require explicit spawn context
    is_windows = platform.system() == "Windows"
    effective_workers = num_workers
    mp_context = "spawn" if is_windows and effective_workers > 0 else None

    # --- Class-balanced sampling (Group D) ---
    # When enabled, oversample images containing rare classes (e.g. pothole)
    # via WeightedRandomSampler. Default off preserves existing behavior.
    class_balanced_sampling = bool(training_config.get("class_balanced_sampling", False))
    train_sampler = None
    if class_balanced_sampling:
        sample_weights = train_torch.compute_sample_weights()
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        logger.info(
            "Class-balanced sampling ENABLED (WeightedRandomSampler over %d images)",
            len(sample_weights),
        )

    # Create data loaders. shuffle must be False when a sampler is provided.
    train_loader = torch.utils.data.DataLoader(
        train_torch,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=effective_workers,
        pin_memory=True,
        persistent_workers=effective_workers > 0,
        multiprocessing_context=mp_context,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_torch,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=True,
        persistent_workers=effective_workers > 0,
        multiprocessing_context=mp_context,
        collate_fn=collate_fn,
    )

    # --- Construct optimizer params (discriminative LR if configured) ---
    # When config provides backbone_lr/head_lr and the model exposes
    # get_parameter_groups, use discriminative learning rates (Group B).
    # Otherwise fall back to a flat parameter list (legacy behavior).
    backbone_lr = training_config.get("backbone_lr")
    head_lr = training_config.get("head_lr")
    backbone_layers = int(training_config.get("backbone_layers", 10))
    if (
        backbone_lr is not None
        and head_lr is not None
        and hasattr(model, "get_parameter_groups")
    ):
        params = model.get_parameter_groups(
            backbone_lr=float(backbone_lr),
            head_lr=float(head_lr),
            backbone_layers=backbone_layers,
        )
        logger.info(
            "Using discriminative learning rates (backbone=%.2e, head=%.2e)",
            float(backbone_lr), float(head_lr),
        )
    else:
        params = model.get_parameters()

    if optimizer_name.upper() == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer_name.upper() == "ADAM":
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.upper() == "ADAMW":
        optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.upper() == "MUSGD":
        try:
            from ultralytics.optim.muon import MuSGD
            # MuSGD needs parameter groups: use_muon=True only for ndim >= 2
            muon_params = []
            sgd_params = []
            for p in params:
                if p.ndim >= 2:
                    muon_params.append(p)
                else:
                    sgd_params.append(p)
            param_groups = [
                {"params": muon_params, "use_muon": True},
                {"params": sgd_params, "use_muon": False},
            ]
            optimizer = MuSGD(
                param_groups, lr=learning_rate, momentum=momentum,
                weight_decay=weight_decay, nesterov=True,
                muon=0.2, sgd=1.0,
            )
            logger.info("Using MuSGD optimizer (Muon + SGD hybrid)")
        except ImportError:
            logger.warning(
                "MuSGD requested but ultralytics is not installed. Falling back to SGD."
            )
            optimizer = torch.optim.SGD(
                params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
            )
    else:
        # Fallback to SGD for unknown optimizer values
        logger.warning("Unknown optimizer '%s', falling back to SGD", optimizer_name)
        optimizer = torch.optim.SGD(
            params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )

    # --- Learning rate scheduler: cosine annealing (stepped only after warmup) ---
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=learning_rate * 0.01
    )

    # --- Mixed precision training (AMP) ---
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if use_amp:
        logger.info("Mixed precision training (AMP) enabled")
    else:
        logger.info("Mixed precision training (AMP) disabled, using full precision")

    # --- SIGINT handling ---
    interrupted = False
    original_sigint_handler = signal.getsignal(signal.SIGINT)

    def _sigint_handler(signum, frame):
        nonlocal interrupted
        if interrupted:
            # Second SIGINT: raise immediately
            raise KeyboardInterrupt
        interrupted = True
        logger.info("SIGINT received. Completing current epoch then stopping...")

    signal.signal(signal.SIGINT, _sigint_handler)

    # --- Experiment tracking ---
    tracker = ExperimentTracker(output_dir=checkpoint_dir)
    dataset_name = dataset_config.get("name", Path(dataset_path).name)

    # --- Determine if save_checkpoint accepts extra params ---
    save_sig = inspect.signature(model.save_checkpoint)
    save_params = list(save_sig.parameters.keys())
    supports_extra_checkpoint_params = "optimizer" in save_params

    def _save_checkpoint(path: Path, optimizer_obj=None, epoch_num=None, metrics_dict=None):
        """Save checkpoint, passing extra params if the model supports them."""
        if supports_extra_checkpoint_params:
            model.save_checkpoint(path, optimizer=optimizer_obj, epoch=epoch_num, metrics=metrics_dict)
        else:
            model.save_checkpoint(path)

    # --- Resume handling ---
    start_epoch = 0
    if resume_from:
        try:
            resume_path = _resolve_resume_path(resume_from, checkpoint_dir, model_type)
            logger.info("Loading resume state from: %s", resume_path)
            state = _load_training_state(resume_path, model, optimizer, cosine_scheduler, scaler, device)
            run_id = state.get("run_id")
            if run_id:
                # Full training state: resume with same run ID and skip completed epochs
                start_epoch = state["epoch"] + 1
                best_val_loss = state["best_val_loss"]
                best_epoch = state["best_epoch"]
                epochs_without_improvement = state["epochs_without_improvement"]
                best_map = state.get("best_map", 0.0)
                run_checkpoint_dir = Path(checkpoint_dir) / run_id
                logger.info(
                    "Resuming run %s from epoch %d (best_val_loss=%.4f at epoch %d)",
                    run_id, start_epoch, best_val_loss, best_epoch,
                )
            else:
                # Old-format checkpoint (model weights only): start fresh with loaded weights
                logger.info(
                    "Old-format checkpoint loaded (model weights only). Starting new training run."
                )
                run_id = tracker.start_run(config, model_type, dataset_name)
                run_checkpoint_dir = checkpoint_dir / run_id
                run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                best_val_loss = float("inf")
                best_epoch = 0
                epochs_without_improvement = 0
                best_map = 0.0
        except Exception as e:
            logger.error("Failed to resume from '%s': %s", resume_from, e)
            signal.signal(signal.SIGINT, original_sigint_handler)
            return {}
    else:
        try:
            run_id = tracker.start_run(config, model_type, dataset_name)
        except Exception as e:
            logger.error("Failed to start experiment run: %s", e)
            signal.signal(signal.SIGINT, original_sigint_handler)
            return {}
        run_checkpoint_dir = checkpoint_dir / run_id
        run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        best_map = 0.0

    # --- Epoch loop ---
    logger.info("Starting training: %d epochs, batch_size=%d, lr=%.6f", epochs, batch_size, learning_rate)

    avg_train_loss = 0.0
    avg_val_loss = float("inf")
    completed_epochs = 0

    try:
        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()

            # --- Mosaic off for final N epochs (fine-tune on clean images) ---
            if mosaic_off_epochs > 0 and epoch >= (epochs - mosaic_off_epochs):
                if train_torch._mosaic_enabled:
                    train_torch.set_mosaic_enabled(False)
                    logger.info(
                        "Epoch %d: Disabling Mosaic for final %d epochs",
                        epoch + 1, mosaic_off_epochs,
                    )

            # --- Linear warmup: LR = learning_rate * (epoch + 1) / warmup_epochs ---
            if epoch < warmup_epochs:
                warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group["lr"] = warmup_lr
                current_lr = warmup_lr
            else:
                current_lr = optimizer.param_groups[0]["lr"]

            # --- Training phase ---
            model.set_train_mode()
            train_loss_sum = 0.0
            train_batches = 0

            for batch_idx, (images, targets) in enumerate(train_loader):
                # Move data to device
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()

                try:
                    # Forward pass with optional AMP autocast
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            loss_dict = model.train_step(images, targets)
                    else:
                        loss_dict = model.train_step(images, targets)

                    loss_tensor = loss_dict["loss_tensor"]

                except Exception as e:
                    # Handle exceptions in train_step: log warning, skip batch
                    logger.warning(
                        "Exception in train_step at epoch %d, batch %d: %s. Skipping batch.",
                        epoch, batch_idx, e,
                    )
                    continue

                # Handle zero-loss batches: skip backward/step
                if loss_tensor.item() == 0.0:
                    train_batches += 1
                    continue

                # Backward pass with gradient scaling
                scaler.scale(loss_tensor).backward()

                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.get_parameters(), max_norm=10.0)

                # Optimizer step (scaler.step is a no-op if inf/NaN detected)
                scaler.step(optimizer)
                scaler.update()

                train_loss_sum += loss_tensor.item()
                train_batches += 1

                if (batch_idx + 1) % log_interval == 0:
                    avg_loss = train_loss_sum / max(train_batches, 1)
                    logger.info(
                        "Epoch %d/%d | Batch %d/%d | Loss: %.4f | LR: %.6f",
                        epoch + 1, epochs, batch_idx + 1, len(train_loader),
                        avg_loss, current_lr,
                    )

            # Step cosine scheduler only at epochs >= warmup_epochs
            if epoch >= warmup_epochs:
                cosine_scheduler.step()

            # Compute epoch training metrics
            avg_train_loss = train_loss_sum / max(train_batches, 1)

            # --- Validation phase ---
            model.set_eval_mode()
            val_loss_sum = 0.0
            val_batches = 0

            with torch.no_grad():
                for images, targets in val_loader:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    try:
                        loss_dict = model.train_step(images, targets)
                        loss_tensor = loss_dict["loss_tensor"]
                        val_loss_sum += loss_tensor.item()
                        val_batches += 1
                    except Exception as e:
                        logger.warning(
                            "Exception in validation train_step at epoch %d: %s. Skipping batch.",
                            epoch, e,
                        )
                        continue

            avg_val_loss = val_loss_sum / max(val_batches, 1)

            # --- mAP validation (Group A): compute every eval_interval epochs ---
            current_map = None
            if use_map_selection and (
                (epoch + 1) % eval_interval == 0 or epoch == epochs - 1
            ):
                try:
                    eval_max_batches = (
                        int(eval_subset_batches) if eval_subset_batches else None
                    )
                    map_results = evaluate_map(
                        model, val_loader, class_names_list, input_size, device,
                        confidence_threshold=eval_confidence,
                        max_batches=eval_max_batches,
                    )
                    current_map = map_results["map_50"]
                    logger.info(
                        "Epoch %d: mAP@0.5=%.4f | per-class AP: %s",
                        epoch + 1, current_map,
                        {k: round(v, 3) for k, v in map_results["per_class_ap"].items()},
                    )
                except Exception as e:
                    logger.warning("mAP evaluation failed at epoch %d: %s", epoch + 1, e)

            epoch_time = time.time() - epoch_start
            completed_epochs = epoch + 1

            logger.info(
                "Epoch %d/%d complete | Train Loss: %.4f | Val Loss: %.4f | Time: %.1fs | LR: %.6f",
                epoch + 1, epochs, avg_train_loss, avg_val_loss, epoch_time, current_lr,
            )

            # --- Experiment tracking: log metrics per epoch ---
            epoch_metrics = {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learning_rate": current_lr,
                "epoch_time_s": epoch_time,
            }
            if current_map is not None:
                epoch_metrics["map_50"] = current_map
            try:
                tracker.log_metrics(run_id, step=epoch, metrics=epoch_metrics)
            except Exception as e:
                logger.warning("Failed to log metrics for epoch %d: %s", epoch, e)

            # --- Checkpointing ---
            current_metrics = {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            }
            if current_map is not None:
                current_metrics["map_50"] = current_map

            # Best checkpoint selection.
            # When use_map_selection: select by highest mAP@0.5 (only on epochs
            # where mAP was computed). Otherwise: legacy val_loss selection.
            if use_map_selection:
                if current_map is not None:
                    if current_map > best_map:
                        best_map = current_map
                        best_val_loss = avg_val_loss
                        best_epoch = epoch + 1
                        epochs_without_improvement = 0
                        try:
                            _save_checkpoint(
                                run_checkpoint_dir / "best_model.pt",
                                optimizer_obj=optimizer,
                                epoch_num=epoch,
                                metrics_dict=current_metrics,
                            )
                            logger.info("Saved best model checkpoint (mAP@0.5=%.4f)", current_map)
                        except (IOError, OSError) as e:
                            logger.warning("Failed to save best checkpoint: %s", e)
                    else:
                        # Count improvement gaps only on eval epochs
                        epochs_without_improvement += 1
                # Non-eval epochs: do not change patience counter
            else:
                # Legacy: save when val_loss improves
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch + 1  # 1-indexed
                    epochs_without_improvement = 0
                    try:
                        _save_checkpoint(
                            run_checkpoint_dir / "best_model.pt",
                            optimizer_obj=optimizer,
                            epoch_num=epoch,
                            metrics_dict=current_metrics,
                        )
                        logger.info("Saved best model checkpoint (val_loss=%.4f)", avg_val_loss)
                    except (IOError, OSError) as e:
                        logger.warning("Failed to save best checkpoint: %s", e)
                else:
                    epochs_without_improvement += 1

            # Recovery checkpoint: every 5 epochs (1-indexed, so epoch+1 % 5 == 0)
            if (epoch + 1) % 5 == 0:
                try:
                    _save_checkpoint(
                        run_checkpoint_dir / "recovery.pt",
                        optimizer_obj=optimizer,
                        epoch_num=epoch,
                        metrics_dict=current_metrics,
                    )
                    logger.info("Saved recovery checkpoint at epoch %d", epoch + 1)
                except (IOError, OSError) as e:
                    logger.warning("Failed to save recovery checkpoint: %s", e)

            # Training state (always, for resume): save every epoch (overwrites previous)
            try:
                _save_training_state(
                    run_checkpoint_dir / "training_state.pt",
                    model, optimizer, cosine_scheduler, scaler,
                    epoch, best_val_loss, best_epoch, epochs_without_improvement,
                    run_id, config, best_map=best_map,
                )
            except (IOError, OSError) as e:
                logger.warning("Failed to save training state at epoch %d: %s", epoch + 1, e)

            # --- Early stopping check ---
            if epochs_without_improvement >= early_stopping_patience:
                logger.info(
                    "Early stopping triggered: no improvement for %d epochs. "
                    "Best val_loss=%.4f at epoch %d (patience=%d)",
                    epochs_without_improvement, best_val_loss, best_epoch, early_stopping_patience,
                )
                break

            # --- Check for interruption ---
            if interrupted:
                logger.info("Training interrupted after epoch %d", epoch + 1)
                break

    except KeyboardInterrupt:
        # Second SIGINT caused immediate termination
        logger.warning("Training forcefully interrupted (double SIGINT)")

    finally:
        # --- End experiment tracking (always, even on interrupt/crash) ---
        # This ensures final_results is populated in the run JSON so the
        # dashboard never shows N/A for completed epochs.
        final_metrics = {
            "final_train_loss": avg_train_loss,
            "final_val_loss": avg_val_loss,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "total_epochs": completed_epochs,
            "run_id": run_id,
        }
        try:
            tracker.end_run(run_id, final_metrics)
        except Exception as e:
            logger.warning("Failed to end experiment run: %s", e)

    # --- Final checkpoint ---
    final_metrics_dict = {
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
    }
    try:
        _save_checkpoint(
            run_checkpoint_dir / "final_model.pt",
            optimizer_obj=optimizer,
            epoch_num=completed_epochs - 1 if completed_epochs > 0 else 0,
            metrics_dict=final_metrics_dict,
        )
        logger.info("Saved final model checkpoint")
    except (IOError, OSError) as e:
        logger.error("Failed to save final checkpoint: %s", e)
        signal.signal(signal.SIGINT, original_sigint_handler)
        sys.exit(1)

    # --- Restore original SIGINT handler ---
    signal.signal(signal.SIGINT, original_sigint_handler)

    logger.info("Training complete!")
    return final_metrics


# -------------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train detection models on RDD2022")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume training from a checkpoint. Provide a path to a .pt file or a run ID.",
    )
    args = parser.parse_args()

    train(args.config, verbose=args.verbose, resume_from=args.resume)
