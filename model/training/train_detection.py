"""Real PyTorch training loop for detection models.

This module provides a training function that performs actual gradient-based
training using torchvision detection models. It works with the RDD2022Dataset
and the SSDMobileNetV3 wrapper.

Usage:
    python -m model.training.train_detection --config model/configs/train_ssd_mobilenet.yaml
"""

import argparse
import logging
import math
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.utils.data
from torchvision import transforms as T
from PIL import Image
import numpy as np

from model.config.manager import ConfigManager
from model.training.augmentation import build_augmentation_pipeline
from model.datasets.rdd2022 import RDD2022Dataset
from model.models import ModelRegistry
from model.models.ssd_mobilenet import SSDMobileNetV3
from model.tracking.tracker import ExperimentTracker

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# PyTorch Dataset adapter
# -------------------------------------------------------------------------


class RDD2022TorchDataset(torch.utils.data.Dataset):
    """PyTorch Dataset adapter for RDD2022Dataset.

    Converts the framework's Annotation objects into the format expected
    by torchvision detection models: (image_tensor, target_dict).
    """

    def __init__(self, dataset: RDD2022Dataset, input_size: int = 320, augmentation=None):
        self._annotations = dataset.get_annotations()
        self._input_size = input_size
        self._class_names = dataset.get_class_names()
        self._augmentation = augmentation  # augmentation.Compose pipeline or None
        # Map class names to 1-indexed labels (0 = background in torchvision)
        self._class_to_idx = {
            name: idx + 1 for idx, name in enumerate(self._class_names)
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

    def __len__(self) -> int:
        return len(self._annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        annotation = self._annotations[idx]

        # Load image
        try:
            image = Image.open(annotation.image_path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            # Return a blank image with no targets if file can't be loaded
            logger.warning("Could not load image %s: %s", annotation.image_path, e)
            image = Image.new("RGB", (self._input_size, self._input_size))
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            }
            return self._transform(image), target

        orig_w, orig_h = image.size

        # Apply augmentation if configured (operates on numpy array + normalized bboxes)
        if self._augmentation is not None:
            # Convert PIL to numpy for augmentation
            image_np = np.array(image)
            # Build normalized bbox list: [x_min, y_min, x_max, y_max, class_label]
            aug_bboxes = []
            for bbox in annotation.bounding_boxes:
                aug_bboxes.append([bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max, bbox.class_label])

            # Apply augmentation pipeline
            image_np, aug_bboxes = self._augmentation(image_np, aug_bboxes)

            # Convert back to PIL for torchvision transforms
            image = Image.fromarray(image_np)

            # Convert augmented bboxes to pixel coords at input_size
            image_tensor = self._transform(image)
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
            # No augmentation — original path
            image_tensor = self._transform(image)

            # Convert normalized bounding boxes to pixel coordinates at the resized scale
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
# Training function
# -------------------------------------------------------------------------


def _train_ultralytics(config: dict, model_config: dict, dataset_config: dict,
                       training_config: dict, verbose: bool) -> dict:
    """Run training using Ultralytics native API (YOLO26 with ProgLoss + MuSGD).

    Translates the framework's YAML config into Ultralytics model.train() arguments.

    Args:
        config: Full experiment config dict.
        model_config: model.config section.
        dataset_config: dataset section.
        training_config: training section.
        verbose: Enable verbose output.

    Returns:
        Dict with final training metrics.
    """
    from ultralytics import YOLO

    # Model weights
    model_size = model_config.get("model_size", "m")
    pretrained_weights = model_config.get("pretrained_weights", f"yolo26{model_size}.pt")

    # Dataset - expects a data.yaml path for Ultralytics
    data_path = dataset_config.get("data_yaml", dataset_config.get("path", "") + "/data.yaml")

    # Training hyperparameters → Ultralytics arguments
    epochs = training_config.get("epochs", 100)
    batch_size = training_config.get("batch_size", 16)
    imgsz = model_config.get("input_size", 640)
    lr0 = training_config.get("learning_rate", 0.01)
    weight_decay = training_config.get("weight_decay", 0.0005)
    momentum = training_config.get("momentum", 0.937)
    warmup_epochs = training_config.get("warmup_epochs", 3)
    patience = training_config.get("early_stopping_patience", 50)

    # Output
    output_config = config.get("output", {})
    project = output_config.get("checkpoint_dir", "./checkpoints/yolo26")
    name = config.get("name", "run")

    logger.info("Using Ultralytics native training for YOLO26")
    logger.info("  Model: %s", pretrained_weights)
    logger.info("  Data: %s", data_path)
    logger.info("  Epochs: %d, Batch: %d, ImgSz: %d", epochs, batch_size, imgsz)

    # Load model
    model = YOLO(pretrained_weights)

    # Train using Ultralytics API (includes ProgLoss, MuSGD, STAL)
    results = model.train(
        data=data_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        lr0=lr0,
        weight_decay=weight_decay,
        momentum=momentum,
        warmup_epochs=warmup_epochs,
        patience=patience,
        project=project,
        name=name,
        exist_ok=True,
        verbose=verbose,
    )

    # Extract metrics from results
    final_metrics = {
        "model_type": "yolo26",
        "training_mode": "ultralytics_native",
        "epochs": epochs,
        "project": project,
        "name": name,
    }

    logger.info("Training complete! Results saved to: %s/%s", project, name)
    return final_metrics


def train(config_path: str, verbose: bool = False) -> dict:
    """Run the full training loop.

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

    # --- Ultralytics native training for YOLO26 ---
    if model_type == "yolo26":
        return _train_ultralytics(config, model_config, dataset_config, training_config, verbose)

    # --- Framework training loop (SSD, YOLOv6, etc.) ---

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
    checkpoint_dir = Path(training_config.get("checkpoint_dir", "./checkpoints/ssd_mobilenetv3"))
    log_interval = training_config.get("log_interval", 10)

    # Reproducibility seed
    seed = training_config.get("seed", 42)
    import random as _random
    import numpy as _np
    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d", seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load dataset
    logger.info("Loading dataset from %s", dataset_path)
    rdd_dataset = RDD2022Dataset(country_filter=country_filter)
    rdd_dataset.load(Path(dataset_path))
    logger.info("Dataset loaded: %d images, classes: %s", len(rdd_dataset), rdd_dataset.get_class_names())

    # Split into train/val
    train_ratio = 1.0 - val_split
    train_ds, val_ds, _ = rdd_dataset.split(train_ratio, val_split, 0.0, seed=42)
    logger.info("Train: %d, Val: %d", len(train_ds), len(val_ds))

    # Create PyTorch datasets
    # Build augmentation pipeline from config (only applied to training set)
    aug_config = training_config.get("augmentation", {})
    augmentation_pipeline = build_augmentation_pipeline(aug_config) if aug_config else None
    logger.info("Augmentation pipeline: %s", augmentation_pipeline)

    train_torch = RDD2022TorchDataset(train_ds, input_size=input_size, augmentation=augmentation_pipeline)
    val_torch = RDD2022TorchDataset(val_ds, input_size=input_size)  # No augmentation for validation

    # Update num_classes from actual dataset
    actual_num_classes = train_torch.num_classes
    if actual_num_classes != num_classes:
        logger.warning(
            "Config num_classes=%d but dataset has %d classes. Using %d.",
            num_classes, actual_num_classes, actual_num_classes,
        )
        num_classes = actual_num_classes

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_torch,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for Windows compatibility
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_torch,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Build model
    model_type = config.get("model", {}).get("type", "ssd_mobilenetv3")
    model_cfg = dict(model_config)
    model_cfg["num_classes"] = num_classes
    logger.info("Building model '%s' (num_classes=%d)", model_type, num_classes)
    model = ModelRegistry.create(model_type, model_cfg)

    # Create optimizer
    params = model.get_parameters()
    if optimizer_name.upper() == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer_name.upper() == "ADAM":
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.upper() == "ADAMW":
        optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(
            params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=learning_rate * 0.01
    )

    # Experiment tracking
    output_config = config.get("output", {})
    results_dir = Path(output_config.get("results_dir", "./results/ssd_mobilenetv3"))
    tracker = ExperimentTracker(output_dir=results_dir)
    run_id = tracker.start_run(
        config=config,
        model_name=model_type,
        dataset_name=f"rdd2022_{len(rdd_dataset)}imgs",
    )
    logger.info("Experiment run ID: %s", run_id)

    # Use run_id as the checkpoint subdirectory so each run is isolated
    checkpoint_dir = checkpoint_dir / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Also save experiment results inside the run's checkpoint folder
    import shutil as _shutil
    run_results_path = checkpoint_dir / "experiment.json"

    logger.info("Checkpoints and results will be saved to: %s", checkpoint_dir)
    best_val_loss = float("inf")
    best_epoch = -1
    interrupted = False
    patience = training_config.get("early_stopping_patience", 15)
    epochs_without_improvement = 0

    def handle_sigint(signum, frame):
        nonlocal interrupted
        logger.info("SIGINT received. Finishing current epoch...")
        interrupted = True

    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handle_sigint)

    try:
        for epoch in range(epochs):
            if interrupted:
                break

            epoch_start = time.time()

            # Warmup learning rate
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
                if interrupted:
                    break

                # Skip batches with no valid targets
                valid = [i for i, t in enumerate(targets) if t["boxes"].shape[0] > 0]
                if len(valid) < 2:
                    continue

                images = [images[i].to(device) for i in valid]
                targets = [{k: v.to(device) for k, v in targets[i].items()} for i in valid]

                # Forward + backward
                optimizer.zero_grad()
                loss_dict = model._model(images, targets)
                total_loss = sum(loss for loss in loss_dict.values())

                total_loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.get_parameters(), max_norm=10.0)
                optimizer.step()

                train_loss_sum += total_loss.item()
                train_batches += 1

                if (batch_idx + 1) % log_interval == 0:
                    avg_loss = train_loss_sum / train_batches
                    logger.info(
                        "Epoch %d/%d | Batch %d/%d | Loss: %.4f | LR: %.6f",
                        epoch + 1, epochs, batch_idx + 1, len(train_loader),
                        avg_loss, current_lr,
                    )

            # --- Validation phase ---
            model.set_eval_mode()
            val_loss_sum = 0.0
            val_batches = 0

            with torch.no_grad():
                for images, targets in val_loader:
                    valid = [i for i, t in enumerate(targets) if t["boxes"].shape[0] > 0]
                    # Need at least 2 samples for BatchNorm in train mode
                    if len(valid) < 2:
                        continue

                    images = [images[i].to(device) for i in valid]
                    targets = [{k: v.to(device) for k, v in targets[i].items()} for i in valid]

                    # Set to train mode for loss computation, but freeze BN stats
                    model._model.train()
                    for module in model._model.modules():
                        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
                            module.eval()
                    loss_dict = model._model(images, targets)
                    model._model.eval()

                    val_loss = sum(loss for loss in loss_dict.values())
                    val_loss_sum += val_loss.item()
                    val_batches += 1

            # Compute epoch metrics
            avg_train_loss = train_loss_sum / max(train_batches, 1)
            avg_val_loss = val_loss_sum / max(val_batches, 1)
            epoch_time = time.time() - epoch_start

            # Step scheduler (after warmup)
            if epoch >= warmup_epochs:
                lr_scheduler.step()

            logger.info(
                "Epoch %d/%d complete | Train Loss: %.4f | Val Loss: %.4f | Time: %.1fs | LR: %.6f",
                epoch + 1, epochs, avg_train_loss, avg_val_loss, epoch_time, current_lr,
            )

            # Log metrics
            tracker.log_metrics(run_id, step=epoch, metrics={
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learning_rate": current_lr,
                "epoch_time_s": epoch_time,
            })

            # Save best checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                model.save_checkpoint(
                    checkpoint_dir / "best_model.pt",
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics={"val_loss": avg_val_loss, "train_loss": avg_train_loss},
                )
                logger.info("New best model saved (val_loss=%.4f)", avg_val_loss)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(
                        "Early stopping triggered: no improvement for %d epochs. "
                        "Best val_loss=%.4f at epoch %d.",
                        patience, best_val_loss, best_epoch + 1,
                    )
                    break

            # Save recovery checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                model.save_checkpoint(
                    checkpoint_dir / "recovery.pt",
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics={"val_loss": avg_val_loss, "train_loss": avg_train_loss},
                )

    finally:
        signal.signal(signal.SIGINT, original_handler)

    # Save final checkpoint
    model.save_checkpoint(
        checkpoint_dir / "final_model.pt",
        optimizer=optimizer,
        epoch=epoch if 'epoch' in dir() else 0,
        metrics={"val_loss": avg_val_loss if 'avg_val_loss' in dir() else 0,
                 "train_loss": avg_train_loss if 'avg_train_loss' in dir() else 0},
    )

    # End experiment
    final_metrics = {
        "final_train_loss": avg_train_loss if 'avg_train_loss' in dir() else 0,
        "final_val_loss": avg_val_loss if 'avg_val_loss' in dir() else 0,
        "best_val_loss": best_val_loss if best_val_loss != float("inf") else 0,
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1 if 'epoch' in dir() else 0,
        "run_id": run_id,
    }
    tracker.end_run(run_id, final_metrics)

    # Copy experiment results into the run's checkpoint folder
    tracker_json = results_dir / f"{run_id}.json"
    if tracker_json.exists():
        import shutil
        shutil.copy2(tracker_json, run_results_path)
        logger.info("Experiment results saved to: %s", run_results_path)

    # Update global best model if this run is better than previous runs
    global_dir = checkpoint_dir.parent / "global"
    global_dir.mkdir(parents=True, exist_ok=True)
    global_best_path = global_dir / "best_model.pt"
    global_best_meta_path = global_dir / "best.json"
    current_best = best_val_loss if best_val_loss != float("inf") else float("inf")

    should_update_global = True
    if global_best_meta_path.exists():
        import json as _json
        with open(global_best_meta_path, "r") as f:
            prev_best = _json.load(f)
        if prev_best.get("best_val_loss", float("inf")) <= current_best:
            should_update_global = False
            logger.info(
                "Global best unchanged (previous: %.4f from run %s, current: %.4f)",
                prev_best["best_val_loss"], prev_best["run_id"], current_best,
            )

    if should_update_global and current_best < float("inf"):
        import shutil
        run_best_path = checkpoint_dir / "best_model.pt"
        if run_best_path.exists():
            shutil.copy2(run_best_path, global_best_path)
            import json as _json
            with open(global_best_meta_path, "w") as f:
                _json.dump({
                    "run_id": run_id,
                    "best_val_loss": current_best,
                    "best_epoch": best_epoch,
                    "config": config,
                }, f, indent=2, default=str)
            logger.info(
                "Global best model updated (val_loss=%.4f, run=%s)",
                current_best, run_id,
            )

    logger.info("Training complete!")
    logger.info("  Best val loss: %.4f (epoch %d)", best_val_loss, best_epoch + 1)
    logger.info("  Checkpoints: %s", checkpoint_dir)
    logger.info("  Global best: %s", global_best_path)
    logger.info("  Experiment: %s", run_id)

    return final_metrics


# -------------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train detection models on RDD2022")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    train(args.config, verbose=args.verbose)
