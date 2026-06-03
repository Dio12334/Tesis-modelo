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
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import platform
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
    """

    def __init__(self, dataset: RDD2022Dataset, input_size: int = 320, augmentation=None):
        self._annotations = dataset.get_annotations()
        self._input_size = input_size
        self._class_names = dataset.get_class_names()
        self._augmentation = augmentation  # augmentation.Compose pipeline or None
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
# Unified training function
# -------------------------------------------------------------------------


def train(config_path: str, verbose: bool = False) -> dict:
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
    if hasattr(torch, "compile") and device.type == "cuda":
        try:
            if hasattr(model, "_model") and hasattr(model._model, "model"):
                model._model.model = torch.compile(model._model.model, mode="reduce-overhead")
            elif hasattr(model, "_model"):
                model._model = torch.compile(model._model, mode="reduce-overhead")
            elif hasattr(model, "model"):
                model.model = torch.compile(model.model, mode="reduce-overhead")
            logger.info("Model compiled with torch.compile (mode=reduce-overhead)")
        except Exception as e:
            logger.warning("torch.compile failed, continuing without compilation: %s", e)


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

    # Create PyTorch datasets
    train_torch = RDD2022TorchDataset(train_ds, input_size=input_size, augmentation=augmentation_pipeline)
    val_torch = RDD2022TorchDataset(val_ds, input_size=input_size)  # No augmentation for validation

    # On Windows, DataLoader workers require explicit spawn context
    is_windows = platform.system() == "Windows"
    effective_workers = num_workers
    mp_context = "spawn" if is_windows and effective_workers > 0 else None

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_torch,
        batch_size=batch_size,
        shuffle=True,
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

    # --- Construct optimizer from model.get_parameters() ---
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

    try:
        run_id = tracker.start_run(config, model_type, dataset_name)
    except Exception as e:
        logger.error("Failed to start experiment run: %s", e)
        signal.signal(signal.SIGINT, original_sigint_handler)
        return {}

    # Create checkpoint directory for this run
    run_checkpoint_dir = checkpoint_dir / run_id
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)

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

    # --- Epoch loop ---
    logger.info("Starting training: %d epochs, batch_size=%d, lr=%.6f", epochs, batch_size, learning_rate)

    avg_train_loss = 0.0
    avg_val_loss = float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    completed_epochs = 0

    try:
        for epoch in range(epochs):
            epoch_start = time.time()

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
            try:
                tracker.log_metrics(run_id, step=epoch, metrics=epoch_metrics)
            except Exception as e:
                logger.warning("Failed to log metrics for epoch %d: %s", epoch, e)

            # --- Checkpointing ---
            current_metrics = {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            }

            # Best checkpoint: save when val_loss improves
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

    # --- End experiment tracking ---
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
    args = parser.parse_args()

    train(args.config, verbose=args.verbose)
