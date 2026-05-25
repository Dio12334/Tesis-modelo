"""Training pipeline for the Road Damage Evaluation Framework.

Orchestrates model training with configurable hyperparameters, dataset splitting,
optimizer/scheduler setup, validation, checkpointing, and graceful interruption
handling via SIGINT.

The pipeline works with the BaseDetector interface and supports training
resumption from recovery checkpoints.
"""

import json
import logging
import math
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from model.datasets.base import BaseDataset
from model.models.registry import BaseDetector
from model.training.callbacks import (
    BaseCallback,
    CheckpointCallback,
    LoggingCallback,
    RecoveryCheckpointCallback,
)

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates model training with configurable parameters.

    Manages the full training lifecycle including dataset splitting,
    optimizer/scheduler creation, training loop execution, validation,
    checkpointing, and graceful interruption handling.

    Attributes:
        model: The detection model to train.
        dataset: The dataset to train on.
        config: Training configuration dictionary.
        callbacks: List of training callbacks.
    """

    def __init__(self, model: BaseDetector, dataset: BaseDataset, config: dict):
        """Initialize the training pipeline.

        Args:
            model: A BaseDetector instance to train.
            dataset: A BaseDataset instance providing training data.
            config: Training configuration dict. Expected structure:
                training:
                    epochs: int
                    batch_size: int
                    learning_rate: float
                    optimizer: str (SGD, Adam, AdamW)
                    weight_decay: float
                    momentum: float
                    scheduler: str (cosine, step, plateau)
                    warmup_epochs: int
                    val_split: float
                    checkpoint_dir: str
                    log_interval: int
                    augmentation: dict
        """
        self.model = model
        self.dataset = dataset
        self.config = config

        # Extract training config (support both nested and flat)
        self._training_config = config.get("training", config)

        # Training state
        self._current_epoch: int = 0
        self._metrics_history: List[Dict[str, Any]] = []
        self._interrupted: bool = False
        self._original_sigint_handler = None

        # Set up callbacks
        checkpoint_dir = Path(
            self._training_config.get("checkpoint_dir", "./checkpoints")
        )
        log_interval = self._training_config.get("log_interval", 10)

        self.callbacks: List[BaseCallback] = [
            CheckpointCallback(checkpoint_dir=checkpoint_dir),
            LoggingCallback(log_interval=log_interval),
            RecoveryCheckpointCallback(checkpoint_dir=checkpoint_dir),
        ]

    def train(self) -> dict:
        """Execute the training loop.

        Splits the dataset, creates optimizer and scheduler, runs the
        training loop for the configured number of epochs, performs
        validation after each epoch, and invokes callbacks at appropriate
        lifecycle points.

        Returns:
            Dict with final metrics including:
                - final_train_loss: float
                - final_val_loss: float
                - best_val_loss: float
                - best_epoch: int
                - total_epochs_trained: int
                - checkpoint_dir: str
        """
        self._install_signal_handler()

        try:
            return self._run_training()
        finally:
            self._restore_signal_handler()

    def resume(self, checkpoint_path: Path) -> dict:
        """Resume training from a recovery checkpoint.

        Loads the recovery checkpoint, restores epoch counter and metrics
        history, then continues training from where it left off.

        Args:
            checkpoint_path: Path to the recovery checkpoint JSON file.

        Returns:
            Dict with final metrics (same structure as train()).
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_data = self._load_recovery_checkpoint(checkpoint_path)

        # Restore training state
        self._current_epoch = checkpoint_data.get("epoch", 0) + 1
        self._metrics_history = checkpoint_data.get("metrics", {}).get(
            "metrics_history", []
        )
        if not isinstance(self._metrics_history, list):
            self._metrics_history = []

        logger.info(
            f"Resuming training from epoch {self._current_epoch + 1} "
            f"(checkpoint: {checkpoint_path})"
        )

        self._install_signal_handler()

        try:
            return self._run_training()
        finally:
            self._restore_signal_handler()

    def _run_training(self) -> dict:
        """Core training loop implementation.

        Returns:
            Dict with final training metrics.
        """
        # Extract hyperparameters
        epochs = self._training_config.get("epochs", 100)
        batch_size = self._training_config.get("batch_size", 16)
        learning_rate = self._training_config.get("learning_rate", 0.01)
        optimizer_name = self._training_config.get("optimizer", "SGD")
        weight_decay = self._training_config.get("weight_decay", 0.0005)
        momentum = self._training_config.get("momentum", 0.937)
        scheduler_name = self._training_config.get("scheduler", "cosine")
        warmup_epochs = self._training_config.get("warmup_epochs", 3)
        val_split = self._training_config.get("val_split", 0.2)

        # Split dataset into train/val
        train_dataset, val_dataset = self._split_dataset(val_split)

        # Create optimizer and scheduler
        optimizer = self._create_optimizer(
            optimizer_name, learning_rate, weight_decay, momentum
        )
        scheduler = self._create_scheduler(
            scheduler_name, optimizer, epochs, warmup_epochs
        )

        # Compute batch counts
        train_size = len(train_dataset)
        num_batches = max(1, math.ceil(train_size / batch_size))

        # Notify callbacks of training start
        start_metrics = {
            "total_epochs": epochs,
            "model_name": self.model.__class__.__name__,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": optimizer_name,
            "train_size": train_size,
            "val_size": len(val_dataset),
        }
        self._call_callbacks("on_train_start", start_metrics)

        # Training state tracking
        best_val_loss: Optional[float] = None
        best_epoch: int = -1
        final_train_loss: float = 0.0
        final_val_loss: float = 0.0

        # Main training loop
        for epoch in range(self._current_epoch, epochs):
            if self._interrupted:
                self._save_interruption_checkpoint(epoch, best_val_loss, best_epoch)
                break

            self._current_epoch = epoch
            current_lr = self._get_learning_rate(
                optimizer, scheduler, epoch, warmup_epochs, learning_rate
            )

            # Epoch start callback
            epoch_metrics = {
                "total_epochs": epochs,
                "learning_rate": current_lr,
            }
            self._call_callbacks("on_epoch_start", epoch, epoch_metrics)

            # Training phase
            epoch_train_loss = self._train_epoch(
                train_dataset, batch_size, num_batches, optimizer, current_lr
            )

            # Validation phase
            epoch_val_loss = self._validate_epoch(val_dataset, batch_size)

            # Update scheduler
            self._step_scheduler(scheduler, scheduler_name, epoch_val_loss)

            # Track metrics
            final_train_loss = epoch_train_loss
            final_val_loss = epoch_val_loss

            if best_val_loss is None or epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch

            epoch_end_metrics = {
                "total_epochs": epochs,
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
                "learning_rate": current_lr,
                "epoch": epoch,
                "metrics_history": self._metrics_history,
            }
            self._metrics_history.append(
                {
                    "epoch": epoch,
                    "train_loss": epoch_train_loss,
                    "val_loss": epoch_val_loss,
                    "learning_rate": current_lr,
                }
            )

            # Epoch end callback
            self._call_callbacks("on_epoch_end", epoch, epoch_end_metrics)

        # Training complete
        final_metrics = {
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "best_val_loss": best_val_loss if best_val_loss is not None else 0.0,
            "best_epoch": best_epoch,
            "total_epochs_trained": self._current_epoch + 1,
            "checkpoint_dir": str(
                self._training_config.get("checkpoint_dir", "./checkpoints")
            ),
        }

        if not self._interrupted:
            self._call_callbacks("on_train_end", final_metrics)

        return final_metrics

    def _split_dataset(self, val_split: float):
        """Split dataset into training and validation sets.

        Args:
            val_split: Fraction of data to use for validation.

        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        train_ratio = 1.0 - val_split
        val_ratio = val_split
        test_ratio = 0.0

        train_ds, val_ds, _ = self.dataset.split(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=42,
        )
        return train_ds, val_ds

    def _create_optimizer(
        self,
        optimizer_name: str,
        learning_rate: float,
        weight_decay: float,
        momentum: float,
    ) -> Dict[str, Any]:
        """Create an optimizer configuration dict.

        Since torch may not be available, we represent the optimizer as a
        configuration dict that holds the state needed for the training loop.

        Args:
            optimizer_name: One of 'SGD', 'Adam', 'AdamW'.
            learning_rate: Initial learning rate.
            weight_decay: Weight decay coefficient.
            momentum: Momentum factor (for SGD).

        Returns:
            Dict representing the optimizer state.
        """
        optimizer_name_upper = optimizer_name.upper()
        if optimizer_name_upper not in ("SGD", "ADAM", "ADAMW"):
            logger.warning(
                f"Unknown optimizer '{optimizer_name}', defaulting to SGD"
            )
            optimizer_name_upper = "SGD"

        return {
            "type": optimizer_name_upper,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "step": 0,
        }

    def _create_scheduler(
        self,
        scheduler_name: str,
        optimizer: Dict[str, Any],
        total_epochs: int,
        warmup_epochs: int,
    ) -> Dict[str, Any]:
        """Create a scheduler configuration dict.

        Args:
            scheduler_name: One of 'cosine', 'step', 'plateau'.
            optimizer: The optimizer dict.
            total_epochs: Total number of training epochs.
            warmup_epochs: Number of warmup epochs.

        Returns:
            Dict representing the scheduler state.
        """
        scheduler_name_lower = scheduler_name.lower()
        if scheduler_name_lower not in ("cosine", "step", "plateau"):
            logger.warning(
                f"Unknown scheduler '{scheduler_name}', defaulting to cosine"
            )
            scheduler_name_lower = "cosine"

        return {
            "type": scheduler_name_lower,
            "total_epochs": total_epochs,
            "warmup_epochs": warmup_epochs,
            "base_lr": optimizer["learning_rate"],
        }

    def _get_learning_rate(
        self,
        optimizer: Dict[str, Any],
        scheduler: Dict[str, Any],
        epoch: int,
        warmup_epochs: int,
        base_lr: float,
    ) -> float:
        """Compute the current learning rate based on scheduler and warmup.

        During warmup, learning rate linearly increases from 0 to base_lr.
        After warmup, the scheduler determines the learning rate.

        Args:
            optimizer: Optimizer state dict.
            scheduler: Scheduler state dict.
            epoch: Current epoch number.
            warmup_epochs: Number of warmup epochs.
            base_lr: Base learning rate.

        Returns:
            Current learning rate value.
        """
        if epoch < warmup_epochs and warmup_epochs > 0:
            # Linear warmup
            return base_lr * (epoch + 1) / warmup_epochs

        # Post-warmup scheduling
        scheduler_type = scheduler["type"]
        total_epochs = scheduler["total_epochs"]
        effective_epoch = epoch - warmup_epochs
        effective_total = total_epochs - warmup_epochs

        if effective_total <= 0:
            return base_lr

        if scheduler_type == "cosine":
            # Cosine annealing
            return base_lr * 0.5 * (
                1.0 + math.cos(math.pi * effective_epoch / effective_total)
            )
        elif scheduler_type == "step":
            # Step decay: reduce by 0.1 at 1/3 and 2/3 of training
            decay = 1.0
            if effective_epoch >= effective_total * 2 / 3:
                decay = 0.01
            elif effective_epoch >= effective_total / 3:
                decay = 0.1
            return base_lr * decay
        elif scheduler_type == "plateau":
            # Plateau: reduce on plateau (simplified - reduce every 10 epochs
            # without improvement, handled externally)
            return optimizer.get("learning_rate", base_lr)
        else:
            return base_lr

    def _step_scheduler(
        self,
        scheduler: Dict[str, Any],
        scheduler_name: str,
        val_loss: float,
    ) -> None:
        """Step the scheduler after an epoch.

        For plateau scheduler, tracks validation loss for reduction decisions.

        Args:
            scheduler: Scheduler state dict.
            scheduler_name: Name of the scheduler type.
            val_loss: Current epoch's validation loss.
        """
        # For plateau scheduler, track best loss
        if scheduler_name.lower() == "plateau":
            best_loss = scheduler.get("best_loss", float("inf"))
            patience_counter = scheduler.get("patience_counter", 0)
            patience = scheduler.get("patience", 10)

            if val_loss < best_loss:
                scheduler["best_loss"] = val_loss
                scheduler["patience_counter"] = 0
            else:
                scheduler["patience_counter"] = patience_counter + 1
                if scheduler["patience_counter"] >= patience:
                    # Reduce learning rate
                    scheduler["base_lr"] = scheduler["base_lr"] * 0.1
                    scheduler["patience_counter"] = 0
                    logger.info(
                        f"Plateau scheduler: reducing LR to {scheduler['base_lr']}"
                    )

    def _train_epoch(
        self,
        train_dataset: BaseDataset,
        batch_size: int,
        num_batches: int,
        optimizer: Dict[str, Any],
        current_lr: float,
    ) -> float:
        """Execute one training epoch.

        Iterates over the training dataset in batches, simulating the
        forward/backward pass. When torch is available, this would perform
        actual gradient computation and parameter updates.

        Args:
            train_dataset: Training dataset split.
            batch_size: Number of samples per batch.
            num_batches: Total number of batches in the epoch.
            optimizer: Optimizer state dict.
            current_lr: Current learning rate.

        Returns:
            Average training loss for the epoch.
        """
        total_loss = 0.0
        batch_count = 0

        # Simulate training batches
        samples = list(train_dataset)
        for batch_idx in range(num_batches):
            if self._interrupted:
                break

            # Get batch samples
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(samples))
            batch_samples = samples[start_idx:end_idx]

            if not batch_samples:
                continue

            # Simulate forward pass and loss computation
            # In a real implementation, this would:
            # 1. Prepare image tensors from batch_samples
            # 2. Run model.forward(images)
            # 3. Compute detection loss
            # 4. Backward pass and optimizer step
            batch_loss = self._compute_batch_loss(batch_samples, batch_idx, num_batches)
            total_loss += batch_loss
            batch_count += 1

            # Update optimizer step count
            optimizer["step"] = optimizer.get("step", 0) + 1

            # Batch end callback
            batch_metrics = {
                "loss": batch_loss,
                "learning_rate": current_lr,
                "batch_size": len(batch_samples),
            }
            self._call_callbacks("on_batch_end", batch_idx, batch_metrics)

        avg_loss = total_loss / max(batch_count, 1)
        return avg_loss

    def _validate_epoch(self, val_dataset: BaseDataset, batch_size: int) -> float:
        """Execute validation for one epoch.

        Iterates over the validation dataset and computes validation loss
        without gradient updates.

        Args:
            val_dataset: Validation dataset split.
            batch_size: Number of samples per batch.

        Returns:
            Average validation loss for the epoch.
        """
        if len(val_dataset) == 0:
            return 0.0

        total_loss = 0.0
        batch_count = 0
        samples = list(val_dataset)
        num_batches = max(1, math.ceil(len(samples) / batch_size))

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(samples))
            batch_samples = samples[start_idx:end_idx]

            if not batch_samples:
                continue

            # Simulate validation loss computation
            batch_loss = self._compute_batch_loss(
                batch_samples, batch_idx, num_batches, is_validation=True
            )
            total_loss += batch_loss
            batch_count += 1

        avg_loss = total_loss / max(batch_count, 1)
        return avg_loss

    def _compute_batch_loss(
        self,
        batch_samples: list,
        batch_idx: int,
        total_batches: int,
        is_validation: bool = False,
    ) -> float:
        """Compute loss for a batch of samples.

        This is a simulation that produces decreasing loss values over time.
        In a real implementation with torch available, this would perform
        actual model forward pass and loss computation.

        Args:
            batch_samples: List of Annotation samples in the batch.
            batch_idx: Current batch index.
            total_batches: Total number of batches.
            is_validation: Whether this is a validation batch.

        Returns:
            Computed loss value for the batch.
        """
        # Simulate a decreasing loss curve
        # Base loss decreases with epoch progress
        epoch_progress = (self._current_epoch + batch_idx / max(total_batches, 1))
        total_epochs = self._training_config.get("epochs", 100)

        # Exponential decay with some noise
        progress_ratio = epoch_progress / max(total_epochs, 1)
        base_loss = 2.0 * math.exp(-3.0 * progress_ratio) + 0.1

        # Validation loss is slightly higher
        if is_validation:
            base_loss *= 1.1

        return base_loss

    def _save_interruption_checkpoint(
        self, epoch: int, best_val_loss: Optional[float], best_epoch: int
    ) -> None:
        """Save a recovery checkpoint when training is interrupted.

        Args:
            epoch: The epoch at which interruption occurred.
            best_val_loss: Best validation loss seen so far.
            best_epoch: Epoch with the best validation loss.
        """
        checkpoint_dir = Path(
            self._training_config.get("checkpoint_dir", "./checkpoints")
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "checkpoint_recovery.json"

        checkpoint_data = {
            "type": "recovery",
            "epoch": epoch,
            "metrics": {
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "metrics_history": self._metrics_history,
            },
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(
            f"Training interrupted at epoch {epoch + 1}. "
            f"Recovery checkpoint saved to {checkpoint_path}"
        )

    def _load_recovery_checkpoint(self, checkpoint_path: Path) -> dict:
        """Load a recovery checkpoint from disk.

        Args:
            checkpoint_path: Path to the recovery checkpoint JSON file.

        Returns:
            Checkpoint data dictionary.

        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist.
            json.JSONDecodeError: If the checkpoint file is malformed.
        """
        with open(checkpoint_path, "r") as f:
            return json.load(f)

    def _install_signal_handler(self) -> None:
        """Install SIGINT handler for graceful interruption.

        Saves the original handler so it can be restored after training.
        """
        self._interrupted = False
        self._original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _restore_signal_handler(self) -> None:
        """Restore the original SIGINT handler."""
        if self._original_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._original_sigint_handler)
            self._original_sigint_handler = None

    def _handle_sigint(self, signum, frame) -> None:
        """Handle SIGINT by setting the interrupted flag.

        The training loop checks this flag and saves a recovery checkpoint
        before exiting gracefully.
        """
        logger.info("SIGINT received. Finishing current batch and saving checkpoint...")
        self._interrupted = True

    def _call_callbacks(self, method_name: str, *args) -> None:
        """Invoke a callback method on all registered callbacks.

        Args:
            method_name: Name of the callback method to invoke.
            *args: Arguments to pass to the callback method.
        """
        for callback in self.callbacks:
            try:
                method = getattr(callback, method_name)
                method(*args)
            except Exception as e:
                logger.warning(
                    f"Callback {callback.__class__.__name__}.{method_name} "
                    f"raised an exception: {e}"
                )
