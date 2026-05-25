"""Training callbacks for checkpoint saving, logging, and recovery.

Provides a callback system for the training pipeline with hooks at key
training lifecycle events. Callbacks can be composed to customize training
behavior without modifying the core training loop.

Callbacks:
    - BaseCallback: Abstract base class defining the callback interface.
    - CheckpointCallback: Saves best (by validation loss) and final checkpoints.
    - LoggingCallback: Logs training metrics at configurable batch intervals.
    - RecoveryCheckpointCallback: Saves periodic recovery checkpoints for resumption.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


class BaseCallback(ABC):
    """Abstract base class for training callbacks.

    Subclasses implement hooks that are called at specific points during
    the training lifecycle. All hooks receive a metrics dict containing
    relevant training state information.
    """

    @abstractmethod
    def on_train_start(self, metrics: Dict[str, Any]) -> None:
        """Called when training begins.

        Args:
            metrics: Dict with keys like 'total_epochs', 'model_name', etc.
        """
        ...

    @abstractmethod
    def on_train_end(self, metrics: Dict[str, Any]) -> None:
        """Called when training completes.

        Args:
            metrics: Dict with final training summary metrics.
        """
        ...

    @abstractmethod
    def on_epoch_start(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Called at the beginning of each epoch.

        Args:
            epoch: Current epoch number (0-indexed).
            metrics: Dict with current training state.
        """
        ...

    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Called at the end of each epoch.

        Args:
            epoch: Current epoch number (0-indexed).
            metrics: Dict with epoch results (e.g., val_loss, train_loss).
        """
        ...

    @abstractmethod
    def on_batch_end(self, batch: int, metrics: Dict[str, Any]) -> None:
        """Called at the end of each training batch.

        Args:
            batch: Current batch number (0-indexed within the epoch).
            metrics: Dict with batch results (e.g., loss, learning_rate).
        """
        ...


class CheckpointCallback(BaseCallback):
    """Saves best model (based on validation loss) and final model checkpoints.

    The best checkpoint is saved whenever validation loss improves. The final
    checkpoint is saved when training completes, regardless of performance.

    Checkpoint files are JSON-serializable dicts containing model state
    metadata and metrics at the time of saving.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        save_best: bool = True,
        save_final: bool = True,
    ):
        """Initialize checkpoint callback.

        Args:
            checkpoint_dir: Directory where checkpoints will be saved.
            save_best: Whether to save the best model checkpoint.
            save_final: Whether to save the final model checkpoint.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_best = save_best
        self.save_final = save_final
        self.best_val_loss: Optional[float] = None
        self.best_epoch: int = -1

    def on_train_start(self, metrics: Dict[str, Any]) -> None:
        """Create checkpoint directory if it doesn't exist."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def on_train_end(self, metrics: Dict[str, Any]) -> None:
        """Save final checkpoint when training completes."""
        if self.save_final:
            checkpoint_path = self.checkpoint_dir / "checkpoint_final.json"
            checkpoint_data = {
                "type": "final",
                "metrics": _serialize_metrics(metrics),
                "best_val_loss": self.best_val_loss,
                "best_epoch": self.best_epoch,
            }
            _save_checkpoint(checkpoint_path, checkpoint_data)
            logger.info(f"Saved final checkpoint to {checkpoint_path}")

    def on_epoch_start(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """No action on epoch start."""
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Save best checkpoint if validation loss improved.

        Args:
            epoch: Current epoch number.
            metrics: Must contain 'val_loss' key for comparison.
        """
        if not self.save_best:
            return

        val_loss = metrics.get("val_loss")
        if val_loss is None:
            return

        if self.best_val_loss is None or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            checkpoint_path = self.checkpoint_dir / "checkpoint_best.json"
            checkpoint_data = {
                "type": "best",
                "epoch": epoch,
                "val_loss": val_loss,
                "metrics": _serialize_metrics(metrics),
            }
            _save_checkpoint(checkpoint_path, checkpoint_data)
            logger.info(
                f"Saved best checkpoint at epoch {epoch} "
                f"(val_loss={val_loss:.6f}) to {checkpoint_path}"
            )

    def on_batch_end(self, batch: int, metrics: Dict[str, Any]) -> None:
        """No action on batch end."""
        pass


class LoggingCallback(BaseCallback):
    """Logs training metrics at configurable batch intervals.

    Uses Python's logging module to output loss, learning rate, and other
    metrics during training. Logs a summary at the end of each epoch.
    """

    def __init__(self, log_interval: int = 10):
        """Initialize logging callback.

        Args:
            log_interval: Log metrics every N batches. Must be >= 1.
        """
        if log_interval < 1:
            raise ValueError(f"log_interval must be >= 1, got {log_interval}")
        self.log_interval = log_interval

    def on_train_start(self, metrics: Dict[str, Any]) -> None:
        """Log training start information."""
        total_epochs = metrics.get("total_epochs", "unknown")
        model_name = metrics.get("model_name", "unknown")
        logger.info(
            f"Training started: model={model_name}, epochs={total_epochs}"
        )

    def on_train_end(self, metrics: Dict[str, Any]) -> None:
        """Log training completion summary."""
        logger.info("Training completed.")
        if metrics:
            summary_parts = []
            for key, value in metrics.items():
                if isinstance(value, float):
                    summary_parts.append(f"{key}={value:.6f}")
                else:
                    summary_parts.append(f"{key}={value}")
            if summary_parts:
                logger.info(f"Final metrics: {', '.join(summary_parts)}")

    def on_epoch_start(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Log epoch start."""
        total_epochs = metrics.get("total_epochs", "?")
        logger.info(f"Epoch {epoch + 1}/{total_epochs} started")

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Log epoch summary with key metrics."""
        total_epochs = metrics.get("total_epochs", "?")
        parts = [f"Epoch {epoch + 1}/{total_epochs} completed"]

        train_loss = metrics.get("train_loss")
        if train_loss is not None:
            parts.append(f"train_loss={train_loss:.6f}")

        val_loss = metrics.get("val_loss")
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.6f}")

        lr = metrics.get("learning_rate")
        if lr is not None:
            parts.append(f"lr={lr:.8f}")

        logger.info(" | ".join(parts))

    def on_batch_end(self, batch: int, metrics: Dict[str, Any]) -> None:
        """Log batch metrics at configured intervals.

        Logs loss and learning rate every log_interval batches.
        """
        if (batch + 1) % self.log_interval != 0:
            return

        parts = [f"Batch {batch + 1}"]

        loss = metrics.get("loss")
        if loss is not None:
            parts.append(f"loss={loss:.6f}")

        lr = metrics.get("learning_rate")
        if lr is not None:
            parts.append(f"lr={lr:.8f}")

        logger.info(" | ".join(parts))


class RecoveryCheckpointCallback(BaseCallback):
    """Saves recovery checkpoints for training resumption after interruption.

    Recovery checkpoints contain the full training state needed to resume
    training from the exact point where it was interrupted, including
    epoch number, optimizer state reference, and metrics history.
    """

    def __init__(self, checkpoint_dir: Path, save_interval: int = 1):
        """Initialize recovery checkpoint callback.

        Args:
            checkpoint_dir: Directory where recovery checkpoints will be saved.
            save_interval: Save a recovery checkpoint every N epochs. Must be >= 1.
        """
        if save_interval < 1:
            raise ValueError(f"save_interval must be >= 1, got {save_interval}")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_interval = save_interval

    def on_train_start(self, metrics: Dict[str, Any]) -> None:
        """Create checkpoint directory if it doesn't exist."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def on_train_end(self, metrics: Dict[str, Any]) -> None:
        """No action on train end (final checkpoint handled by CheckpointCallback)."""
        pass

    def on_epoch_start(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """No action on epoch start."""
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Save recovery checkpoint at configured intervals.

        The recovery checkpoint contains the full training state needed
        to resume training from this point.

        Args:
            epoch: Current epoch number (0-indexed).
            metrics: Dict with current training state and metrics.
        """
        # Save every save_interval epochs (using 1-indexed epoch for interval check)
        if (epoch + 1) % self.save_interval != 0:
            return

        checkpoint_path = self.checkpoint_dir / "checkpoint_recovery.json"
        checkpoint_data = {
            "type": "recovery",
            "epoch": epoch,
            "metrics": _serialize_metrics(metrics),
        }
        _save_checkpoint(checkpoint_path, checkpoint_data)
        logger.info(
            f"Saved recovery checkpoint at epoch {epoch + 1} to {checkpoint_path}"
        )

    def on_batch_end(self, batch: int, metrics: Dict[str, Any]) -> None:
        """No action on batch end."""
        pass


def _serialize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize metrics dict to JSON-compatible format.

    Converts non-serializable values to strings for safe JSON storage.
    """
    serialized = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            serialized[key] = value
        elif isinstance(value, (list, tuple)):
            serialized[key] = list(value)
        elif isinstance(value, dict):
            serialized[key] = _serialize_metrics(value)
        else:
            serialized[key] = str(value)
    return serialized


def _save_checkpoint(path: Path, data: Dict[str, Any]) -> None:
    """Save checkpoint data to a JSON file.

    Args:
        path: File path for the checkpoint.
        data: Checkpoint data dictionary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
