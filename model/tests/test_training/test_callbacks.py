"""Unit tests for training callbacks."""

import json
import logging
from pathlib import Path

import pytest

from model.training.callbacks import (
    BaseCallback,
    CheckpointCallback,
    LoggingCallback,
    RecoveryCheckpointCallback,
)


class TestBaseCallback:
    """Tests for the BaseCallback abstract class."""

    def test_cannot_instantiate_directly(self):
        """BaseCallback cannot be instantiated without implementing all hooks."""
        with pytest.raises(TypeError):
            BaseCallback()

    def test_concrete_subclass_can_be_instantiated(self):
        """A concrete subclass implementing all hooks can be instantiated."""

        class ConcreteCallback(BaseCallback):
            def on_train_start(self, metrics):
                pass

            def on_train_end(self, metrics):
                pass

            def on_epoch_start(self, epoch, metrics):
                pass

            def on_epoch_end(self, epoch, metrics):
                pass

            def on_batch_end(self, batch, metrics):
                pass

        cb = ConcreteCallback()
        assert cb is not None


class TestCheckpointCallback:
    """Tests for the CheckpointCallback."""

    def test_creates_checkpoint_dir_on_train_start(self, tmp_path):
        """Checkpoint directory is created when training starts."""
        checkpoint_dir = tmp_path / "checkpoints" / "nested"
        cb = CheckpointCallback(checkpoint_dir=checkpoint_dir)
        cb.on_train_start({})
        assert checkpoint_dir.exists()

    def test_saves_best_checkpoint_on_improved_val_loss(self, tmp_path):
        """Best checkpoint is saved when val_loss improves."""
        cb = CheckpointCallback(checkpoint_dir=tmp_path)
        cb.on_train_start({})

        cb.on_epoch_end(0, {"val_loss": 0.5, "train_loss": 0.6})
        best_path = tmp_path / "checkpoint_best.json"
        assert best_path.exists()

        data = json.loads(best_path.read_text())
        assert data["type"] == "best"
        assert data["epoch"] == 0
        assert data["val_loss"] == 0.5

    def test_updates_best_checkpoint_on_further_improvement(self, tmp_path):
        """Best checkpoint is updated when val_loss improves further."""
        cb = CheckpointCallback(checkpoint_dir=tmp_path)
        cb.on_train_start({})

        cb.on_epoch_end(0, {"val_loss": 0.5})
        cb.on_epoch_end(1, {"val_loss": 0.3})

        data = json.loads((tmp_path / "checkpoint_best.json").read_text())
        assert data["epoch"] == 1
        assert data["val_loss"] == 0.3

    def test_does_not_save_best_when_val_loss_worsens(self, tmp_path):
        """Best checkpoint is not overwritten when val_loss worsens."""
        cb = CheckpointCallback(checkpoint_dir=tmp_path)
        cb.on_train_start({})

        cb.on_epoch_end(0, {"val_loss": 0.3})
        cb.on_epoch_end(1, {"val_loss": 0.5})

        data = json.loads((tmp_path / "checkpoint_best.json").read_text())
        assert data["epoch"] == 0
        assert data["val_loss"] == 0.3

    def test_does_not_save_best_when_no_val_loss(self, tmp_path):
        """No best checkpoint is saved if metrics lack val_loss."""
        cb = CheckpointCallback(checkpoint_dir=tmp_path)
        cb.on_train_start({})

        cb.on_epoch_end(0, {"train_loss": 0.5})
        assert not (tmp_path / "checkpoint_best.json").exists()

    def test_saves_final_checkpoint_on_train_end(self, tmp_path):
        """Final checkpoint is saved when training ends."""
        cb = CheckpointCallback(checkpoint_dir=tmp_path)
        cb.on_train_start({})

        cb.on_epoch_end(0, {"val_loss": 0.4})
        cb.on_train_end({"final_loss": 0.35})

        final_path = tmp_path / "checkpoint_final.json"
        assert final_path.exists()

        data = json.loads(final_path.read_text())
        assert data["type"] == "final"
        assert data["best_val_loss"] == 0.4
        assert data["best_epoch"] == 0

    def test_save_best_disabled(self, tmp_path):
        """No best checkpoint is saved when save_best=False."""
        cb = CheckpointCallback(checkpoint_dir=tmp_path, save_best=False)
        cb.on_train_start({})

        cb.on_epoch_end(0, {"val_loss": 0.3})
        assert not (tmp_path / "checkpoint_best.json").exists()

    def test_save_final_disabled(self, tmp_path):
        """No final checkpoint is saved when save_final=False."""
        cb = CheckpointCallback(checkpoint_dir=tmp_path, save_final=False)
        cb.on_train_start({})
        cb.on_train_end({"loss": 0.3})
        assert not (tmp_path / "checkpoint_final.json").exists()


class TestLoggingCallback:
    """Tests for the LoggingCallback."""

    def test_invalid_log_interval_raises(self):
        """log_interval < 1 raises ValueError."""
        with pytest.raises(ValueError, match="log_interval must be >= 1"):
            LoggingCallback(log_interval=0)

    def test_logs_on_train_start(self, caplog):
        """Training start is logged."""
        cb = LoggingCallback(log_interval=5)
        with caplog.at_level(logging.INFO, logger="model.training.callbacks"):
            cb.on_train_start({"total_epochs": 10, "model_name": "yolov6"})
        assert "Training started" in caplog.text
        assert "yolov6" in caplog.text

    def test_logs_on_train_end(self, caplog):
        """Training completion is logged."""
        cb = LoggingCallback(log_interval=5)
        with caplog.at_level(logging.INFO, logger="model.training.callbacks"):
            cb.on_train_end({"final_loss": 0.25})
        assert "Training completed" in caplog.text

    def test_logs_epoch_summary(self, caplog):
        """Epoch summary is logged with train_loss, val_loss, and lr."""
        cb = LoggingCallback(log_interval=5)
        with caplog.at_level(logging.INFO, logger="model.training.callbacks"):
            cb.on_epoch_end(
                2, {"total_epochs": 10, "train_loss": 0.4, "val_loss": 0.35, "learning_rate": 0.001}
            )
        assert "Epoch 3/10" in caplog.text
        assert "train_loss" in caplog.text
        assert "val_loss" in caplog.text

    def test_logs_batch_at_interval(self, caplog):
        """Batch metrics are logged at the configured interval."""
        cb = LoggingCallback(log_interval=5)
        with caplog.at_level(logging.INFO, logger="model.training.callbacks"):
            # Batch 4 (0-indexed) is the 5th batch, should log
            cb.on_batch_end(4, {"loss": 0.5, "learning_rate": 0.01})
        assert "Batch 5" in caplog.text
        assert "loss=" in caplog.text

    def test_does_not_log_batch_between_intervals(self, caplog):
        """Batch metrics are not logged between intervals."""
        cb = LoggingCallback(log_interval=5)
        with caplog.at_level(logging.INFO, logger="model.training.callbacks"):
            cb.on_batch_end(2, {"loss": 0.5, "learning_rate": 0.01})
        assert "Batch" not in caplog.text

    def test_logs_every_batch_when_interval_is_1(self, caplog):
        """Every batch is logged when log_interval=1."""
        cb = LoggingCallback(log_interval=1)
        with caplog.at_level(logging.INFO, logger="model.training.callbacks"):
            cb.on_batch_end(0, {"loss": 0.5})
            cb.on_batch_end(1, {"loss": 0.4})
        assert "Batch 1" in caplog.text
        assert "Batch 2" in caplog.text


class TestRecoveryCheckpointCallback:
    """Tests for the RecoveryCheckpointCallback."""

    def test_invalid_save_interval_raises(self):
        """save_interval < 1 raises ValueError."""
        with pytest.raises(ValueError, match="save_interval must be >= 1"):
            RecoveryCheckpointCallback(checkpoint_dir=Path("/tmp"), save_interval=0)

    def test_creates_checkpoint_dir_on_train_start(self, tmp_path):
        """Checkpoint directory is created when training starts."""
        checkpoint_dir = tmp_path / "recovery"
        cb = RecoveryCheckpointCallback(checkpoint_dir=checkpoint_dir)
        cb.on_train_start({})
        assert checkpoint_dir.exists()

    def test_saves_recovery_checkpoint_every_epoch_by_default(self, tmp_path):
        """Recovery checkpoint is saved every epoch with default interval."""
        cb = RecoveryCheckpointCallback(checkpoint_dir=tmp_path, save_interval=1)
        cb.on_train_start({})

        cb.on_epoch_end(0, {"train_loss": 0.5, "val_loss": 0.4})
        recovery_path = tmp_path / "checkpoint_recovery.json"
        assert recovery_path.exists()

        data = json.loads(recovery_path.read_text())
        assert data["type"] == "recovery"
        assert data["epoch"] == 0

    def test_saves_recovery_at_configured_interval(self, tmp_path):
        """Recovery checkpoint is saved only at the configured interval."""
        cb = RecoveryCheckpointCallback(checkpoint_dir=tmp_path, save_interval=3)
        cb.on_train_start({})

        # Epochs 0, 1 should not trigger save (1st, 2nd epoch)
        cb.on_epoch_end(0, {"train_loss": 0.5})
        assert not (tmp_path / "checkpoint_recovery.json").exists()

        cb.on_epoch_end(1, {"train_loss": 0.4})
        assert not (tmp_path / "checkpoint_recovery.json").exists()

        # Epoch 2 is the 3rd epoch, should trigger save
        cb.on_epoch_end(2, {"train_loss": 0.3})
        assert (tmp_path / "checkpoint_recovery.json").exists()

        data = json.loads((tmp_path / "checkpoint_recovery.json").read_text())
        assert data["epoch"] == 2

    def test_recovery_checkpoint_overwrites_previous(self, tmp_path):
        """Each recovery checkpoint overwrites the previous one."""
        cb = RecoveryCheckpointCallback(checkpoint_dir=tmp_path, save_interval=1)
        cb.on_train_start({})

        cb.on_epoch_end(0, {"train_loss": 0.5})
        cb.on_epoch_end(1, {"train_loss": 0.4})

        data = json.loads((tmp_path / "checkpoint_recovery.json").read_text())
        assert data["epoch"] == 1

    def test_recovery_checkpoint_contains_full_state(self, tmp_path):
        """Recovery checkpoint contains metrics for resumption."""
        cb = RecoveryCheckpointCallback(checkpoint_dir=tmp_path, save_interval=1)
        cb.on_train_start({})

        metrics = {
            "train_loss": 0.35,
            "val_loss": 0.30,
            "learning_rate": 0.001,
            "total_epochs": 100,
        }
        cb.on_epoch_end(5, metrics)

        data = json.loads((tmp_path / "checkpoint_recovery.json").read_text())
        assert data["epoch"] == 5
        assert data["metrics"]["train_loss"] == 0.35
        assert data["metrics"]["val_loss"] == 0.30
        assert data["metrics"]["learning_rate"] == 0.001
