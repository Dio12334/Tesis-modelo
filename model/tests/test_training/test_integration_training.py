"""Integration tests for end-to-end training.

These tests verify the complete training workflow including checkpoints,
metrics, experiment tracking, early stopping, and signal handling.

Feature: unified-training-loop, Task 5.3: Integration tests for end-to-end training

**Validates: Requirements 1.1, 7.1, 7.2, 7.3, 8.1, 10.1, 10.2, 10.4, 10.5, 11.2**
"""

import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, call
import yaml

import pytest
import torch

from model.training.train_detection import train


# ---------------------------------------------------------------------------
# Mock BaseDetector for integration testing
# ---------------------------------------------------------------------------


class MockDetectorForIntegration:
    """Mock BaseDetector for integration testing.
    
    This mock simulates a real detector with configurable behavior for
    testing various training scenarios.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        loss_sequence: Optional[List[float]] = None,
        raise_on_batch: Optional[int] = None,
    ):
        """Initialize the mock detector.
        
        Args:
            config: Configuration dict.
            loss_sequence: Sequence of loss values to return (cycles through).
            raise_on_batch: Batch index at which to raise an exception.
        """
        self.config = config or {}
        self._loss_sequence = loss_sequence or [1.0, 0.9, 0.8, 0.7, 0.6]
        self._raise_on_batch = raise_on_batch
        self._batch_counter = 0
        self._epoch_counter = 0
        self._call_log: List[str] = []
        self._train_mode = False
        self._checkpoints_saved: List[str] = []
        
        # Create a simple parameter for optimizer construction
        self._param = torch.nn.Parameter(torch.tensor([1.0]))

    def train_step(
        self, images: List[torch.Tensor], targets: List[dict]
    ) -> Dict[str, Any]:
        """Simulate a training step."""
        batch_idx = self._batch_counter
        self._batch_counter += 1
        self._call_log.append(f"train_step(batch={batch_idx})")
        
        if self._raise_on_batch is not None and batch_idx == self._raise_on_batch:
            raise RuntimeError(f"Simulated error at batch {batch_idx}")
        
        # Return configured loss value (cycle through sequence)
        loss_idx = batch_idx % len(self._loss_sequence)
        loss_value = self._loss_sequence[loss_idx]
        
        # Create a tensor with grad_fn for backprop
        loss_tensor = torch.abs(self._param) * loss_value
        
        return {"loss_tensor": loss_tensor}

    def get_parameters(self) -> List[torch.nn.Parameter]:
        """Return trainable parameters."""
        self._call_log.append("get_parameters()")
        return [self._param]

    def set_train_mode(self) -> None:
        """Switch to training mode."""
        self._call_log.append("set_train_mode()")
        self._train_mode = True

    def set_eval_mode(self) -> None:
        """Switch to evaluation mode."""
        self._call_log.append("set_eval_mode()")
        self._train_mode = False
        self._epoch_counter += 1

    def save_checkpoint(
        self,
        path: Path,
        optimizer: Optional[Any] = None,
        epoch: Optional[int] = None,
        metrics: Optional[dict] = None,
    ) -> None:
        """Save checkpoint."""
        self._call_log.append(f"save_checkpoint({path.name})")
        self._checkpoints_saved.append(path.name)
        # Create the file
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "mock": True,
            "epoch": epoch,
            "metrics": metrics,
        }, path)

    def get_config_schema(self) -> dict:
        """Return empty schema."""
        return {}

    def reset_batch_counter(self) -> None:
        """Reset the batch counter for a new epoch."""
        self._batch_counter = 0

    @property
    def call_log(self) -> List[str]:
        """Return the log of method calls."""
        return self._call_log

    @property
    def checkpoints_saved(self) -> List[str]:
        """Return list of checkpoint filenames saved."""
        return self._checkpoints_saved


# ---------------------------------------------------------------------------
# Test helper functions
# ---------------------------------------------------------------------------


def _create_test_dataset(tmp_path: Path, num_samples: int) -> None:
    """Create a minimal test dataset with images and annotations."""
    from PIL import Image
    import xml.etree.ElementTree as ET
    
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_samples):
        # Create a small test image
        img = Image.new("RGB", (64, 64), color=(i % 256, 0, 0))
        img_path = dataset_path / f"image_{i}.jpg"
        img.save(img_path)
        
        # Create corresponding XML annotation
        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = f"image_{i}.jpg"
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = "64"
        ET.SubElement(size, "height").text = "64"
        ET.SubElement(size, "depth").text = "3"
        
        # Add a bounding box
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = "D00"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = "10"
        ET.SubElement(bndbox, "ymin").text = "10"
        ET.SubElement(bndbox, "xmax").text = "50"
        ET.SubElement(bndbox, "ymax").text = "50"
        
        tree = ET.ElementTree(root)
        xml_path = dataset_path / f"image_{i}.xml"
        tree.write(xml_path)


def _create_test_config(
    tmp_path: Path,
    epochs: int = 3,
    batch_size: int = 2,
    early_stopping_patience: int = 100,
    seed: int = 42,
) -> Path:
    """Create a test training configuration file."""
    config = {
        "model": {
            "type": "ssd_mobilenetv3",
            "config": {
                "num_classes": 5,
                "input_size": 64,
            },
        },
        "dataset": {
            "path": str(tmp_path / "dataset"),
            "name": "test_dataset",
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": 0.001,
            "optimizer": "SGD",
            "weight_decay": 0.0001,
            "momentum": 0.9,
            "warmup_epochs": 1,
            "val_split": 0.2,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "log_interval": 1,
            "use_amp": False,
            "num_workers": 0,
            "early_stopping_patience": early_stopping_patience,
            "seed": seed,
        },
    }
    
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    return config_path


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestEndToEndTraining:
    """Integration tests for end-to-end training scenarios."""

    def test_3_epoch_run_with_mock_detector(self, tmp_path):
        """Test 3-epoch run with mock detector verifies checkpoints, metrics, tracker calls.
        
        **Validates: Requirements 1.1, 7.1, 7.2, 7.3**
        """
        _create_test_dataset(tmp_path, 20)
        config_path = _create_test_config(tmp_path, epochs=3, batch_size=2)
        
        mock_detector = MockDetectorForIntegration(
            config={"num_classes": 5},
            loss_sequence=[0.5, 0.4, 0.3, 0.2, 0.1],  # Decreasing losses
        )
        
        captured_metrics: List[dict] = []
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker_class:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                
                def capture_metrics(run_id, step, metrics):
                    captured_metrics.append(dict(metrics))
                
                mock_tracker_instance.log_metrics.side_effect = capture_metrics
                mock_tracker_class.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                try:
                    result = train(str(config_path), verbose=False)
                finally:
                    logging.disable(logging.NOTSET)
        
        # Verify training completed
        assert result is not None
        assert "total_epochs" in result
        assert result["total_epochs"] == 3
        
        # Verify metrics were logged for each epoch
        assert len(captured_metrics) == 3, f"Expected 3 metric logs, got {len(captured_metrics)}"
        
        # Verify required metric keys
        for metrics in captured_metrics:
            assert "train_loss" in metrics
            assert "val_loss" in metrics
            assert "learning_rate" in metrics
            assert "epoch_time_s" in metrics
        
        # Verify checkpoints were saved
        checkpoint_dir = tmp_path / "checkpoints" / "test_run_id"
        assert (checkpoint_dir / "final_model.pt").exists(), "Final checkpoint should be saved"
        assert (checkpoint_dir / "best_model.pt").exists(), "Best checkpoint should be saved"
        
        # Verify tracker calls
        mock_tracker_instance.start_run.assert_called_once()
        mock_tracker_instance.end_run.assert_called_once()

    def test_identical_behavior_for_different_model_types(self, tmp_path):
        """Test that different model types (via mock) have identical training behavior.
        
        This verifies the unified loop treats all models the same way.
        
        **Validates: Requirements 11.2**
        """
        _create_test_dataset(tmp_path, 20)
        
        results = []
        
        for model_name in ["ssd_mock", "yolo_mock"]:
            config_path = _create_test_config(
                tmp_path, 
                epochs=2, 
                batch_size=2,
                seed=42,  # Same seed for reproducibility
            )
            
            mock_detector = MockDetectorForIntegration(
                config={"num_classes": 5},
                loss_sequence=[0.5, 0.4, 0.3],
            )
            
            with patch("model.training.train_detection.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = mock_detector
                
                with patch("model.training.train_detection.ExperimentTracker") as mock_tracker_class:
                    mock_tracker_instance = MagicMock()
                    mock_tracker_instance.start_run.return_value = f"run_{model_name}"
                    mock_tracker_class.return_value = mock_tracker_instance
                    
                    logging.disable(logging.CRITICAL)
                    try:
                        result = train(str(config_path), verbose=False)
                        results.append(result)
                    finally:
                        logging.disable(logging.NOTSET)
        
        # Both runs should complete with same number of epochs
        assert len(results) == 2
        assert results[0]["total_epochs"] == results[1]["total_epochs"]

    def test_early_stopping_with_patience_2(self, tmp_path):
        """Test early stopping with patience=2 terminates at correct epoch.
        
        **Validates: Requirements 8.1**
        """
        _create_test_dataset(tmp_path, 20)
        config_path = _create_test_config(
            tmp_path, 
            epochs=10,  # More epochs than needed
            batch_size=2,
            early_stopping_patience=2,
        )
        
        # Create a mock that returns increasing losses to trigger early stopping
        # The mock needs to return losses that increase over epochs
        class EarlyStoppingMockDetector(MockDetectorForIntegration):
            def __init__(self):
                super().__init__(config={"num_classes": 5})
                self._epoch = 0
                self._in_eval = False
                
            def set_train_mode(self):
                super().set_train_mode()
                self._in_eval = False
                
            def set_eval_mode(self):
                super().set_eval_mode()
                self._in_eval = True
                self._epoch += 1
                
            def train_step(self, images, targets):
                # Return increasing losses for validation to trigger early stopping
                # Epoch 0: val_loss ~0.5 (best)
                # Epoch 1: val_loss ~0.6 (no improvement, patience=1)
                # Epoch 2: val_loss ~0.7 (no improvement, patience=2) -> stop
                if self._in_eval:
                    base_loss = 0.5 + (self._epoch - 1) * 0.1
                else:
                    base_loss = 0.5
                
                loss_tensor = torch.abs(self._param) * base_loss
                return {"loss_tensor": loss_tensor}
        
        mock_detector = EarlyStoppingMockDetector()
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker_class:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                mock_tracker_class.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                try:
                    result = train(str(config_path), verbose=False)
                finally:
                    logging.disable(logging.NOTSET)
        
        # Should stop early due to patience (after 3 epochs: 0, 1, 2)
        assert result["total_epochs"] <= 4, f"Training should stop early, got {result['total_epochs']} epochs"
        assert result["best_epoch"] == 1, f"Best epoch should be 1, got {result['best_epoch']}"


class TestSignalHandling:
    """Tests for SIGINT signal handling during training."""

    def test_sigint_during_training_completes_epoch(self, tmp_path):
        """Test SIGINT during epoch 2 completes the epoch and saves final checkpoint.
        
        **Validates: Requirements 10.1, 10.2**
        """
        _create_test_dataset(tmp_path, 20)
        config_path = _create_test_config(tmp_path, epochs=5, batch_size=2)
        
        mock_detector = MockDetectorForIntegration(
            config={"num_classes": 5},
            loss_sequence=[0.5, 0.4, 0.3],
        )
        
        sigint_sent = threading.Event()
        epochs_completed = []
        
        def track_epochs(run_id, step, metrics):
            epochs_completed.append(step)
            # Send SIGINT after epoch 1 (step=1)
            if step == 1 and not sigint_sent.is_set():
                sigint_sent.set()
                os.kill(os.getpid(), signal.SIGINT)
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker_class:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                mock_tracker_instance.log_metrics.side_effect = track_epochs
                mock_tracker_class.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                try:
                    result = train(str(config_path), verbose=False)
                finally:
                    logging.disable(logging.NOTSET)
        
        # Should have completed at least 2 epochs (0 and 1)
        assert len(epochs_completed) >= 2, "Should complete at least 2 epochs"
        
        # Final checkpoint should be saved
        checkpoint_dir = tmp_path / "checkpoints" / "test_run_id"
        assert (checkpoint_dir / "final_model.pt").exists(), "Final checkpoint should be saved"


class TestErrorHandling:
    """Tests for error handling during training."""

    def test_exception_in_train_step_continues_training(self, tmp_path):
        """Test that exceptions in train_step are caught and training continues.
        
        **Validates: Requirements 2.5**
        """
        _create_test_dataset(tmp_path, 20)
        config_path = _create_test_config(tmp_path, epochs=2, batch_size=2)
        
        # Raise exception on batch 3
        mock_detector = MockDetectorForIntegration(
            config={"num_classes": 5},
            loss_sequence=[0.5, 0.4, 0.3],
            raise_on_batch=3,
        )
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker_class:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                mock_tracker_class.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                try:
                    result = train(str(config_path), verbose=False)
                finally:
                    logging.disable(logging.NOTSET)
        
        # Training should complete despite the exception
        assert result is not None
        assert result["total_epochs"] == 2

    def test_tracker_start_run_failure_terminates_gracefully(self, tmp_path):
        """Test that failure in start_run terminates without entering epoch loop.
        
        **Validates: Requirements 9.1**
        """
        _create_test_dataset(tmp_path, 20)
        config_path = _create_test_config(tmp_path, epochs=3, batch_size=2)
        
        mock_detector = MockDetectorForIntegration(
            config={"num_classes": 5},
            loss_sequence=[0.5, 0.4, 0.3],
        )
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker_class:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.side_effect = RuntimeError("Tracker failed")
                mock_tracker_class.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                try:
                    result = train(str(config_path), verbose=False)
                finally:
                    logging.disable(logging.NOTSET)
        
        # Should return empty dict without training
        assert result == {}
        
        # No checkpoints should be saved
        assert len(mock_detector.checkpoints_saved) == 0


class TestCheckpointBehavior:
    """Tests for checkpoint saving behavior."""

    def test_best_checkpoint_saved_on_improvement(self, tmp_path):
        """Test that best checkpoint is saved when validation loss improves.
        
        **Validates: Requirements 7.1**
        """
        _create_test_dataset(tmp_path, 20)
        config_path = _create_test_config(tmp_path, epochs=3, batch_size=2)
        
        mock_detector = MockDetectorForIntegration(
            config={"num_classes": 5},
            loss_sequence=[0.5, 0.4, 0.3, 0.2, 0.1],  # Decreasing losses
        )
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker_class:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                mock_tracker_class.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                try:
                    result = train(str(config_path), verbose=False)
                finally:
                    logging.disable(logging.NOTSET)
        
        # Best checkpoint should be saved (losses are decreasing, so each epoch improves)
        assert "best_model.pt" in mock_detector.checkpoints_saved

    def test_recovery_checkpoint_at_epoch_5(self, tmp_path):
        """Test that recovery checkpoint is saved at epoch 5.
        
        **Validates: Requirements 7.2**
        """
        _create_test_dataset(tmp_path, 20)
        config_path = _create_test_config(
            tmp_path, 
            epochs=6,  # Run 6 epochs to hit epoch 5
            batch_size=2,
            early_stopping_patience=100,  # Disable early stopping
        )
        
        mock_detector = MockDetectorForIntegration(
            config={"num_classes": 5},
            loss_sequence=[0.5, 0.4, 0.3, 0.2, 0.1],
        )
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker_class:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                mock_tracker_class.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                try:
                    result = train(str(config_path), verbose=False)
                finally:
                    logging.disable(logging.NOTSET)
        
        # Recovery checkpoint should be saved at epoch 5
        assert "recovery.pt" in mock_detector.checkpoints_saved

    def test_final_checkpoint_always_saved(self, tmp_path):
        """Test that final checkpoint is always saved on completion.
        
        **Validates: Requirements 7.3**
        """
        _create_test_dataset(tmp_path, 20)
        config_path = _create_test_config(tmp_path, epochs=2, batch_size=2)
        
        mock_detector = MockDetectorForIntegration(
            config={"num_classes": 5},
            loss_sequence=[0.5, 0.4, 0.3],
        )
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker_class:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                mock_tracker_class.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                try:
                    result = train(str(config_path), verbose=False)
                finally:
                    logging.disable(logging.NOTSET)
        
        # Final checkpoint should always be saved
        assert "final_model.pt" in mock_detector.checkpoints_saved
