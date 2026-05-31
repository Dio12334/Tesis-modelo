"""Property-based tests for metrics logging per epoch.

Tests Property 13 from the design document: Metrics logged per epoch with required keys.

For any training run of N completed epochs, `ExperimentTracker.log_metrics()` SHALL be
called exactly N times, and each call's metrics dict SHALL contain at minimum the keys
`train_loss`, `val_loss`, `learning_rate`, and `epoch_time_s`.

Feature: unified-training-loop, Property 13: Metrics logged per epoch with required keys

**Validates: Requirements 9.2**
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock, patch, call

import pytest
import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from model.training.train_detection import train


# ---------------------------------------------------------------------------
# Mock BaseDetector for property testing
# ---------------------------------------------------------------------------


class MockDetector:
    """Mock BaseDetector that records method calls and returns configurable loss values.
    
    This mock is designed to test the training loop's behavior without actual
    model computation.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        loss_values: Optional[List[float]] = None,
    ):
        """Initialize the mock detector.
        
        Args:
            config: Configuration dict (stored but not used).
            loss_values: List of loss values to return per batch (cycles if shorter).
        """
        self.config = config or {}
        self._loss_values = loss_values or [1.0]
        self._batch_counter = 0
        self._call_log: List[str] = []
        self._train_mode = False
        
        # Create a simple parameter for optimizer construction
        self._param = torch.nn.Parameter(torch.tensor([1.0]))

    def train_step(
        self, images: List[torch.Tensor], targets: List[dict]
    ) -> Dict[str, Any]:
        """Simulate a training step.
        
        Args:
            images: List of image tensors.
            targets: List of target dicts.
            
        Returns:
            Dict with 'loss_tensor' key.
        """
        batch_idx = self._batch_counter
        self._batch_counter += 1
        self._call_log.append(f"train_step(batch={batch_idx})")
        
        # Return configured loss value (cycle through list)
        loss_idx = batch_idx % len(self._loss_values)
        loss_value = self._loss_values[loss_idx]
        
        # Create a tensor with grad_fn for backprop
        # Use abs() to ensure loss is always non-negative (parameter can become negative during training)
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

    def save_checkpoint(
        self,
        path: Path,
        optimizer: Optional[Any] = None,
        epoch: Optional[int] = None,
        metrics: Optional[dict] = None,
    ) -> None:
        """Save checkpoint (no-op for mock)."""
        self._call_log.append(f"save_checkpoint({path.name})")
        # Create the file to satisfy the training loop
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mock": True}, path)

    def get_config_schema(self) -> dict:
        """Return empty schema (no required params)."""
        return {}

    def reset_batch_counter(self) -> None:
        """Reset the batch counter for a new epoch."""
        self._batch_counter = 0

    @property
    def call_log(self) -> List[str]:
        """Return the log of method calls."""
        return self._call_log


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
    epochs: int = 1, 
    batch_size: int = 2, 
    seed: int = 42,
    early_stopping_patience: int = 100,
) -> dict:
    """Create a minimal test configuration and write it to a file."""
    import yaml
    
    config = {
        "model": {
            "type": "mock_detector",
            "config": {"num_classes": 5, "input_size": 32},
        },
        "dataset": {
            "path": str(tmp_path / "dataset"),
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": 0.01,
            "optimizer": "SGD",
            "weight_decay": 0.0005,
            "momentum": 0.9,
            "warmup_epochs": 0,
            "val_split": 0.2,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "log_interval": 100,
            "use_amp": False,
            "num_workers": 0,
            "early_stopping_patience": early_stopping_patience,
            "seed": seed,
        },
    }
    
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    return config


# ---------------------------------------------------------------------------
# Required metric keys
# ---------------------------------------------------------------------------

REQUIRED_METRIC_KEYS = {"train_loss", "val_loss", "learning_rate", "epoch_time_s"}


# ---------------------------------------------------------------------------
# Property 13: Metrics logged per epoch with required keys
# ---------------------------------------------------------------------------


class TestProperty13MetricsLoggedPerEpoch:
    """Property 13: Metrics logged per epoch with required keys.
    
    For any training run of N completed epochs, `ExperimentTracker.log_metrics()`
    SHALL be called exactly N times, and each call's metrics dict SHALL contain
    at minimum the keys `train_loss`, `val_loss`, `learning_rate`, and `epoch_time_s`.
    
    **Validates: Requirements 9.2**
    """

    @given(
        num_epochs=st.integers(min_value=1, max_value=10),
        num_samples=st.integers(min_value=10, max_value=30),
        batch_size=st.integers(min_value=2, max_value=4),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None)
    def test_log_metrics_called_n_times_for_n_epochs(
        self,
        num_epochs: int,
        num_samples: int,
        batch_size: int,
        seed: int,
        tmp_path_factory,
    ):
        """Feature: unified-training-loop, Property 13: Metrics logged per epoch with required keys
        
        Generate training runs of N epochs.
        Assert `log_metrics` called N times with keys: train_loss, val_loss, learning_rate, epoch_time_s.
        
        **Validates: Requirements 9.2**
        """
        # Create mock detector
        mock_detector = MockDetector(
            config={"num_classes": 5},
            loss_values=[1.0, 0.9, 0.8, 0.7],  # Decreasing losses to avoid early stopping
        )
        
        # Create a temporary directory for this test
        tmp_path = tmp_path_factory.mktemp(f"test_metrics_{seed}")
        
        # Create minimal dataset
        _create_test_dataset(tmp_path, num_samples)
        
        # Create config with specified epochs
        _create_test_config(
            tmp_path, 
            epochs=num_epochs, 
            batch_size=batch_size, 
            seed=seed,
            early_stopping_patience=num_epochs + 10,  # Prevent early stopping
        )
        
        # Track log_metrics calls
        log_metrics_calls: List[Dict[str, Any]] = []
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker_class:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                
                # Capture log_metrics calls
                def capture_log_metrics(run_id, step, metrics):
                    log_metrics_calls.append({
                        "run_id": run_id,
                        "step": step,
                        "metrics": dict(metrics),
                    })
                
                mock_tracker_instance.log_metrics.side_effect = capture_log_metrics
                mock_tracker_class.return_value = mock_tracker_instance
                
                # Suppress logging during test
                logging.disable(logging.CRITICAL)
                
                try:
                    # Run training
                    result = train(str(tmp_path / "config.yaml"), verbose=False)
                    
                    # Verify training completed
                    assert isinstance(result, dict), "Training should return a metrics dict"
                    
                    # Get the actual number of completed epochs from the result
                    completed_epochs = result.get("total_epochs", num_epochs)
                    
                    # Assert log_metrics was called exactly N times (once per epoch)
                    assert len(log_metrics_calls) == completed_epochs, (
                        f"Expected log_metrics to be called {completed_epochs} times "
                        f"(once per epoch), but was called {len(log_metrics_calls)} times"
                    )
                    
                    # Assert each call contains the required keys
                    for i, call_data in enumerate(log_metrics_calls):
                        metrics = call_data["metrics"]
                        missing_keys = REQUIRED_METRIC_KEYS - set(metrics.keys())
                        
                        assert not missing_keys, (
                            f"log_metrics call {i} (epoch {call_data['step']}) is missing "
                            f"required keys: {missing_keys}. Got keys: {set(metrics.keys())}"
                        )
                        
                        # Verify the step matches the epoch index (0-indexed)
                        assert call_data["step"] == i, (
                            f"Expected step={i} for call {i}, got step={call_data['step']}"
                        )
                        
                        # Verify values are numeric
                        for key in REQUIRED_METRIC_KEYS:
                            value = metrics[key]
                            assert isinstance(value, (int, float)), (
                                f"Metric '{key}' should be numeric, got {type(value).__name__}"
                            )
                    
                finally:
                    logging.disable(logging.NOTSET)

    @given(
        num_epochs=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None)
    def test_metrics_contain_all_required_keys(
        self,
        num_epochs: int,
        seed: int,
        tmp_path_factory,
    ):
        """Feature: unified-training-loop, Property 13: Each metrics dict contains required keys
        
        Verify that every log_metrics call includes train_loss, val_loss, 
        learning_rate, and epoch_time_s.
        
        **Validates: Requirements 9.2**
        """
        mock_detector = MockDetector(
            config={"num_classes": 5},
            loss_values=[1.0],
        )
        
        tmp_path = tmp_path_factory.mktemp(f"test_keys_{seed}")
        _create_test_dataset(tmp_path, 15)
        _create_test_config(
            tmp_path, 
            epochs=num_epochs, 
            batch_size=2, 
            seed=seed,
            early_stopping_patience=num_epochs + 10,
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
                    train(str(tmp_path / "config.yaml"), verbose=False)
                    
                    # Verify each captured metrics dict has all required keys
                    for epoch_idx, metrics in enumerate(captured_metrics):
                        for key in REQUIRED_METRIC_KEYS:
                            assert key in metrics, (
                                f"Epoch {epoch_idx}: Missing required key '{key}'. "
                                f"Got keys: {list(metrics.keys())}"
                            )
                finally:
                    logging.disable(logging.NOTSET)

    @given(
        num_epochs=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None)
    def test_metrics_values_are_valid_numbers(
        self,
        num_epochs: int,
        seed: int,
        tmp_path_factory,
    ):
        """Feature: unified-training-loop, Property 13: Metric values are valid numbers
        
        Verify that all required metric values are valid numeric types (not NaN, not None).
        
        **Validates: Requirements 9.2**
        """
        import math
        
        mock_detector = MockDetector(
            config={"num_classes": 5},
            loss_values=[0.5, 0.4, 0.3],  # Valid loss values
        )
        
        tmp_path = tmp_path_factory.mktemp(f"test_values_{seed}")
        _create_test_dataset(tmp_path, 20)
        _create_test_config(
            tmp_path, 
            epochs=num_epochs, 
            batch_size=2, 
            seed=seed,
            early_stopping_patience=num_epochs + 10,
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
                    train(str(tmp_path / "config.yaml"), verbose=False)
                    
                    for epoch_idx, metrics in enumerate(captured_metrics):
                        for key in REQUIRED_METRIC_KEYS:
                            value = metrics.get(key)
                            
                            # Value should not be None
                            assert value is not None, (
                                f"Epoch {epoch_idx}: '{key}' is None"
                            )
                            
                            # Value should be a number
                            assert isinstance(value, (int, float)), (
                                f"Epoch {epoch_idx}: '{key}' is not numeric: {type(value)}"
                            )
                            
                            # Value should not be NaN
                            assert not math.isnan(value), (
                                f"Epoch {epoch_idx}: '{key}' is NaN"
                            )
                            
                            # Loss values should be non-negative
                            if key in ("train_loss", "val_loss"):
                                assert value >= 0, (
                                    f"Epoch {epoch_idx}: '{key}' is negative: {value}"
                                )
                            
                            # Learning rate should be positive
                            if key == "learning_rate":
                                assert value > 0, (
                                    f"Epoch {epoch_idx}: '{key}' should be positive: {value}"
                                )
                            
                            # Epoch time should be non-negative
                            if key == "epoch_time_s":
                                assert value >= 0, (
                                    f"Epoch {epoch_idx}: '{key}' should be non-negative: {value}"
                                )
                finally:
                    logging.disable(logging.NOTSET)

    @given(
        num_epochs=st.integers(min_value=1, max_value=6),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None)
    def test_log_metrics_step_matches_epoch_index(
        self,
        num_epochs: int,
        seed: int,
        tmp_path_factory,
    ):
        """Feature: unified-training-loop, Property 13: log_metrics step parameter matches epoch
        
        Verify that the step parameter passed to log_metrics is the zero-based epoch index.
        
        **Validates: Requirements 9.2**
        """
        mock_detector = MockDetector(
            config={"num_classes": 5},
            loss_values=[1.0],
        )
        
        tmp_path = tmp_path_factory.mktemp(f"test_step_{seed}")
        _create_test_dataset(tmp_path, 15)
        _create_test_config(
            tmp_path, 
            epochs=num_epochs, 
            batch_size=2, 
            seed=seed,
            early_stopping_patience=num_epochs + 10,
        )
        
        captured_steps: List[int] = []
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker_class:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                
                def capture_step(run_id, step, metrics):
                    captured_steps.append(step)
                
                mock_tracker_instance.log_metrics.side_effect = capture_step
                mock_tracker_class.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                try:
                    result = train(str(tmp_path / "config.yaml"), verbose=False)
                    completed_epochs = result.get("total_epochs", num_epochs)
                    
                    # Verify steps are sequential 0-indexed epoch numbers
                    expected_steps = list(range(completed_epochs))
                    assert captured_steps == expected_steps, (
                        f"Expected steps {expected_steps}, got {captured_steps}"
                    )
                finally:
                    logging.disable(logging.NOTSET)

    @given(
        num_epochs=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None)
    def test_log_metrics_uses_correct_run_id(
        self,
        num_epochs: int,
        seed: int,
        tmp_path_factory,
    ):
        """Feature: unified-training-loop, Property 13: log_metrics uses the correct run_id
        
        Verify that log_metrics is called with the run_id returned by start_run.
        
        **Validates: Requirements 9.2**
        """
        mock_detector = MockDetector(
            config={"num_classes": 5},
            loss_values=[1.0],
        )
        
        tmp_path = tmp_path_factory.mktemp(f"test_runid_{seed}")
        _create_test_dataset(tmp_path, 15)
        _create_test_config(
            tmp_path, 
            epochs=num_epochs, 
            batch_size=2, 
            seed=seed,
            early_stopping_patience=num_epochs + 10,
        )
        
        expected_run_id = f"test_run_{seed}"
        captured_run_ids: List[str] = []
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker_class:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = expected_run_id
                
                def capture_run_id(run_id, step, metrics):
                    captured_run_ids.append(run_id)
                
                mock_tracker_instance.log_metrics.side_effect = capture_run_id
                mock_tracker_class.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                try:
                    train(str(tmp_path / "config.yaml"), verbose=False)
                    
                    # All log_metrics calls should use the same run_id
                    for i, run_id in enumerate(captured_run_ids):
                        assert run_id == expected_run_id, (
                            f"Call {i}: Expected run_id '{expected_run_id}', got '{run_id}'"
                        )
                finally:
                    logging.disable(logging.NOTSET)


class TestMetricsLoggingWithEarlyStopping:
    """Test metrics logging behavior when early stopping triggers."""

    @given(
        patience=st.integers(min_value=1, max_value=3),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None)
    def test_metrics_logged_for_completed_epochs_with_early_stopping(
        self,
        patience: int,
        seed: int,
        tmp_path_factory,
    ):
        """Feature: unified-training-loop, Property 13: Metrics logged for completed epochs with early stopping
        
        When early stopping triggers, log_metrics should have been called for each
        completed epoch up to and including the stopping epoch.
        
        **Validates: Requirements 9.2**
        """
        # Use increasing loss values to trigger early stopping
        mock_detector = MockDetector(
            config={"num_classes": 5},
            loss_values=[1.0, 1.1, 1.2, 1.3, 1.4],  # Increasing losses
        )
        
        tmp_path = tmp_path_factory.mktemp(f"test_early_stop_{seed}")
        _create_test_dataset(tmp_path, 15)
        
        max_epochs = patience + 5  # More epochs than patience allows
        _create_test_config(
            tmp_path, 
            epochs=max_epochs, 
            batch_size=2, 
            seed=seed,
            early_stopping_patience=patience,
        )
        
        log_metrics_calls: List[Dict[str, Any]] = []
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker_class:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                
                def capture_log_metrics(run_id, step, metrics):
                    log_metrics_calls.append({
                        "run_id": run_id,
                        "step": step,
                        "metrics": dict(metrics),
                    })
                
                mock_tracker_instance.log_metrics.side_effect = capture_log_metrics
                mock_tracker_class.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                try:
                    result = train(str(tmp_path / "config.yaml"), verbose=False)
                    
                    completed_epochs = result.get("total_epochs", 0)
                    
                    # log_metrics should be called exactly once per completed epoch
                    assert len(log_metrics_calls) == completed_epochs, (
                        f"Expected {completed_epochs} log_metrics calls, "
                        f"got {len(log_metrics_calls)}"
                    )
                    
                    # Each call should have all required keys
                    for i, call_data in enumerate(log_metrics_calls):
                        metrics = call_data["metrics"]
                        missing_keys = REQUIRED_METRIC_KEYS - set(metrics.keys())
                        assert not missing_keys, (
                            f"Call {i} missing keys: {missing_keys}"
                        )
                finally:
                    logging.disable(logging.NOTSET)
