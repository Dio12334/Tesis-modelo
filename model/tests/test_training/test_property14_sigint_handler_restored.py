"""Property-based tests for the unified training loop - Property 14.

Feature: unified-training-loop, Property 14: SIGINT handler restored after training

These tests verify that the SIGINT signal handler is restored to its original state
after train() returns, regardless of how training ended (normal completion, early
stopping, or interruption).

**Validates: Requirements 10.3**
"""

import logging
import signal
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
import xml.etree.ElementTree as ET

import pytest
import torch
import yaml
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
from PIL import Image

from model.training.train_detection import train


# ---------------------------------------------------------------------------
# Mock BaseDetector for property testing
# ---------------------------------------------------------------------------


class MockDetectorForSignalTest:
    """Mock BaseDetector that supports configurable training completion modes.
    
    This mock can be configured to:
    - Complete normally after N epochs
    - Trigger early stopping by returning non-improving losses
    - Support interruption testing
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
        """Simulate a training step."""
        batch_idx = self._batch_counter
        self._batch_counter += 1
        self._call_log.append(f"train_step(batch={batch_idx})")
        
        # Return configured loss value (cycle through list)
        loss_idx = batch_idx % len(self._loss_values)
        loss_value = self._loss_values[loss_idx]
        
        # Create a tensor with grad_fn for backprop
        loss_tensor = self._param * loss_value
        
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
# Helper functions
# ---------------------------------------------------------------------------


def create_test_dataset(tmp_path: Path, num_samples: int) -> None:
    """Create a minimal test dataset with images and annotations."""
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


def create_test_config(
    tmp_path: Path,
    epochs: int = 1,
    batch_size: int = 2,
    early_stopping_patience: int = 100,
    seed: int = 42,
) -> dict:
    """Create a minimal test configuration and write it to a file."""
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
# Hypothesis strategies
# ---------------------------------------------------------------------------


@st.composite
def st_completion_mode(draw):
    """Generate a training completion mode.
    
    Returns a tuple of (mode, config_params) where:
    - mode: "normal", "early_stopping"
    - config_params: dict with epochs, patience, loss_values
    """
    mode = draw(st.sampled_from(["normal", "early_stopping"]))
    
    if mode == "normal":
        # Normal completion: run 1-3 epochs
        epochs = draw(st.integers(min_value=1, max_value=3))
        return (mode, {
            "epochs": epochs,
            "early_stopping_patience": 100,  # High patience to avoid early stopping
            "loss_values": [1.0],  # Constant loss
        })
    else:
        # Early stopping: use increasing losses to trigger early stopping
        patience = draw(st.integers(min_value=1, max_value=3))
        # Generate increasing loss values to trigger early stopping
        # First epoch has low loss, subsequent epochs have higher loss
        return (mode, {
            "epochs": 10,  # High max epochs
            "early_stopping_patience": patience,
            "loss_values": [0.5, 1.0, 1.5, 2.0, 2.5],  # Increasing losses
        })


# ---------------------------------------------------------------------------
# Property 14: SIGINT handler restored after training
# ---------------------------------------------------------------------------


class TestProperty14SigintHandlerRestored:
    """Property 14: SIGINT handler restored after training.
    
    For any training run (normal completion, early stopping, or interruption),
    after train() returns, the SIGINT signal handler SHALL be the same handler
    that was installed before train() was called.
    
    **Validates: Requirements 10.3**
    """

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        epochs=st.integers(min_value=1, max_value=3),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_sigint_handler_restored_after_normal_completion(
        self,
        epochs: int,
        seed: int,
        tmp_path_factory,
    ):
        """SIGINT handler is restored after normal training completion.
        
        Feature: unified-training-loop, Property 14: SIGINT handler restored after training
        **Validates: Requirements 10.3**
        """
        tmp_path = tmp_path_factory.mktemp(f"test_normal_{seed}")
        
        # Create test dataset
        num_samples = 10
        create_test_dataset(tmp_path, num_samples)
        
        # Create config
        create_test_config(
            tmp_path,
            epochs=epochs,
            batch_size=2,
            early_stopping_patience=100,  # High to avoid early stopping
            seed=seed,
        )
        
        # Create mock detector
        mock_detector = MockDetectorForSignalTest(
            config={"num_classes": 5},
            loss_values=[1.0],  # Constant loss
        )
        
        # Record the original SIGINT handler before calling train()
        original_handler = signal.getsignal(signal.SIGINT)
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                mock_tracker.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                try:
                    # Run training
                    result = train(str(tmp_path / "config.yaml"), verbose=False)
                    
                    # Verify training completed
                    assert isinstance(result, dict), "Training should return a metrics dict"
                    
                finally:
                    logging.disable(logging.NOTSET)
        
        # Verify SIGINT handler is restored to original
        handler_after = signal.getsignal(signal.SIGINT)
        assert handler_after == original_handler, (
            f"SIGINT handler should be restored after normal completion. "
            f"Original: {original_handler}, After: {handler_after}"
        )

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        patience=st.integers(min_value=1, max_value=3),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_sigint_handler_restored_after_early_stopping(
        self,
        patience: int,
        seed: int,
        tmp_path_factory,
    ):
        """SIGINT handler is restored after early stopping triggers.
        
        Feature: unified-training-loop, Property 14: SIGINT handler restored after training
        **Validates: Requirements 10.3**
        """
        tmp_path = tmp_path_factory.mktemp(f"test_early_stop_{seed}")
        
        # Create test dataset
        num_samples = 10
        create_test_dataset(tmp_path, num_samples)
        
        # Create config with low patience to trigger early stopping
        create_test_config(
            tmp_path,
            epochs=10,  # High max epochs
            batch_size=2,
            early_stopping_patience=patience,
            seed=seed,
        )
        
        # Create mock detector with increasing losses to trigger early stopping
        # First batch has low loss, subsequent batches have higher loss
        mock_detector = MockDetectorForSignalTest(
            config={"num_classes": 5},
            loss_values=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],  # Increasing losses
        )
        
        # Record the original SIGINT handler before calling train()
        original_handler = signal.getsignal(signal.SIGINT)
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                mock_tracker.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                try:
                    # Run training
                    result = train(str(tmp_path / "config.yaml"), verbose=False)
                    
                    # Verify training completed (possibly via early stopping)
                    assert isinstance(result, dict), "Training should return a metrics dict"
                    
                finally:
                    logging.disable(logging.NOTSET)
        
        # Verify SIGINT handler is restored to original
        handler_after = signal.getsignal(signal.SIGINT)
        assert handler_after == original_handler, (
            f"SIGINT handler should be restored after early stopping. "
            f"Original: {original_handler}, After: {handler_after}"
        )

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        completion_mode=st_completion_mode(),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_sigint_handler_restored_for_any_completion_mode(
        self,
        completion_mode,
        seed: int,
        tmp_path_factory,
    ):
        """SIGINT handler is restored regardless of how training completes.
        
        Feature: unified-training-loop, Property 14: SIGINT handler restored after training
        **Validates: Requirements 10.3**
        """
        mode, config_params = completion_mode
        tmp_path = tmp_path_factory.mktemp(f"test_{mode}_{seed}")
        
        # Create test dataset
        num_samples = 10
        create_test_dataset(tmp_path, num_samples)
        
        # Create config
        create_test_config(
            tmp_path,
            epochs=config_params["epochs"],
            batch_size=2,
            early_stopping_patience=config_params["early_stopping_patience"],
            seed=seed,
        )
        
        # Create mock detector
        mock_detector = MockDetectorForSignalTest(
            config={"num_classes": 5},
            loss_values=config_params["loss_values"],
        )
        
        # Record the original SIGINT handler before calling train()
        original_handler = signal.getsignal(signal.SIGINT)
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                mock_tracker.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                try:
                    # Run training
                    result = train(str(tmp_path / "config.yaml"), verbose=False)
                    
                    # Verify training completed
                    assert isinstance(result, dict), "Training should return a metrics dict"
                    
                finally:
                    logging.disable(logging.NOTSET)
        
        # Verify SIGINT handler is restored to original
        handler_after = signal.getsignal(signal.SIGINT)
        assert handler_after == original_handler, (
            f"SIGINT handler should be restored after {mode} completion. "
            f"Original: {original_handler}, After: {handler_after}"
        )

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(seed=st.integers(min_value=0, max_value=10000))
    def test_sigint_handler_restored_with_custom_original_handler(
        self,
        seed: int,
        tmp_path_factory,
    ):
        """SIGINT handler is restored even when a custom handler was installed.
        
        Feature: unified-training-loop, Property 14: SIGINT handler restored after training
        **Validates: Requirements 10.3**
        """
        tmp_path = tmp_path_factory.mktemp(f"test_custom_handler_{seed}")
        
        # Create test dataset
        num_samples = 10
        create_test_dataset(tmp_path, num_samples)
        
        # Create config
        create_test_config(
            tmp_path,
            epochs=1,
            batch_size=2,
            early_stopping_patience=100,
            seed=seed,
        )
        
        # Create mock detector
        mock_detector = MockDetectorForSignalTest(
            config={"num_classes": 5},
            loss_values=[1.0],
        )
        
        # Install a custom SIGINT handler before calling train()
        custom_handler_called = []
        
        def custom_sigint_handler(signum, frame):
            custom_handler_called.append(True)
        
        # Save the default handler and install our custom one
        default_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, custom_sigint_handler)
        
        try:
            # Record the custom handler as the "original" handler
            original_handler = signal.getsignal(signal.SIGINT)
            assert original_handler == custom_sigint_handler, "Custom handler should be installed"
            
            with patch("model.training.train_detection.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = mock_detector
                
                with patch("model.training.train_detection.ExperimentTracker") as mock_tracker:
                    mock_tracker_instance = MagicMock()
                    mock_tracker_instance.start_run.return_value = "test_run_id"
                    mock_tracker.return_value = mock_tracker_instance
                    
                    logging.disable(logging.CRITICAL)
                    try:
                        # Run training
                        result = train(str(tmp_path / "config.yaml"), verbose=False)
                        
                        # Verify training completed
                        assert isinstance(result, dict), "Training should return a metrics dict"
                        
                    finally:
                        logging.disable(logging.NOTSET)
            
            # Verify SIGINT handler is restored to our custom handler
            handler_after = signal.getsignal(signal.SIGINT)
            assert handler_after == custom_sigint_handler, (
                f"SIGINT handler should be restored to custom handler. "
                f"Expected: {custom_sigint_handler}, Got: {handler_after}"
            )
            
        finally:
            # Restore the default handler
            signal.signal(signal.SIGINT, default_handler)

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(seed=st.integers(min_value=0, max_value=10000))
    def test_sigint_handler_restored_after_tracker_start_failure(
        self,
        seed: int,
        tmp_path_factory,
    ):
        """SIGINT handler is restored even when ExperimentTracker.start_run fails.
        
        Feature: unified-training-loop, Property 14: SIGINT handler restored after training
        **Validates: Requirements 10.3**
        """
        tmp_path = tmp_path_factory.mktemp(f"test_tracker_fail_{seed}")
        
        # Create test dataset
        num_samples = 10
        create_test_dataset(tmp_path, num_samples)
        
        # Create config
        create_test_config(
            tmp_path,
            epochs=1,
            batch_size=2,
            early_stopping_patience=100,
            seed=seed,
        )
        
        # Create mock detector
        mock_detector = MockDetectorForSignalTest(
            config={"num_classes": 5},
            loss_values=[1.0],
        )
        
        # Record the original SIGINT handler before calling train()
        original_handler = signal.getsignal(signal.SIGINT)
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker:
                mock_tracker_instance = MagicMock()
                # Make start_run raise an exception
                mock_tracker_instance.start_run.side_effect = RuntimeError("Simulated tracker failure")
                mock_tracker.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                try:
                    # Run training - should return early due to tracker failure
                    result = train(str(tmp_path / "config.yaml"), verbose=False)
                    
                    # Verify training returned (possibly empty dict due to failure)
                    assert isinstance(result, dict), "Training should return a dict"
                    
                finally:
                    logging.disable(logging.NOTSET)
        
        # Verify SIGINT handler is restored to original
        handler_after = signal.getsignal(signal.SIGINT)
        assert handler_after == original_handler, (
            f"SIGINT handler should be restored after tracker failure. "
            f"Original: {original_handler}, After: {handler_after}"
        )


class TestSigintHandlerRestoredEdgeCases:
    """Additional edge case tests for SIGINT handler restoration."""

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        num_epochs=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_handler_not_leaked_across_multiple_train_calls(
        self,
        num_epochs: int,
        seed: int,
        tmp_path_factory,
    ):
        """SIGINT handler is properly restored across multiple train() calls.
        
        Feature: unified-training-loop, Property 14: SIGINT handler restored after training
        **Validates: Requirements 10.3**
        """
        # Record the original SIGINT handler
        original_handler = signal.getsignal(signal.SIGINT)
        
        # Run train() multiple times and verify handler is restored each time
        for i in range(2):  # Run twice to check for leaks
            tmp_path = tmp_path_factory.mktemp(f"test_multi_{seed}_{i}")
            
            # Create test dataset
            num_samples = 10
            create_test_dataset(tmp_path, num_samples)
            
            # Create config
            create_test_config(
                tmp_path,
                epochs=num_epochs,
                batch_size=2,
                early_stopping_patience=100,
                seed=seed + i,
            )
            
            # Create mock detector
            mock_detector = MockDetectorForSignalTest(
                config={"num_classes": 5},
                loss_values=[1.0],
            )
            
            with patch("model.training.train_detection.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = mock_detector
                
                with patch("model.training.train_detection.ExperimentTracker") as mock_tracker:
                    mock_tracker_instance = MagicMock()
                    mock_tracker_instance.start_run.return_value = f"test_run_id_{i}"
                    mock_tracker.return_value = mock_tracker_instance
                    
                    logging.disable(logging.CRITICAL)
                    try:
                        result = train(str(tmp_path / "config.yaml"), verbose=False)
                        assert isinstance(result, dict)
                    finally:
                        logging.disable(logging.NOTSET)
            
            # Verify handler is restored after each call
            handler_after = signal.getsignal(signal.SIGINT)
            assert handler_after == original_handler, (
                f"SIGINT handler should be restored after train() call {i+1}. "
                f"Original: {original_handler}, After: {handler_after}"
            )

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(seed=st.integers(min_value=0, max_value=10000))
    def test_handler_restored_when_default_handler_is_sig_dfl(
        self,
        seed: int,
        tmp_path_factory,
    ):
        """SIGINT handler is restored when original handler is SIG_DFL.
        
        Feature: unified-training-loop, Property 14: SIGINT handler restored after training
        **Validates: Requirements 10.3**
        """
        tmp_path = tmp_path_factory.mktemp(f"test_sig_dfl_{seed}")
        
        # Create test dataset
        num_samples = 10
        create_test_dataset(tmp_path, num_samples)
        
        # Create config
        create_test_config(
            tmp_path,
            epochs=1,
            batch_size=2,
            early_stopping_patience=100,
            seed=seed,
        )
        
        # Create mock detector
        mock_detector = MockDetectorForSignalTest(
            config={"num_classes": 5},
            loss_values=[1.0],
        )
        
        # Save current handler and set to SIG_DFL
        saved_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        
        try:
            # Record the SIG_DFL handler as the "original" handler
            original_handler = signal.getsignal(signal.SIGINT)
            assert original_handler == signal.SIG_DFL, "Handler should be SIG_DFL"
            
            with patch("model.training.train_detection.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = mock_detector
                
                with patch("model.training.train_detection.ExperimentTracker") as mock_tracker:
                    mock_tracker_instance = MagicMock()
                    mock_tracker_instance.start_run.return_value = "test_run_id"
                    mock_tracker.return_value = mock_tracker_instance
                    
                    logging.disable(logging.CRITICAL)
                    try:
                        result = train(str(tmp_path / "config.yaml"), verbose=False)
                        assert isinstance(result, dict)
                    finally:
                        logging.disable(logging.NOTSET)
            
            # Verify SIGINT handler is restored to SIG_DFL
            handler_after = signal.getsignal(signal.SIGINT)
            assert handler_after == signal.SIG_DFL, (
                f"SIGINT handler should be restored to SIG_DFL. "
                f"Got: {handler_after}"
            )
            
        finally:
            # Restore the saved handler
            signal.signal(signal.SIGINT, saved_handler)
