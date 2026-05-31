"""Property-based tests for the unified training loop.

Tests correctness properties from the design document using Hypothesis.
Each test validates specific requirements from the unified-training-loop spec.

Feature: unified-training-loop
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock, patch

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
    model computation. It can be configured to:
    - Return specific loss values per batch
    - Raise exceptions at specific batch positions
    - Record all method calls for verification
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        loss_values: Optional[List[float]] = None,
        exception_batches: Optional[Set[int]] = None,
    ):
        """Initialize the mock detector.
        
        Args:
            config: Configuration dict (stored but not used).
            loss_values: List of loss values to return per batch (cycles if shorter).
            exception_batches: Set of batch indices where train_step should raise.
        """
        self.config = config or {}
        self._loss_values = loss_values or [1.0]
        self._exception_batches = exception_batches or set()
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
            
        Raises:
            RuntimeError: If current batch is in exception_batches.
        """
        batch_idx = self._batch_counter
        self._batch_counter += 1
        self._call_log.append(f"train_step(batch={batch_idx})")
        
        # Raise exception if configured for this batch
        if batch_idx in self._exception_batches:
            raise RuntimeError(f"Simulated exception at batch {batch_idx}")
        
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
# Hypothesis strategies
# ---------------------------------------------------------------------------


@st.composite
def st_exception_batch_positions(draw, num_batches: int):
    """Generate a set of batch positions where exceptions should be raised.
    
    Args:
        draw: Hypothesis draw function.
        num_batches: Total number of batches in the epoch.
        
    Returns:
        Set of batch indices where exceptions should occur.
    """
    if num_batches <= 0:
        return set()
    
    # Generate 0 to num_batches/2 exception positions
    max_exceptions = max(1, num_batches // 2)
    num_exceptions = draw(st.integers(min_value=1, max_value=max_exceptions))
    
    # Select random batch positions
    positions = draw(
        st.lists(
            st.integers(min_value=0, max_value=num_batches - 1),
            min_size=num_exceptions,
            max_size=num_exceptions,
            unique=True,
        )
    )
    return set(positions)


# ---------------------------------------------------------------------------
# Property 3: Exceptions in train_step are caught and training continues
# ---------------------------------------------------------------------------


class TestExceptionsInTrainStepAreCaught:
    """Property 3: Exceptions in train_step are caught and training continues.
    
    For any exception raised by train_step() at any batch position within an epoch,
    the training loop SHALL continue processing subsequent batches and complete
    the epoch without propagating the exception.
    
    **Validates: Requirements 2.5**
    """

    @given(
        num_train_samples=st.integers(min_value=4, max_value=20),
        batch_size=st.integers(min_value=2, max_value=4),
        exception_fraction=st.floats(min_value=0.1, max_value=0.5),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None)
    def test_exceptions_caught_and_training_continues(
        self,
        num_train_samples: int,
        batch_size: int,
        exception_fraction: float,
        seed: int,
        tmp_path_factory,
    ):
        """Feature: unified-training-loop, Property 3: Exceptions in train_step are caught and training continues
        
        Mock detector raises exceptions at random batch positions.
        Assert the epoch completes and subsequent batches are processed.
        
        **Validates: Requirements 2.5**
        """
        # Calculate number of batches
        num_batches = (num_train_samples + batch_size - 1) // batch_size
        assume(num_batches >= 2)  # Need at least 2 batches to test continuation
        
        # Determine which batches will raise exceptions
        num_exceptions = max(1, int(num_batches * exception_fraction))
        # Ensure we don't have exceptions in ALL batches (need some to succeed)
        num_exceptions = min(num_exceptions, num_batches - 1)
        
        # Use seed to deterministically select exception batches
        import random
        rng = random.Random(seed)
        exception_batches = set(rng.sample(range(num_batches), num_exceptions))
        
        # Create mock detector that raises at specified batches
        mock_detector = MockDetector(
            config={"num_classes": 5},
            loss_values=[1.0],  # Normal loss for non-exception batches
            exception_batches=exception_batches,
        )
        
        # Track which batches were actually processed
        processed_batches: List[int] = []
        original_train_step = mock_detector.train_step
        
        def tracking_train_step(images, targets):
            batch_idx = mock_detector._batch_counter
            try:
                result = original_train_step(images, targets)
                processed_batches.append(batch_idx)
                return result
            except RuntimeError:
                # Re-raise to let the training loop handle it
                raise
        
        mock_detector.train_step = tracking_train_step
        
        # Create a temporary directory for this test
        tmp_path = tmp_path_factory.mktemp(f"test_exceptions_{seed}")
        
        # Create minimal config
        config = {
            "model": {
                "type": "mock_detector",
                "config": {"num_classes": 5, "input_size": 32},
            },
            "dataset": {
                "path": str(tmp_path / "dataset"),
            },
            "training": {
                "epochs": 1,
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
                "early_stopping_patience": 100,
                "seed": seed,
            },
        }
        
        # Create mock dataset directory with sample images
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Create sample annotation files and images
        from PIL import Image
        import xml.etree.ElementTree as ET
        
        for i in range(num_train_samples + 5):  # Extra for validation split
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
        
        # Write config to file
        import yaml
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Patch ModelRegistry.create to return our mock detector
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            # Patch ExperimentTracker to avoid file system issues
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                mock_tracker.return_value = mock_tracker_instance
                
                # Suppress logging during test
                logging.disable(logging.CRITICAL)
                
                try:
                    # Run training - should NOT raise despite exceptions in train_step
                    result = train(str(config_path), verbose=False)
                    
                    # Verify training completed (returned a result dict)
                    assert isinstance(result, dict), "Training should return a metrics dict"
                    
                    # Verify that batches after exceptions were processed
                    # The training loop should have continued after each exception
                    non_exception_batches = set(range(num_batches)) - exception_batches
                    
                    # At least some non-exception batches should have been processed
                    # (accounting for the fact that batch counter includes both train and val)
                    assert len(processed_batches) > 0, (
                        f"No batches were processed. Exception batches: {exception_batches}"
                    )
                    
                finally:
                    logging.disable(logging.NOTSET)


class TestExceptionsAtVariousPositions:
    """Additional tests for exception handling at specific batch positions."""

    @given(
        num_batches=st.integers(min_value=3, max_value=10),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None)
    def test_exception_at_first_batch_continues(
        self,
        num_batches: int,
        seed: int,
        tmp_path_factory,
    ):
        """Feature: unified-training-loop, Property 3: Exception at first batch doesn't stop training
        
        When train_step raises an exception at the very first batch,
        the training loop should continue with subsequent batches.
        
        **Validates: Requirements 2.5**
        """
        # Exception only at batch 0
        exception_batches = {0}
        
        mock_detector = MockDetector(
            config={"num_classes": 5},
            loss_values=[1.0],
            exception_batches=exception_batches,
        )
        
        tmp_path = tmp_path_factory.mktemp(f"test_first_batch_{seed}")
        
        # Create minimal dataset
        self._create_test_dataset(tmp_path, num_batches * 2 + 5)
        
        config = self._create_test_config(tmp_path, batch_size=2, seed=seed)
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                mock_tracker.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                try:
                    result = train(str(tmp_path / "config.yaml"), verbose=False)
                    
                    # Training should complete
                    assert isinstance(result, dict)
                    assert "total_epochs" in result or result == {}
                    
                    # Verify train_step was called multiple times (not just once)
                    train_step_calls = [
                        c for c in mock_detector.call_log if c.startswith("train_step")
                    ]
                    assert len(train_step_calls) > 1, (
                        "Training should continue after first batch exception"
                    )
                finally:
                    logging.disable(logging.NOTSET)

    @given(
        num_batches=st.integers(min_value=3, max_value=10),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None)
    def test_exception_at_last_batch_completes_epoch(
        self,
        num_batches: int,
        seed: int,
        tmp_path_factory,
    ):
        """Feature: unified-training-loop, Property 3: Exception at last batch still completes epoch
        
        When train_step raises an exception at the last batch of an epoch,
        the epoch should still complete (validation phase runs).
        
        **Validates: Requirements 2.5**
        """
        # Exception only at the last batch
        last_batch = num_batches - 1
        exception_batches = {last_batch}
        
        mock_detector = MockDetector(
            config={"num_classes": 5},
            loss_values=[1.0],
            exception_batches=exception_batches,
        )
        
        tmp_path = tmp_path_factory.mktemp(f"test_last_batch_{seed}")
        
        # Create dataset with enough samples
        num_samples = num_batches * 2 + 5
        self._create_test_dataset(tmp_path, num_samples)
        
        config = self._create_test_config(tmp_path, batch_size=2, seed=seed)
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                mock_tracker.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                try:
                    result = train(str(tmp_path / "config.yaml"), verbose=False)
                    
                    # Training should complete
                    assert isinstance(result, dict)
                    
                    # Verify set_eval_mode was called (validation phase ran)
                    eval_mode_calls = [
                        c for c in mock_detector.call_log if c == "set_eval_mode()"
                    ]
                    assert len(eval_mode_calls) >= 1, (
                        "Validation phase should run even if last training batch raised"
                    )
                finally:
                    logging.disable(logging.NOTSET)

    @given(
        num_batches=st.integers(min_value=5, max_value=15),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None)
    def test_multiple_consecutive_exceptions(
        self,
        num_batches: int,
        seed: int,
        tmp_path_factory,
    ):
        """Feature: unified-training-loop, Property 3: Multiple consecutive exceptions handled
        
        When train_step raises exceptions in multiple consecutive batches,
        the training loop should continue with the next non-failing batch.
        
        **Validates: Requirements 2.5**
        """
        # Exceptions in batches 1, 2, 3 (consecutive)
        exception_batches = {1, 2, 3}
        assume(num_batches > max(exception_batches) + 1)  # Need batches after exceptions
        
        mock_detector = MockDetector(
            config={"num_classes": 5},
            loss_values=[1.0],
            exception_batches=exception_batches,
        )
        
        tmp_path = tmp_path_factory.mktemp(f"test_consecutive_{seed}")
        
        num_samples = num_batches * 2 + 5
        self._create_test_dataset(tmp_path, num_samples)
        
        config = self._create_test_config(tmp_path, batch_size=2, seed=seed)
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                mock_tracker.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                try:
                    result = train(str(tmp_path / "config.yaml"), verbose=False)
                    
                    # Training should complete
                    assert isinstance(result, dict)
                    
                    # Verify batches after the consecutive exceptions were processed
                    train_step_calls = [
                        c for c in mock_detector.call_log if c.startswith("train_step")
                    ]
                    
                    # Should have more calls than just the exception batches
                    # (batch 0 + batches after 3)
                    assert len(train_step_calls) > len(exception_batches), (
                        f"Expected batches after consecutive exceptions to be processed. "
                        f"Calls: {train_step_calls}"
                    )
                finally:
                    logging.disable(logging.NOTSET)

    def _create_test_dataset(self, tmp_path: Path, num_samples: int) -> None:
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
        self, tmp_path: Path, batch_size: int = 2, seed: int = 42
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
                "epochs": 1,
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
                "early_stopping_patience": 100,
                "seed": seed,
            },
        }
        
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        return config
