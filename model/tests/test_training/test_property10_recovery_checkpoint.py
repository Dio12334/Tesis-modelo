"""Property-based test for recovery checkpoint at epoch multiples of 5.

**Property 10: Recovery checkpoint at epoch multiples of 5**

For any training run of E epochs, a recovery checkpoint SHALL be saved at epochs
{5, 10, 15, ...} ∩ {1, ..., E} (using 1-indexed epoch numbers).

**Validates: Requirements 7.2**

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


class MockDetectorForRecoveryCheckpoint:
    """Mock BaseDetector that tracks checkpoint saves.
    
    This mock records all save_checkpoint calls to verify that recovery.pt
    is saved at the correct epochs (multiples of 5).
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        loss_values: Optional[List[float]] = None,
    ):
        """Initialize the mock detector.
        
        Args:
            config: Configuration dict (stored but not used).
            loss_values: List of loss values to return per epoch (cycles if shorter).
        """
        self.config = config or {}
        self._loss_values = loss_values or [1.0]
        self._batch_counter = 0
        self._epoch_counter = 0
        self._call_log: List[str] = []
        self._checkpoint_saves: List[Dict[str, Any]] = []
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
        
        # Return configured loss value (cycle through list based on epoch)
        loss_idx = self._epoch_counter % len(self._loss_values)
        loss_value = self._loss_values[loss_idx]
        
        # Create a tensor with grad_fn for backprop
        loss_tensor = self._param * loss_value
        
        return {"loss_tensor": loss_tensor}

    def get_parameters(self) -> List[torch.nn.Parameter]:
        """Return trainable parameters."""
        return [self._param]

    def set_train_mode(self) -> None:
        """Switch to training mode."""
        self._call_log.append(f"set_train_mode(epoch={self._epoch_counter})")
        self._train_mode = True

    def set_eval_mode(self) -> None:
        """Switch to evaluation mode."""
        self._call_log.append(f"set_eval_mode(epoch={self._epoch_counter})")
        self._train_mode = False
        # Increment epoch counter after validation phase
        self._epoch_counter += 1

    def save_checkpoint(
        self,
        path: Path,
        optimizer: Optional[Any] = None,
        epoch: Optional[int] = None,
        metrics: Optional[dict] = None,
    ) -> None:
        """Save checkpoint and record the call."""
        checkpoint_name = path.name
        self._call_log.append(f"save_checkpoint({checkpoint_name}, epoch={epoch})")
        self._checkpoint_saves.append({
            "path": path,
            "name": checkpoint_name,
            "epoch": epoch,  # 0-indexed epoch from training loop
            "metrics": metrics,
        })
        # Create the file to satisfy the training loop
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mock": True, "epoch": epoch}, path)

    def get_config_schema(self) -> dict:
        """Return empty schema (no required params)."""
        return {}

    def reset_counters(self) -> None:
        """Reset counters for a new run."""
        self._batch_counter = 0
        self._epoch_counter = 0

    @property
    def call_log(self) -> List[str]:
        """Return the log of method calls."""
        return self._call_log

    @property
    def checkpoint_saves(self) -> List[Dict[str, Any]]:
        """Return the list of checkpoint saves."""
        return self._checkpoint_saves

    def get_recovery_checkpoint_epochs(self) -> Set[int]:
        """Return the set of 1-indexed epochs where recovery.pt was saved."""
        recovery_epochs = set()
        for save in self._checkpoint_saves:
            if save["name"] == "recovery.pt":
                # Convert 0-indexed epoch to 1-indexed
                recovery_epochs.add(save["epoch"] + 1)
        return recovery_epochs


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def create_test_dataset(tmp_path: Path, num_samples: int) -> None:
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


def create_test_config(
    tmp_path: Path,
    epochs: int,
    batch_size: int = 2,
    seed: int = 42,
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
            "early_stopping_patience": 1000,  # Disable early stopping
            "seed": seed,
        },
    }
    
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    return config


def expected_recovery_epochs(total_epochs: int) -> Set[int]:
    """Calculate the expected set of epochs where recovery.pt should be saved.
    
    Recovery checkpoints are saved at epochs that are multiples of 5 (1-indexed).
    For E total epochs, this is {5, 10, 15, ...} ∩ {1, ..., E}.
    
    Args:
        total_epochs: Total number of epochs in the training run.
        
    Returns:
        Set of 1-indexed epoch numbers where recovery.pt should be saved.
    """
    return {e for e in range(5, total_epochs + 1, 5)}


# ---------------------------------------------------------------------------
# Property 10: Recovery checkpoint at epoch multiples of 5
# ---------------------------------------------------------------------------


class TestProperty10RecoveryCheckpointAtEpochMultiplesOf5:
    """Property 10: Recovery checkpoint at epoch multiples of 5.
    
    For any training run of E epochs, a recovery checkpoint SHALL be saved at
    epochs {5, 10, 15, ...} ∩ {1, ..., E} (using 1-indexed epoch numbers).
    
    **Validates: Requirements 7.2**
    """

    @given(
        total_epochs=st.integers(min_value=1, max_value=25),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None)
    def test_recovery_checkpoint_saved_at_multiples_of_5(
        self,
        total_epochs: int,
        seed: int,
        tmp_path_factory,
    ):
        """Feature: unified-training-loop, Property 10: Recovery checkpoint at epoch multiples of 5
        
        Generate training runs of E epochs.
        Assert recovery checkpoint saved at epochs {5, 10, 15, ...} ∩ {1, ..., E}.
        
        **Validates: Requirements 7.2**
        """
        # Create mock detector that tracks checkpoint saves
        mock_detector = MockDetectorForRecoveryCheckpoint(
            config={"num_classes": 5},
            loss_values=[1.0],  # Constant loss to avoid early stopping
        )
        
        # Create a temporary directory for this test
        tmp_path = tmp_path_factory.mktemp(f"test_recovery_{total_epochs}_{seed}")
        
        # Create minimal dataset (enough samples for train/val split)
        num_samples = 20
        create_test_dataset(tmp_path, num_samples)
        
        # Create config with specified number of epochs
        create_test_config(tmp_path, epochs=total_epochs, seed=seed)
        
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
                    # Run training
                    result = train(str(tmp_path / "config.yaml"), verbose=False)
                    
                    # Verify training completed
                    assert isinstance(result, dict), "Training should return a metrics dict"
                    
                    # Get the actual epochs where recovery.pt was saved
                    actual_recovery_epochs = mock_detector.get_recovery_checkpoint_epochs()
                    
                    # Calculate expected epochs
                    expected_epochs = expected_recovery_epochs(total_epochs)
                    
                    # Assert the sets are equal
                    assert actual_recovery_epochs == expected_epochs, (
                        f"Recovery checkpoint epochs mismatch for {total_epochs} total epochs.\n"
                        f"Expected: {sorted(expected_epochs)}\n"
                        f"Actual: {sorted(actual_recovery_epochs)}\n"
                        f"Missing: {sorted(expected_epochs - actual_recovery_epochs)}\n"
                        f"Extra: {sorted(actual_recovery_epochs - expected_epochs)}"
                    )
                    
                finally:
                    logging.disable(logging.NOTSET)

    @given(
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None)
    def test_no_recovery_checkpoint_before_epoch_5(
        self,
        seed: int,
        tmp_path_factory,
    ):
        """Feature: unified-training-loop, Property 10: No recovery checkpoint before epoch 5
        
        For training runs with fewer than 5 epochs, no recovery checkpoint should be saved.
        
        **Validates: Requirements 7.2**
        """
        # Test with 1-4 epochs (none should have recovery checkpoints)
        for total_epochs in range(1, 5):
            mock_detector = MockDetectorForRecoveryCheckpoint(
                config={"num_classes": 5},
                loss_values=[1.0],
            )
            
            tmp_path = tmp_path_factory.mktemp(f"test_no_recovery_{total_epochs}_{seed}")
            
            num_samples = 20
            create_test_dataset(tmp_path, num_samples)
            create_test_config(tmp_path, epochs=total_epochs, seed=seed)
            
            with patch("model.training.train_detection.ModelRegistry") as mock_registry:
                mock_registry.create.return_value = mock_detector
                
                with patch("model.training.train_detection.ExperimentTracker") as mock_tracker:
                    mock_tracker_instance = MagicMock()
                    mock_tracker_instance.start_run.return_value = "test_run_id"
                    mock_tracker.return_value = mock_tracker_instance
                    
                    logging.disable(logging.CRITICAL)
                    
                    try:
                        result = train(str(tmp_path / "config.yaml"), verbose=False)
                        
                        actual_recovery_epochs = mock_detector.get_recovery_checkpoint_epochs()
                        
                        assert len(actual_recovery_epochs) == 0, (
                            f"No recovery checkpoint should be saved for {total_epochs} epochs, "
                            f"but found saves at epochs: {sorted(actual_recovery_epochs)}"
                        )
                        
                    finally:
                        logging.disable(logging.NOTSET)

    @given(
        num_multiples=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None)
    def test_recovery_checkpoint_at_exact_multiples(
        self,
        num_multiples: int,
        seed: int,
        tmp_path_factory,
    ):
        """Feature: unified-training-loop, Property 10: Recovery checkpoint at exact multiples
        
        For training runs of exactly 5*N epochs, recovery checkpoints should be saved
        at epochs 5, 10, 15, ..., 5*N.
        
        **Validates: Requirements 7.2**
        """
        total_epochs = 5 * num_multiples
        
        mock_detector = MockDetectorForRecoveryCheckpoint(
            config={"num_classes": 5},
            loss_values=[1.0],
        )
        
        tmp_path = tmp_path_factory.mktemp(f"test_exact_multiples_{num_multiples}_{seed}")
        
        num_samples = 20
        create_test_dataset(tmp_path, num_samples)
        create_test_config(tmp_path, epochs=total_epochs, seed=seed)
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                mock_tracker.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                
                try:
                    result = train(str(tmp_path / "config.yaml"), verbose=False)
                    
                    actual_recovery_epochs = mock_detector.get_recovery_checkpoint_epochs()
                    expected_epochs = {5 * i for i in range(1, num_multiples + 1)}
                    
                    assert actual_recovery_epochs == expected_epochs, (
                        f"For {total_epochs} epochs (5*{num_multiples}), expected recovery "
                        f"checkpoints at {sorted(expected_epochs)}, but got {sorted(actual_recovery_epochs)}"
                    )
                    
                    # Also verify the count
                    assert len(actual_recovery_epochs) == num_multiples, (
                        f"Expected {num_multiples} recovery checkpoints, "
                        f"but got {len(actual_recovery_epochs)}"
                    )
                    
                finally:
                    logging.disable(logging.NOTSET)

    @given(
        base_epochs=st.integers(min_value=1, max_value=4),
        num_full_multiples=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None)
    def test_recovery_checkpoint_with_partial_interval(
        self,
        base_epochs: int,
        num_full_multiples: int,
        seed: int,
        tmp_path_factory,
    ):
        """Feature: unified-training-loop, Property 10: Recovery checkpoint with partial interval
        
        For training runs of 5*N + K epochs (where 1 <= K < 5), recovery checkpoints
        should be saved at epochs 5, 10, ..., 5*N (not at the final partial interval).
        
        **Validates: Requirements 7.2**
        """
        # Total epochs = 5*N + K where K is base_epochs (1-4)
        total_epochs = 5 * num_full_multiples + base_epochs
        
        mock_detector = MockDetectorForRecoveryCheckpoint(
            config={"num_classes": 5},
            loss_values=[1.0],
        )
        
        tmp_path = tmp_path_factory.mktemp(f"test_partial_{total_epochs}_{seed}")
        
        num_samples = 20
        create_test_dataset(tmp_path, num_samples)
        create_test_config(tmp_path, epochs=total_epochs, seed=seed)
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                mock_tracker.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                
                try:
                    result = train(str(tmp_path / "config.yaml"), verbose=False)
                    
                    actual_recovery_epochs = mock_detector.get_recovery_checkpoint_epochs()
                    expected_epochs = {5 * i for i in range(1, num_full_multiples + 1)}
                    
                    assert actual_recovery_epochs == expected_epochs, (
                        f"For {total_epochs} epochs (5*{num_full_multiples} + {base_epochs}), "
                        f"expected recovery checkpoints at {sorted(expected_epochs)}, "
                        f"but got {sorted(actual_recovery_epochs)}"
                    )
                    
                finally:
                    logging.disable(logging.NOTSET)

    @given(
        total_epochs=st.integers(min_value=5, max_value=20),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None)
    def test_recovery_checkpoint_overwrites_previous(
        self,
        total_epochs: int,
        seed: int,
        tmp_path_factory,
    ):
        """Feature: unified-training-loop, Property 10: Recovery checkpoint overwrites previous
        
        Each recovery checkpoint should overwrite the previous one (same file path).
        
        **Validates: Requirements 7.2**
        """
        assume(total_epochs >= 5)  # Need at least one recovery checkpoint
        
        mock_detector = MockDetectorForRecoveryCheckpoint(
            config={"num_classes": 5},
            loss_values=[1.0],
        )
        
        tmp_path = tmp_path_factory.mktemp(f"test_overwrite_{total_epochs}_{seed}")
        
        num_samples = 20
        create_test_dataset(tmp_path, num_samples)
        create_test_config(tmp_path, epochs=total_epochs, seed=seed)
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                mock_tracker.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                
                try:
                    result = train(str(tmp_path / "config.yaml"), verbose=False)
                    
                    # Get all recovery checkpoint saves
                    recovery_saves = [
                        save for save in mock_detector.checkpoint_saves
                        if save["name"] == "recovery.pt"
                    ]
                    
                    # All recovery saves should use the same path
                    if len(recovery_saves) > 1:
                        first_path = recovery_saves[0]["path"]
                        for save in recovery_saves[1:]:
                            assert save["path"] == first_path, (
                                f"Recovery checkpoints should all use the same path. "
                                f"Expected {first_path}, got {save['path']}"
                            )
                    
                finally:
                    logging.disable(logging.NOTSET)

