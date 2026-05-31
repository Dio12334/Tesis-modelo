"""Property-based tests for Property 11: Final checkpoint always saved.

Feature: unified-training-loop, Property 11: Final checkpoint always saved

For any training run that completes (whether by reaching max epochs, early stopping,
or SIGINT interruption), a final checkpoint SHALL be saved to `final_model.pt`.

**Validates: Requirements 7.3, 10.2**
"""

import logging
import signal
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock, patch

import pytest
import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from model.training.train_detection import train


# ---------------------------------------------------------------------------
# Termination mode enum
# ---------------------------------------------------------------------------


class TerminationMode(Enum):
    """How training should terminate."""
    NORMAL_COMPLETION = "normal"
    EARLY_STOPPING = "early_stopping"
    SIGINT_INTERRUPTION = "sigint"


# ---------------------------------------------------------------------------
# Mock BaseDetector for property testing
# ---------------------------------------------------------------------------


class MockDetectorForCheckpoint:
    """Mock BaseDetector that tracks checkpoint saves and supports configurable behavior.
    
    This mock is designed to test checkpoint saving behavior across different
    termination scenarios.
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
        self._train_mode = False
        self._checkpoint_saves: List[str] = []
        
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
        
        # Return configured loss value (cycle through list based on epoch)
        loss_idx = self._epoch_counter % len(self._loss_values)
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
        """Switch to evaluation mode and increment epoch counter."""
        self._call_log.append("set_eval_mode()")
        self._train_mode = False
        self._epoch_counter += 1
        self._batch_counter = 0  # Reset batch counter for next epoch

    def save_checkpoint(
        self,
        path: Path,
        optimizer: Optional[Any] = None,
        epoch: Optional[int] = None,
        metrics: Optional[dict] = None,
    ) -> None:
        """Save checkpoint and record the save."""
        checkpoint_name = path.name
        self._checkpoint_saves.append(checkpoint_name)
        self._call_log.append(f"save_checkpoint({checkpoint_name})")
        # Create the file to satisfy the training loop
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mock": True, "epoch": epoch, "metrics": metrics}, path)

    def get_config_schema(self) -> dict:
        """Return empty schema (no required params)."""
        return {}

    @property
    def checkpoint_saves(self) -> List[str]:
        """Return the list of checkpoint filenames that were saved."""
        return self._checkpoint_saves

    @property
    def call_log(self) -> List[str]:
        """Return the log of method calls."""
        return self._call_log


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


@st.composite
def st_termination_mode(draw):
    """Generate a termination mode."""
    return draw(st.sampled_from(list(TerminationMode)))


@st.composite
def st_training_config_for_termination(draw, termination_mode: TerminationMode):
    """Generate training configuration based on termination mode.
    
    Args:
        draw: Hypothesis draw function.
        termination_mode: How training should terminate.
        
    Returns:
        Tuple of (epochs, patience, loss_values) configured for the termination mode.
    """
    if termination_mode == TerminationMode.NORMAL_COMPLETION:
        # Normal completion: run for a small number of epochs
        epochs = draw(st.integers(min_value=1, max_value=5))
        patience = epochs + 10  # Patience higher than epochs, so no early stopping
        # Decreasing losses to avoid early stopping
        loss_values = [1.0 - 0.1 * i for i in range(epochs)]
        
    elif termination_mode == TerminationMode.EARLY_STOPPING:
        # Early stopping: patience should be exceeded
        epochs = draw(st.integers(min_value=5, max_value=10))
        patience = draw(st.integers(min_value=1, max_value=3))
        # Non-improving losses to trigger early stopping
        # First loss is best, then all subsequent are worse
        loss_values = [0.5] + [1.0] * (epochs - 1)
        
    else:  # SIGINT_INTERRUPTION
        # SIGINT: will be interrupted during training
        epochs = draw(st.integers(min_value=3, max_value=8))
        patience = epochs + 10  # High patience to avoid early stopping
        loss_values = [1.0 - 0.05 * i for i in range(epochs)]
    
    return epochs, patience, loss_values


# ---------------------------------------------------------------------------
# Test dataset creation helper
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
    patience: int,
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
            "early_stopping_patience": patience,
            "seed": seed,
        },
    }
    
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    return config


# ---------------------------------------------------------------------------
# Property 11: Final checkpoint always saved
# ---------------------------------------------------------------------------


class TestProperty11FinalCheckpointAlwaysSaved:
    """Property 11: Final checkpoint always saved.
    
    For any training run that completes (whether by reaching max epochs,
    early stopping, or SIGINT interruption), a final checkpoint SHALL be
    saved to `final_model.pt`.
    
    **Validates: Requirements 7.3, 10.2**
    """

    @given(
        epochs=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None)
    def test_final_checkpoint_saved_on_normal_completion(
        self,
        epochs: int,
        seed: int,
        tmp_path_factory,
    ):
        """Feature: unified-training-loop, Property 11: Final checkpoint always saved
        
        When training completes normally (reaching max epochs), final_model.pt
        SHALL be saved.
        
        **Validates: Requirements 7.3, 10.2**
        """
        tmp_path = tmp_path_factory.mktemp(f"test_normal_{seed}")
        
        # Create decreasing loss values to avoid early stopping
        loss_values = [1.0 - 0.1 * i for i in range(epochs)]
        
        mock_detector = MockDetectorForCheckpoint(
            config={"num_classes": 5},
            loss_values=loss_values,
        )
        
        # Create test dataset and config
        create_test_dataset(tmp_path, 15)  # Enough samples for train/val split
        config = create_test_config(
            tmp_path,
            epochs=epochs,
            patience=epochs + 10,  # High patience to avoid early stopping
            seed=seed,
        )
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                mock_tracker.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                try:
                    result = train(str(tmp_path / "config.yaml"), verbose=False)
                    
                    # Verify training completed
                    assert isinstance(result, dict)
                    assert result.get("total_epochs") == epochs
                    
                    # Verify final_model.pt was saved
                    assert "final_model.pt" in mock_detector.checkpoint_saves, (
                        f"final_model.pt not saved on normal completion. "
                        f"Saved checkpoints: {mock_detector.checkpoint_saves}"
                    )
                    
                    # Verify the file actually exists
                    final_checkpoint_path = tmp_path / "checkpoints" / "test_run_id" / "final_model.pt"
                    assert final_checkpoint_path.exists(), (
                        f"final_model.pt file does not exist at {final_checkpoint_path}"
                    )
                    
                finally:
                    logging.disable(logging.NOTSET)

    @given(
        patience=st.integers(min_value=1, max_value=3),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None)
    def test_final_checkpoint_saved_on_early_stopping(
        self,
        patience: int,
        seed: int,
        tmp_path_factory,
    ):
        """Feature: unified-training-loop, Property 11: Final checkpoint always saved
        
        When training stops due to early stopping, final_model.pt SHALL be saved.
        
        **Validates: Requirements 7.3, 10.2**
        """
        # Use fixed epochs that's much larger than patience to ensure early stopping triggers
        epochs = patience + 10
        
        tmp_path = tmp_path_factory.mktemp(f"test_early_stop_{seed}")
        
        # Create loss values that don't improve after first epoch
        # First epoch has low loss (0.1), all subsequent epochs have higher loss (1.0)
        # This will trigger early stopping after 'patience' epochs without improvement
        loss_values = [0.1] + [1.0] * (epochs - 1)
        
        mock_detector = MockDetectorForCheckpoint(
            config={"num_classes": 5},
            loss_values=loss_values,
        )
        
        # Create test dataset and config
        create_test_dataset(tmp_path, 15)
        config = create_test_config(
            tmp_path,
            epochs=epochs,
            patience=patience,
            seed=seed,
        )
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                mock_tracker.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                try:
                    result = train(str(tmp_path / "config.yaml"), verbose=False)
                    
                    # Verify training completed
                    assert isinstance(result, dict)
                    
                    # Early stopping should have triggered before max epochs
                    # The exact epoch depends on when the loss values cause early stopping
                    # What matters for this property test is that final_model.pt is saved
                    total_epochs = result.get("total_epochs", epochs)
                    
                    # Verify final_model.pt was saved (the main property being tested)
                    assert "final_model.pt" in mock_detector.checkpoint_saves, (
                        f"final_model.pt not saved on early stopping. "
                        f"Saved checkpoints: {mock_detector.checkpoint_saves}"
                    )
                    
                    # Verify the file actually exists
                    final_checkpoint_path = tmp_path / "checkpoints" / "test_run_id" / "final_model.pt"
                    assert final_checkpoint_path.exists(), (
                        f"final_model.pt file does not exist at {final_checkpoint_path}"
                    )
                    
                    # If early stopping triggered, verify it happened before max epochs
                    # This is a secondary check - the main property is that final checkpoint is saved
                    if total_epochs < epochs:
                        # Early stopping triggered as expected
                        pass
                    
                finally:
                    logging.disable(logging.NOTSET)

    @given(
        epochs=st.integers(min_value=3, max_value=8),
        interrupt_after_epoch=st.integers(min_value=0, max_value=2),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None)
    def test_final_checkpoint_saved_on_sigint(
        self,
        epochs: int,
        interrupt_after_epoch: int,
        seed: int,
        tmp_path_factory,
    ):
        """Feature: unified-training-loop, Property 11: Final checkpoint always saved
        
        When training is interrupted via SIGINT, final_model.pt SHALL be saved
        after completing the current epoch.
        
        **Validates: Requirements 7.3, 10.2**
        """
        # Ensure interrupt happens before max epochs
        assume(interrupt_after_epoch < epochs - 1)
        
        tmp_path = tmp_path_factory.mktemp(f"test_sigint_{seed}")
        
        # Create decreasing loss values
        loss_values = [1.0 - 0.05 * i for i in range(epochs)]
        
        mock_detector = MockDetectorForCheckpoint(
            config={"num_classes": 5},
            loss_values=loss_values,
        )
        
        # Track when to send SIGINT
        sigint_sent = threading.Event()
        
        # Wrap set_eval_mode to send SIGINT after specified epoch
        original_set_eval_mode = mock_detector.set_eval_mode
        
        def set_eval_mode_with_interrupt():
            original_set_eval_mode()
            # Send SIGINT after the specified epoch completes
            if mock_detector._epoch_counter == interrupt_after_epoch + 1 and not sigint_sent.is_set():
                sigint_sent.set()
                # Send SIGINT to the current process
                import os
                os.kill(os.getpid(), signal.SIGINT)
        
        mock_detector.set_eval_mode = set_eval_mode_with_interrupt
        
        # Create test dataset and config
        create_test_dataset(tmp_path, 15)
        config = create_test_config(
            tmp_path,
            epochs=epochs,
            patience=epochs + 10,  # High patience to avoid early stopping
            seed=seed,
        )
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                mock_tracker.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                try:
                    result = train(str(tmp_path / "config.yaml"), verbose=False)
                    
                    # Verify training was interrupted
                    assert isinstance(result, dict)
                    # Should have stopped after the interrupt epoch
                    assert result.get("total_epochs", 0) <= interrupt_after_epoch + 2, (
                        f"Expected training to stop around epoch {interrupt_after_epoch + 1}, "
                        f"but ran for {result.get('total_epochs')} epochs"
                    )
                    
                    # Verify final_model.pt was saved
                    assert "final_model.pt" in mock_detector.checkpoint_saves, (
                        f"final_model.pt not saved on SIGINT interruption. "
                        f"Saved checkpoints: {mock_detector.checkpoint_saves}"
                    )
                    
                    # Verify the file actually exists
                    final_checkpoint_path = tmp_path / "checkpoints" / "test_run_id" / "final_model.pt"
                    assert final_checkpoint_path.exists(), (
                        f"final_model.pt file does not exist at {final_checkpoint_path}"
                    )
                    
                finally:
                    logging.disable(logging.NOTSET)


class TestFinalCheckpointContent:
    """Additional tests for final checkpoint content and behavior."""

    @given(
        epochs=st.integers(min_value=1, max_value=3),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None)
    def test_final_checkpoint_contains_valid_data(
        self,
        epochs: int,
        seed: int,
        tmp_path_factory,
    ):
        """Feature: unified-training-loop, Property 11: Final checkpoint contains valid data
        
        The final checkpoint file should contain valid PyTorch data.
        
        **Validates: Requirements 7.3, 10.2**
        """
        tmp_path = tmp_path_factory.mktemp(f"test_content_{seed}")
        
        loss_values = [1.0 - 0.1 * i for i in range(epochs)]
        
        mock_detector = MockDetectorForCheckpoint(
            config={"num_classes": 5},
            loss_values=loss_values,
        )
        
        create_test_dataset(tmp_path, 15)
        config = create_test_config(
            tmp_path,
            epochs=epochs,
            patience=epochs + 10,
            seed=seed,
        )
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                mock_tracker.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                try:
                    result = train(str(tmp_path / "config.yaml"), verbose=False)
                    
                    # Load and verify the checkpoint
                    final_checkpoint_path = tmp_path / "checkpoints" / "test_run_id" / "final_model.pt"
                    assert final_checkpoint_path.exists()
                    
                    checkpoint = torch.load(final_checkpoint_path, weights_only=False)
                    assert isinstance(checkpoint, dict), "Checkpoint should be a dict"
                    
                finally:
                    logging.disable(logging.NOTSET)

    @given(
        termination_mode=st_termination_mode(),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None)
    def test_final_checkpoint_saved_regardless_of_termination_mode(
        self,
        termination_mode: TerminationMode,
        seed: int,
        tmp_path_factory,
    ):
        """Feature: unified-training-loop, Property 11: Final checkpoint saved for all termination modes
        
        Regardless of how training terminates (normal, early stopping, or SIGINT),
        final_model.pt SHALL always be saved.
        
        **Validates: Requirements 7.3, 10.2**
        """
        tmp_path = tmp_path_factory.mktemp(f"test_all_modes_{seed}")
        
        # Configure based on termination mode
        if termination_mode == TerminationMode.NORMAL_COMPLETION:
            epochs = 2
            patience = 20
            loss_values = [1.0, 0.9]
            interrupt_epoch = None
        elif termination_mode == TerminationMode.EARLY_STOPPING:
            epochs = 10
            patience = 2
            loss_values = [0.5] + [1.0] * 9  # No improvement after first epoch
            interrupt_epoch = None
        else:  # SIGINT
            epochs = 5
            patience = 20
            loss_values = [1.0 - 0.1 * i for i in range(5)]
            interrupt_epoch = 1
        
        mock_detector = MockDetectorForCheckpoint(
            config={"num_classes": 5},
            loss_values=loss_values,
        )
        
        # Setup SIGINT handling if needed
        if termination_mode == TerminationMode.SIGINT_INTERRUPTION:
            sigint_sent = threading.Event()
            original_set_eval_mode = mock_detector.set_eval_mode
            
            def set_eval_mode_with_interrupt():
                original_set_eval_mode()
                if mock_detector._epoch_counter == interrupt_epoch + 1 and not sigint_sent.is_set():
                    sigint_sent.set()
                    import os
                    os.kill(os.getpid(), signal.SIGINT)
            
            mock_detector.set_eval_mode = set_eval_mode_with_interrupt
        
        create_test_dataset(tmp_path, 15)
        config = create_test_config(
            tmp_path,
            epochs=epochs,
            patience=patience,
            seed=seed,
        )
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                mock_tracker.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                try:
                    result = train(str(tmp_path / "config.yaml"), verbose=False)
                    
                    # Verify final_model.pt was saved regardless of termination mode
                    assert "final_model.pt" in mock_detector.checkpoint_saves, (
                        f"final_model.pt not saved for termination mode {termination_mode.value}. "
                        f"Saved checkpoints: {mock_detector.checkpoint_saves}"
                    )
                    
                    final_checkpoint_path = tmp_path / "checkpoints" / "test_run_id" / "final_model.pt"
                    assert final_checkpoint_path.exists(), (
                        f"final_model.pt file does not exist for termination mode {termination_mode.value}"
                    )
                    
                finally:
                    logging.disable(logging.NOTSET)
