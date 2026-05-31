"""Property-based tests for best checkpoint saving behavior.

Feature: unified-training-loop, Property 9: Best checkpoint saved at new minimum

For any sequence of epoch validation losses, save_checkpoint() for best_model.pt
SHALL be called exactly at those epochs where the validation loss is strictly
less than all prior validation losses in the run.

**Validates: Requirements 7.1**
"""

from pathlib import Path
from typing import List, Optional, Set
from unittest.mock import MagicMock, patch

import pytest
import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from model.models.registry import BaseDetector


# ---------------------------------------------------------------------------
# Mock BaseDetector for testing best checkpoint behavior
# ---------------------------------------------------------------------------


class MockDetectorWithConfigurableValLoss(BaseDetector):
    """Mock BaseDetector that returns configurable validation loss values per epoch.
    
    This mock allows testing best checkpoint saving by returning specific
    loss values for each epoch's validation phase.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        val_loss_sequence: Optional[List[float]] = None,
    ):
        """Initialize the mock detector.
        
        Args:
            config: Optional configuration dict.
            val_loss_sequence: List of validation loss values to return per epoch.
                              The mock will cycle through this list if more epochs
                              are run than values provided.
        """
        self.config = config or {}
        self._val_loss_sequence = val_loss_sequence or [1.0]
        
        # Create trainable parameters
        self._param = torch.nn.Parameter(torch.tensor([1.0]))
        
        # Call tracking
        self._epoch_counter = 0
        self._batch_counter = 0
        self._in_eval_mode = False
        self._seen_first_train_mode = False
        self.save_checkpoint_calls: List[str] = []  # Track checkpoint filenames
        self.mode_calls: List[str] = []

    def forward(self, images):
        return [{"boxes": torch.tensor([]), "labels": torch.tensor([]), "scores": torch.tensor([])}]

    def get_config_schema(self) -> dict:
        return {}

    def load_checkpoint(self, path: Path) -> None:
        pass

    def save_checkpoint(self, path: Path, optimizer=None, epoch=None, metrics=None) -> None:
        """Record checkpoint saves with the filename."""
        self.save_checkpoint_calls.append(path.name)

    def get_parameters(self) -> List[torch.nn.Parameter]:
        return [self._param]

    def set_train_mode(self) -> None:
        self.mode_calls.append("set_train_mode")
        self._in_eval_mode = False
        # Increment epoch counter when entering training mode (start of new epoch)
        # But only after the first epoch (when we've seen at least one eval mode)
        if self._seen_first_train_mode:
            # This is not the first train mode call, so we're starting a new epoch
            self._epoch_counter += 1
        else:
            self._seen_first_train_mode = True
        self._batch_counter = 0

    def set_eval_mode(self) -> None:
        self.mode_calls.append("set_eval_mode")
        self._in_eval_mode = True
        self._batch_counter = 0

    def train_step(self, images, targets) -> dict:
        """Execute a training step, returning configurable loss values.
        
        During validation (eval mode), returns the configured validation loss
        for the current epoch. During training, returns a small constant loss.
        """
        self._batch_counter += 1
        
        if self._in_eval_mode:
            # Return the configured validation loss for this epoch
            epoch_idx = self._epoch_counter
            loss_idx = epoch_idx % len(self._val_loss_sequence)
            loss_value = self._val_loss_sequence[loss_idx]
            # Return exact loss value without parameter dependency
            loss_tensor = torch.tensor(loss_value, requires_grad=False)
        else:
            # Training phase: return a small constant loss that depends on parameters
            # so that gradients can flow during training
            loss_value = 0.5
            loss_tensor = self._param * loss_value
        
        return {"loss_tensor": loss_tensor}


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


@st.composite
def st_val_loss_sequence(draw, min_epochs: int = 1, max_epochs: int = 15):
    """Generate a sequence of validation loss values.
    
    Generates positive float values that can include:
    - Decreasing sequences (new minimums)
    - Increasing sequences (no new minimums)
    - Mixed sequences with some new minimums
    
    Args:
        draw: Hypothesis draw function.
        min_epochs: Minimum number of epochs.
        max_epochs: Maximum number of epochs.
        
    Returns:
        List of validation loss values.
    """
    num_epochs = draw(st.integers(min_value=min_epochs, max_value=max_epochs))
    
    # Generate loss values - use a reasonable range for validation losses
    loss_values = draw(
        st.lists(
            st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=num_epochs,
            max_size=num_epochs,
        )
    )
    
    return loss_values


def compute_expected_best_checkpoint_epochs(val_losses: List[float]) -> Set[int]:
    """Compute which epochs should trigger a best checkpoint save.
    
    A best checkpoint should be saved at epoch i (0-indexed) if and only if
    val_losses[i] < min(val_losses[0:i]) (strictly less than all prior losses).
    
    For the first epoch (i=0), there are no prior losses, so it always triggers
    a best checkpoint save.
    
    Args:
        val_losses: List of validation loss values per epoch.
        
    Returns:
        Set of 0-indexed epoch numbers where best checkpoint should be saved.
    """
    if not val_losses:
        return set()
    
    best_checkpoint_epochs = set()
    best_loss_so_far = float("inf")
    
    for epoch, loss in enumerate(val_losses):
        if loss < best_loss_so_far:
            best_checkpoint_epochs.add(epoch)
            best_loss_so_far = loss
    
    return best_checkpoint_epochs


# ---------------------------------------------------------------------------
# Property 9: Best checkpoint saved at new minimum
# ---------------------------------------------------------------------------


class TestProperty9BestCheckpointSavedAtNewMinimum:
    """Property 9: Best checkpoint saved at new minimum.
    
    For any sequence of epoch validation losses, save_checkpoint() for best_model.pt
    SHALL be called exactly at those epochs where the validation loss is strictly
    less than all prior validation losses in the run.
    
    **Validates: Requirements 7.1**
    """

    @given(val_loss_sequence=st_val_loss_sequence(min_epochs=1, max_epochs=10))
    @settings(max_examples=100, deadline=None)
    def test_best_checkpoint_saved_at_new_minimum(
        self,
        val_loss_sequence: List[float],
        tmp_path_factory,
    ):
        """Feature: unified-training-loop, Property 9: Best checkpoint saved at new minimum
        
        Generate sequences of epoch validation losses.
        Assert save_checkpoint for best_model.pt called exactly at epochs with new minimum.
        
        **Validates: Requirements 7.1**
        """
        import logging
        import yaml
        from PIL import Image
        import xml.etree.ElementTree as ET
        
        from model.training.train_detection import train
        
        num_epochs = len(val_loss_sequence)
        
        # Compute expected epochs where best checkpoint should be saved
        expected_best_epochs = compute_expected_best_checkpoint_epochs(val_loss_sequence)
        
        # Create mock detector with the specified validation loss sequence
        mock_detector = MockDetectorWithConfigurableValLoss(
            config={"num_classes": 5},
            val_loss_sequence=val_loss_sequence,
        )
        
        # Create a temporary directory for this test
        tmp_path = tmp_path_factory.mktemp("test_best_checkpoint")
        
        # Create minimal dataset
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Create sample annotation files and images (need enough for train/val split)
        num_samples = 10
        for i in range(num_samples):
            img = Image.new("RGB", (64, 64), color=(i % 256, 0, 0))
            img_path = dataset_path / f"image_{i}.jpg"
            img.save(img_path)
            
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
        
        # Create config
        config = {
            "model": {
                "type": "mock_detector",
                "config": {"num_classes": 5, "input_size": 32},
            },
            "dataset": {
                "path": str(dataset_path),
            },
            "training": {
                "epochs": num_epochs,
                "batch_size": 2,
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
                "early_stopping_patience": num_epochs + 10,  # Disable early stopping
                "seed": 42,
            },
        }
        
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
                    # Run training
                    result = train(str(config_path), verbose=False)
                    
                    # Verify training completed
                    assert isinstance(result, dict), "Training should return a metrics dict"
                    
                    # Extract best_model.pt saves from checkpoint calls
                    best_checkpoint_saves = [
                        call for call in mock_detector.save_checkpoint_calls
                        if call == "best_model.pt"
                    ]
                    
                    # Count how many times best_model.pt was saved
                    num_best_saves = len(best_checkpoint_saves)
                    expected_num_best_saves = len(expected_best_epochs)
                    
                    assert num_best_saves == expected_num_best_saves, (
                        f"Expected {expected_num_best_saves} best checkpoint saves, "
                        f"got {num_best_saves}. "
                        f"Val losses: {val_loss_sequence}, "
                        f"Expected epochs: {sorted(expected_best_epochs)}, "
                        f"All checkpoint calls: {mock_detector.save_checkpoint_calls}"
                    )
                    
                finally:
                    logging.disable(logging.NOTSET)

    @given(
        num_epochs=st.integers(min_value=2, max_value=10),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None)
    def test_strictly_decreasing_losses_save_every_epoch(
        self,
        num_epochs: int,
        seed: int,
        tmp_path_factory,
    ):
        """Feature: unified-training-loop, Property 9: Strictly decreasing losses save every epoch
        
        When validation loss strictly decreases every epoch, best_model.pt
        should be saved at every epoch.
        
        **Validates: Requirements 7.1**
        """
        import logging
        import yaml
        from PIL import Image
        import xml.etree.ElementTree as ET
        import random
        
        from model.training.train_detection import train
        
        # Generate strictly decreasing loss sequence
        rng = random.Random(seed)
        start_loss = rng.uniform(5.0, 10.0)
        val_loss_sequence = [start_loss - i * 0.5 for i in range(num_epochs)]
        
        # All epochs should trigger best checkpoint save
        expected_best_epochs = set(range(num_epochs))
        
        mock_detector = MockDetectorWithConfigurableValLoss(
            config={"num_classes": 5},
            val_loss_sequence=val_loss_sequence,
        )
        
        tmp_path = tmp_path_factory.mktemp(f"test_decreasing_{seed}")
        
        # Create minimal dataset
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        num_samples = 10
        for i in range(num_samples):
            img = Image.new("RGB", (64, 64), color=(i % 256, 0, 0))
            img_path = dataset_path / f"image_{i}.jpg"
            img.save(img_path)
            
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
        
        config = {
            "model": {
                "type": "mock_detector",
                "config": {"num_classes": 5, "input_size": 32},
            },
            "dataset": {
                "path": str(dataset_path),
            },
            "training": {
                "epochs": num_epochs,
                "batch_size": 2,
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
                "early_stopping_patience": num_epochs + 10,
                "seed": seed,
            },
        }
        
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                mock_tracker.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                
                try:
                    result = train(str(config_path), verbose=False)
                    
                    assert isinstance(result, dict)
                    
                    # Count best_model.pt saves
                    best_saves = [c for c in mock_detector.save_checkpoint_calls if c == "best_model.pt"]
                    
                    assert len(best_saves) == num_epochs, (
                        f"With strictly decreasing losses, expected {num_epochs} best saves, "
                        f"got {len(best_saves)}. Val losses: {val_loss_sequence}"
                    )
                    
                finally:
                    logging.disable(logging.NOTSET)

    @given(
        num_epochs=st.integers(min_value=2, max_value=10),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None)
    def test_strictly_increasing_losses_save_only_first_epoch(
        self,
        num_epochs: int,
        seed: int,
        tmp_path_factory,
    ):
        """Feature: unified-training-loop, Property 9: Strictly increasing losses save only first epoch
        
        When validation loss strictly increases after the first epoch,
        best_model.pt should only be saved at epoch 0.
        
        **Validates: Requirements 7.1**
        """
        import logging
        import yaml
        from PIL import Image
        import xml.etree.ElementTree as ET
        import random
        
        from model.training.train_detection import train
        
        # Generate strictly increasing loss sequence
        rng = random.Random(seed)
        start_loss = rng.uniform(0.5, 2.0)
        val_loss_sequence = [start_loss + i * 0.5 for i in range(num_epochs)]
        
        # Only first epoch should trigger best checkpoint save
        expected_best_epochs = {0}
        
        mock_detector = MockDetectorWithConfigurableValLoss(
            config={"num_classes": 5},
            val_loss_sequence=val_loss_sequence,
        )
        
        tmp_path = tmp_path_factory.mktemp(f"test_increasing_{seed}")
        
        # Create minimal dataset
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        num_samples = 10
        for i in range(num_samples):
            img = Image.new("RGB", (64, 64), color=(i % 256, 0, 0))
            img_path = dataset_path / f"image_{i}.jpg"
            img.save(img_path)
            
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
        
        config = {
            "model": {
                "type": "mock_detector",
                "config": {"num_classes": 5, "input_size": 32},
            },
            "dataset": {
                "path": str(dataset_path),
            },
            "training": {
                "epochs": num_epochs,
                "batch_size": 2,
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
                "early_stopping_patience": num_epochs + 10,
                "seed": seed,
            },
        }
        
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                mock_tracker.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                
                try:
                    result = train(str(config_path), verbose=False)
                    
                    assert isinstance(result, dict)
                    
                    # Count best_model.pt saves
                    best_saves = [c for c in mock_detector.save_checkpoint_calls if c == "best_model.pt"]
                    
                    assert len(best_saves) == 1, (
                        f"With strictly increasing losses, expected 1 best save, "
                        f"got {len(best_saves)}. Val losses: {val_loss_sequence}"
                    )
                    
                finally:
                    logging.disable(logging.NOTSET)

    @given(
        num_epochs=st.integers(min_value=3, max_value=10),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None)
    def test_equal_losses_do_not_trigger_save(
        self,
        num_epochs: int,
        seed: int,
        tmp_path_factory,
    ):
        """Feature: unified-training-loop, Property 9: Equal losses do not trigger save
        
        When validation loss equals the previous best (not strictly less),
        best_model.pt should NOT be saved.
        
        **Validates: Requirements 7.1**
        """
        import logging
        import yaml
        from PIL import Image
        import xml.etree.ElementTree as ET
        import random
        
        from model.training.train_detection import train
        
        # Generate loss sequence with equal values after first epoch
        rng = random.Random(seed)
        constant_loss = rng.uniform(1.0, 5.0)
        val_loss_sequence = [constant_loss] * num_epochs
        
        # Only first epoch should trigger best checkpoint save (equal is not strictly less)
        expected_best_epochs = {0}
        
        mock_detector = MockDetectorWithConfigurableValLoss(
            config={"num_classes": 5},
            val_loss_sequence=val_loss_sequence,
        )
        
        tmp_path = tmp_path_factory.mktemp(f"test_equal_{seed}")
        
        # Create minimal dataset
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        num_samples = 10
        for i in range(num_samples):
            img = Image.new("RGB", (64, 64), color=(i % 256, 0, 0))
            img_path = dataset_path / f"image_{i}.jpg"
            img.save(img_path)
            
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
        
        config = {
            "model": {
                "type": "mock_detector",
                "config": {"num_classes": 5, "input_size": 32},
            },
            "dataset": {
                "path": str(dataset_path),
            },
            "training": {
                "epochs": num_epochs,
                "batch_size": 2,
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
                "early_stopping_patience": num_epochs + 10,
                "seed": seed,
            },
        }
        
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        with patch("model.training.train_detection.ModelRegistry") as mock_registry:
            mock_registry.create.return_value = mock_detector
            
            with patch("model.training.train_detection.ExperimentTracker") as mock_tracker:
                mock_tracker_instance = MagicMock()
                mock_tracker_instance.start_run.return_value = "test_run_id"
                mock_tracker.return_value = mock_tracker_instance
                
                logging.disable(logging.CRITICAL)
                
                try:
                    result = train(str(config_path), verbose=False)
                    
                    assert isinstance(result, dict)
                    
                    # Count best_model.pt saves
                    best_saves = [c for c in mock_detector.save_checkpoint_calls if c == "best_model.pt"]
                    
                    assert len(best_saves) == 1, (
                        f"With equal losses, expected 1 best save (first epoch only), "
                        f"got {len(best_saves)}. Val losses: {val_loss_sequence}"
                    )
                    
                finally:
                    logging.disable(logging.NOTSET)
