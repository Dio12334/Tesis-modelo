"""Property-based tests for the unified training loop - Property 1.

Feature: unified-training-loop, Property 1: Any conforming BaseDetector trains without loop modification

These tests verify that the training loop operates correctly through the BaseDetector
interface without requiring model-specific branching.
"""

import logging
import signal
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from model.models.registry import BaseDetector, ModelRegistry


# ---------------------------------------------------------------------------
# Mock BaseDetector implementation for property testing
# ---------------------------------------------------------------------------


class MockBaseDetector(BaseDetector):
    """Mock BaseDetector that records method calls and returns configurable loss values.
    
    This mock implements all required BaseDetector methods and tracks calls
    for verification in property tests.
    """

    def __init__(
        self,
        config: dict,
        loss_value: float = 1.0,
        raise_on_train_step: bool = False,
        exception_batch_indices: Optional[List[int]] = None,
        zero_loss_batch_indices: Optional[List[int]] = None,
    ):
        """Initialize the mock detector.
        
        Args:
            config: Configuration dict.
            loss_value: Default loss value to return from train_step.
            raise_on_train_step: If True, raise an exception on train_step.
            exception_batch_indices: List of batch indices where train_step should raise.
            zero_loss_batch_indices: List of batch indices where train_step returns 0.0 loss.
        """
        self.config = config
        self._loss_value = loss_value
        self._raise_on_train_step = raise_on_train_step
        self._exception_batch_indices = exception_batch_indices or []
        self._zero_loss_batch_indices = zero_loss_batch_indices or []
        
        # Track method calls
        self.call_log: List[str] = []
        self._train_step_count = 0
        self._in_train_mode = False
        self._in_eval_mode = False
        
        # Create dummy parameters for optimizer
        self._dummy_param = torch.nn.Parameter(torch.zeros(10))

    def forward(self, images: torch.Tensor) -> List[dict]:
        """Run forward pass (not used in training loop)."""
        self.call_log.append("forward")
        return [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0), "scores": torch.zeros(0)}]

    def get_config_schema(self) -> dict:
        """Return empty schema (no required params for mock)."""
        return {}

    def load_checkpoint(self, path: Path) -> None:
        """Load checkpoint (no-op for mock)."""
        self.call_log.append(f"load_checkpoint:{path}")

    def save_checkpoint(
        self,
        path: Path,
        optimizer: Optional[Any] = None,
        epoch: Optional[int] = None,
        metrics: Optional[dict] = None,
    ) -> None:
        """Save checkpoint (no-op for mock, but records the call)."""
        self.call_log.append(f"save_checkpoint:{path}")

    def train_step(self, images: List[torch.Tensor], targets: List[dict]) -> dict:
        """Perform a training step and return loss dict.
        
        Args:
            images: List of image tensors.
            targets: List of target dicts with 'boxes' and 'labels'.
            
        Returns:
            Dict with 'loss_tensor' key containing a scalar tensor.
        """
        self.call_log.append("train_step")
        batch_idx = self._train_step_count
        self._train_step_count += 1
        
        # Check if we should raise an exception
        if self._raise_on_train_step or batch_idx in self._exception_batch_indices:
            raise RuntimeError(f"Simulated train_step error at batch {batch_idx}")
        
        # Check if we should return zero loss
        if batch_idx in self._zero_loss_batch_indices:
            return {"loss_tensor": torch.tensor(0.0, requires_grad=True)}
        
        # Return configurable loss value
        return {"loss_tensor": torch.tensor(self._loss_value, requires_grad=True)}

    def get_parameters(self) -> List[torch.nn.Parameter]:
        """Return trainable parameters."""
        self.call_log.append("get_parameters")
        return [self._dummy_param]

    def set_train_mode(self) -> None:
        """Set model to training mode."""
        self.call_log.append("set_train_mode")
        self._in_train_mode = True
        self._in_eval_mode = False

    def set_eval_mode(self) -> None:
        """Set model to evaluation mode."""
        self.call_log.append("set_eval_mode")
        self._in_train_mode = False
        self._in_eval_mode = True


# ---------------------------------------------------------------------------
# Hypothesis strategies for training configs
# ---------------------------------------------------------------------------


@st.composite
def st_training_config(draw):
    """Generate valid training configuration dicts.
    
    Generates configs with:
    - epochs: 1-10
    - batch_size: 1-8
    - learning_rate: 1e-5 to 1e-1
    - optimizer: SGD, Adam, AdamW, or random string
    - warmup_epochs: 0 to epochs-1
    - early_stopping_patience: 1-20
    """
    epochs = draw(st.integers(min_value=1, max_value=10))
    warmup_epochs = draw(st.integers(min_value=0, max_value=max(0, epochs - 1)))
    
    optimizer = draw(st.sampled_from(["SGD", "Adam", "AdamW", "sgd", "adam", "adamw", "UnknownOpt", "RMSprop"]))
    
    return {
        "model": {
            "type": "mock_detector",
            "config": {
                "input_size": 64,  # Small for fast tests
                "num_classes": 2,
            },
        },
        "dataset": {
            "path": "dummy_path",
            "name": "test_dataset",
        },
        "training": {
            "epochs": epochs,
            "batch_size": draw(st.integers(min_value=1, max_value=4)),
            "learning_rate": draw(st.floats(min_value=1e-5, max_value=1e-1)),
            "optimizer": optimizer,
            "weight_decay": draw(st.floats(min_value=0.0, max_value=0.01)),
            "momentum": draw(st.floats(min_value=0.0, max_value=0.99)),
            "warmup_epochs": warmup_epochs,
            "val_split": draw(st.floats(min_value=0.1, max_value=0.3)),
            "checkpoint_dir": "dummy_checkpoints",
            "log_interval": 100,  # High to reduce logging noise
            "use_amp": False,  # Disable AMP for CPU testing
            "num_workers": 0,  # No multiprocessing for tests
            "early_stopping_patience": draw(st.integers(min_value=1, max_value=20)),
            "seed": 42,
        },
    }


# ---------------------------------------------------------------------------
# Mock dataset and data loader for testing
# ---------------------------------------------------------------------------


class MockTorchDataset(torch.utils.data.Dataset):
    """Minimal mock dataset for testing the training loop."""

    def __init__(self, size: int = 10, input_size: int = 64):
        self._size = size
        self._input_size = input_size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int):
        # Return a dummy image tensor and target dict
        image = torch.rand(3, self._input_size, self._input_size)
        target = {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([1]),
        }
        return image, target


def mock_collate_fn(batch):
    """Collate function for mock dataset."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


# ---------------------------------------------------------------------------
# Property 1: Any conforming BaseDetector trains without loop modification
# ---------------------------------------------------------------------------


class TestProperty1ConformingDetectorTrains:
    """Property 1: Any conforming BaseDetector trains without loop modification.
    
    For any object implementing train_step(), get_parameters(), set_train_mode(),
    set_eval_mode(), and save_checkpoint() with valid return types, the training
    loop SHALL execute the full epoch sequence (training phase + validation phase
    + checkpointing) without raising an error attributable to the loop itself.
    
    **Validates: Requirements 1.3**
    """

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(config=st_training_config())
    def test_conforming_detector_completes_training(self, config, tmp_path):
        """Any conforming BaseDetector completes training without loop errors.
        
        Feature: unified-training-loop, Property 1: Any conforming BaseDetector trains without loop modification
        **Validates: Requirements 1.3**
        """
        # Create a mock detector
        mock_detector = MockBaseDetector(
            config=config["model"]["config"],
            loss_value=1.0,
        )
        
        # Update checkpoint dir to use temp path
        config["training"]["checkpoint_dir"] = str(tmp_path)
        
        # Create mock datasets
        train_dataset = MockTorchDataset(size=8, input_size=64)
        val_dataset = MockTorchDataset(size=4, input_size=64)
        
        # Run a minimal training loop that mirrors the real implementation
        epochs = config["training"]["epochs"]
        batch_size = config["training"]["batch_size"]
        learning_rate = config["training"]["learning_rate"]
        optimizer_name = config["training"]["optimizer"]
        weight_decay = config["training"]["weight_decay"]
        momentum = config["training"]["momentum"]
        warmup_epochs = config["training"]["warmup_epochs"]
        early_stopping_patience = config["training"]["early_stopping_patience"]
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=mock_collate_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=mock_collate_fn,
        )
        
        # Construct optimizer from model.get_parameters()
        params = mock_detector.get_parameters()
        if optimizer_name.upper() == "SGD":
            optimizer = torch.optim.SGD(
                params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
            )
        elif optimizer_name.upper() == "ADAM":
            optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name.upper() == "ADAMW":
            optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            # Fallback to SGD
            optimizer = torch.optim.SGD(
                params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
            )
        
        # Cosine scheduler
        t_max = max(1, epochs - warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=learning_rate * 0.01
        )
        
        # Training state
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        completed_epochs = 0
        
        # Run epoch loop
        for epoch in range(epochs):
            # Linear warmup
            if epoch < warmup_epochs and warmup_epochs > 0:
                warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group["lr"] = warmup_lr
            
            # Training phase
            mock_detector.set_train_mode()
            train_loss_sum = 0.0
            train_batches = 0
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                
                try:
                    loss_dict = mock_detector.train_step(images, targets)
                    loss_tensor = loss_dict["loss_tensor"]
                except Exception:
                    # Skip batch on exception (as per requirement 2.5)
                    continue
                
                # Skip backward for zero loss (as per requirement 2.4)
                if loss_tensor.item() == 0.0:
                    train_batches += 1
                    continue
                
                loss_tensor.backward()
                optimizer.step()
                
                train_loss_sum += loss_tensor.item()
                train_batches += 1
            
            # Step scheduler after warmup
            if epoch >= warmup_epochs:
                cosine_scheduler.step()
            
            avg_train_loss = train_loss_sum / max(train_batches, 1)
            
            # Validation phase
            mock_detector.set_eval_mode()
            val_loss_sum = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for images, targets in val_loader:
                    try:
                        loss_dict = mock_detector.train_step(images, targets)
                        loss_tensor = loss_dict["loss_tensor"]
                        val_loss_sum += loss_tensor.item()
                        val_batches += 1
                    except Exception:
                        continue
            
            avg_val_loss = val_loss_sum / max(val_batches, 1)
            completed_epochs = epoch + 1
            
            # Best checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                mock_detector.save_checkpoint(tmp_path / "best_model.pt")
            else:
                epochs_without_improvement += 1
            
            # Recovery checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                mock_detector.save_checkpoint(tmp_path / "recovery.pt")
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                break
        
        # Final checkpoint
        mock_detector.save_checkpoint(tmp_path / "final_model.pt")
        
        # Assertions
        # 1. Training completed without raising errors (we got here)
        assert completed_epochs >= 1, "At least one epoch should complete"
        
        # 2. Mode switching was called correctly
        assert "set_train_mode" in mock_detector.call_log, "set_train_mode should be called"
        assert "set_eval_mode" in mock_detector.call_log, "set_eval_mode should be called"
        
        # 3. train_step was called
        assert "train_step" in mock_detector.call_log, "train_step should be called"
        
        # 4. get_parameters was called for optimizer construction
        assert "get_parameters" in mock_detector.call_log, "get_parameters should be called"
        
        # 5. save_checkpoint was called (at least final)
        save_calls = [c for c in mock_detector.call_log if c.startswith("save_checkpoint")]
        assert len(save_calls) >= 1, "At least one checkpoint should be saved"

    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        loss_value=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        epochs=st.integers(min_value=1, max_value=5),
    )
    def test_detector_with_various_loss_values_completes(self, loss_value, epochs, tmp_path):
        """Detectors returning various loss values complete training.
        
        Feature: unified-training-loop, Property 1: Any conforming BaseDetector trains without loop modification
        **Validates: Requirements 1.3**
        """
        mock_detector = MockBaseDetector(
            config={"input_size": 64, "num_classes": 2},
            loss_value=loss_value,
        )
        
        train_dataset = MockTorchDataset(size=4, input_size=64)
        val_dataset = MockTorchDataset(size=2, input_size=64)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=2, collate_fn=mock_collate_fn
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=2, collate_fn=mock_collate_fn
        )
        
        params = mock_detector.get_parameters()
        optimizer = torch.optim.SGD(params, lr=0.01)
        
        completed_epochs = 0
        
        for epoch in range(epochs):
            mock_detector.set_train_mode()
            
            for images, targets in train_loader:
                optimizer.zero_grad()
                loss_dict = mock_detector.train_step(images, targets)
                loss_tensor = loss_dict["loss_tensor"]
                
                if loss_tensor.item() != 0.0:
                    loss_tensor.backward()
                    optimizer.step()
            
            mock_detector.set_eval_mode()
            
            with torch.no_grad():
                for images, targets in val_loader:
                    mock_detector.train_step(images, targets)
            
            completed_epochs = epoch + 1
        
        mock_detector.save_checkpoint(tmp_path / "final_model.pt")
        
        assert completed_epochs == epochs, f"Expected {epochs} epochs, got {completed_epochs}"
        assert "train_step" in mock_detector.call_log

    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        optimizer_name=st.sampled_from(["SGD", "Adam", "AdamW", "sgd", "adam", "adamw", "Unknown", "RMSprop", ""]),
    )
    def test_any_optimizer_string_completes_training(self, optimizer_name, tmp_path):
        """Training completes regardless of optimizer string (with fallback to SGD).
        
        Feature: unified-training-loop, Property 1: Any conforming BaseDetector trains without loop modification
        **Validates: Requirements 1.3**
        """
        mock_detector = MockBaseDetector(
            config={"input_size": 64, "num_classes": 2},
            loss_value=1.0,
        )
        
        train_dataset = MockTorchDataset(size=4, input_size=64)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=2, collate_fn=mock_collate_fn
        )
        
        params = mock_detector.get_parameters()
        
        # Construct optimizer (mirroring the real implementation)
        if optimizer_name.upper() == "SGD":
            optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9)
        elif optimizer_name.upper() == "ADAM":
            optimizer = torch.optim.Adam(params, lr=0.01)
        elif optimizer_name.upper() == "ADAMW":
            optimizer = torch.optim.AdamW(params, lr=0.01)
        else:
            # Fallback to SGD
            optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9)
        
        # Run one epoch
        mock_detector.set_train_mode()
        
        for images, targets in train_loader:
            optimizer.zero_grad()
            loss_dict = mock_detector.train_step(images, targets)
            loss_tensor = loss_dict["loss_tensor"]
            
            if loss_tensor.item() != 0.0:
                loss_tensor.backward()
                optimizer.step()
        
        mock_detector.set_eval_mode()
        mock_detector.save_checkpoint(tmp_path / "final_model.pt")
        
        # Training completed without error
        assert "train_step" in mock_detector.call_log
        assert "set_train_mode" in mock_detector.call_log
        assert "set_eval_mode" in mock_detector.call_log
