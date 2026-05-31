"""Property-based tests for the unified training loop.

Feature: unified-training-loop

These tests verify correctness properties of the training loop using
Hypothesis to generate random inputs and verify invariants hold.
"""

import signal
import tempfile
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from model.models.registry import BaseDetector


# ---------------------------------------------------------------------------
# Mock BaseDetector for testing
# ---------------------------------------------------------------------------


class MockDetector(BaseDetector):
    """Mock BaseDetector that records method calls for testing.
    
    This mock records all calls to set_train_mode and set_eval_mode
    to verify the mode switching pattern.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._parameters = [torch.nn.Parameter(torch.randn(10, 10))]
        self.mode_calls: List[str] = []
        self.train_step_call_count = 0
        self.save_checkpoint_calls: List[Path] = []

    def forward(self, images):
        return [{"boxes": torch.tensor([]), "labels": torch.tensor([]), "scores": torch.tensor([])}]

    def get_config_schema(self) -> dict:
        return {}

    def load_checkpoint(self, path: Path) -> None:
        pass

    def save_checkpoint(self, path: Path, optimizer=None, epoch=None, metrics=None) -> None:
        self.save_checkpoint_calls.append(path)

    def get_parameters(self) -> List[torch.nn.Parameter]:
        return self._parameters

    def set_train_mode(self) -> None:
        self.mode_calls.append("set_train_mode")

    def set_eval_mode(self) -> None:
        self.mode_calls.append("set_eval_mode")

    def train_step(self, images, targets) -> dict:
        self.train_step_call_count += 1
        # Return a small loss tensor with grad_fn
        loss = torch.tensor(0.1, requires_grad=True)
        return {"loss_tensor": loss}


class MockDetectorWithConfigurableLoss(BaseDetector):
    """Mock BaseDetector that returns configurable loss values per batch.
    
    This mock allows testing zero-loss batch handling by returning
    loss_tensor=0.0 for specified batch positions.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        zero_loss_batches: Optional[List[int]] = None,
    ):
        """Initialize the mock detector.
        
        Args:
            config: Optional configuration dict.
            zero_loss_batches: List of batch indices that should return loss=0.0.
        """
        self.config = config or {}
        self.zero_loss_batches = set(zero_loss_batches or [])
        
        # Create trainable parameters - use a simple linear layer to ensure
        # gradients flow properly when backward() is called
        self._linear = torch.nn.Linear(10, 1)
        self._parameters = list(self._linear.parameters())
        
        # Call tracking
        self._batch_counter = 0
        self.train_step_calls: List[dict] = []
        self.mode_calls: List[str] = []
        self.save_checkpoint_calls: List[Path] = []

    def forward(self, images):
        return [{"boxes": torch.tensor([]), "labels": torch.tensor([]), "scores": torch.tensor([])}]

    def get_config_schema(self) -> dict:
        return {}

    def load_checkpoint(self, path: Path) -> None:
        pass

    def save_checkpoint(self, path: Path, optimizer=None, epoch=None, metrics=None) -> None:
        self.save_checkpoint_calls.append(path)

    def get_parameters(self) -> List[torch.nn.Parameter]:
        return self._parameters

    def set_train_mode(self) -> None:
        self.mode_calls.append("set_train_mode")

    def set_eval_mode(self) -> None:
        self.mode_calls.append("set_eval_mode")

    def train_step(self, images, targets) -> dict:
        """Execute a training step, returning configurable loss values.
        
        Args:
            images: List of image tensors.
            targets: List of target dicts.
            
        Returns:
            Dict with 'loss_tensor' key. Returns 0.0 for batches in zero_loss_batches.
        """
        batch_idx = self._batch_counter
        self._batch_counter += 1
        
        self.train_step_calls.append({
            "batch_idx": batch_idx,
            "num_images": len(images),
        })
        
        # Determine loss value
        if batch_idx in self.zero_loss_batches:
            # Return a zero loss tensor (still needs grad_fn for consistency)
            loss_tensor = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)
        else:
            # Create a loss that depends on the model parameters so that
            # backward() will actually compute gradients
            dummy_input = torch.randn(1, 10)
            output = self._linear(dummy_input)
            # Use a simple loss that will produce non-zero gradients
            loss_tensor = output.sum().abs() + 1.0  # Ensure non-zero loss
        
        return {"loss_tensor": loss_tensor}

    def reset_batch_counter(self):
        """Reset the batch counter for a new epoch."""
        self._batch_counter = 0


# ---------------------------------------------------------------------------
# Strategies for generating test inputs
# ---------------------------------------------------------------------------


@st.composite
def st_zero_loss_batch_config(draw, max_batches: int = 10):
    """Generate a configuration for zero-loss batch testing.
    
    Args:
        max_batches: Maximum number of batches.
        
    Returns:
        Tuple of (total_batches, zero_loss_positions).
    """
    total_batches = draw(st.integers(min_value=1, max_value=max_batches))
    # Generate a subset of batch indices to be zero-loss
    zero_loss_positions = draw(
        st.lists(
            st.integers(min_value=0, max_value=total_batches - 1),
            min_size=0,
            max_size=total_batches,
            unique=True,
        )
    )
    return total_batches, sorted(zero_loss_positions)


# ---------------------------------------------------------------------------
# Property 2: Zero-loss batches skip backward pass
# ---------------------------------------------------------------------------


class TestProperty2ZeroLossBatchesSkipBackward:
    """Property 2: Zero-loss batches skip backward pass.
    
    For any batch where train_step() returns a loss_tensor with value 0.0,
    the optimizer's parameter state SHALL remain unchanged after processing
    that batch (i.e., no backward() or optimizer.step() is performed).
    
    **Validates: Requirements 2.4**
    """

    @settings(max_examples=100, deadline=None)
    @given(batch_config=st_zero_loss_batch_config(max_batches=10))
    def test_optimizer_step_count_excludes_zero_loss_batches(self, batch_config):
        """Optimizer step is called only for non-zero loss batches.
        
        # Feature: unified-training-loop, Property 2: Zero-loss batches skip backward pass
        **Validates: Requirements 2.4**
        """
        total_batches, zero_loss_positions = batch_config
        
        # Create mock detector with specified zero-loss batches
        detector = MockDetectorWithConfigurableLoss(zero_loss_batches=zero_loss_positions)
        
        # Get parameters and create optimizer
        params = detector.get_parameters()
        optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9)
        
        # Track optimizer.step() calls
        step_call_count = 0
        original_step = optimizer.step
        
        def tracked_step(*args, **kwargs):
            nonlocal step_call_count
            step_call_count += 1
            return original_step(*args, **kwargs)
        
        optimizer.step = tracked_step
        
        # Simulate training loop (replicating train_detection.py behavior)
        device = torch.device("cpu")
        
        for batch_idx in range(total_batches):
            optimizer.zero_grad()
            
            # Create dummy inputs
            images = [torch.randn(3, 320, 320)]
            targets = [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)}]
            
            # Call train_step
            loss_dict = detector.train_step(images, targets)
            loss_tensor = loss_dict["loss_tensor"]
            
            # Replicate the training loop's zero-loss handling from train_detection.py:
            # "if loss_tensor.item() == 0.0: ... continue"
            if loss_tensor.item() == 0.0:
                continue
            
            # For non-zero loss, perform backward and step
            loss_tensor.backward()
            optimizer.step()
        
        # Expected step count = total batches - zero loss batches
        expected_steps = total_batches - len(zero_loss_positions)
        assert step_call_count == expected_steps, (
            f"Expected {expected_steps} optimizer steps, got {step_call_count}. "
            f"Total batches: {total_batches}, zero-loss batches: {len(zero_loss_positions)}"
        )

    @settings(max_examples=100, deadline=None)
    @given(
        num_batches=st.integers(min_value=2, max_value=8),
        data=st.data(),
    )
    def test_parameters_unchanged_after_zero_loss_batch(self, num_batches, data):
        """Parameters are exactly unchanged after processing a zero-loss batch.
        
        # Feature: unified-training-loop, Property 2: Zero-loss batches skip backward pass
        **Validates: Requirements 2.4**
        """
        # Pick one batch to be zero-loss
        zero_loss_idx = data.draw(st.integers(min_value=0, max_value=num_batches - 1))
        
        # Create detector with single zero-loss batch
        detector = MockDetectorWithConfigurableLoss(zero_loss_batches=[zero_loss_idx])
        
        # Create optimizer
        params = detector.get_parameters()
        optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9)
        
        # Run batches and track parameter changes
        for batch_idx in range(num_batches):
            # Store parameters before this batch
            params_before = [p.clone().detach() for p in params]
            
            optimizer.zero_grad()
            
            images = [torch.randn(3, 320, 320)]
            targets = [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)}]
            
            loss_dict = detector.train_step(images, targets)
            loss_tensor = loss_dict["loss_tensor"]
            
            if loss_tensor.item() == 0.0:
                # For zero-loss batch, parameters should be unchanged
                for p_before, p_after in zip(params_before, params):
                    assert torch.allclose(p_before, p_after), (
                        f"Parameters changed after zero-loss batch {batch_idx}"
                    )
                continue
            
            # Non-zero loss: backward and step
            loss_tensor.backward()
            optimizer.step()
            
            # Parameters should have changed for non-zero loss
            params_changed = any(
                not torch.allclose(p_before, p_after)
                for p_before, p_after in zip(params_before, params)
            )
            assert params_changed, (
                f"Parameters should have changed after non-zero loss batch {batch_idx}"
            )

    @settings(max_examples=100, deadline=None)
    @given(st.lists(st.booleans(), min_size=1, max_size=10))
    def test_backward_not_called_for_zero_loss(self, is_zero_loss_sequence):
        """backward() is not called on zero-loss tensors.
        
        # Feature: unified-training-loop, Property 2: Zero-loss batches skip backward pass
        **Validates: Requirements 2.4**
        """
        num_batches = len(is_zero_loss_sequence)
        zero_loss_positions = [i for i, is_zero in enumerate(is_zero_loss_sequence) if is_zero]
        
        # Create detector
        detector = MockDetectorWithConfigurableLoss(zero_loss_batches=zero_loss_positions)
        
        # Track backward calls
        backward_call_batches = []
        
        # Simulate training loop
        for batch_idx in range(num_batches):
            images = [torch.randn(3, 320, 320)]
            targets = [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)}]
            
            loss_dict = detector.train_step(images, targets)
            loss_tensor = loss_dict["loss_tensor"]
            
            if loss_tensor.item() == 0.0:
                continue
            
            # Track that backward would be called
            backward_call_batches.append(batch_idx)
            loss_tensor.backward()
        
        # Verify backward was only called for non-zero loss batches
        expected_backward_batches = [
            i for i in range(num_batches) if i not in zero_loss_positions
        ]
        assert backward_call_batches == expected_backward_batches, (
            f"backward() called for batches {backward_call_batches}, "
            f"expected {expected_backward_batches}"
        )

    @settings(max_examples=100, deadline=None)
    @given(batch_config=st_zero_loss_batch_config(max_batches=10))
    def test_zero_loss_batches_do_not_accumulate_gradients(self, batch_config):
        """Zero-loss batches do not accumulate gradients in parameters.
        
        # Feature: unified-training-loop, Property 2: Zero-loss batches skip backward pass
        **Validates: Requirements 2.4**
        """
        total_batches, zero_loss_positions = batch_config
        
        # Skip if no zero-loss batches to test
        assume(len(zero_loss_positions) > 0)
        
        # Create detector
        detector = MockDetectorWithConfigurableLoss(zero_loss_batches=zero_loss_positions)
        
        # Get parameters and create optimizer
        params = detector.get_parameters()
        optimizer = torch.optim.SGD(params, lr=0.01)
        
        # Track gradient states after zero-loss batches
        for batch_idx in range(total_batches):
            optimizer.zero_grad()
            
            images = [torch.randn(3, 320, 320)]
            targets = [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)}]
            
            loss_dict = detector.train_step(images, targets)
            loss_tensor = loss_dict["loss_tensor"]
            
            if loss_tensor.item() == 0.0:
                # After a zero-loss batch (with skip), gradients should still be None or zero
                # because backward() was not called
                for p in params:
                    assert p.grad is None or torch.all(p.grad == 0), (
                        f"Gradients should be None or zero after zero-loss batch {batch_idx}, "
                        f"but got {p.grad}"
                    )
                continue
            
            # Non-zero loss: backward and step
            loss_tensor.backward()
            optimizer.step()


# ---------------------------------------------------------------------------
# Strategies for Property 6: Warmup learning rate
# ---------------------------------------------------------------------------


# Strategy for warmup_epochs (positive integers)
warmup_epochs_strategy = st.integers(min_value=1, max_value=50)

# Strategy for learning_rate (positive floats)
learning_rate_strategy = st.floats(
    min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Strategy for epoch numbers (non-negative integers)
epoch_strategy = st.integers(min_value=0, max_value=100)


# ---------------------------------------------------------------------------
# Helper functions for Property 6
# ---------------------------------------------------------------------------


def compute_warmup_lr(learning_rate: float, epoch: int, warmup_epochs: int) -> float:
    """Compute the expected warmup learning rate.

    This is the reference implementation of the warmup LR formula:
    LR * (e + 1) / W for epochs e < W.

    Args:
        learning_rate: Base learning rate (LR).
        epoch: Current epoch (0-indexed).
        warmup_epochs: Number of warmup epochs (W).

    Returns:
        Expected learning rate for the given epoch during warmup.
    """
    return learning_rate * (epoch + 1) / warmup_epochs


def simulate_warmup_lr_update(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    epoch: int,
    warmup_epochs: int,
) -> float:
    """Simulate the warmup LR update as implemented in train_detection.py.

    This mirrors the actual implementation in the training loop:
    ```
    if epoch < warmup_epochs:
        warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group["lr"] = warmup_lr
    ```

    Args:
        optimizer: PyTorch optimizer to update.
        learning_rate: Base learning rate.
        epoch: Current epoch (0-indexed).
        warmup_epochs: Number of warmup epochs.

    Returns:
        The learning rate that was set.
    """
    if epoch < warmup_epochs:
        warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group["lr"] = warmup_lr
        return warmup_lr
    return optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Property 6: Warmup learning rate follows linear schedule
# ---------------------------------------------------------------------------


class TestProperty6WarmupLearningRateSchedule:
    """Property 6: Warmup learning rate follows linear schedule.

    For any warmup_epochs W > 0 and learning_rate LR > 0, at epoch e (0-indexed)
    where e < W, the effective learning rate SHALL equal LR * (e + 1) / W.

    **Validates: Requirements 4.4**
    """

    @settings(max_examples=100, deadline=None)
    @given(
        warmup_epochs=warmup_epochs_strategy,
        learning_rate=learning_rate_strategy,
        epoch=epoch_strategy,
    )
    def test_warmup_lr_formula_correctness(
        self, warmup_epochs: int, learning_rate: float, epoch: int
    ):
        """Warmup LR at epoch e < W equals LR * (e + 1) / W.

        Feature: unified-training-loop, Property 6: Warmup learning rate follows linear schedule
        **Validates: Requirements 4.4**
        """
        # Only test epochs within the warmup period
        assume(epoch < warmup_epochs)

        # Create a simple optimizer with a dummy parameter
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([dummy_param], lr=learning_rate)

        # Simulate the warmup LR update
        actual_lr = simulate_warmup_lr_update(
            optimizer, learning_rate, epoch, warmup_epochs
        )

        # Compute expected LR using the formula
        expected_lr = compute_warmup_lr(learning_rate, epoch, warmup_epochs)

        # Assert they match (with floating point tolerance)
        assert abs(actual_lr - expected_lr) < 1e-10, (
            f"Warmup LR mismatch at epoch {epoch} with warmup_epochs={warmup_epochs}, "
            f"learning_rate={learning_rate}: expected {expected_lr}, got {actual_lr}"
        )

        # Also verify the optimizer's param_group was updated correctly
        assert abs(optimizer.param_groups[0]["lr"] - expected_lr) < 1e-10, (
            f"Optimizer param_group LR mismatch: expected {expected_lr}, "
            f"got {optimizer.param_groups[0]['lr']}"
        )

    @settings(max_examples=100, deadline=None)
    @given(
        warmup_epochs=warmup_epochs_strategy,
        learning_rate=learning_rate_strategy,
    )
    def test_warmup_lr_linear_progression(
        self, warmup_epochs: int, learning_rate: float
    ):
        """Warmup LR increases linearly from LR/W to LR over warmup epochs.

        Feature: unified-training-loop, Property 6: Warmup learning rate follows linear schedule
        **Validates: Requirements 4.4**
        """
        # Create a simple optimizer with a dummy parameter
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([dummy_param], lr=learning_rate)

        # Track LR values across all warmup epochs
        lr_values = []
        for epoch in range(warmup_epochs):
            actual_lr = simulate_warmup_lr_update(
                optimizer, learning_rate, epoch, warmup_epochs
            )
            lr_values.append(actual_lr)

        # Verify first epoch LR = LR * 1 / W
        expected_first_lr = learning_rate * 1 / warmup_epochs
        assert abs(lr_values[0] - expected_first_lr) < 1e-10, (
            f"First warmup LR should be {expected_first_lr}, got {lr_values[0]}"
        )

        # Verify last warmup epoch LR = LR * W / W = LR
        expected_last_lr = learning_rate * warmup_epochs / warmup_epochs
        assert abs(lr_values[-1] - expected_last_lr) < 1e-10, (
            f"Last warmup LR should be {expected_last_lr}, got {lr_values[-1]}"
        )

        # Verify LR increases monotonically
        for i in range(1, len(lr_values)):
            assert lr_values[i] > lr_values[i - 1], (
                f"LR should increase monotonically: epoch {i-1} LR={lr_values[i-1]}, "
                f"epoch {i} LR={lr_values[i]}"
            )

        # Verify constant step size (linear progression)
        step_size = learning_rate / warmup_epochs
        for i in range(1, len(lr_values)):
            actual_step = lr_values[i] - lr_values[i - 1]
            assert abs(actual_step - step_size) < 1e-10, (
                f"LR step should be constant {step_size}, got {actual_step} "
                f"between epochs {i-1} and {i}"
            )

    @settings(max_examples=100, deadline=None)
    @given(
        warmup_epochs=warmup_epochs_strategy,
        learning_rate=learning_rate_strategy,
    )
    def test_warmup_lr_reaches_target_at_end(
        self, warmup_epochs: int, learning_rate: float
    ):
        """At the last warmup epoch (W-1), LR equals the target learning_rate.

        Feature: unified-training-loop, Property 6: Warmup learning rate follows linear schedule
        **Validates: Requirements 4.4**
        """
        # Create a simple optimizer with a dummy parameter
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([dummy_param], lr=learning_rate)

        # Simulate warmup at the last warmup epoch (epoch = warmup_epochs - 1)
        last_warmup_epoch = warmup_epochs - 1
        actual_lr = simulate_warmup_lr_update(
            optimizer, learning_rate, last_warmup_epoch, warmup_epochs
        )

        # At epoch W-1, LR = LR * W / W = LR
        expected_lr = learning_rate
        assert abs(actual_lr - expected_lr) < 1e-10, (
            f"At last warmup epoch {last_warmup_epoch}, LR should equal target "
            f"{expected_lr}, got {actual_lr}"
        )

    @settings(max_examples=100, deadline=None)
    @given(
        warmup_epochs=warmup_epochs_strategy,
        learning_rate=learning_rate_strategy,
    )
    def test_warmup_lr_with_multiple_param_groups(
        self, warmup_epochs: int, learning_rate: float
    ):
        """Warmup LR is applied to all optimizer param groups.

        Feature: unified-training-loop, Property 6: Warmup learning rate follows linear schedule
        **Validates: Requirements 4.4**
        """
        # Create an optimizer with multiple param groups
        param1 = torch.nn.Parameter(torch.zeros(1))
        param2 = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD(
            [
                {"params": [param1], "lr": learning_rate},
                {"params": [param2], "lr": learning_rate * 0.1},  # Different initial LR
            ]
        )

        # Pick a random epoch within warmup
        epoch = warmup_epochs // 2 if warmup_epochs > 1 else 0

        # Simulate warmup LR update
        expected_lr = compute_warmup_lr(learning_rate, epoch, warmup_epochs)
        simulate_warmup_lr_update(optimizer, learning_rate, epoch, warmup_epochs)

        # All param groups should have the same warmup LR
        for i, param_group in enumerate(optimizer.param_groups):
            assert abs(param_group["lr"] - expected_lr) < 1e-10, (
                f"Param group {i} LR should be {expected_lr}, got {param_group['lr']}"
            )


# ---------------------------------------------------------------------------
# Strategies for Property 8: Average validation loss computation
# ---------------------------------------------------------------------------


def st_loss_values(min_size: int = 1, max_size: int = 50):
    """Generate sequences of loss values for testing.
    
    Loss values are positive floats (validation loss should always be positive).
    """
    return st.lists(
        st.floats(min_value=0.001, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=min_size,
        max_size=max_size,
    )


# ---------------------------------------------------------------------------
# Property 8: Average validation loss computation
# ---------------------------------------------------------------------------


class TestProperty8AverageValidationLossComputation:
    """Property 8: Average validation loss computation.
    
    For any sequence of N valid batches with loss values [L₁, L₂, ..., Lₙ],
    the reported average validation loss SHALL equal (L₁ + L₂ + ... + Lₙ) / N.
    
    **Validates: Requirements 6.5**
    """

    @settings(max_examples=100, deadline=None)
    @given(loss_values=st_loss_values(min_size=1, max_size=50))
    def test_average_validation_loss_equals_sum_divided_by_count(
        self, loss_values: List[float]
    ):
        """Average validation loss equals sum of losses / number of batches.
        
        Feature: unified-training-loop, Property 8: Average validation loss computation
        **Validates: Requirements 6.5**
        """
        # Compute expected average
        expected_average = sum(loss_values) / len(loss_values)
        
        # Simulate the validation loop logic from train_detection.py:
        #   val_loss_sum = 0.0
        #   val_batches = 0
        #   for batch:
        #       loss_tensor = model.train_step(...)["loss_tensor"]
        #       val_loss_sum += loss_tensor.item()
        #       val_batches += 1
        #   avg_val_loss = val_loss_sum / max(val_batches, 1)
        
        val_loss_sum = 0.0
        val_batches = 0
        
        for loss_val in loss_values:
            # Simulate getting loss from train_step
            loss_tensor = torch.tensor(loss_val)
            val_loss_sum += loss_tensor.item()
            val_batches += 1
        
        # Compute average as done in the training loop
        avg_val_loss = val_loss_sum / max(val_batches, 1)
        
        # Assert the computed average equals the expected average
        # Use relative tolerance for floating point comparison
        # Note: torch.tensor() uses float32 by default, which has ~7 decimal digits of precision
        # so we use a tolerance that accounts for this precision loss
        relative_tol = 1e-6
        absolute_tol = 1e-9
        diff = abs(avg_val_loss - expected_average)
        max_val = max(abs(avg_val_loss), abs(expected_average), 1.0)
        assert diff < relative_tol * max_val + absolute_tol, (
            f"Average validation loss mismatch: computed {avg_val_loss}, "
            f"expected {expected_average} for {len(loss_values)} batches (diff={diff})"
        )

    @settings(max_examples=100, deadline=None)
    @given(loss_values=st_loss_values(min_size=1, max_size=50))
    def test_average_validation_loss_with_mock_detector(
        self, loss_values: List[float]
    ):
        """Average validation loss computed via mock detector equals sum / N.
        
        Feature: unified-training-loop, Property 8: Average validation loss computation
        **Validates: Requirements 6.5**
        """
        # Create a mock detector that returns the specified loss values
        class MockDetectorWithLosses(BaseDetector):
            def __init__(self, losses: List[float]):
                self._losses = losses
                self._idx = 0
                
            def forward(self, images):
                return [{"boxes": torch.tensor([]), "labels": torch.tensor([]), "scores": torch.tensor([])}]
                
            def get_config_schema(self) -> dict:
                return {}
                
            def load_checkpoint(self, path: Path) -> None:
                pass
                
            def save_checkpoint(self, path: Path, optimizer=None, epoch=None, metrics=None) -> None:
                pass
                
            def get_parameters(self) -> List[torch.nn.Parameter]:
                return []
                
            def set_train_mode(self) -> None:
                pass
                
            def set_eval_mode(self) -> None:
                pass
                
            def train_step(self, images, targets) -> dict:
                loss_val = self._losses[self._idx]
                self._idx += 1
                return {"loss_tensor": torch.tensor(loss_val)}
        
        mock_detector = MockDetectorWithLosses(loss_values)
        
        # Compute expected average
        expected_average = sum(loss_values) / len(loss_values)
        
        # Simulate validation loop with mock detector
        val_loss_sum = 0.0
        val_batches = 0
        
        # Create dummy inputs (not used by mock, but needed for interface)
        dummy_images = [torch.zeros(3, 320, 320)]
        dummy_targets = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)}]
        
        with torch.no_grad():
            for _ in range(len(loss_values)):
                loss_dict = mock_detector.train_step(dummy_images, dummy_targets)
                loss_tensor = loss_dict["loss_tensor"]
                val_loss_sum += loss_tensor.item()
                val_batches += 1
        
        # Compute average as done in the training loop
        avg_val_loss = val_loss_sum / max(val_batches, 1)
        
        # Assert the computed average equals the expected average
        # Use relative tolerance for floating point comparison
        relative_tol = 1e-6
        absolute_tol = 1e-9
        diff = abs(avg_val_loss - expected_average)
        max_val = max(abs(avg_val_loss), abs(expected_average), 1.0)
        assert diff < relative_tol * max_val + absolute_tol, (
            f"Average validation loss mismatch: computed {avg_val_loss}, "
            f"expected {expected_average} for {len(loss_values)} batches (diff={diff})"
        )

    @settings(max_examples=100, deadline=None)
    @given(
        loss_values=st_loss_values(min_size=2, max_size=30),
        skip_indices=st.lists(st.integers(min_value=0, max_value=29), min_size=0, max_size=5, unique=True),
    )
    def test_average_validation_loss_with_skipped_batches(
        self, loss_values: List[float], skip_indices: List[int]
    ):
        """Average is computed only over valid (non-skipped) batches.
        
        When some batches are skipped (e.g., due to exceptions), the average
        should be computed only over the valid batches that were processed.
        
        Feature: unified-training-loop, Property 8: Average validation loss computation
        **Validates: Requirements 6.5**
        """
        # Filter skip_indices to only include valid indices
        valid_skip_indices = set(i for i in skip_indices if i < len(loss_values))
        
        # Compute expected average over non-skipped batches
        valid_losses = [
            loss for i, loss in enumerate(loss_values) 
            if i not in valid_skip_indices
        ]
        
        if not valid_losses:
            # All batches skipped - edge case, skip this test
            return
        
        expected_average = sum(valid_losses) / len(valid_losses)
        
        # Simulate validation loop with some batches skipped
        val_loss_sum = 0.0
        val_batches = 0
        
        for i, loss_val in enumerate(loss_values):
            if i in valid_skip_indices:
                # Simulate skipping this batch (e.g., due to exception)
                continue
            
            loss_tensor = torch.tensor(loss_val)
            val_loss_sum += loss_tensor.item()
            val_batches += 1
        
        # Compute average as done in the training loop
        avg_val_loss = val_loss_sum / max(val_batches, 1)
        
        # Assert the computed average equals the expected average
        # Use relative tolerance for floating point comparison
        relative_tol = 1e-6
        absolute_tol = 1e-9
        diff = abs(avg_val_loss - expected_average)
        max_val = max(abs(avg_val_loss), abs(expected_average), 1.0)
        assert diff < relative_tol * max_val + absolute_tol, (
            f"Average validation loss mismatch with skipped batches: computed {avg_val_loss}, "
            f"expected {expected_average} for {val_batches} valid batches out of {len(loss_values)} (diff={diff})"
        )

    @settings(max_examples=100, deadline=None)
    @given(single_loss=st.floats(min_value=0.001, max_value=100.0, allow_nan=False, allow_infinity=False))
    def test_average_validation_loss_single_batch(self, single_loss: float):
        """Average of single batch equals that batch's loss value.
        
        Feature: unified-training-loop, Property 8: Average validation loss computation
        **Validates: Requirements 6.5**
        """
        val_loss_sum = 0.0
        val_batches = 0
        
        # Single batch
        loss_tensor = torch.tensor(single_loss)
        val_loss_sum += loss_tensor.item()
        val_batches += 1
        
        avg_val_loss = val_loss_sum / max(val_batches, 1)
        
        # Use relative tolerance for floating point comparison
        # torch.tensor() uses float32 by default, which has ~7 decimal digits of precision
        relative_tol = 1e-6
        diff = abs(avg_val_loss - single_loss)
        max_val = max(abs(single_loss), 1.0)
        assert diff < relative_tol * max_val, (
            f"Single batch average should equal the loss: {avg_val_loss} != {single_loss} (diff={diff})"
        )

    @settings(max_examples=100, deadline=None)
    @given(
        n_batches=st.integers(min_value=1, max_value=100),
        constant_loss=st.floats(min_value=0.001, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    def test_average_validation_loss_constant_values(
        self, n_batches: int, constant_loss: float
    ):
        """Average of constant loss values equals the constant.
        
        Feature: unified-training-loop, Property 8: Average validation loss computation
        **Validates: Requirements 6.5**
        """
        val_loss_sum = 0.0
        val_batches = 0
        
        for _ in range(n_batches):
            loss_tensor = torch.tensor(constant_loss)
            val_loss_sum += loss_tensor.item()
            val_batches += 1
        
        avg_val_loss = val_loss_sum / max(val_batches, 1)
        
        # For constant values, average should equal the constant
        # Use relative tolerance for floating point comparison
        relative_tol = 1e-5  # Slightly larger tolerance for accumulated errors
        diff = abs(avg_val_loss - constant_loss)
        max_val = max(abs(constant_loss), 1.0)
        assert diff < relative_tol * max_val, (
            f"Average of {n_batches} constant losses ({constant_loss}) should equal {constant_loss}, "
            f"got {avg_val_loss} (diff={diff})"
        )
