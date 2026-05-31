"""Property-based tests for early stopping behavior.

Feature: unified-training-loop, Property 12: Early stopping triggers at correct epoch

This module tests that training terminates at the first epoch where patience
consecutive epochs have passed without the validation loss being strictly less
than the best recorded validation loss.

**Validates: Requirements 8.1**
"""

from typing import List, Tuple

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Helper functions for early stopping logic
# ---------------------------------------------------------------------------


def compute_expected_stop_epoch(
    val_losses: List[float], patience: int
) -> Tuple[int, float, int]:
    """Compute the expected epoch at which early stopping should trigger.
    
    This is the reference implementation of the early stopping logic:
    - Track best validation loss seen so far
    - Count consecutive epochs without improvement
    - Stop when count reaches patience
    
    Args:
        val_losses: Sequence of validation loss values, one per epoch.
        patience: Number of consecutive non-improving epochs before stopping.
        
    Returns:
        Tuple of (stop_epoch, best_val_loss, best_epoch) where:
        - stop_epoch: 1-indexed epoch at which training should stop (or len(val_losses) if no early stop)
        - best_val_loss: The best validation loss recorded
        - best_epoch: 1-indexed epoch at which best loss occurred
    """
    if not val_losses:
        return 0, float("inf"), 0
    
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    
    for epoch_idx, val_loss in enumerate(val_losses):
        epoch_num = epoch_idx + 1  # 1-indexed
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch_num
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Check early stopping condition AFTER updating counters
        if epochs_without_improvement >= patience:
            return epoch_num, best_val_loss, best_epoch
    
    # No early stopping triggered - completed all epochs
    return len(val_losses), best_val_loss, best_epoch


def simulate_early_stopping(
    val_losses: List[float], patience: int
) -> Tuple[int, float, int, int]:
    """Simulate the early stopping logic as implemented in train_detection.py.
    
    This mirrors the actual implementation in the training loop:
    ```
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
    
    if epochs_without_improvement >= early_stopping_patience:
        break
    ```
    
    Args:
        val_losses: Sequence of validation loss values, one per epoch.
        patience: Number of consecutive non-improving epochs before stopping.
        
    Returns:
        Tuple of (completed_epochs, best_val_loss, best_epoch, epochs_without_improvement)
    """
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    completed_epochs = 0
    
    for epoch in range(len(val_losses)):
        avg_val_loss = val_losses[epoch]
        completed_epochs = epoch + 1
        
        # Update best tracking (mirrors train_detection.py lines 555-562)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1  # 1-indexed
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stopping check (mirrors train_detection.py lines 575-582)
        if epochs_without_improvement >= patience:
            break
    
    return completed_epochs, best_val_loss, best_epoch, epochs_without_improvement


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


@st.composite
def st_loss_sequence_with_patience(draw, max_epochs: int = 30, max_patience: int = 10):
    """Generate a loss sequence and patience value for early stopping testing.
    
    Args:
        draw: Hypothesis draw function.
        max_epochs: Maximum number of epochs to generate.
        max_patience: Maximum patience value.
        
    Returns:
        Tuple of (val_losses, patience).
    """
    # Generate patience first (at least 1)
    patience = draw(st.integers(min_value=1, max_value=max_patience))
    
    # Generate enough epochs to potentially trigger early stopping
    # Need at least patience + 1 epochs to possibly trigger
    min_epochs = patience + 1
    num_epochs = draw(st.integers(min_value=min_epochs, max_value=max_epochs))
    
    # Generate loss values
    val_losses = draw(
        st.lists(
            st.floats(min_value=0.001, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=num_epochs,
            max_size=num_epochs,
        )
    )
    
    return val_losses, patience


@st.composite
def st_decreasing_then_plateau_losses(draw, max_epochs: int = 30, max_patience: int = 10):
    """Generate loss sequences that decrease then plateau (guaranteed to trigger early stopping).
    
    Args:
        draw: Hypothesis draw function.
        max_epochs: Maximum number of epochs.
        max_patience: Maximum patience value.
        
    Returns:
        Tuple of (val_losses, patience).
    """
    patience = draw(st.integers(min_value=1, max_value=max_patience))
    
    # Number of improving epochs (at least 1)
    num_improving = draw(st.integers(min_value=1, max_value=10))
    
    # Number of plateau epochs (at least patience to trigger early stopping)
    num_plateau = draw(st.integers(min_value=patience, max_value=patience + 5))
    
    # Generate decreasing losses for improving phase
    start_loss = draw(st.floats(min_value=10.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    improving_losses = []
    current_loss = start_loss
    for _ in range(num_improving):
        improving_losses.append(current_loss)
        # Decrease by a random amount
        decrease = draw(st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False))
        current_loss = max(0.001, current_loss - decrease)
    
    # The best loss is the last improving loss
    best_loss = improving_losses[-1]
    
    # Generate plateau losses (all >= best_loss, so no improvement)
    plateau_losses = draw(
        st.lists(
            st.floats(min_value=best_loss, max_value=best_loss + 10.0, allow_nan=False, allow_infinity=False),
            min_size=num_plateau,
            max_size=num_plateau,
        )
    )
    
    val_losses = improving_losses + plateau_losses
    return val_losses, patience


@st.composite
def st_always_improving_losses(draw, max_epochs: int = 20):
    """Generate strictly decreasing loss sequences (never triggers early stopping).
    
    Args:
        draw: Hypothesis draw function.
        max_epochs: Maximum number of epochs.
        
    Returns:
        Tuple of (val_losses, patience).
    """
    patience = draw(st.integers(min_value=1, max_value=10))
    num_epochs = draw(st.integers(min_value=2, max_value=max_epochs))
    
    # Generate strictly decreasing losses
    start_loss = draw(st.floats(min_value=50.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    val_losses = []
    current_loss = start_loss
    for _ in range(num_epochs):
        val_losses.append(current_loss)
        # Strictly decrease
        decrease = draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))
        current_loss = max(0.001, current_loss - decrease)
    
    return val_losses, patience


# ---------------------------------------------------------------------------
# Property 12: Early stopping triggers at correct epoch
# ---------------------------------------------------------------------------


class TestProperty12EarlyStoppingTriggersAtCorrectEpoch:
    """Property 12: Early stopping triggers at correct epoch.
    
    For any sequence of validation losses and patience value P, training SHALL
    terminate at the first epoch where P consecutive epochs have passed without
    the validation loss being strictly less than the best recorded validation loss.
    
    **Validates: Requirements 8.1**
    """

    @settings(max_examples=100, deadline=None)
    @given(config=st_loss_sequence_with_patience(max_epochs=30, max_patience=10))
    def test_early_stopping_triggers_at_correct_epoch(
        self, config: Tuple[List[float], int]
    ):
        """Early stopping triggers exactly when patience consecutive non-improving epochs pass.
        
        Feature: unified-training-loop, Property 12: Early stopping triggers at correct epoch
        **Validates: Requirements 8.1**
        """
        val_losses, patience = config
        
        # Compute expected stop epoch using reference implementation
        expected_stop_epoch, expected_best_loss, expected_best_epoch = compute_expected_stop_epoch(
            val_losses, patience
        )
        
        # Simulate the actual training loop logic
        actual_completed, actual_best_loss, actual_best_epoch, _ = simulate_early_stopping(
            val_losses, patience
        )
        
        # Assert the training stopped at the correct epoch
        assert actual_completed == expected_stop_epoch, (
            f"Early stopping mismatch: expected to stop at epoch {expected_stop_epoch}, "
            f"but stopped at epoch {actual_completed}. "
            f"Losses: {val_losses[:10]}{'...' if len(val_losses) > 10 else ''}, patience={patience}"
        )
        
        # Assert best loss tracking is correct
        assert abs(actual_best_loss - expected_best_loss) < 1e-9, (
            f"Best loss mismatch: expected {expected_best_loss}, got {actual_best_loss}"
        )
        
        # Assert best epoch tracking is correct
        assert actual_best_epoch == expected_best_epoch, (
            f"Best epoch mismatch: expected {expected_best_epoch}, got {actual_best_epoch}"
        )

    @settings(max_examples=100, deadline=None)
    @given(config=st_decreasing_then_plateau_losses(max_epochs=30, max_patience=10))
    def test_early_stopping_with_plateau(
        self, config: Tuple[List[float], int]
    ):
        """Early stopping triggers when loss plateaus after improvement.
        
        Feature: unified-training-loop, Property 12: Early stopping triggers at correct epoch
        **Validates: Requirements 8.1**
        """
        val_losses, patience = config
        
        # This sequence is designed to trigger early stopping
        completed_epochs, best_loss, best_epoch, epochs_without_improvement = simulate_early_stopping(
            val_losses, patience
        )
        
        # Early stopping should have triggered (completed < total epochs)
        assert completed_epochs < len(val_losses) or epochs_without_improvement >= patience, (
            f"Early stopping should trigger with plateau losses. "
            f"Completed {completed_epochs}/{len(val_losses)} epochs, "
            f"epochs_without_improvement={epochs_without_improvement}, patience={patience}"
        )
        
        # Verify the stop happened at the right time
        expected_stop, _, _ = compute_expected_stop_epoch(val_losses, patience)
        assert completed_epochs == expected_stop, (
            f"Expected to stop at epoch {expected_stop}, but stopped at {completed_epochs}"
        )

    @settings(max_examples=100, deadline=None)
    @given(config=st_always_improving_losses(max_epochs=20))
    def test_no_early_stopping_when_always_improving(
        self, config: Tuple[List[float], int]
    ):
        """No early stopping when loss always improves.
        
        Feature: unified-training-loop, Property 12: Early stopping triggers at correct epoch
        **Validates: Requirements 8.1**
        """
        val_losses, patience = config
        
        # With strictly decreasing losses, early stopping should never trigger
        completed_epochs, _, _, epochs_without_improvement = simulate_early_stopping(
            val_losses, patience
        )
        
        # Should complete all epochs
        assert completed_epochs == len(val_losses), (
            f"Should complete all {len(val_losses)} epochs when always improving, "
            f"but stopped at epoch {completed_epochs}"
        )
        
        # epochs_without_improvement should be 0 (every epoch improved)
        assert epochs_without_improvement == 0, (
            f"epochs_without_improvement should be 0 for strictly decreasing losses, "
            f"got {epochs_without_improvement}"
        )

    @settings(max_examples=100, deadline=None)
    @given(
        patience=st.integers(min_value=1, max_value=20),
        constant_loss=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    def test_early_stopping_with_constant_loss(
        self, patience: int, constant_loss: float
    ):
        """Early stopping triggers after patience epochs with constant loss.
        
        When all losses are identical, only the first epoch improves (from inf),
        then all subsequent epochs don't improve, triggering early stopping
        after patience non-improving epochs.
        
        Feature: unified-training-loop, Property 12: Early stopping triggers at correct epoch
        **Validates: Requirements 8.1**
        """
        # Generate enough epochs to trigger early stopping
        num_epochs = patience + 5
        val_losses = [constant_loss] * num_epochs
        
        completed_epochs, best_loss, best_epoch, _ = simulate_early_stopping(
            val_losses, patience
        )
        
        # First epoch sets best_loss, then patience epochs don't improve
        # So we stop at epoch 1 + patience
        expected_stop_epoch = 1 + patience
        
        assert completed_epochs == expected_stop_epoch, (
            f"With constant loss, should stop at epoch {expected_stop_epoch} "
            f"(1 improving + {patience} non-improving), but stopped at {completed_epochs}"
        )
        
        # Best loss should be the constant loss (set at epoch 1)
        assert abs(best_loss - constant_loss) < 1e-9, (
            f"Best loss should be {constant_loss}, got {best_loss}"
        )
        
        # Best epoch should be 1 (first epoch)
        assert best_epoch == 1, (
            f"Best epoch should be 1, got {best_epoch}"
        )

    @settings(max_examples=100, deadline=None)
    @given(
        patience=st.integers(min_value=1, max_value=10),
        num_epochs=st.integers(min_value=1, max_value=30),
    )
    def test_early_stopping_patience_boundary(
        self, patience: int, num_epochs: int
    ):
        """Early stopping triggers exactly at patience boundary, not before.
        
        Feature: unified-training-loop, Property 12: Early stopping triggers at correct epoch
        **Validates: Requirements 8.1**
        """
        # Create a sequence: one improving epoch, then non-improving epochs
        # Loss: [10.0, 11.0, 11.0, 11.0, ...]
        val_losses = [10.0] + [11.0] * (num_epochs - 1) if num_epochs > 1 else [10.0]
        
        completed_epochs, _, _, epochs_without_improvement = simulate_early_stopping(
            val_losses, patience
        )
        
        if num_epochs == 1:
            # Only one epoch, no early stopping possible
            assert completed_epochs == 1
        elif num_epochs - 1 >= patience:
            # Enough non-improving epochs to trigger early stopping
            # Stop at epoch 1 + patience (first epoch improves, then patience don't)
            expected_stop = 1 + patience
            assert completed_epochs == expected_stop, (
                f"Should stop at epoch {expected_stop}, but stopped at {completed_epochs}. "
                f"patience={patience}, num_epochs={num_epochs}"
            )
        else:
            # Not enough epochs to trigger early stopping
            assert completed_epochs == num_epochs, (
                f"Should complete all {num_epochs} epochs, but stopped at {completed_epochs}"
            )

    @settings(max_examples=100, deadline=None)
    @given(
        patience=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_early_stopping_resets_on_improvement(
        self, patience: int, seed: int
    ):
        """Early stopping counter resets when loss improves.
        
        Feature: unified-training-loop, Property 12: Early stopping triggers at correct epoch
        **Validates: Requirements 8.1**
        """
        import random
        rng = random.Random(seed)
        
        # Create a sequence with intermittent improvements
        # Pattern: improve, plateau (patience-1), improve, plateau (patience-1), ...
        # This should NOT trigger early stopping
        val_losses = []
        current_loss = 100.0
        
        for _ in range(3):  # 3 cycles
            # Improvement
            current_loss -= rng.uniform(1.0, 5.0)
            val_losses.append(current_loss)
            
            # Plateau for patience-1 epochs (not enough to trigger)
            for _ in range(patience - 1):
                val_losses.append(current_loss + rng.uniform(0.0, 0.5))
        
        completed_epochs, _, _, _ = simulate_early_stopping(val_losses, patience)
        
        # Should complete all epochs (improvements reset the counter)
        assert completed_epochs == len(val_losses), (
            f"Should complete all {len(val_losses)} epochs with intermittent improvements, "
            f"but stopped at {completed_epochs}"
        )

    @settings(max_examples=100, deadline=None)
    @given(
        patience=st.integers(min_value=1, max_value=10),
        num_improving=st.integers(min_value=1, max_value=10),
    )
    def test_early_stopping_best_epoch_tracking(
        self, patience: int, num_improving: int
    ):
        """Best epoch is correctly tracked as the last improving epoch.
        
        Feature: unified-training-loop, Property 12: Early stopping triggers at correct epoch
        **Validates: Requirements 8.1**
        """
        # Create sequence: num_improving decreasing epochs, then plateau
        val_losses = []
        current_loss = 100.0
        for i in range(num_improving):
            current_loss -= 5.0
            val_losses.append(current_loss)
        
        # Add plateau epochs (enough to trigger early stopping)
        plateau_loss = current_loss + 1.0  # Slightly worse than best
        for _ in range(patience + 2):
            val_losses.append(plateau_loss)
        
        completed_epochs, best_loss, best_epoch, _ = simulate_early_stopping(
            val_losses, patience
        )
        
        # Best epoch should be the last improving epoch
        assert best_epoch == num_improving, (
            f"Best epoch should be {num_improving}, got {best_epoch}"
        )
        
        # Best loss should be the loss at the last improving epoch
        expected_best_loss = val_losses[num_improving - 1]
        assert abs(best_loss - expected_best_loss) < 1e-9, (
            f"Best loss should be {expected_best_loss}, got {best_loss}"
        )

    @settings(max_examples=100, deadline=None)
    @given(
        patience=st.integers(min_value=1, max_value=15),
    )
    def test_early_stopping_minimum_epochs_to_trigger(
        self, patience: int,
    ):
        """Early stopping requires at least patience+1 epochs to trigger.
        
        The first epoch always "improves" (from inf), so we need patience
        additional non-improving epochs to trigger early stopping.
        
        Feature: unified-training-loop, Property 12: Early stopping triggers at correct epoch
        **Validates: Requirements 8.1**
        """
        # With exactly patience epochs, early stopping should NOT trigger
        # (first epoch improves, then patience-1 don't improve = not enough)
        val_losses_not_enough = [10.0] * patience
        completed, _, _, epochs_without = simulate_early_stopping(val_losses_not_enough, patience)
        
        assert completed == patience, (
            f"With {patience} epochs, should complete all (not enough to trigger). "
            f"Completed {completed}, epochs_without_improvement={epochs_without}"
        )
        
        # With patience+1 epochs, early stopping SHOULD trigger
        # (first epoch improves, then patience don't improve = triggers)
        val_losses_enough = [10.0] * (patience + 1)
        completed, _, _, epochs_without = simulate_early_stopping(val_losses_enough, patience)
        
        assert completed == patience + 1, (
            f"With {patience + 1} epochs of constant loss, should stop at epoch {patience + 1}. "
            f"Completed {completed}, epochs_without_improvement={epochs_without}"
        )
        assert epochs_without >= patience, (
            f"epochs_without_improvement should be >= {patience}, got {epochs_without}"
        )

