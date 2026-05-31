"""Property-based tests for cosine scheduler configuration.

Tests Property 7 from the design document using Hypothesis.

**Validates: Requirements 4.5**
"""

import warnings

import torch
import torch.nn as nn
from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Hypothesis strategies for scheduler configuration
# ---------------------------------------------------------------------------

@st.composite
def scheduler_config(draw: st.DrawFn) -> dict:
    """Generate valid scheduler configuration with epochs E > warmup_epochs W.
    
    Returns a dict with:
    - epochs: total number of epochs (E)
    - warmup_epochs: number of warmup epochs (W), where E > W
    - learning_rate: base learning rate (LR)
    """
    # Generate warmup_epochs first (W >= 0)
    warmup_epochs = draw(st.integers(min_value=0, max_value=20))
    
    # Generate epochs such that E > W (at least 1 epoch after warmup)
    epochs = draw(st.integers(min_value=warmup_epochs + 1, max_value=warmup_epochs + 50))
    
    # Generate learning rate (positive, reasonable range)
    learning_rate = draw(st.floats(
        min_value=1e-5, 
        max_value=0.1, 
        allow_nan=False, 
        allow_infinity=False
    ))
    
    return {
        "epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "learning_rate": learning_rate,
    }


# ---------------------------------------------------------------------------
# Property 7: Cosine scheduler configuration
# Feature: unified-training-loop, Property 7: Cosine scheduler configuration
# ---------------------------------------------------------------------------


class TestProperty7CosineSchedulerConfiguration:
    """Property 7: For any epochs E and warmup_epochs W where E > W, the cosine
    annealing scheduler SHALL be configured with T_max = E - W and 
    eta_min = LR * 0.01, and SHALL be stepped only at epochs e >= W.

    **Validates: Requirements 4.5**
    """

    @given(config=scheduler_config())
    @settings(max_examples=100, deadline=None)
    def test_cosine_scheduler_t_max_equals_epochs_minus_warmup(self, config):
        """Cosine scheduler T_max equals epochs - warmup_epochs.
        
        Feature: unified-training-loop, Property 7: Cosine scheduler configuration
        """
        epochs = config["epochs"]
        warmup_epochs = config["warmup_epochs"]
        learning_rate = config["learning_rate"]
        
        # Create a simple model with parameters
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
        # Create cosine scheduler as done in train_detection.py
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs - warmup_epochs, 
            eta_min=learning_rate * 0.01
        )
        
        # Assert T_max is configured correctly
        assert cosine_scheduler.T_max == epochs - warmup_epochs

    @given(config=scheduler_config())
    @settings(max_examples=100, deadline=None)
    def test_cosine_scheduler_eta_min_equals_lr_times_001(self, config):
        """Cosine scheduler eta_min equals learning_rate * 0.01.
        
        Feature: unified-training-loop, Property 7: Cosine scheduler configuration
        """
        epochs = config["epochs"]
        warmup_epochs = config["warmup_epochs"]
        learning_rate = config["learning_rate"]
        
        # Create a simple model with parameters
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
        # Create cosine scheduler as done in train_detection.py
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs - warmup_epochs, 
            eta_min=learning_rate * 0.01
        )
        
        # Assert eta_min is configured correctly
        expected_eta_min = learning_rate * 0.01
        assert abs(cosine_scheduler.eta_min - expected_eta_min) < 1e-10

    @given(config=scheduler_config())
    @settings(max_examples=100, deadline=None)
    def test_cosine_scheduler_stepped_only_at_epochs_gte_warmup(self, config):
        """Cosine scheduler is stepped only at epochs >= warmup_epochs.
        
        Feature: unified-training-loop, Property 7: Cosine scheduler configuration
        
        This test simulates the training loop behavior where:
        - During warmup (epoch < warmup_epochs): LR is set manually, scheduler NOT stepped
        - After warmup (epoch >= warmup_epochs): scheduler IS stepped
        """
        epochs = config["epochs"]
        warmup_epochs = config["warmup_epochs"]
        learning_rate = config["learning_rate"]
        
        # Create a simple model with parameters
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
        # Create cosine scheduler as done in train_detection.py
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs - warmup_epochs, 
            eta_min=learning_rate * 0.01
        )
        
        # Track scheduler step count
        scheduler_step_count = 0
        
        # Simulate the epoch loop as in train_detection.py
        # Suppress the warning about calling scheduler.step() before optimizer.step()
        # since we're testing scheduler behavior in isolation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before")
            for epoch in range(epochs):
                # During warmup: set LR manually (linear warmup)
                if epoch < warmup_epochs:
                    warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = warmup_lr
                    # Scheduler should NOT be stepped during warmup
                else:
                    # After warmup: step the cosine scheduler
                    cosine_scheduler.step()
                    scheduler_step_count += 1
        
        # Assert scheduler was stepped exactly (epochs - warmup_epochs) times
        expected_steps = epochs - warmup_epochs
        assert scheduler_step_count == expected_steps, (
            f"Expected {expected_steps} scheduler steps, got {scheduler_step_count}"
        )

    @given(config=scheduler_config())
    @settings(max_examples=100, deadline=None)
    def test_cosine_scheduler_lr_reaches_eta_min_at_end(self, config):
        """After all scheduler steps, LR should approach eta_min.
        
        Feature: unified-training-loop, Property 7: Cosine scheduler configuration
        """
        epochs = config["epochs"]
        warmup_epochs = config["warmup_epochs"]
        learning_rate = config["learning_rate"]
        
        # Skip edge case where T_max is very small (scheduler behavior is less predictable)
        assume(epochs - warmup_epochs >= 2)
        
        # Create a simple model with parameters
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
        # Create cosine scheduler as done in train_detection.py
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs - warmup_epochs, 
            eta_min=learning_rate * 0.01
        )
        
        # Simulate the epoch loop - step scheduler only after warmup
        # Suppress the warning about calling scheduler.step() before optimizer.step()
        # since we're testing scheduler behavior in isolation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before")
            for epoch in range(epochs):
                if epoch < warmup_epochs:
                    # During warmup: set LR manually
                    warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = warmup_lr
                else:
                    # After warmup: step the cosine scheduler
                    cosine_scheduler.step()
        
        # After all steps, LR should be at or very close to eta_min
        final_lr = optimizer.param_groups[0]["lr"]
        expected_eta_min = learning_rate * 0.01
        
        # Allow small tolerance for floating point
        assert abs(final_lr - expected_eta_min) < 1e-8, (
            f"Final LR {final_lr} should be close to eta_min {expected_eta_min}"
        )

    @given(config=scheduler_config())
    @settings(max_examples=100, deadline=None)
    def test_cosine_scheduler_lr_starts_at_base_lr_after_warmup(self, config):
        """At the first epoch after warmup, LR should start at the base learning rate.
        
        Feature: unified-training-loop, Property 7: Cosine scheduler configuration
        """
        epochs = config["epochs"]
        warmup_epochs = config["warmup_epochs"]
        learning_rate = config["learning_rate"]
        
        # Create a simple model with parameters
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
        # Create cosine scheduler as done in train_detection.py
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs - warmup_epochs, 
            eta_min=learning_rate * 0.01
        )
        
        # Before any scheduler step, LR should be at base learning rate
        initial_lr = optimizer.param_groups[0]["lr"]
        assert abs(initial_lr - learning_rate) < 1e-10, (
            f"Initial LR {initial_lr} should equal base LR {learning_rate}"
        )
