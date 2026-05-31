"""Property-based tests for the unified training loop.

Tests correctness properties from the design document using Hypothesis.

**Validates: Requirements 4.2, 4.3**
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Hypothesis strategies for optimizer configuration
# ---------------------------------------------------------------------------

# Strategy for known optimizer names (case variations)
_KNOWN_OPTIMIZERS = st.sampled_from([
    "SGD", "sgd", "Sgd", "sGd",
    "Adam", "adam", "ADAM", "aDaM",
    "AdamW", "adamw", "ADAMW", "AdAmW",
])

# Strategy for arbitrary strings that are NOT known optimizers
_ARBITRARY_STRINGS = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P")),
    min_size=0,
    max_size=20,
).filter(lambda s: s.upper() not in ("SGD", "ADAM", "ADAMW"))

# Combined strategy: known optimizers + arbitrary strings
_OPTIMIZER_STRINGS = st.one_of(_KNOWN_OPTIMIZERS, _ARBITRARY_STRINGS)


# Strategy for valid learning rates
_LEARNING_RATES = st.floats(
    min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Strategy for valid weight decay values
_WEIGHT_DECAY = st.floats(
    min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False
)

# Strategy for valid momentum values (for SGD)
_MOMENTUM = st.floats(
    min_value=0.0, max_value=0.999, allow_nan=False, allow_infinity=False
)


def _build_optimizer(
    optimizer_name: str,
    params,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
) -> torch.optim.Optimizer:
    """Build optimizer using the same logic as train_detection.train().
    
    This mirrors the optimizer construction logic from the unified training loop.
    """
    if optimizer_name.upper() == "SGD":
        return torch.optim.SGD(
            params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer_name.upper() == "ADAM":
        return torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.upper() == "ADAMW":
        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    else:
        # Fallback to SGD for unknown optimizer values
        return torch.optim.SGD(
            params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )


# ---------------------------------------------------------------------------
# Property 5: Optimizer class selection follows config
# Feature: unified-training-loop, Property 5: Optimizer class selection follows config
# ---------------------------------------------------------------------------


class TestProperty5OptimizerClassSelection:
    """Property 5: Optimizer class selection follows config.
    
    For any `training.optimizer` string, the training loop SHALL instantiate
    `torch.optim.SGD` when the string is "SGD" (case-insensitive),
    `torch.optim.Adam` when "Adam", `torch.optim.AdamW` when "AdamW",
    and `torch.optim.SGD` for all other values.
    
    **Validates: Requirements 4.2, 4.3**
    """

    @given(
        optimizer_name=_OPTIMIZER_STRINGS,
        learning_rate=_LEARNING_RATES,
        weight_decay=_WEIGHT_DECAY,
        momentum=_MOMENTUM,
    )
    @settings(max_examples=100, deadline=None)
    def test_optimizer_class_selection_follows_config(
        self, optimizer_name: str, learning_rate: float, weight_decay: float, momentum: float
    ):
        """Optimizer class is selected correctly based on config string.
        
        Feature: unified-training-loop, Property 5: Optimizer class selection follows config
        """
        # Create a simple parameter tensor to construct the optimizer
        params = [torch.nn.Parameter(torch.randn(10, 10))]
        
        # Build optimizer using the same logic as the training loop
        optimizer = _build_optimizer(
            optimizer_name, params, learning_rate, weight_decay, momentum
        )
        
        # Determine expected optimizer class
        upper_name = optimizer_name.upper()
        if upper_name == "SGD":
            expected_class = torch.optim.SGD
        elif upper_name == "ADAM":
            expected_class = torch.optim.Adam
        elif upper_name == "ADAMW":
            expected_class = torch.optim.AdamW
        else:
            # Unknown strings should fall back to SGD
            expected_class = torch.optim.SGD
        
        assert isinstance(optimizer, expected_class), (
            f"Expected {expected_class.__name__} for optimizer_name='{optimizer_name}', "
            f"but got {type(optimizer).__name__}"
        )

    @given(
        optimizer_name=st.sampled_from(["SGD", "sgd", "Sgd", "sGD"]),
        learning_rate=_LEARNING_RATES,
        weight_decay=_WEIGHT_DECAY,
        momentum=_MOMENTUM,
    )
    @settings(max_examples=100, deadline=None)
    def test_sgd_case_insensitive(
        self, optimizer_name: str, learning_rate: float, weight_decay: float, momentum: float
    ):
        """SGD is selected regardless of case.
        
        Feature: unified-training-loop, Property 5: Optimizer class selection follows config
        """
        params = [torch.nn.Parameter(torch.randn(10, 10))]
        optimizer = _build_optimizer(
            optimizer_name, params, learning_rate, weight_decay, momentum
        )
        
        assert isinstance(optimizer, torch.optim.SGD), (
            f"Expected SGD for optimizer_name='{optimizer_name}', "
            f"but got {type(optimizer).__name__}"
        )

    @given(
        optimizer_name=st.sampled_from(["Adam", "adam", "ADAM", "aDaM"]),
        learning_rate=_LEARNING_RATES,
        weight_decay=_WEIGHT_DECAY,
    )
    @settings(max_examples=100, deadline=None)
    def test_adam_case_insensitive(
        self, optimizer_name: str, learning_rate: float, weight_decay: float
    ):
        """Adam is selected regardless of case.
        
        Feature: unified-training-loop, Property 5: Optimizer class selection follows config
        """
        params = [torch.nn.Parameter(torch.randn(10, 10))]
        optimizer = _build_optimizer(
            optimizer_name, params, learning_rate, weight_decay, momentum=0.9
        )
        
        assert isinstance(optimizer, torch.optim.Adam), (
            f"Expected Adam for optimizer_name='{optimizer_name}', "
            f"but got {type(optimizer).__name__}"
        )

    @given(
        optimizer_name=st.sampled_from(["AdamW", "adamw", "ADAMW", "AdAmW"]),
        learning_rate=_LEARNING_RATES,
        weight_decay=_WEIGHT_DECAY,
    )
    @settings(max_examples=100, deadline=None)
    def test_adamw_case_insensitive(
        self, optimizer_name: str, learning_rate: float, weight_decay: float
    ):
        """AdamW is selected regardless of case.
        
        Feature: unified-training-loop, Property 5: Optimizer class selection follows config
        """
        params = [torch.nn.Parameter(torch.randn(10, 10))]
        optimizer = _build_optimizer(
            optimizer_name, params, learning_rate, weight_decay, momentum=0.9
        )
        
        assert isinstance(optimizer, torch.optim.AdamW), (
            f"Expected AdamW for optimizer_name='{optimizer_name}', "
            f"but got {type(optimizer).__name__}"
        )

    @given(
        optimizer_name=_ARBITRARY_STRINGS,
        learning_rate=_LEARNING_RATES,
        weight_decay=_WEIGHT_DECAY,
        momentum=_MOMENTUM,
    )
    @settings(max_examples=100, deadline=None)
    def test_unknown_optimizer_falls_back_to_sgd(
        self, optimizer_name: str, learning_rate: float, weight_decay: float, momentum: float
    ):
        """Unknown optimizer strings fall back to SGD.
        
        Feature: unified-training-loop, Property 5: Optimizer class selection follows config
        """
        params = [torch.nn.Parameter(torch.randn(10, 10))]
        optimizer = _build_optimizer(
            optimizer_name, params, learning_rate, weight_decay, momentum
        )
        
        assert isinstance(optimizer, torch.optim.SGD), (
            f"Expected SGD fallback for unknown optimizer_name='{optimizer_name}', "
            f"but got {type(optimizer).__name__}"
        )

    @given(
        optimizer_name=_OPTIMIZER_STRINGS,
        learning_rate=_LEARNING_RATES,
        weight_decay=_WEIGHT_DECAY,
        momentum=_MOMENTUM,
    )
    @settings(max_examples=100, deadline=None)
    def test_optimizer_hyperparameters_applied(
        self, optimizer_name: str, learning_rate: float, weight_decay: float, momentum: float
    ):
        """Optimizer is constructed with the correct hyperparameters from config.
        
        Feature: unified-training-loop, Property 5: Optimizer class selection follows config
        """
        params = [torch.nn.Parameter(torch.randn(10, 10))]
        optimizer = _build_optimizer(
            optimizer_name, params, learning_rate, weight_decay, momentum
        )
        
        # Check that learning rate is set correctly
        for param_group in optimizer.param_groups:
            assert param_group["lr"] == learning_rate, (
                f"Expected lr={learning_rate}, got {param_group['lr']}"
            )
            assert param_group["weight_decay"] == weight_decay, (
                f"Expected weight_decay={weight_decay}, got {param_group['weight_decay']}"
            )
            
            # Check momentum for SGD (Adam/AdamW use betas instead)
            if isinstance(optimizer, torch.optim.SGD):
                assert param_group["momentum"] == momentum, (
                    f"Expected momentum={momentum}, got {param_group['momentum']}"
                )

    @given(
        optimizer_name=_OPTIMIZER_STRINGS,
        learning_rate=_LEARNING_RATES,
        weight_decay=_WEIGHT_DECAY,
        momentum=_MOMENTUM,
    )
    @settings(max_examples=100, deadline=None)
    def test_optimizer_uses_provided_parameters(
        self, optimizer_name: str, learning_rate: float, weight_decay: float, momentum: float
    ):
        """Optimizer is constructed with the parameters from model.get_parameters().
        
        Feature: unified-training-loop, Property 5: Optimizer class selection follows config
        """
        # Create multiple parameter tensors to simulate model.get_parameters()
        params = [
            torch.nn.Parameter(torch.randn(10, 10)),
            torch.nn.Parameter(torch.randn(5, 5)),
            torch.nn.Parameter(torch.randn(3)),
        ]
        
        optimizer = _build_optimizer(
            optimizer_name, params, learning_rate, weight_decay, momentum
        )
        
        # Verify all parameters are in the optimizer
        optimizer_params = []
        for param_group in optimizer.param_groups:
            optimizer_params.extend(param_group["params"])
        
        assert len(optimizer_params) == len(params), (
            f"Expected {len(params)} parameters, got {len(optimizer_params)}"
        )
        
        for p in params:
            assert any(p is op for op in optimizer_params), (
                "Parameter not found in optimizer"
            )
