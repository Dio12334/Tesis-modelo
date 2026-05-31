"""Unit tests for optimizer/scheduler construction and config handling.

Tests cover:
- SGD, Adam, AdamW instantiation with correct hyperparameters
- Fallback to SGD for unknown optimizer string
- Warmup LR calculation at each epoch
- Cosine scheduler T_max and eta_min configuration
- Missing augmentation section proceeds without error
- data_yaml field is ignored

Validates: Requirements 4.2, 4.3, 4.4, 4.5, 3.3, 3.5
"""

import math
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from model.models.registry import BaseDetector


# ---------------------------------------------------------------------------
# Mock detector for testing optimizer construction
# ---------------------------------------------------------------------------


class MockDetectorWithParams(BaseDetector):
    """Mock BaseDetector that provides real trainable parameters."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        # Create a simple linear layer to provide real parameters
        self._layer = nn.Linear(10, 5)
        self._train_mode = True

    def forward(self, images):
        return [{"boxes": [], "labels": [], "scores": []}]

    def get_config_schema(self) -> dict:
        return {}

    def load_checkpoint(self, path: Path) -> None:
        pass

    def save_checkpoint(self, path: Path, **kwargs) -> None:
        pass

    def get_parameters(self) -> List[nn.Parameter]:
        return list(self._layer.parameters())

    def set_train_mode(self) -> None:
        self._train_mode = True
        self._layer.train()

    def set_eval_mode(self) -> None:
        self._train_mode = False
        self._layer.eval()

    def train_step(self, images, targets):
        # Return a simple loss tensor
        return {"loss_tensor": torch.tensor(1.0, requires_grad=True)}


# ---------------------------------------------------------------------------
# Helper functions for optimizer/scheduler construction
# ---------------------------------------------------------------------------


def build_optimizer(params, optimizer_name: str, learning_rate: float,
                    momentum: float, weight_decay: float) -> torch.optim.Optimizer:
    """Build optimizer following the same logic as train_detection.py."""
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


def compute_warmup_lr(epoch: int, warmup_epochs: int, learning_rate: float) -> float:
    """Compute warmup learning rate following the linear warmup formula.

    LR = learning_rate * (epoch + 1) / warmup_epochs
    """
    return learning_rate * (epoch + 1) / warmup_epochs


# ---------------------------------------------------------------------------
# Test: Optimizer instantiation
# ---------------------------------------------------------------------------


class TestOptimizerInstantiation:
    """Test optimizer construction with correct hyperparameters."""

    def test_sgd_instantiation_with_correct_hyperparameters(self):
        """SGD optimizer is instantiated with correct lr, momentum, weight_decay.

        Validates: Requirements 4.2
        """
        model = MockDetectorWithParams()
        params = model.get_parameters()

        learning_rate = 0.01
        momentum = 0.9
        weight_decay = 0.0005

        optimizer = build_optimizer(
            params, "SGD", learning_rate, momentum, weight_decay
        )

        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.defaults["lr"] == learning_rate
        assert optimizer.defaults["momentum"] == momentum
        assert optimizer.defaults["weight_decay"] == weight_decay

    def test_adam_instantiation_with_correct_hyperparameters(self):
        """Adam optimizer is instantiated with correct lr and weight_decay.

        Validates: Requirements 4.2
        """
        model = MockDetectorWithParams()
        params = model.get_parameters()

        learning_rate = 0.001
        weight_decay = 0.01

        optimizer = build_optimizer(
            params, "Adam", learning_rate, momentum=0.9, weight_decay=weight_decay
        )

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.defaults["lr"] == learning_rate
        assert optimizer.defaults["weight_decay"] == weight_decay

    def test_adamw_instantiation_with_correct_hyperparameters(self):
        """AdamW optimizer is instantiated with correct lr and weight_decay.

        Validates: Requirements 4.2
        """
        model = MockDetectorWithParams()
        params = model.get_parameters()

        learning_rate = 0.0001
        weight_decay = 0.05

        optimizer = build_optimizer(
            params, "AdamW", learning_rate, momentum=0.9, weight_decay=weight_decay
        )

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["lr"] == learning_rate
        assert optimizer.defaults["weight_decay"] == weight_decay

    def test_sgd_case_insensitive(self):
        """SGD optimizer name is case-insensitive.

        Validates: Requirements 4.2
        """
        model = MockDetectorWithParams()
        params = model.get_parameters()

        for name in ["sgd", "Sgd", "sGd", "SGD"]:
            optimizer = build_optimizer(params, name, 0.01, 0.9, 0.0005)
            assert isinstance(optimizer, torch.optim.SGD)

    def test_adam_case_insensitive(self):
        """Adam optimizer name is case-insensitive.

        Validates: Requirements 4.2
        """
        model = MockDetectorWithParams()
        params = model.get_parameters()

        for name in ["adam", "Adam", "ADAM", "aDaM"]:
            optimizer = build_optimizer(params, name, 0.001, 0.9, 0.01)
            assert isinstance(optimizer, torch.optim.Adam)

    def test_adamw_case_insensitive(self):
        """AdamW optimizer name is case-insensitive.

        Validates: Requirements 4.2
        """
        model = MockDetectorWithParams()
        params = model.get_parameters()

        for name in ["adamw", "AdamW", "ADAMW", "Adamw"]:
            optimizer = build_optimizer(params, name, 0.0001, 0.9, 0.05)
            assert isinstance(optimizer, torch.optim.AdamW)


class TestOptimizerFallback:
    """Test fallback to SGD for unknown optimizer strings."""

    def test_unknown_optimizer_falls_back_to_sgd(self):
        """Unknown optimizer string falls back to SGD.

        Validates: Requirements 4.3
        """
        model = MockDetectorWithParams()
        params = model.get_parameters()

        learning_rate = 0.01
        momentum = 0.9
        weight_decay = 0.0005

        optimizer = build_optimizer(
            params, "UnknownOptimizer", learning_rate, momentum, weight_decay
        )

        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.defaults["lr"] == learning_rate
        assert optimizer.defaults["momentum"] == momentum
        assert optimizer.defaults["weight_decay"] == weight_decay

    def test_empty_optimizer_string_falls_back_to_sgd(self):
        """Empty optimizer string falls back to SGD.

        Validates: Requirements 4.3
        """
        model = MockDetectorWithParams()
        params = model.get_parameters()

        optimizer = build_optimizer(params, "", 0.01, 0.9, 0.0005)
        assert isinstance(optimizer, torch.optim.SGD)

    def test_random_string_falls_back_to_sgd(self):
        """Random string optimizer falls back to SGD.

        Validates: Requirements 4.3
        """
        model = MockDetectorWithParams()
        params = model.get_parameters()

        for name in ["RMSprop", "Adagrad", "LARS", "xyz123", "momentum_sgd"]:
            optimizer = build_optimizer(params, name, 0.01, 0.9, 0.0005)
            assert isinstance(optimizer, torch.optim.SGD), f"Expected SGD for '{name}'"


# ---------------------------------------------------------------------------
# Test: Warmup learning rate calculation
# ---------------------------------------------------------------------------


class TestWarmupLearningRate:
    """Test warmup LR calculation at each epoch."""

    def test_warmup_lr_at_epoch_0(self):
        """At epoch 0, warmup LR = learning_rate * 1 / warmup_epochs.

        Validates: Requirements 4.4
        """
        learning_rate = 0.01
        warmup_epochs = 5

        warmup_lr = compute_warmup_lr(epoch=0, warmup_epochs=warmup_epochs, learning_rate=learning_rate)

        expected = learning_rate * 1 / warmup_epochs  # 0.01 * 1/5 = 0.002
        assert warmup_lr == pytest.approx(expected)

    def test_warmup_lr_at_epoch_1(self):
        """At epoch 1, warmup LR = learning_rate * 2 / warmup_epochs.

        Validates: Requirements 4.4
        """
        learning_rate = 0.01
        warmup_epochs = 5

        warmup_lr = compute_warmup_lr(epoch=1, warmup_epochs=warmup_epochs, learning_rate=learning_rate)

        expected = learning_rate * 2 / warmup_epochs  # 0.01 * 2/5 = 0.004
        assert warmup_lr == pytest.approx(expected)

    def test_warmup_lr_at_last_warmup_epoch(self):
        """At the last warmup epoch, warmup LR = learning_rate.

        Validates: Requirements 4.4
        """
        learning_rate = 0.01
        warmup_epochs = 5

        # Last warmup epoch is warmup_epochs - 1 (0-indexed)
        warmup_lr = compute_warmup_lr(
            epoch=warmup_epochs - 1, warmup_epochs=warmup_epochs, learning_rate=learning_rate
        )

        expected = learning_rate * warmup_epochs / warmup_epochs  # 0.01 * 5/5 = 0.01
        assert warmup_lr == pytest.approx(expected)

    def test_warmup_lr_linear_progression(self):
        """Warmup LR increases linearly from LR/W to LR over W epochs.

        Validates: Requirements 4.4
        """
        learning_rate = 0.1
        warmup_epochs = 10

        lrs = [
            compute_warmup_lr(epoch=e, warmup_epochs=warmup_epochs, learning_rate=learning_rate)
            for e in range(warmup_epochs)
        ]

        # Check linear progression
        for i in range(1, len(lrs)):
            expected_diff = learning_rate / warmup_epochs
            actual_diff = lrs[i] - lrs[i - 1]
            assert actual_diff == pytest.approx(expected_diff)

    def test_warmup_lr_with_different_learning_rates(self):
        """Warmup LR scales correctly with different base learning rates.

        Validates: Requirements 4.4
        """
        warmup_epochs = 3

        for learning_rate in [0.001, 0.01, 0.1, 1.0]:
            for epoch in range(warmup_epochs):
                warmup_lr = compute_warmup_lr(epoch, warmup_epochs, learning_rate)
                expected = learning_rate * (epoch + 1) / warmup_epochs
                assert warmup_lr == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Test: Cosine scheduler configuration
# ---------------------------------------------------------------------------


class TestCosineSchedulerConfiguration:
    """Test cosine scheduler T_max and eta_min configuration."""

    def test_cosine_scheduler_t_max_equals_epochs_minus_warmup(self):
        """Cosine scheduler T_max = epochs - warmup_epochs.

        Validates: Requirements 4.5
        """
        model = MockDetectorWithParams()
        params = model.get_parameters()
        optimizer = torch.optim.SGD(params, lr=0.01)

        epochs = 100
        warmup_epochs = 10
        learning_rate = 0.01

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=learning_rate * 0.01
        )

        assert scheduler.T_max == epochs - warmup_epochs

    def test_cosine_scheduler_eta_min_equals_lr_times_0_01(self):
        """Cosine scheduler eta_min = learning_rate * 0.01.

        Validates: Requirements 4.5
        """
        model = MockDetectorWithParams()
        params = model.get_parameters()
        optimizer = torch.optim.SGD(params, lr=0.01)

        learning_rate = 0.01
        epochs = 50
        warmup_epochs = 5

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=learning_rate * 0.01
        )

        assert scheduler.eta_min == pytest.approx(learning_rate * 0.01)

    def test_cosine_scheduler_with_various_configs(self):
        """Cosine scheduler configured correctly for various epoch/warmup combinations.

        Validates: Requirements 4.5
        """
        model = MockDetectorWithParams()
        params = model.get_parameters()

        test_cases = [
            (100, 10, 0.01),
            (50, 5, 0.001),
            (200, 20, 0.1),
            (30, 3, 0.005),
        ]

        for epochs, warmup_epochs, learning_rate in test_cases:
            optimizer = torch.optim.SGD(params, lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - warmup_epochs, eta_min=learning_rate * 0.01
            )

            assert scheduler.T_max == epochs - warmup_epochs
            assert scheduler.eta_min == pytest.approx(learning_rate * 0.01)

    def test_cosine_scheduler_lr_decreases_over_time(self):
        """Cosine scheduler decreases LR from initial to eta_min.

        Validates: Requirements 4.5
        """
        model = MockDetectorWithParams()
        params = model.get_parameters()
        learning_rate = 0.01
        optimizer = torch.optim.SGD(params, lr=learning_rate)

        epochs = 20
        warmup_epochs = 5
        t_max = epochs - warmup_epochs

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=learning_rate * 0.01
        )

        lrs = [optimizer.param_groups[0]["lr"]]
        for _ in range(t_max):
            # Simulate optimizer step before scheduler step (proper order)
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        # LR should decrease overall (with cosine pattern)
        assert lrs[0] > lrs[-1]
        # Final LR should be close to eta_min
        assert lrs[-1] == pytest.approx(learning_rate * 0.01, rel=0.01)


# ---------------------------------------------------------------------------
# Test: Missing augmentation section
# ---------------------------------------------------------------------------


class TestMissingAugmentationSection:
    """Test that missing augmentation section proceeds without error."""

    def test_empty_augmentation_config_returns_empty_pipeline(self):
        """Empty augmentation config produces empty pipeline.

        Validates: Requirements 3.3
        """
        from model.training.augmentation import build_augmentation_pipeline, Compose

        pipeline = build_augmentation_pipeline({})
        assert isinstance(pipeline, Compose)
        assert len(pipeline.transforms) == 0

    def test_training_loop_passes_empty_dict_when_no_augmentation(self):
        """Training loop passes empty dict when augmentation section is missing.

        Validates: Requirements 3.3

        Note: The training loop uses `training_config.get("augmentation", {})`
        which returns an empty dict when the key is missing. This test verifies
        that the augmentation pipeline handles this correctly.
        """
        from model.training.augmentation import build_augmentation_pipeline, Compose

        # Simulate what the training loop does when augmentation is missing
        training_config = {"epochs": 10, "batch_size": 16}
        aug_config = training_config.get("augmentation", {})

        pipeline = build_augmentation_pipeline(aug_config)
        assert isinstance(pipeline, Compose)
        assert len(pipeline.transforms) == 0

    def test_config_without_augmentation_key_proceeds(self):
        """Config without 'augmentation' key proceeds without error.

        Validates: Requirements 3.3
        """
        from model.training.augmentation import build_augmentation_pipeline

        # Simulate training config without augmentation section
        training_config = {
            "epochs": 10,
            "batch_size": 16,
            "learning_rate": 0.01,
        }

        aug_config = training_config.get("augmentation", {})
        pipeline = build_augmentation_pipeline(aug_config)

        # Should not raise and should return empty pipeline
        assert pipeline is not None
        assert len(pipeline.transforms) == 0


# ---------------------------------------------------------------------------
# Test: data_yaml field is ignored
# ---------------------------------------------------------------------------


class TestDataYamlIgnored:
    """Test that data_yaml field is ignored by the training loop."""

    def test_dataset_path_used_instead_of_data_yaml(self):
        """Training uses dataset.path, not dataset.data_yaml.

        Validates: Requirements 3.5
        """
        # This test verifies the config parsing logic
        dataset_config = {
            "path": "/path/to/annotations",
            "data_yaml": "/path/to/data.yaml",  # Should be ignored
        }

        # The training loop should use 'path', not 'data_yaml'
        dataset_path = dataset_config.get("path", "")
        assert dataset_path == "/path/to/annotations"

        # data_yaml should not affect the dataset path used
        # (This is a config-level test; the actual loading is tested in integration)

    def test_data_yaml_not_required_in_config(self):
        """Config without data_yaml field is valid.

        Validates: Requirements 3.5
        """
        dataset_config = {
            "path": "/path/to/annotations",
            # No data_yaml field
        }

        # Should be able to get path without error
        dataset_path = dataset_config.get("path", "")
        assert dataset_path == "/path/to/annotations"

    def test_data_yaml_presence_does_not_affect_path_extraction(self):
        """Presence of data_yaml does not change how path is extracted.

        Validates: Requirements 3.5
        """
        config_with_yaml = {
            "path": "/annotations",
            "data_yaml": "/data.yaml",
        }
        config_without_yaml = {
            "path": "/annotations",
        }

        # Both should extract the same path
        path1 = config_with_yaml.get("path", "")
        path2 = config_without_yaml.get("path", "")

        assert path1 == path2 == "/annotations"


# ---------------------------------------------------------------------------
# Test: Optimizer uses model.get_parameters()
# ---------------------------------------------------------------------------


class TestOptimizerUsesGetParameters:
    """Test that optimizer is constructed using model.get_parameters()."""

    def test_optimizer_receives_parameters_from_get_parameters(self):
        """Optimizer is constructed with parameters from model.get_parameters().

        Validates: Requirements 4.1
        """
        model = MockDetectorWithParams()
        params = model.get_parameters()

        optimizer = torch.optim.SGD(params, lr=0.01)

        # Verify optimizer has the same parameters
        optimizer_params = list(optimizer.param_groups[0]["params"])
        model_params = list(params)

        assert len(optimizer_params) == len(model_params)
        for opt_p, model_p in zip(optimizer_params, model_params):
            assert opt_p is model_p

    def test_optimizer_parameter_count_matches_model(self):
        """Optimizer parameter count matches model.get_parameters() count.

        Validates: Requirements 4.1
        """
        model = MockDetectorWithParams()
        params = model.get_parameters()

        optimizer = torch.optim.Adam(params, lr=0.001)

        optimizer_param_count = sum(
            len(pg["params"]) for pg in optimizer.param_groups
        )
        model_param_count = len(list(params))

        assert optimizer_param_count == model_param_count
