"""Checkpoint verification script for Task 4: Ensure all core implementation is complete."""

import sys

print("=" * 60)
print("RT-DETR Detector - Checkpoint Verification")
print("=" * 60)

# 1. Verify module imports correctly (no syntax errors)
print("\n[1] Testing module import...")
try:
    from model.models import RT_DETR_Detector, ModelRegistry
    print("    ✓ Import OK: RT_DETR_Detector and ModelRegistry imported successfully")
except Exception as e:
    print(f"    ✗ Import FAILED: {e}")
    sys.exit(1)

# 2. Verify class is registered in ModelRegistry
print("\n[2] Testing ModelRegistry registration...")
models = ModelRegistry.list_models()
print(f"    Registered models: {models}")
if "rt_detr" in models:
    print("    ✓ 'rt_detr' is registered in ModelRegistry")
else:
    print("    ✗ 'rt_detr' NOT found in ModelRegistry")
    sys.exit(1)

# 3. Verify YAML config loads correctly
print("\n[3] Testing YAML config loading...")
try:
    import yaml
    from pathlib import Path

    config_path = Path("model/configs/train_rt_detr.yaml")
    if not config_path.exists():
        print(f"    ✗ Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"    ✓ YAML config loaded from: {config_path}")
    print(f"    Model type: {config.get('model', {}).get('type', 'N/A')}")
    print(f"    Model size: {config.get('model', {}).get('config', {}).get('model_size', 'N/A')}")
    print(f"    Num classes: {config.get('model', {}).get('config', {}).get('num_classes', 'N/A')}")
    print(f"    Batch size: {config.get('training', {}).get('batch_size', 'N/A')}")
    print(f"    Input size: {config.get('model', {}).get('config', {}).get('input_size', 'N/A')}")

    # Validate key values
    assert config["model"]["type"] == "rt_detr", f"Expected model type 'rt_detr', got '{config['model']['type']}'"
    assert config["model"]["config"]["model_size"] == "l", "Expected model_size 'l'"
    assert config["model"]["config"]["num_classes"] == 5, "Expected num_classes 5"
    assert config["training"]["batch_size"] == 4, "Expected batch_size 4"
    assert config["model"]["config"]["input_size"] == 640, "Expected input_size 640"
    print("    ✓ All YAML config values validated correctly")
except Exception as e:
    print(f"    ✗ YAML config test FAILED: {e}")
    sys.exit(1)

# 4. Verify RT_DETR_Detector is a subclass of BaseDetector
print("\n[4] Testing class hierarchy...")
from model.models.registry import BaseDetector
if issubclass(RT_DETR_Detector, BaseDetector):
    print("    ✓ RT_DETR_Detector is a subclass of BaseDetector")
else:
    print("    ✗ RT_DETR_Detector is NOT a subclass of BaseDetector")
    sys.exit(1)

# 5. Verify configuration validation works
print("\n[5] Testing configuration validation...")
from model.exceptions import ConfigurationError

# Test valid config raises no error (with mocked RTDETR)
try:
    from unittest.mock import patch, MagicMock
    import torch

    mock_model = MagicMock()
    mock_model.model.parameters.return_value = [torch.nn.Parameter(torch.randn(3, 3))]
    mock_model.model.init_criterion.return_value = MagicMock()

    with patch("model.models.rt_detr_wrapper.RTDETR", return_value=mock_model):
        detector = RT_DETR_Detector({
            "model_size": "l",
            "num_classes": 5,
        })
    print("    ✓ Valid config accepted without error")
except Exception as e:
    print(f"    ✗ Valid config raised unexpected error: {e}")
    sys.exit(1)

# Test invalid config raises ConfigurationError
try:
    with patch("model.models.rt_detr_wrapper.RTDETR", return_value=mock_model):
        RT_DETR_Detector({"model_size": "invalid", "num_classes": -1})
    print("    ✗ Invalid config did NOT raise ConfigurationError")
    sys.exit(1)
except ConfigurationError as e:
    print(f"    ✓ Invalid config raises ConfigurationError: {e.violations}")
except Exception as e:
    print(f"    ✗ Invalid config raised wrong exception: {type(e).__name__}: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL CHECKPOINT VERIFICATIONS PASSED ✓")
print("=" * 60)
