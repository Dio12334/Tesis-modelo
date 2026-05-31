"""Unit tests verifying legacy code removal from train_detection.py.

These tests ensure that the deprecated `_train_ultralytics()` function and all
YOLO26 special-case logic have been removed from the training module.

Feature: unified-training-loop, Task 5.2: Verify legacy code removal

**Validates: Requirements 11.1, 1.2, 2.3**
"""

import ast
import inspect
import re
from pathlib import Path

import pytest

from model.training import train_detection


class TestLegacyCodeRemoval:
    """Tests verifying that legacy code has been removed from train_detection.py."""

    def test_train_ultralytics_not_defined(self):
        """Verify that _train_ultralytics is not defined in train_detection.py.
        
        **Validates: Requirements 11.1**
        """
        # Check that the function doesn't exist as an attribute
        assert not hasattr(train_detection, "_train_ultralytics"), (
            "_train_ultralytics function should not exist in train_detection module"
        )
        
        # Also verify by checking the module's __dict__
        assert "_train_ultralytics" not in dir(train_detection), (
            "_train_ultralytics should not be in train_detection's namespace"
        )

    def test_no_model_internal_access(self):
        """Verify that no model._model access exists in the training module source.
        
        **Validates: Requirements 11.2**
        """
        # Get the source code of the module
        source_file = Path(train_detection.__file__)
        source_code = source_file.read_text(encoding="utf-8")
        
        # Check for model._model pattern
        pattern = r"model\._model"
        matches = re.findall(pattern, source_code)
        
        assert len(matches) == 0, (
            f"Found {len(matches)} occurrences of 'model._model' in train_detection.py. "
            "All internal model access should go through the BaseDetector interface."
        )

    def test_no_model_type_conditionals(self):
        """Verify that no model-type conditionals exist in the training loop.
        
        **Validates: Requirements 2.3, 11.3**
        """
        source_file = Path(train_detection.__file__)
        source_code = source_file.read_text(encoding="utf-8")
        
        # Patterns that indicate model-type branching
        patterns = [
            r'if\s+model_type\s*==',
            r'if\s+["\']yolo',
            r'if\s+["\']ssd',
            r'if\s+model_type\s+in\s+\[',
            r'model_type\s*==\s*["\']yolo26["\']',
            r'model_type\s*==\s*["\']YOLO26["\']',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, source_code, re.IGNORECASE)
            assert len(matches) == 0, (
                f"Found model-type conditional matching pattern '{pattern}' in train_detection.py. "
                "The unified training loop should not branch based on model type."
            )

    def test_no_ultralytics_imports(self):
        """Verify that no Ultralytics-specific imports exist in train_detection.py.
        
        **Validates: Requirements 11.4**
        """
        source_file = Path(train_detection.__file__)
        source_code = source_file.read_text(encoding="utf-8")
        
        # Parse the AST to find import statements
        tree = ast.parse(source_code)
        
        ultralytics_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "ultralytics" in alias.name.lower():
                        ultralytics_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and "ultralytics" in node.module.lower():
                    ultralytics_imports.append(node.module)
        
        assert len(ultralytics_imports) == 0, (
            f"Found Ultralytics imports in train_detection.py: {ultralytics_imports}. "
            "The unified training loop should not depend on Ultralytics internals."
        )

    def test_no_yolo26_references_in_code(self):
        """Verify that no YOLO26 special-case references exist in the code.
        
        **Validates: Requirements 11.3**
        """
        source_file = Path(train_detection.__file__)
        source_code = source_file.read_text(encoding="utf-8")
        
        # Check for yolo26 references (case-insensitive)
        # Exclude comments and docstrings by checking actual code patterns
        patterns = [
            r'["\']yolo26["\']',  # String literals
            r'yolo26\s*=',        # Variable assignments
            r'_train_ultralytics',  # Legacy function name
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, source_code, re.IGNORECASE)
            assert len(matches) == 0, (
                f"Found YOLO26 reference matching pattern '{pattern}' in train_detection.py. "
                "All YOLO26 special-case logic should be removed."
            )

    def test_train_function_is_model_agnostic(self):
        """Verify that the train() function signature is model-agnostic.
        
        The function should only accept config_path and verbose, not model_type.
        
        **Validates: Requirements 1.2**
        """
        sig = inspect.signature(train_detection.train)
        param_names = list(sig.parameters.keys())
        
        # Should have config_path and verbose
        assert "config_path" in param_names, "train() should accept config_path parameter"
        assert "verbose" in param_names, "train() should accept verbose parameter"
        
        # Should NOT have model_type as a parameter
        assert "model_type" not in param_names, (
            "train() should not accept model_type parameter - "
            "model type should be determined from config"
        )

    def test_no_data_yaml_special_handling(self):
        """Verify that data_yaml field is not specially handled for YOLO models.
        
        The unified loop should use the standard dataset path, not data.yaml.
        
        **Validates: Requirements 3.5**
        """
        source_file = Path(train_detection.__file__)
        source_code = source_file.read_text(encoding="utf-8")
        
        # Check that data_yaml is not used to construct paths
        # It's OK to read it from config, but it shouldn't be used for data loading
        patterns = [
            r'data_yaml\s*=.*data\.yaml',
            r'YOLO\s*\(',  # Direct YOLO instantiation
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, source_code)
            assert len(matches) == 0, (
                f"Found data_yaml special handling matching pattern '{pattern}'. "
                "The unified loop should use RDD2022Dataset, not YOLO's data.yaml format."
            )


class TestModuleStructure:
    """Tests verifying the module structure after legacy code removal."""

    def test_module_has_train_function(self):
        """Verify that the train() function exists and is callable."""
        assert hasattr(train_detection, "train"), "train_detection should have train function"
        assert callable(train_detection.train), "train should be callable"

    def test_module_has_dataset_classes(self):
        """Verify that the dataset adapter classes exist."""
        assert hasattr(train_detection, "RDD2022TorchDataset"), (
            "train_detection should have RDD2022TorchDataset class"
        )
        assert hasattr(train_detection, "collate_fn"), (
            "train_detection should have collate_fn function"
        )

    def test_no_private_training_functions(self):
        """Verify that no private training functions exist (like _train_ultralytics).
        
        The only training entry point should be the public train() function.
        """
        # Get all functions in the module
        functions = [
            name for name in dir(train_detection)
            if callable(getattr(train_detection, name))
            and name.startswith("_train")
        ]
        
        # Filter out dunder methods
        private_train_funcs = [f for f in functions if not f.startswith("__")]
        
        assert len(private_train_funcs) == 0, (
            f"Found private training functions: {private_train_funcs}. "
            "Only the public train() function should exist."
        )
