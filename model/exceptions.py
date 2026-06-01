"""Custom exception hierarchy for the Road Damage Evaluation Framework."""

from pathlib import Path
from typing import List, Optional


class FrameworkError(Exception):
    """Base exception for all framework errors."""

    pass


class DatasetNotFoundError(FrameworkError):
    """Raised when a dataset path does not exist."""

    def __init__(self, path: Path):
        self.path = path
        super().__init__(f"Dataset not found: {path}")


class ParseError(FrameworkError):
    """Raised when an annotation file cannot be parsed."""

    def __init__(self, file_path: Path, line_number: int, description: str):
        self.file_path = file_path
        self.line_number = line_number
        self.description = description
        super().__init__(
            f"Parse error in {file_path} at line {line_number}: {description}"
        )


class UnmappedClassError(FrameworkError):
    """Raised when a source class has no mapping (strict mode)."""

    def __init__(self, source_class: str, available_classes: List[str]):
        self.source_class = source_class
        self.available_classes = available_classes
        super().__init__(
            f"No mapping for class '{source_class}'. Available: {available_classes}"
        )


class ModelNotFoundError(FrameworkError):
    """Raised when requesting an unregistered model.

    The message lists the available models in alphabetical order and, when
    provided, appends a "Did you mean" suggestion and the underlying cause of
    a failed instantiation. The original two-argument constructor remains
    supported for backward compatibility.
    """

    def __init__(
        self,
        model_name: str,
        available_models: List[str],
        suggestion: Optional[str] = None,
        cause: Optional[str] = None,
    ):
        self.model_name = model_name
        self.available_models = sorted(available_models)
        self.suggestion = suggestion
        self.cause = cause
        lines = [
            f"Model '{model_name}' not found.",
            f"Available models: {self.available_models}",
        ]
        if suggestion:
            lines.append(f"Did you mean: {suggestion}?")
        if cause:
            lines.append(f"Underlying error: {cause}")
        super().__init__(" ".join(lines))


class ConfigurationError(FrameworkError):
    """Raised for configuration validation failures."""

    def __init__(self, violations: List[str]):
        self.violations = violations
        super().__init__(f"Configuration errors: {'; '.join(violations)}")


class ValidationError(FrameworkError):
    """Raised when a config file fails schema validation."""

    def __init__(self, schema_violations: List[str]):
        self.schema_violations = schema_violations
        super().__init__(f"Schema validation failed: {'; '.join(schema_violations)}")
