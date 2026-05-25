"""Configuration management: loading, validation, merging, and saving YAML configs."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List

import yaml

from model.exceptions import ValidationError


class ConfigManager:
    """Handles YAML configuration loading, validation, and inheritance."""

    _ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

    def load(self, path: Path) -> dict:
        """Load and parse a YAML config file.

        Args:
            path: Path to the YAML file.

        Returns:
            Parsed configuration dictionary.

        Raises:
            ValidationError: If the file cannot be read or contains invalid YAML.
        """
        path = Path(path)
        if not path.exists():
            raise ValidationError([f"Configuration file not found: {path}"])
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValidationError([f"Invalid YAML in {path}: {e}"])

        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValidationError(
                [f"Expected a YAML mapping at top level in {path}, got {type(data).__name__}"]
            )
        return data

    def merge(self, parent: dict, child: dict) -> dict:
        """Deep merge parent and child configs with child taking precedence.

        For nested dicts, merge is applied recursively. For all other types
        (including lists), the child value wins outright.

        Args:
            parent: Base configuration dictionary.
            child: Override configuration dictionary.

        Returns:
            Merged configuration dictionary.
        """
        result = parent.copy()
        for key, child_value in child.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(child_value, dict)
            ):
                result[key] = self.merge(result[key], child_value)
            else:
                result[key] = child_value
        return result

    def validate(self, config: dict, schema: dict) -> None:
        """Validate config against a schema definition.

        The schema is a dictionary describing expected structure:
        - "required": list of required top-level keys
        - "properties": dict mapping key names to property schemas
          Each property schema can have:
            - "type": expected Python type name (str, int, float, bool, list, dict)
            - "min": minimum numeric value
            - "max": maximum numeric value
            - "enum": list of allowed values
            - "required": for nested dicts, list of required sub-keys
            - "properties": for nested dicts, nested property schemas

        All violations are collected and raised together.

        Args:
            config: Configuration dictionary to validate.
            schema: Schema dictionary describing constraints.

        Raises:
            ValidationError: With a list of all schema violations found.
        """
        violations: List[str] = []
        self._validate_node(config, schema, "", violations)
        if violations:
            raise ValidationError(violations)

    def save(self, config: dict, path: Path) -> None:
        """Save configuration dictionary to a YAML file.

        Args:
            config: Configuration dictionary to save.
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def resolve_env_vars(self, config: Any) -> Any:
        """Recursively substitute ${VAR_NAME} patterns with environment variable values.

        Walks the config structure (dicts, lists, strings) and replaces
        occurrences of ${VAR_NAME} with the value of the corresponding
        environment variable. If the variable is not set, the pattern is
        left unchanged.

        Args:
            config: Configuration value (dict, list, or scalar).

        Returns:
            Configuration with environment variables resolved.
        """
        if isinstance(config, dict):
            return {key: self.resolve_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self.resolve_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._substitute_env_vars(config)
        else:
            return config

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _substitute_env_vars(self, value: str) -> str:
        """Replace all ${VAR_NAME} patterns in a string with env var values."""

        def _replacer(match: re.Match) -> str:
            var_name = match.group(1)
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            # Leave unset variables as-is
            return match.group(0)

        return self._ENV_VAR_PATTERN.sub(_replacer, value)

    def _validate_node(
        self, config: Any, schema: dict, path_prefix: str, violations: List[str]
    ) -> None:
        """Recursively validate a config node against its schema."""
        # Check required fields
        required_fields = schema.get("required", [])
        if isinstance(config, dict):
            for field in required_fields:
                if field not in config:
                    field_path = f"{path_prefix}.{field}" if path_prefix else field
                    violations.append(f"Missing required field: '{field_path}'")

        # Check properties
        properties = schema.get("properties", {})
        if isinstance(config, dict):
            for key, prop_schema in properties.items():
                if key in config:
                    key_path = f"{path_prefix}.{key}" if path_prefix else key
                    self._validate_property(config[key], prop_schema, key_path, violations)

    def _validate_property(
        self, value: Any, prop_schema: dict, path: str, violations: List[str]
    ) -> None:
        """Validate a single property value against its schema."""
        # Type check
        if "type" in prop_schema:
            expected_type = self._resolve_type(prop_schema["type"])
            if expected_type is not None and not isinstance(value, expected_type):
                violations.append(
                    f"Field '{path}' expected type '{prop_schema['type']}', "
                    f"got '{type(value).__name__}'"
                )
                return  # Skip further checks if type is wrong

        # Enum check
        if "enum" in prop_schema:
            if value not in prop_schema["enum"]:
                violations.append(
                    f"Field '{path}' value '{value}' not in allowed values: {prop_schema['enum']}"
                )

        # Min/max checks
        if "min" in prop_schema:
            if isinstance(value, (int, float)) and value < prop_schema["min"]:
                violations.append(
                    f"Field '{path}' value {value} is below minimum {prop_schema['min']}"
                )

        if "max" in prop_schema:
            if isinstance(value, (int, float)) and value > prop_schema["max"]:
                violations.append(
                    f"Field '{path}' value {value} is above maximum {prop_schema['max']}"
                )

        # Nested dict validation
        if isinstance(value, dict) and ("properties" in prop_schema or "required" in prop_schema):
            self._validate_node(value, prop_schema, path, violations)

    @staticmethod
    def _resolve_type(type_name: str):
        """Map type name string to Python type."""
        type_map = {
            "str": str,
            "int": int,
            "float": (int, float),  # Accept int as float
            "bool": bool,
            "list": list,
            "dict": dict,
        }
        return type_map.get(type_name)
