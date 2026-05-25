"""Target class mapping between datasets using YAML configuration."""

from pathlib import Path
from typing import List

import yaml

from model.exceptions import ConfigurationError, UnmappedClassError


class TargetMapper:
    """Maps damage class labels between datasets using YAML configuration.

    Supports many-to-one mappings and configurable default class behavior.
    """

    def __init__(self, config_path: Path, strict: bool = True):
        """Load mapping configuration from a YAML file.

        Args:
            config_path: Path to YAML mapping file containing taxonomy,
                mappings, and default_class fields.
            strict: If True, raises UnmappedClassError for unknown classes.
                If False, returns the configured default class (or raises
                UnmappedClassError if default_class is null/None).
        """
        self.config_path = Path(config_path)
        self.strict = strict

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        self.taxonomy: List[str] = config.get("taxonomy", [])
        self.mappings: dict = config.get("mappings", {})
        self.default_class: str | None = config.get("default_class", None)

    def map_class(self, source_class: str) -> str:
        """Map a source class to its target class.

        Args:
            source_class: The source class label to map.

        Returns:
            The corresponding target class label.

        Raises:
            UnmappedClassError: If source_class has no mapping and strict mode
                is enabled, or if default_class is None in non-strict mode.
        """
        if source_class in self.mappings:
            return self.mappings[source_class]

        if self.strict:
            raise UnmappedClassError(
                source_class=source_class,
                available_classes=list(self.mappings.keys()),
            )

        # Non-strict mode: return default_class if configured
        if self.default_class is not None:
            return self.default_class

        # default_class is None even in non-strict mode
        raise UnmappedClassError(
            source_class=source_class,
            available_classes=list(self.mappings.keys()),
        )

    def reverse_map(self, target_class: str) -> List[str]:
        """Return all source classes that map to the given target class.

        Args:
            target_class: The target class to reverse-lookup.

        Returns:
            List of source classes that map to target_class.
        """
        return [
            source
            for source, target in self.mappings.items()
            if target == target_class
        ]

    def validate(self, target_taxonomy: List[str]) -> None:
        """Validate that all target classes in mappings exist in the taxonomy.

        Args:
            target_taxonomy: List of valid target class names.

        Raises:
            ConfigurationError: If any mapped target classes are not present
                in target_taxonomy.
        """
        invalid_classes = []
        for source, target in self.mappings.items():
            if target not in target_taxonomy:
                invalid_classes.append(target)

        if invalid_classes:
            # Deduplicate while preserving order
            seen = set()
            unique_invalid = []
            for cls in invalid_classes:
                if cls not in seen:
                    seen.add(cls)
                    unique_invalid.append(cls)

            raise ConfigurationError(
                violations=[
                    f"Invalid target class '{cls}' not in taxonomy"
                    for cls in unique_invalid
                ]
            )
