"""Abstract dataset interface and core data models for the evaluation framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple


@dataclass
class BoundingBox:
    """A bounding box with normalized coordinates and class information.

    All coordinates are normalized to the [0, 1] range relative to image dimensions.
    """

    x_min: float  # Normalized [0, 1]
    y_min: float  # Normalized [0, 1]
    x_max: float  # Normalized [0, 1]
    y_max: float  # Normalized [0, 1]
    class_label: str
    confidence: float = 1.0  # Ground truth defaults to 1.0


@dataclass
class Annotation:
    """A single image annotation containing bounding boxes and metadata.

    Attributes:
        image_path: Path to the annotated image file.
        bounding_boxes: List of bounding boxes for detected objects.
        metadata: Flexible metadata dict (e.g., country, source, dataset name).
    """

    image_path: Path
    bounding_boxes: List[BoundingBox] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class BaseDataset(ABC):
    """Abstract base class for all dataset implementations.

    Concrete subclasses must implement all abstract methods to provide
    a unified interface for loading, iterating, and splitting datasets.
    """

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load dataset from the given path.

        Args:
            path: Root directory of the dataset.

        Raises:
            DatasetNotFoundError: If the path does not exist.
            ParseError: If annotation files are malformed.
        """
        ...

    @abstractmethod
    def get_annotations(self) -> List[Annotation]:
        """Return all annotations in unified format.

        Returns:
            List of Annotation objects for all loaded samples.
        """
        ...

    @abstractmethod
    def split(
        self,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int = 42,
    ) -> Tuple["BaseDataset", "BaseDataset", "BaseDataset"]:
        """Split dataset into train/validation/test subsets.

        Args:
            train_ratio: Fraction of data for training (0-1).
            val_ratio: Fraction of data for validation (0-1).
            test_ratio: Fraction of data for testing (0-1).
            seed: Random seed for reproducible splits.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).

        Raises:
            ValueError: If ratios do not sum to 1.0.
        """
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[Annotation]:
        """Iterate over dataset samples.

        Yields:
            Annotation objects one at a time.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        ...

    @abstractmethod
    def get_class_names(self) -> List[str]:
        """Return list of class names present in this dataset.

        Returns:
            Sorted list of unique class label strings.
        """
        ...
