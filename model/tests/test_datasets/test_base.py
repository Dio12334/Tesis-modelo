# Feature: road-damage-evaluation-framework, Property 1: Dataset split is a valid partition
"""Property-based tests for the BaseDataset split partition property.

Validates: Requirements 1.3, 1.4
"""

import random
from pathlib import Path
from typing import Iterator, List, Tuple

from hypothesis import given, settings
from hypothesis import strategies as st

from model.datasets.base import Annotation, BaseDataset, BoundingBox


class ConcreteDataset(BaseDataset):
    """Minimal concrete BaseDataset subclass for testing split behavior."""

    def __init__(self, annotations: List[Annotation] | None = None):
        self._annotations: List[Annotation] = annotations or []

    def load(self, path: Path) -> None:
        """Not used in split tests."""
        pass

    def get_annotations(self) -> List[Annotation]:
        return list(self._annotations)

    def split(
        self,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int = 42,
    ) -> Tuple["ConcreteDataset", "ConcreteDataset", "ConcreteDataset"]:
        """Split dataset using deterministic seeded shuffling."""
        annotations = list(self._annotations)
        n = len(annotations)

        # Deterministic shuffle
        rng = random.Random(seed)
        rng.shuffle(annotations)

        train_end = int(round(n * train_ratio))
        val_end = train_end + int(round(n * val_ratio))

        train_annotations = annotations[:train_end]
        val_annotations = annotations[train_end:val_end]
        test_annotations = annotations[val_end:]

        return (
            ConcreteDataset(train_annotations),
            ConcreteDataset(val_annotations),
            ConcreteDataset(test_annotations),
        )

    def __iter__(self) -> Iterator[Annotation]:
        return iter(self._annotations)

    def __len__(self) -> int:
        return len(self._annotations)

    def get_class_names(self) -> List[str]:
        classes = set()
        for ann in self._annotations:
            for bb in ann.bounding_boxes:
                classes.add(bb.class_label)
        return sorted(classes)


def _make_annotations(n: int) -> List[Annotation]:
    """Create n distinct Annotation objects for testing."""
    return [
        Annotation(
            image_path=Path(f"image_{i}.jpg"),
            bounding_boxes=[
                BoundingBox(
                    x_min=0.1,
                    y_min=0.1,
                    x_max=0.5,
                    y_max=0.5,
                    class_label="damage",
                )
            ],
            metadata={"index": i},
        )
        for i in range(n)
    ]


@st.composite
def valid_split_ratios(draw: st.DrawFn) -> Tuple[float, float, float]:
    """Generate valid split ratios that sum to 1.0.

    Generates train and val ratios in [0.1, 0.8] ensuring test >= 0.1.
    """
    # Cap train at 0.7999 to avoid floating-point edge case where
    # 1.0 - 0.8 - 0.1 < 0.1 due to IEEE 754 representation.
    train = draw(st.floats(min_value=0.1, max_value=0.7999))
    val_max = max(0.1, 1.0 - train - 0.1)
    val = draw(st.floats(min_value=0.1, max_value=min(0.8, val_max)))
    test = 1.0 - train - val
    # Guard against floating point issues
    if test < 0.0:
        test = 0.0
    return (train, val, test)


@settings(max_examples=100)
@given(
    n=st.integers(min_value=1, max_value=100),
    ratios=valid_split_ratios(),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_dataset_split_is_valid_partition(
    n: int,
    ratios: Tuple[float, float, float],
    seed: int,
) -> None:
    """Property 1: Dataset split is a valid partition.

    For any dataset with N samples and any valid split ratios
    (train + val + test = 1.0), splitting the dataset SHALL produce
    three non-overlapping subsets whose combined size equals N,
    and iterating over each subset SHALL yield exactly len(subset) items.

    **Validates: Requirements 1.3, 1.4**
    """
    train_ratio, val_ratio, test_ratio = ratios
    annotations = _make_annotations(n)
    dataset = ConcreteDataset(annotations)

    train_ds, val_ds, test_ds = dataset.split(train_ratio, val_ratio, test_ratio, seed=seed)

    # Combined size equals original dataset size N
    assert len(train_ds) + len(val_ds) + len(test_ds) == n

    # Subsets are non-overlapping (check by image_path which is unique per annotation)
    train_paths = {ann.image_path for ann in train_ds.get_annotations()}
    val_paths = {ann.image_path for ann in val_ds.get_annotations()}
    test_paths = {ann.image_path for ann in test_ds.get_annotations()}

    assert train_paths.isdisjoint(val_paths), "Train and val overlap"
    assert train_paths.isdisjoint(test_paths), "Train and test overlap"
    assert val_paths.isdisjoint(test_paths), "Val and test overlap"

    # Union covers all original annotations
    all_paths = train_paths | val_paths | test_paths
    original_paths = {ann.image_path for ann in annotations}
    assert all_paths == original_paths

    # Iterating over each subset yields exactly len(subset) items
    assert len(list(iter(train_ds))) == len(train_ds)
    assert len(list(iter(val_ds))) == len(val_ds)
    assert len(list(iter(test_ds))) == len(test_ds)
