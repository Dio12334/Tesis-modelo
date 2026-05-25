"""Shared test configuration, Hypothesis strategies, and pytest fixtures.

This conftest.py provides reusable Hypothesis strategies and pytest fixtures
for all test files in model/tests/ and its subdirectories.

Validates: Requirements 1.1, 2.1
"""

import random
import tempfile
from pathlib import Path
from typing import Iterator, List, Tuple

import pytest
import yaml
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st

from model.datasets.base import Annotation, BaseDataset, BoundingBox

# ---------------------------------------------------------------------------
# Hypothesis settings profile configuration
# ---------------------------------------------------------------------------

settings.register_profile("ci", max_examples=100, suppress_health_check=[HealthCheck.too_slow])
settings.register_profile("dev", max_examples=20, suppress_health_check=[HealthCheck.too_slow])
settings.load_profile("ci")


# ---------------------------------------------------------------------------
# Reusable Hypothesis Strategies
# ---------------------------------------------------------------------------


@st.composite
def bounding_boxes(draw: st.DrawFn) -> BoundingBox:
    """Generate valid BoundingBox instances with normalized coordinates.

    Ensures x_min < x_max and y_min < y_max, all within [0, 1].
    """
    x_min = draw(st.floats(min_value=0.0, max_value=0.89, allow_nan=False, allow_infinity=False))
    y_min = draw(st.floats(min_value=0.0, max_value=0.89, allow_nan=False, allow_infinity=False))
    x_max = draw(
        st.floats(
            min_value=x_min + 0.01,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    y_max = draw(
        st.floats(
            min_value=y_min + 0.01,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    class_label = draw(
        st.sampled_from(
            ["bache", "fisura_longitudinal", "fisura_transversal", "piel_de_cocodrilo"]
        )
    )
    confidence = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))

    return BoundingBox(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        class_label=class_label,
        confidence=confidence,
    )


@st.composite
def annotations(draw: st.DrawFn) -> Annotation:
    """Generate valid Annotation instances with bounding boxes and metadata."""
    num_boxes = draw(st.integers(min_value=0, max_value=5))
    boxes = [draw(bounding_boxes()) for _ in range(num_boxes)]

    image_name = draw(
        st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(whitelist_categories=("L", "N")),
        )
    )
    image_path = Path(f"{image_name}.jpg")

    country = draw(st.sampled_from(["Japan", "India", "Czech", "Norway", "USA", "China"]))
    metadata = {"country": country}

    return Annotation(
        image_path=image_path,
        bounding_boxes=boxes,
        metadata=metadata,
    )


@st.composite
def mapping_configs(draw: st.DrawFn) -> dict:
    """Generate valid target mapping configuration dicts.

    Produces a dict with 'taxonomy', 'mappings', and 'default_class' keys,
    where all mapped target classes are present in the taxonomy.
    """
    class_name_strategy = st.text(
        min_size=1,
        max_size=12,
        alphabet=st.characters(whitelist_categories=("L",)),
    )

    # Generate taxonomy (at least 1 target class)
    taxonomy = draw(st.lists(class_name_strategy, min_size=1, max_size=6, unique=True))

    # Generate source classes (distinct from each other)
    source_classes = draw(st.lists(class_name_strategy, min_size=1, max_size=8, unique=True))

    # Map each source to a random target from the taxonomy
    mappings = {}
    for source in source_classes:
        target = draw(st.sampled_from(taxonomy))
        mappings[source] = target

    return {
        "taxonomy": taxonomy,
        "mappings": mappings,
        "default_class": None,
    }


@st.composite
def training_configs(draw: st.DrawFn) -> dict:
    """Generate valid training configuration dicts.

    Produces a dict matching the structure of default_training.yaml with
    randomized but valid hyperparameter values.
    """
    epochs = draw(st.integers(min_value=1, max_value=300))
    batch_size = draw(st.sampled_from([4, 8, 16, 32, 64]))
    learning_rate = draw(
        st.floats(min_value=1e-5, max_value=0.1, allow_nan=False, allow_infinity=False)
    )
    optimizer = draw(st.sampled_from(["SGD", "Adam", "AdamW"]))
    weight_decay = draw(
        st.floats(min_value=0.0, max_value=0.01, allow_nan=False, allow_infinity=False)
    )
    momentum = draw(
        st.floats(min_value=0.8, max_value=0.999, allow_nan=False, allow_infinity=False)
    )
    scheduler = draw(st.sampled_from(["cosine", "step", "plateau"]))
    warmup_epochs = draw(st.integers(min_value=0, max_value=10))
    val_split = draw(
        st.floats(min_value=0.1, max_value=0.4, allow_nan=False, allow_infinity=False)
    )
    log_interval = draw(st.integers(min_value=1, max_value=100))

    horizontal_flip = draw(st.booleans())
    vertical_flip = draw(st.booleans())
    rotation_range = draw(st.integers(min_value=0, max_value=45))
    brightness_low = draw(
        st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    brightness_high = draw(
        st.floats(
            min_value=brightness_low,
            max_value=1.5,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    mosaic = draw(st.booleans())

    return {
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": optimizer,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "scheduler": scheduler,
            "warmup_epochs": warmup_epochs,
            "val_split": val_split,
            "checkpoint_dir": "./checkpoints",
            "log_interval": log_interval,
            "augmentation": {
                "horizontal_flip": horizontal_flip,
                "vertical_flip": vertical_flip,
                "rotation_range": rotation_range,
                "brightness_range": [brightness_low, brightness_high],
                "mosaic": mosaic,
            },
        }
    }


# ---------------------------------------------------------------------------
# Concrete Dataset for testing (reusable across test modules)
# ---------------------------------------------------------------------------


class ConcreteDataset(BaseDataset):
    """Minimal concrete BaseDataset subclass for testing purposes."""

    def __init__(self, annotations_list: List[Annotation] | None = None):
        self._annotations: List[Annotation] = annotations_list or []

    def load(self, path: Path) -> None:
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
        annotations_copy = list(self._annotations)
        n = len(annotations_copy)

        rng = random.Random(seed)
        rng.shuffle(annotations_copy)

        train_end = int(round(n * train_ratio))
        val_end = train_end + int(round(n * val_ratio))

        return (
            ConcreteDataset(annotations_copy[:train_end]),
            ConcreteDataset(annotations_copy[train_end:val_end]),
            ConcreteDataset(annotations_copy[val_end:]),
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


# ---------------------------------------------------------------------------
# Pytest Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dataset_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample XML annotations and images.

    Provides a minimal RDD2022-like dataset structure with:
    - 3 sample XML annotation files
    - 3 corresponding (empty) image files
    """
    # Create sample images (empty files)
    images = ["Japan_000001.jpg", "Japan_000002.jpg", "India_000001.jpg"]
    for img_name in images:
        (tmp_path / img_name).write_bytes(b"\x00" * 100)

    # Create sample XML annotations
    xml_template = """\
<annotation>
    <filename>{filename}</filename>
    <size>
        <width>600</width>
        <height>600</height>
    </size>
    <object>
        <name>{class_name}</name>
        <bndbox>
            <xmin>{xmin}</xmin>
            <ymin>{ymin}</ymin>
            <xmax>{xmax}</xmax>
            <ymax>{ymax}</ymax>
        </bndbox>
    </object>
</annotation>"""

    annotation_data = [
        ("Japan_000001.jpg", "D00", 100, 150, 300, 350),
        ("Japan_000002.jpg", "D10", 50, 60, 200, 250),
        ("India_000001.jpg", "D40", 200, 200, 400, 500),
    ]

    for filename, class_name, xmin, ymin, xmax, ymax in annotation_data:
        xml_content = xml_template.format(
            filename=filename,
            class_name=class_name,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )
        xml_name = filename.replace(".jpg", ".xml")
        (tmp_path / xml_name).write_text(xml_content)

    return tmp_path


@pytest.fixture
def sample_yaml_config(tmp_path: Path) -> Path:
    """Create a sample YAML config file in a temporary directory.

    Returns the path to the YAML file containing a valid training configuration.
    """
    config = {
        "training": {
            "epochs": 50,
            "batch_size": 16,
            "learning_rate": 0.01,
            "optimizer": "SGD",
            "weight_decay": 0.0005,
            "momentum": 0.937,
            "scheduler": "cosine",
            "warmup_epochs": 3,
            "val_split": 0.2,
            "checkpoint_dir": "./checkpoints",
            "log_interval": 10,
            "augmentation": {
                "horizontal_flip": True,
                "vertical_flip": False,
                "rotation_range": 15,
                "brightness_range": [0.8, 1.2],
                "mosaic": True,
            },
        }
    }

    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path


@pytest.fixture
def mock_dataset() -> "ConcreteDataset":
    """Create a ConcreteDataset with sample annotations for testing.

    Returns a dataset with 5 annotations spanning multiple classes and countries.
    """
    sample_annotations = [
        Annotation(
            image_path=Path("Japan_000001.jpg"),
            bounding_boxes=[
                BoundingBox(
                    x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.6,
                    class_label="fisura_longitudinal", confidence=1.0,
                ),
                BoundingBox(
                    x_min=0.3, y_min=0.4, x_max=0.7, y_max=0.8,
                    class_label="bache", confidence=0.95,
                ),
            ],
            metadata={"country": "Japan"},
        ),
        Annotation(
            image_path=Path("Japan_000002.jpg"),
            bounding_boxes=[
                BoundingBox(
                    x_min=0.05, y_min=0.1, x_max=0.4, y_max=0.5,
                    class_label="fisura_transversal", confidence=1.0,
                ),
            ],
            metadata={"country": "Japan"},
        ),
        Annotation(
            image_path=Path("India_000001.jpg"),
            bounding_boxes=[
                BoundingBox(
                    x_min=0.2, y_min=0.3, x_max=0.6, y_max=0.7,
                    class_label="piel_de_cocodrilo", confidence=1.0,
                ),
            ],
            metadata={"country": "India"},
        ),
        Annotation(
            image_path=Path("Czech_000001.jpg"),
            bounding_boxes=[
                BoundingBox(
                    x_min=0.15, y_min=0.25, x_max=0.55, y_max=0.65,
                    class_label="bache", confidence=0.88,
                ),
            ],
            metadata={"country": "Czech"},
        ),
        Annotation(
            image_path=Path("Norway_000001.jpg"),
            bounding_boxes=[
                BoundingBox(
                    x_min=0.0, y_min=0.0, x_max=0.3, y_max=0.4,
                    class_label="fisura_longitudinal", confidence=1.0,
                ),
                BoundingBox(
                    x_min=0.5, y_min=0.5, x_max=0.9, y_max=0.9,
                    class_label="fisura_transversal", confidence=0.92,
                ),
            ],
            metadata={"country": "Norway"},
        ),
    ]

    return ConcreteDataset(sample_annotations)
