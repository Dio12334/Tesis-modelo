"""Tests for strict class-label resolution in ``RDD2022TorchDataset``.

The dataset adapter previously coerced unknown raw class labels to index 0
via ``self._class_to_idx.get(label, 0)``. That silently mislabeled
out-of-vocabulary boxes as the first taxonomy class, hiding real bugs (e.g.,
augmentation pipelines that inject foreign labels, or a stale class map
shipped with a checkpoint) under a poisoned training signal.

These tests pin the new contract:

* A bbox carrying a label present in ``_class_to_idx`` is resolved to its
  integer index.
* A bbox carrying a label *not* in ``_class_to_idx`` raises ``ValueError``
  with the offending label, the available labels, and the source context
  (image path or mosaic index) embedded in the message.

The tests use a synthetic single-image dataset to keep the exercise focused
on the resolution path; the surrounding ``__getitem__`` plumbing (resize,
augmentation, mosaic toggle) is exercised by ``test_dataset_alignment.py``
and ``test_mosaic.py`` and is not under test here.
"""

from pathlib import Path
from typing import List

import numpy as np
import pytest
from PIL import Image as PILImage

from model.datasets.base import Annotation, BoundingBox
from model.training.train_detection import RDD2022TorchDataset


IMG_SIZE = 64
INPUT_SIZE = 32


class _FakeDataset:
    """Minimal duck-typed dataset stub used by these tests only."""

    def __init__(self, annotations: List[Annotation], class_names: List[str]):
        self._annotations = annotations
        self._class_names = class_names

    def get_annotations(self) -> List[Annotation]:
        return self._annotations

    def get_class_names(self) -> List[str]:
        return self._class_names


def _make_image(tmp_path: Path) -> Path:
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    img[16:48, 16:48] = 255
    path = tmp_path / "img.png"
    PILImage.fromarray(img).save(path)
    return path


def _annotation_with_label(image_path: Path, label: str) -> Annotation:
    return Annotation(
        image_path=image_path,
        bounding_boxes=[
            BoundingBox(
                x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75,
                class_label=label,
            )
        ],
    )


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_resolve_class_idx_returns_known_label_index(tmp_path):
    image_path = _make_image(tmp_path)
    fake = _FakeDataset(
        [_annotation_with_label(image_path, "pothole")],
        class_names=["alligator crack", "longitudinal crack", "pothole"],
    )
    ds = RDD2022TorchDataset(fake, input_size=INPUT_SIZE, augmentation=None)

    assert ds._resolve_class_idx("pothole") == 2
    assert ds._resolve_class_idx("alligator crack") == 0


def test_getitem_returns_correct_label_index_for_known_class(tmp_path):
    """The strict path is exercised end-to-end via __getitem__."""
    image_path = _make_image(tmp_path)
    fake = _FakeDataset(
        [_annotation_with_label(image_path, "pothole")],
        class_names=["alligator crack", "longitudinal crack", "pothole"],
    )
    ds = RDD2022TorchDataset(fake, input_size=INPUT_SIZE, augmentation=None)

    _, target = ds[0]
    assert target["labels"].tolist() == [2]


# --------------------------------------------------------------------------
# Strict-failure path
# --------------------------------------------------------------------------


def test_resolve_class_idx_raises_on_unknown_label(tmp_path):
    image_path = _make_image(tmp_path)
    fake = _FakeDataset(
        [_annotation_with_label(image_path, "pothole")],
        class_names=["alligator crack", "longitudinal crack", "pothole"],
    )
    ds = RDD2022TorchDataset(fake, input_size=INPUT_SIZE, augmentation=None)

    with pytest.raises(ValueError) as exc_info:
        ds._resolve_class_idx("lava_crack")

    msg = str(exc_info.value)
    assert "lava_crack" in msg, "Offending label must appear in error message"
    assert "pothole" in msg, "Known labels must appear in error message"


def test_resolve_class_idx_includes_context_when_provided(tmp_path):
    image_path = _make_image(tmp_path)
    fake = _FakeDataset(
        [_annotation_with_label(image_path, "pothole")],
        class_names=["pothole"],
    )
    ds = RDD2022TorchDataset(fake, input_size=INPUT_SIZE, augmentation=None)

    with pytest.raises(ValueError) as exc_info:
        ds._resolve_class_idx("foo", context="mosaic primary idx=42")

    assert "mosaic primary idx=42" in str(exc_info.value)


def test_getitem_raises_when_annotation_carries_unknown_label(tmp_path):
    """End-to-end: dataset built from labels that don't match its taxonomy.

    The annotation says ``mystery_class`` but ``class_names`` only registers
    ``pothole``; the loader must raise rather than silently coerce to 0.
    """
    image_path = _make_image(tmp_path)
    fake = _FakeDataset(
        [_annotation_with_label(image_path, "mystery_class")],
        class_names=["pothole"],  # NOTE: no "mystery_class" here
    )
    ds = RDD2022TorchDataset(fake, input_size=INPUT_SIZE, augmentation=None)

    with pytest.raises(ValueError) as exc_info:
        _ = ds[0]

    msg = str(exc_info.value)
    assert "mystery_class" in msg
    assert str(image_path) in msg or image_path.name in msg
