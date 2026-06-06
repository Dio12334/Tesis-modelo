"""Tests for train↔eval class-index alignment fix (Option B2).

These tests pin the contract introduced by the
``.kiro/specs/eval-class-index-alignment/design.md`` fix:

* The canonical class-index space used by ``load_split`` must equal the one
  ``train_detection.RDD2022TorchDataset`` builds at training time -- the
  sorted list of raw English labels returned by
  ``RDD2022Dataset.get_class_names()`` -- regardless of whether
  ``dataset.class_mapping`` is configured.
* The 4th element of ``load_split``'s return tuple is now a list of display
  class names (Spanish when a class_mapping is configured, English otherwise),
  never a callable label-mapper.
* Ground-truth labels emitted by ``_build_ground_truth`` are the raw English
  ``bbox.class_label`` strings -- unmodified by any taxonomy mapping.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from model.datasets.target_mapper import TargetMapper
from model.training.evaluate_detection import _build_ground_truth, load_split

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_DIR = REPO_ROOT / "model" / "data" / "rdd2022" / "sample"
RDD2022_CLASSES_YAML = REPO_ROOT / "model" / "configs" / "rdd2022_classes.yaml"


# Raw labels deliberately given out-of-order so the sorted-English contract is
# observable. ``other corruption`` is included specifically because it has no
# mapping in the YAML and must therefore fall back to the canonical English
# name in ``display_class_names`` (rather than aborting evaluation).
RAW_ENGLISH_LABELS = [
    "transverse crack",
    "alligator crack",
    "pothole",
    "longitudinal crack",
    "other corruption",
]
SORTED_ENGLISH_LABELS = sorted(RAW_ENGLISH_LABELS)


class _FakeRDD2022Dataset:
    """Conforming stub that returns deterministic raw English class names."""

    def __init__(self, country_filter=None, subset="train"):
        self.country_filter = country_filter
        self.subset = subset

    def load(self, path):  # noqa: D401  (stub)
        return None

    def get_class_names(self):
        # ``RDD2022Dataset.get_class_names`` returns a sorted set of raw
        # ``bbox.class_label`` values; emulate that here.
        return list(SORTED_ENGLISH_LABELS)

    def get_annotations(self):
        return []

    def __len__(self):
        return 0


def _config(split: str = "test", with_mapping: bool = True) -> dict:
    cfg = {
        "dataset": {"path": str(SAMPLE_DIR)},
        "evaluation": {"split": split},
    }
    if with_mapping:
        cfg["dataset"]["class_mapping"] = str(RDD2022_CLASSES_YAML)
    return cfg


@pytest.fixture
def patched_dataset():
    with patch(
        "model.training.evaluate_detection.RDD2022Dataset",
        _FakeRDD2022Dataset,
    ):
        yield


def test_class_names_are_sorted_english_when_class_mapping_set(patched_dataset):
    """class_names must be the canonical sorted English list, NOT YAML order."""
    _, class_names, _, _ = load_split(_config(with_mapping=True))

    assert class_names == SORTED_ENGLISH_LABELS

    # Defensive: ensure no Spanish taxonomy entries leaked in.
    mapper = TargetMapper(RDD2022_CLASSES_YAML, strict=False)
    for taxon in mapper.taxonomy:
        assert taxon not in class_names, (
            f"YAML taxonomy entry {taxon!r} must not appear in canonical "
            "class_names; found leakage from the legacy override."
        )


def test_load_split_returns_display_class_names_not_label_mapper(patched_dataset):
    """4th tuple element is now a list of display names, never a callable."""
    result = load_split(_config(with_mapping=True))

    assert len(result) == 4
    fourth = result[3]
    assert isinstance(fourth, list)
    assert all(isinstance(item, str) for item in fourth)
    assert not callable(fourth)


def test_display_class_names_are_spanish_when_mapping_set(patched_dataset):
    """display_class_names[i] == map_class(class_names[i]) for mappable labels."""
    _, class_names, _, display_class_names = load_split(_config(with_mapping=True))

    mapper = TargetMapper(RDD2022_CLASSES_YAML, strict=False)
    assert len(display_class_names) == len(class_names)

    for english, display in zip(class_names, display_class_names):
        if english in mapper.mappings:
            assert display == mapper.mappings[english], (
                f"display name for {english!r} should be {mapper.mappings[english]!r}, "
                f"got {display!r}"
            )
        else:
            # Unmappable raw label (e.g., "other corruption") falls back to
            # canonical English so evaluation never aborts.
            assert display == english


def test_display_class_names_default_to_canonical_when_no_mapping(patched_dataset):
    """Without class_mapping, display_class_names equals class_names."""
    _, class_names, _, display_class_names = load_split(_config(with_mapping=False))

    assert display_class_names == class_names
    assert display_class_names is not class_names  # parallel list, not alias


def test_ground_truth_labels_preserve_raw_english():
    """_build_ground_truth keeps bbox.class_label untouched (no remapping)."""

    class _Bbox:
        def __init__(self, label: str):
            self.x_min = 0.1
            self.y_min = 0.1
            self.x_max = 0.4
            self.y_max = 0.4
            self.class_label = label

    class _Annotation:
        def __init__(self, labels):
            self.bounding_boxes = [_Bbox(label) for label in labels]

    annotation = _Annotation(["longitudinal crack", "pothole", "other corruption"])

    gt = _build_ground_truth(annotation, image_id="img-001", input_size=320)

    assert gt["image_id"] == "img-001"
    assert gt["labels"] == ["longitudinal crack", "pothole", "other corruption"]
    assert len(gt["boxes"]) == 3


def test_load_split_idx_to_class_aligns_with_canonical_class_names(patched_dataset):
    """idx_to_class[i] is class_names[i] (canonical English), not Spanish."""
    _, class_names, idx_to_class, _ = load_split(_config(with_mapping=True))

    for idx, name in enumerate(class_names):
        assert idx_to_class[idx] == name
