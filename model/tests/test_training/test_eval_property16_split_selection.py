"""Property-based tests for evaluation-split partition selection.

Feature: generic-evaluation-script
Property 16: Split selection uses the correct partition

For ``evaluation.split`` in ``{train, val, test}``, ``load_split`` returns the
corresponding partition produced by the Dataset_Loader:

* ``test`` -> the dataset loaded with the ``test`` subset, evaluated in full;
* ``train`` -> the *train* partition produced by ``RDD2022Dataset.split`` on the
  loaded ``train`` subset;
* ``val`` -> the *validation* partition produced by the same ``.split`` call.

These tests exercise the real ``load_split`` function in
``model/training/evaluate_detection.py``.

To make the property independent of real data and fully deterministic, the
``RDD2022Dataset`` symbol referenced by ``load_split`` is monkeypatched with a
conforming fake whose ``.split`` returns three *distinct, identifiable*
partition markers. This lets each example assert -- by object identity -- that
the returned partition is exactly the one the loader produced for the requested
split, and that ``.split`` was invoked with the expected
``(train_ratio, val_ratio, test_ratio, seed)`` arguments. ``dataset.path`` is
pointed at the existing on-disk sample dataset so the up-front existence check
in ``load_split`` passes without the fake's ``load`` touching real files.

The example-based tests at the bottom run against the real
``model/data/rdd2022/sample`` dataset with a fixed seed to confirm that the
train/val/test partitions selected by ``load_split`` match the partitions the
``RDD2022Dataset`` loader produces directly.

**Validates: Requirements 9.1, 9.2, 9.3**
"""

from pathlib import Path
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from model.datasets.rdd2022 import RDD2022Dataset
from model.training.evaluate_detection import load_split


# ---------------------------------------------------------------------------
# Fixtures / constants
# ---------------------------------------------------------------------------

# Repo-root-relative path to the bundled sample dataset (exists on disk), used
# so load_split's up-front `dataset.path` existence check passes.
SAMPLE_DIR = Path(__file__).resolve().parents[3] / "model" / "data" / "rdd2022" / "sample"


# ---------------------------------------------------------------------------
# Conforming fake dataset
# ---------------------------------------------------------------------------


class _FakePartition:
    """A distinct, identifiable stand-in for a partition returned by ``split``."""

    def __init__(self, name: str):
        self.name = name

    def __len__(self) -> int:  # load_split logs len(split_dataset)
        return 0


def _make_fake_dataset_class():
    """Build a fresh fake ``RDD2022Dataset`` class and an instance registry.

    A fresh class/registry per example keeps Hypothesis examples isolated. The
    fake records the ``subset`` it was constructed with, the path it was asked
    to load, and the arguments passed to ``.split``; its ``.split`` returns three
    distinct ``_FakePartition`` markers so the caller's selection can be checked
    by identity.
    """
    created = []

    class _FakeRDD2022Dataset:
        def __init__(self, country_filter=None, subset="train"):
            self.country_filter = country_filter
            self.subset = subset
            self.loaded_path = None
            self.split_args = None
            # Distinct markers per (train/val/test) partition.
            self.train_partition = _FakePartition("train")
            self.val_partition = _FakePartition("val")
            self.test_partition = _FakePartition("test")
            created.append(self)

        def load(self, path):
            # No real I/O: just record the requested path.
            self.loaded_path = path

        def split(self, train_ratio, val_ratio, test_ratio, seed=42):
            self.split_args = (train_ratio, val_ratio, test_ratio, seed)
            return self.train_partition, self.val_partition, self.test_partition

        def get_class_names(self):
            return ["crack", "pothole"]

        def __len__(self):
            return 10

    return _FakeRDD2022Dataset, created


def _config(split, val_split=None, seed=None, country_filter=None) -> dict:
    """Assemble a config selecting ``split`` with optional partition knobs."""
    evaluation = {"split": split}
    if val_split is not None:
        evaluation["val_split"] = val_split
    if seed is not None:
        evaluation["seed"] = seed

    dataset = {"path": str(SAMPLE_DIR)}
    if country_filter is not None:
        dataset["country_filter"] = country_filter

    return {
        "model": {"type": "yolo26", "config": {"num_classes": 5}},
        "dataset": dataset,
        "evaluation": evaluation,
        "checkpoint": {"path": "/checkpoints/best_model.pt"},
    }


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_SPLITS = st.sampled_from(["train", "val", "test"])

# Optional val_split fraction; None exercises the documented 0.2 default.
_VAL_SPLITS = st.one_of(
    st.none(),
    st.floats(min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False),
)

# Optional seed; None exercises the documented 42 default.
_SEEDS = st.one_of(st.none(), st.integers(min_value=0, max_value=2**31 - 1))

# Optional country filter passed through to the dataset constructor.
_COUNTRY_FILTERS = st.one_of(
    st.none(),
    st.lists(st.sampled_from(["Japan", "India", "Czech", "Norway"]), max_size=3),
)


# ---------------------------------------------------------------------------
# Property 16
# ---------------------------------------------------------------------------


class TestProperty16SplitSelection:
    """Property 16: Split selection uses the correct partition.

    **Validates: Requirements 9.1, 9.2, 9.3**
    """

    @given(
        split=_SPLITS,
        val_split=_VAL_SPLITS,
        seed=_SEEDS,
        country_filter=_COUNTRY_FILTERS,
    )
    @settings(max_examples=100)
    def test_load_split_returns_corresponding_partition(
        self, split, val_split, seed, country_filter
    ):
        # Feature: generic-evaluation-script, Property 16: Split selection
        """``load_split`` returns the loader partition matching ``evaluation.split``.

        **Validates: Requirements 9.1, 9.2, 9.3**
        """
        config = _config(split, val_split=val_split, seed=seed, country_filter=country_filter)
        fake_cls, created = _make_fake_dataset_class()

        with patch("model.training.evaluate_detection.RDD2022Dataset", fake_cls):
            split_dataset, class_names, idx_to_class, _label_mapper = load_split(config)

        # Exactly one source dataset is constructed and loaded from dataset.path.
        assert len(created) == 1
        source = created[0]
        assert source.loaded_path == SAMPLE_DIR
        # The country filter is passed through to the dataset constructor.
        assert source.country_filter == country_filter

        # Class-name mapping is derived from the loaded source dataset, zero-based.
        assert class_names == ["crack", "pothole"]
        assert idx_to_class == {0: "crack", 1: "pothole"}

        expected_val = 0.2 if val_split is None else float(val_split)
        expected_seed = 42 if seed is None else int(seed)

        if split == "test":
            # Req 9.3: the test subset is loaded and evaluated in full -- the
            # returned partition IS the test-subset dataset, with no .split call.
            assert source.subset == "test"
            assert split_dataset is source
            assert source.split_args is None
        else:
            # Req 9.1, 9.2: the train subset is loaded and partitioned via
            # .split(1 - val_split, val_split, 0.0, seed).
            assert source.subset == "train"
            assert source.split_args == (
                1.0 - expected_val,
                expected_val,
                0.0,
                expected_seed,
            )
            if split == "train":
                assert split_dataset is source.train_partition
            else:  # split == "val"
                assert split_dataset is source.val_partition

    @given(split=_SPLITS)
    @settings(max_examples=100)
    def test_returned_partition_is_never_a_sibling(self, split):
        # Feature: generic-evaluation-script, Property 16: Split selection
        """The returned partition is the requested one, never a sibling partition.

        **Validates: Requirements 9.1, 9.2, 9.3**
        """
        config = _config(split, val_split=0.2, seed=7)
        fake_cls, created = _make_fake_dataset_class()

        with patch("model.training.evaluate_detection.RDD2022Dataset", fake_cls):
            split_dataset, _, _, _ = load_split(config)

        source = created[0]
        if split == "train":
            assert split_dataset is source.train_partition
            assert split_dataset is not source.val_partition
            assert split_dataset is not source.test_partition
        elif split == "val":
            assert split_dataset is source.val_partition
            assert split_dataset is not source.train_partition
            assert split_dataset is not source.test_partition
        else:  # test
            assert split_dataset is source
            assert split_dataset is not source.train_partition
            assert split_dataset is not source.val_partition


# ---------------------------------------------------------------------------
# Example-based tests against the real sample dataset
# ---------------------------------------------------------------------------


class TestSplitSelectionRealSample:
    """Concrete examples on the bundled sample dataset with a fixed seed.

    These confirm the partition ``load_split`` selects matches the partition the
    real ``RDD2022Dataset`` loader produces directly, deterministically.

    **Validates: Requirements 9.1, 9.2, 9.3**
    """

    def _expected_partitions(self, val_split=0.2, seed=42):
        """Reproduce the loader's train/val partitions exactly. (Req 9.1, 9.2)"""
        source = RDD2022Dataset(subset="train")
        source.load(SAMPLE_DIR)
        train_ds, val_ds, _ = source.split(1.0 - val_split, val_split, 0.0, seed=seed)
        return train_ds, val_ds

    def test_train_split_matches_loader_train_partition(self):
        """``split == train`` returns the loader's train partition. (Req 9.1)"""
        train_ds, _ = self._expected_partitions(val_split=0.2, seed=42)
        config = _config("train", val_split=0.2, seed=42)

        split_dataset, _, _, _ = load_split(config)

        assert len(split_dataset) == len(train_ds)
        got = [str(a.image_path) for a in split_dataset.get_annotations()]
        expected = [str(a.image_path) for a in train_ds.get_annotations()]
        assert got == expected

    def test_val_split_matches_loader_val_partition(self):
        """``split == val`` returns the loader's validation partition. (Req 9.2)"""
        _, val_ds = self._expected_partitions(val_split=0.2, seed=42)
        config = _config("val", val_split=0.2, seed=42)

        split_dataset, _, _, _ = load_split(config)

        assert len(split_dataset) == len(val_ds)
        got = [str(a.image_path) for a in split_dataset.get_annotations()]
        expected = [str(a.image_path) for a in val_ds.get_annotations()]
        assert got == expected

    def test_test_split_matches_loader_test_subset(self):
        """``split == test`` returns the full test subset. (Req 9.3)"""
        test_source = RDD2022Dataset(subset="test")
        test_source.load(SAMPLE_DIR)
        config = _config("test")

        split_dataset, _, _, _ = load_split(config)

        assert len(split_dataset) == len(test_source)
        got = sorted(str(a.image_path) for a in split_dataset.get_annotations())
        expected = sorted(str(a.image_path) for a in test_source.get_annotations())
        assert got == expected

    def test_train_and_val_partitions_are_disjoint_and_cover_train_subset(self):
        """Train + val partitions are disjoint and together cover the train subset.

        Confirms ``load_split`` selects genuine complementary partitions of the
        loaded train subset (not overlapping or arbitrary slices).

        **Validates: Requirements 9.1, 9.2**
        """
        source = RDD2022Dataset(subset="train")
        source.load(SAMPLE_DIR)
        total = len(source)

        train_ds, _, _, _ = load_split(_config("train", val_split=0.2, seed=42))
        # load_split is deterministic, so re-running for val yields the complement.
        val_ds, _, _, _ = load_split(_config("val", val_split=0.2, seed=42))

        train_paths = {str(a.image_path) for a in train_ds.get_annotations()}
        val_paths = {str(a.image_path) for a in val_ds.get_annotations()}

        assert train_paths.isdisjoint(val_paths)
        assert len(train_paths) + len(val_paths) == total
