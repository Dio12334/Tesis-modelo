"""Unit tests for the RDD2022Dataset implementation."""

import tempfile
from pathlib import Path

import pytest

from model.datasets.rdd2022 import RDD2022Dataset
from model.exceptions import DatasetNotFoundError, ParseError


def _create_xml(filename: str, width: int, height: int, objects: list) -> str:
    """Helper to create a PASCAL VOC XML annotation string."""
    objects_xml = ""
    for obj in objects:
        objects_xml += f"""    <object>
        <name>{obj['name']}</name>
        <bndbox>
            <xmin>{obj['xmin']}</xmin>
            <ymin>{obj['ymin']}</ymin>
            <xmax>{obj['xmax']}</xmax>
            <ymax>{obj['ymax']}</ymax>
        </bndbox>
    </object>
"""
    return f"""<annotation>
    <filename>{filename}</filename>
    <size>
        <width>{width}</width>
        <height>{height}</height>
    </size>
{objects_xml}</annotation>
"""


def _setup_dataset(tmp_path, annotations):
    """Create a dataset directory with XML annotations and image files.

    Args:
        tmp_path: Temporary directory path.
        annotations: List of dicts with keys: filename, width, height, objects, create_image.
    """
    for ann in annotations:
        xml_name = Path(ann["filename"]).stem + ".xml"
        xml_content = _create_xml(
            ann["filename"], ann["width"], ann["height"], ann.get("objects", [])
        )
        xml_path = tmp_path / xml_name
        xml_path.write_text(xml_content)

        if ann.get("create_image", True):
            img_path = tmp_path / ann["filename"]
            img_path.write_bytes(b"\x00" * 10)  # Dummy image file


class TestRDD2022DatasetLoad:
    """Tests for the load method."""

    def test_load_valid_dataset(self, tmp_path):
        """Test loading a valid dataset with annotations."""
        _setup_dataset(tmp_path, [
            {
                "filename": "Japan_000001.jpg",
                "width": 600,
                "height": 600,
                "objects": [
                    {"name": "D00", "xmin": 150, "ymin": 200, "xmax": 350, "ymax": 400},
                ],
            },
        ])

        dataset = RDD2022Dataset()
        dataset.load(tmp_path)

        assert len(dataset) == 1
        ann = dataset.get_annotations()[0]
        assert ann.metadata["country"] == "Japan"
        assert len(ann.bounding_boxes) == 1
        bbox = ann.bounding_boxes[0]
        assert bbox.class_label == "D00"
        assert abs(bbox.x_min - 150 / 600) < 1e-9
        assert abs(bbox.y_min - 200 / 600) < 1e-9
        assert abs(bbox.x_max - 350 / 600) < 1e-9
        assert abs(bbox.y_max - 400 / 600) < 1e-9

    def test_load_nonexistent_path(self):
        """Test that loading from a non-existent path raises DatasetNotFoundError."""
        dataset = RDD2022Dataset()
        with pytest.raises(DatasetNotFoundError) as exc_info:
            dataset.load(Path("/nonexistent/path/to/dataset"))
        assert "/nonexistent/path/to/dataset" in str(exc_info.value)

    def test_load_multiple_objects(self, tmp_path):
        """Test loading annotations with multiple bounding boxes."""
        _setup_dataset(tmp_path, [
            {
                "filename": "India_000002.jpg",
                "width": 800,
                "height": 600,
                "objects": [
                    {"name": "D00", "xmin": 100, "ymin": 100, "xmax": 200, "ymax": 200},
                    {"name": "D10", "xmin": 300, "ymin": 300, "xmax": 500, "ymax": 500},
                    {"name": "D20", "xmin": 600, "ymin": 400, "xmax": 700, "ymax": 550},
                ],
            },
        ])

        dataset = RDD2022Dataset()
        dataset.load(tmp_path)

        assert len(dataset) == 1
        ann = dataset.get_annotations()[0]
        assert len(ann.bounding_boxes) == 3
        assert ann.metadata["country"] == "India"

    def test_load_empty_directory(self, tmp_path):
        """Test loading from an empty directory produces no annotations."""
        dataset = RDD2022Dataset()
        dataset.load(tmp_path)
        assert len(dataset) == 0

    def test_missing_image_excluded(self, tmp_path):
        """Test that annotations with missing image files are excluded."""
        _setup_dataset(tmp_path, [
            {
                "filename": "Japan_000001.jpg",
                "width": 600,
                "height": 600,
                "objects": [{"name": "D00", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50}],
                "create_image": True,
            },
            {
                "filename": "Japan_000002.jpg",
                "width": 600,
                "height": 600,
                "objects": [{"name": "D10", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50}],
                "create_image": False,  # Image does not exist
            },
        ])

        dataset = RDD2022Dataset()
        dataset.load(tmp_path)

        assert len(dataset) == 1
        assert dataset.get_annotations()[0].metadata["filename"] == "Japan_000001.jpg"


class TestRDD2022DatasetParseErrors:
    """Tests for ParseError handling."""

    def test_malformed_xml(self, tmp_path):
        """Test that malformed XML raises ParseError."""
        xml_path = tmp_path / "bad.xml"
        xml_path.write_text("<annotation><unclosed>")

        dataset = RDD2022Dataset()
        with pytest.raises(ParseError) as exc_info:
            dataset.load(tmp_path)
        assert exc_info.value.file_path == xml_path

    def test_missing_filename_element(self, tmp_path):
        """Test that missing <filename> raises ParseError."""
        xml_path = tmp_path / "no_filename.xml"
        xml_path.write_text("""<annotation>
    <size><width>600</width><height>600</height></size>
</annotation>""")

        dataset = RDD2022Dataset()
        with pytest.raises(ParseError) as exc_info:
            dataset.load(tmp_path)
        assert "filename" in exc_info.value.description.lower()

    def test_missing_size_element(self, tmp_path):
        """Test that missing <size> raises ParseError."""
        xml_path = tmp_path / "no_size.xml"
        xml_path.write_text("""<annotation>
    <filename>test.jpg</filename>
</annotation>""")

        dataset = RDD2022Dataset()
        with pytest.raises(ParseError) as exc_info:
            dataset.load(tmp_path)
        assert "size" in exc_info.value.description.lower()

    def test_invalid_width_value(self, tmp_path):
        """Test that non-numeric width raises ParseError."""
        xml_path = tmp_path / "bad_width.xml"
        xml_path.write_text("""<annotation>
    <filename>test.jpg</filename>
    <size><width>abc</width><height>600</height></size>
</annotation>""")

        dataset = RDD2022Dataset()
        with pytest.raises(ParseError) as exc_info:
            dataset.load(tmp_path)
        assert "width" in exc_info.value.description.lower()

    def test_zero_dimensions(self, tmp_path):
        """Test that zero dimensions raise ParseError."""
        xml_path = tmp_path / "zero_dim.xml"
        xml_path.write_text("""<annotation>
    <filename>test.jpg</filename>
    <size><width>0</width><height>600</height></size>
</annotation>""")

        dataset = RDD2022Dataset()
        with pytest.raises(ParseError) as exc_info:
            dataset.load(tmp_path)
        assert "dimension" in exc_info.value.description.lower()

    def test_missing_bndbox_coordinate(self, tmp_path):
        """Test that missing bndbox coordinate raises ParseError."""
        xml_path = tmp_path / "missing_coord.xml"
        xml_path.write_text("""<annotation>
    <filename>test.jpg</filename>
    <size><width>600</width><height>600</height></size>
    <object>
        <name>D00</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>100</ymin>
            <xmax>200</xmax>
        </bndbox>
    </object>
</annotation>""")

        # Create the image file so it doesn't get excluded for missing image
        (tmp_path / "test.jpg").write_bytes(b"\x00")

        dataset = RDD2022Dataset()
        with pytest.raises(ParseError) as exc_info:
            dataset.load(tmp_path)
        assert "ymax" in exc_info.value.description.lower()


class TestRDD2022DatasetCountryFilter:
    """Tests for country filtering."""

    def test_filter_single_country(self, tmp_path):
        """Test filtering by a single country."""
        _setup_dataset(tmp_path, [
            {
                "filename": "Japan_000001.jpg",
                "width": 600,
                "height": 600,
                "objects": [{"name": "D00", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50}],
            },
            {
                "filename": "India_000001.jpg",
                "width": 600,
                "height": 600,
                "objects": [{"name": "D10", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50}],
            },
        ])

        dataset = RDD2022Dataset(country_filter=["Japan"])
        dataset.load(tmp_path)

        assert len(dataset) == 1
        assert dataset.get_annotations()[0].metadata["country"] == "Japan"

    def test_filter_multiple_countries(self, tmp_path):
        """Test filtering by multiple countries."""
        _setup_dataset(tmp_path, [
            {
                "filename": "Japan_000001.jpg",
                "width": 600,
                "height": 600,
                "objects": [{"name": "D00", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50}],
            },
            {
                "filename": "India_000001.jpg",
                "width": 600,
                "height": 600,
                "objects": [{"name": "D10", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50}],
            },
            {
                "filename": "Czech_000001.jpg",
                "width": 600,
                "height": 600,
                "objects": [{"name": "D20", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50}],
            },
        ])

        dataset = RDD2022Dataset(country_filter=["Japan", "India"])
        dataset.load(tmp_path)

        assert len(dataset) == 2
        countries = {a.metadata["country"] for a in dataset.get_annotations()}
        assert countries == {"Japan", "India"}

    def test_no_filter_loads_all(self, tmp_path):
        """Test that no filter loads all annotations."""
        _setup_dataset(tmp_path, [
            {
                "filename": "Japan_000001.jpg",
                "width": 600,
                "height": 600,
                "objects": [{"name": "D00", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50}],
            },
            {
                "filename": "India_000001.jpg",
                "width": 600,
                "height": 600,
                "objects": [{"name": "D10", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50}],
            },
        ])

        dataset = RDD2022Dataset()
        dataset.load(tmp_path)

        assert len(dataset) == 2


class TestRDD2022DatasetSplit:
    """Tests for the split method."""

    def test_split_basic(self, tmp_path):
        """Test basic split produces correct sizes."""
        _setup_dataset(tmp_path, [
            {
                "filename": f"Japan_{i:06d}.jpg",
                "width": 600,
                "height": 600,
                "objects": [{"name": "D00", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50}],
            }
            for i in range(10)
        ])

        dataset = RDD2022Dataset()
        dataset.load(tmp_path)

        train, val, test = dataset.split(0.6, 0.2, 0.2, seed=42)
        assert len(train) + len(val) + len(test) == 10

    def test_split_deterministic(self, tmp_path):
        """Test that split is deterministic with the same seed."""
        _setup_dataset(tmp_path, [
            {
                "filename": f"Japan_{i:06d}.jpg",
                "width": 600,
                "height": 600,
                "objects": [{"name": "D00", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50}],
            }
            for i in range(20)
        ])

        dataset = RDD2022Dataset()
        dataset.load(tmp_path)

        train1, val1, test1 = dataset.split(0.7, 0.15, 0.15, seed=123)
        train2, val2, test2 = dataset.split(0.7, 0.15, 0.15, seed=123)

        # Same seed should produce same splits
        assert [a.metadata["filename"] for a in train1] == [a.metadata["filename"] for a in train2]
        assert [a.metadata["filename"] for a in val1] == [a.metadata["filename"] for a in val2]
        assert [a.metadata["filename"] for a in test1] == [a.metadata["filename"] for a in test2]

    def test_split_different_seeds(self, tmp_path):
        """Test that different seeds produce different splits."""
        _setup_dataset(tmp_path, [
            {
                "filename": f"Japan_{i:06d}.jpg",
                "width": 600,
                "height": 600,
                "objects": [{"name": "D00", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50}],
            }
            for i in range(20)
        ])

        dataset = RDD2022Dataset()
        dataset.load(tmp_path)

        train1, _, _ = dataset.split(0.7, 0.15, 0.15, seed=1)
        train2, _, _ = dataset.split(0.7, 0.15, 0.15, seed=2)

        # Different seeds should (very likely) produce different splits
        filenames1 = [a.metadata["filename"] for a in train1]
        filenames2 = [a.metadata["filename"] for a in train2]
        assert filenames1 != filenames2

    def test_split_invalid_ratios(self, tmp_path):
        """Test that invalid ratios raise ValueError."""
        dataset = RDD2022Dataset()
        dataset._annotations = []

        with pytest.raises(ValueError):
            dataset.split(0.5, 0.3, 0.3)  # Sum > 1.0


class TestRDD2022DatasetIteration:
    """Tests for iteration and utility methods."""

    def test_iter(self, tmp_path):
        """Test iteration over annotations."""
        _setup_dataset(tmp_path, [
            {
                "filename": f"Japan_{i:06d}.jpg",
                "width": 600,
                "height": 600,
                "objects": [{"name": "D00", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50}],
            }
            for i in range(3)
        ])

        dataset = RDD2022Dataset()
        dataset.load(tmp_path)

        items = list(dataset)
        assert len(items) == 3
        assert all(hasattr(item, "image_path") for item in items)

    def test_get_class_names(self, tmp_path):
        """Test get_class_names returns sorted unique labels."""
        _setup_dataset(tmp_path, [
            {
                "filename": "Japan_000001.jpg",
                "width": 600,
                "height": 600,
                "objects": [
                    {"name": "D20", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50},
                    {"name": "D00", "xmin": 60, "ymin": 60, "xmax": 100, "ymax": 100},
                ],
            },
            {
                "filename": "Japan_000002.jpg",
                "width": 600,
                "height": 600,
                "objects": [
                    {"name": "D10", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50},
                    {"name": "D00", "xmin": 60, "ymin": 60, "xmax": 100, "ymax": 100},
                ],
            },
        ])

        dataset = RDD2022Dataset()
        dataset.load(tmp_path)

        class_names = dataset.get_class_names()
        assert class_names == ["D00", "D10", "D20"]

    def test_coordinate_normalization(self, tmp_path):
        """Test that coordinates are properly normalized to [0, 1]."""
        _setup_dataset(tmp_path, [
            {
                "filename": "Japan_000001.jpg",
                "width": 800,
                "height": 400,
                "objects": [
                    {"name": "D00", "xmin": 200, "ymin": 100, "xmax": 600, "ymax": 300},
                ],
            },
        ])

        dataset = RDD2022Dataset()
        dataset.load(tmp_path)

        bbox = dataset.get_annotations()[0].bounding_boxes[0]
        assert abs(bbox.x_min - 200 / 800) < 1e-9
        assert abs(bbox.y_min - 100 / 400) < 1e-9
        assert abs(bbox.x_max - 600 / 800) < 1e-9
        assert abs(bbox.y_max - 300 / 400) < 1e-9
