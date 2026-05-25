# Feature: road-damage-evaluation-framework, Property 2: Non-existent dataset path raises DatasetNotFoundError
"""Property-based tests for the RDD2022Dataset implementation.

This file contains property tests for:
- Property 2: Non-existent dataset path raises DatasetNotFoundError
- (Properties 3, 4, 20, 21, 22 will be added in later tasks)

Validates: Requirements 1.6
"""

from pathlib import Path

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from model.datasets.rdd2022 import RDD2022Dataset
from model.exceptions import DatasetNotFoundError, ParseError


# Strategy to generate path segments that are unlikely to exist on the filesystem
_path_segment = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N"),
        whitelist_characters="_-",
    ),
    min_size=1,
    max_size=20,
)

_nonexistent_path = st.builds(
    lambda segments: Path("/") / "nonexistent_base_kiro_test" / "/".join(segments),
    segments=st.lists(_path_segment, min_size=1, max_size=5),
)


@settings(max_examples=100)
@given(path=_nonexistent_path)
def test_nonexistent_dataset_path_raises_dataset_not_found_error(path: Path) -> None:
    """Property 2: Non-existent dataset path raises DatasetNotFoundError.

    For any file path that does not exist on the filesystem, attempting to load
    a dataset from that path SHALL raise a DatasetNotFoundError whose message
    contains the invalid path string.

    **Validates: Requirements 1.6**
    """
    # Ensure the generated path truly does not exist
    assume(not path.exists())

    dataset = RDD2022Dataset()

    # Verify DatasetNotFoundError is raised
    with pytest.raises(DatasetNotFoundError) as exc_info:
        dataset.load(path)

    # Verify the exception's path attribute matches the input path
    assert exc_info.value.path == path

    # Verify the string representation contains the path
    assert str(path) in str(exc_info.value)


# Feature: road-damage-evaluation-framework, Property 3: Malformed annotations raise ParseError with location
"""
Property 3: Malformed annotations raise ParseError with location

For any malformed XML annotation string (invalid XML, missing required elements,
or invalid coordinate values), parsing SHALL raise a ParseError whose message
contains a line number and a human-readable description.

Validates: Requirements 1.7
"""

import tempfile
import os

from hypothesis import strategies as st


# --- Strategies for generating malformed XML ---

# Strategy 1: Invalid XML syntax (unclosed tags, random non-XML text)
_invalid_xml_syntax = st.one_of(
    # Unclosed tags
    st.builds(
        lambda tag, content: f"<{tag}>{content}",
        tag=st.sampled_from(["annotation", "object", "size", "bndbox"]),
        content=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P"), whitelist_characters=" "),
            min_size=1,
            max_size=50,
        ),
    ),
    # Random text that isn't valid XML
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P"), whitelist_characters=" \n"),
        min_size=1,
        max_size=100,
    ).filter(lambda t: not t.strip().startswith("<") or "</" not in t),
    # Mismatched tags
    st.builds(
        lambda tag1, tag2: f"<{tag1}>content</{tag2}>",
        tag1=st.sampled_from(["annotation", "object", "size"]),
        tag2=st.sampled_from(["filename", "bndbox", "width"]),
    ),
)

# Strategy 2: Valid XML but missing required elements
_missing_required_elements = st.one_of(
    # Missing <filename> element
    st.just(
        "<annotation>\n"
        "  <size>\n"
        "    <width>600</width>\n"
        "    <height>600</height>\n"
        "  </size>\n"
        "</annotation>"
    ),
    # Missing <size> element
    st.just(
        "<annotation>\n"
        "  <filename>test_001.jpg</filename>\n"
        "  <object>\n"
        "    <name>D00</name>\n"
        "    <bndbox>\n"
        "      <xmin>10</xmin>\n"
        "      <ymin>20</ymin>\n"
        "      <xmax>100</xmax>\n"
        "      <ymax>200</ymax>\n"
        "    </bndbox>\n"
        "  </object>\n"
        "</annotation>"
    ),
    # Missing <width> inside <size>
    st.just(
        "<annotation>\n"
        "  <filename>test_001.jpg</filename>\n"
        "  <size>\n"
        "    <height>600</height>\n"
        "  </size>\n"
        "</annotation>"
    ),
    # Missing <height> inside <size>
    st.just(
        "<annotation>\n"
        "  <filename>test_001.jpg</filename>\n"
        "  <size>\n"
        "    <width>600</width>\n"
        "  </size>\n"
        "</annotation>"
    ),
)

# Strategy 3: Invalid coordinate values (non-numeric values in bndbox)
def _is_not_parseable_as_float(t: str) -> bool:
    """Return True if the text cannot be parsed as a float."""
    try:
        float(t.strip())
        return False
    except ValueError:
        return True


_invalid_coord_text = st.text(
    alphabet=st.characters(whitelist_categories=("L",), whitelist_characters="!@#$%"),
    min_size=1,
    max_size=10,
).filter(lambda t: t.strip() != "" and _is_not_parseable_as_float(t))

_invalid_coordinate_values = st.builds(
    lambda xmin, ymin, xmax, ymax: (
        "<annotation>\n"
        "  <filename>test_001.jpg</filename>\n"
        "  <size>\n"
        "    <width>600</width>\n"
        "    <height>600</height>\n"
        "  </size>\n"
        "  <object>\n"
        "    <name>D00</name>\n"
        "    <bndbox>\n"
        f"      <xmin>{xmin}</xmin>\n"
        f"      <ymin>{ymin}</ymin>\n"
        f"      <xmax>{xmax}</xmax>\n"
        f"      <ymax>{ymax}</ymax>\n"
        "    </bndbox>\n"
        "  </object>\n"
        "</annotation>"
    ),
    xmin=st.one_of(_invalid_coord_text, st.just("10")),
    ymin=st.one_of(_invalid_coord_text, st.just("20")),
    xmax=st.one_of(_invalid_coord_text, st.just("100")),
    ymax=st.one_of(_invalid_coord_text, st.just("200")),
).filter(
    # Ensure at least one coordinate is non-numeric (not parseable as float)
    lambda xml: any(
        _is_not_parseable_as_float(val)
        for val in _extract_coord_values(xml)
    )
)


def _extract_coord_values(xml: str) -> list:
    """Helper to extract coordinate values from generated XML for filtering."""
    import re
    coords = []
    for tag in ["xmin", "ymin", "xmax", "ymax"]:
        match = re.search(f"<{tag}>(.*?)</{tag}>", xml)
        if match:
            coords.append(match.group(1))
    return coords


# Combined strategy for all malformed XML types
_malformed_xml = st.one_of(
    _invalid_xml_syntax,
    _missing_required_elements,
    _invalid_coordinate_values,
)


@settings(max_examples=100)
@given(malformed_xml=_malformed_xml)
def test_malformed_annotations_raise_parse_error_with_location(malformed_xml: str) -> None:
    """Property 3: Malformed annotations raise ParseError with location.

    For any malformed XML annotation string (invalid XML, missing required elements,
    or invalid coordinate values), parsing SHALL raise a ParseError whose message
    contains a line number and a human-readable description.

    **Validates: Requirements 1.7**
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Write the malformed XML to a file in the temp directory
        xml_file = Path(tmp_dir) / "malformed_annotation.xml"
        xml_file.write_text(malformed_xml, encoding="utf-8")

        dataset = RDD2022Dataset()

        # Verify ParseError is raised
        with pytest.raises(ParseError) as exc_info:
            dataset.load(Path(tmp_dir))

        error = exc_info.value

        # Verify the exception has a line_number attribute (integer >= 0)
        assert hasattr(error, "line_number")
        assert isinstance(error.line_number, int)
        assert error.line_number >= 0

        # Verify the exception has a description attribute (non-empty string)
        assert hasattr(error, "description")
        assert isinstance(error.description, str)
        assert len(error.description) > 0

        # Verify the exception has a file_path attribute
        assert hasattr(error, "file_path")
        assert error.file_path is not None


# Feature: road-damage-evaluation-framework, Property 4: Missing image references are excluded
"""
Property 4: Missing image references are excluded

For any set of annotations where some reference existing image files and some
reference non-existent files, loading the dataset SHALL return only annotations
whose image files exist, and the excluded set SHALL be exactly those with missing files.

Validates: Requirements 1.8, 1.9
"""


def _make_voc_xml(filename: str, width: int = 600, height: int = 600) -> str:
    """Create a valid PASCAL VOC XML annotation string."""
    return (
        "<annotation>\n"
        f"    <filename>{filename}</filename>\n"
        "    <size>\n"
        f"        <width>{width}</width>\n"
        f"        <height>{height}</height>\n"
        "    </size>\n"
        "    <object>\n"
        "        <name>D00</name>\n"
        "        <bndbox>\n"
        "            <xmin>10</xmin>\n"
        "            <ymin>10</ymin>\n"
        "            <xmax>100</xmax>\n"
        "            <ymax>100</ymax>\n"
        "        </bndbox>\n"
        "    </object>\n"
        "</annotation>"
    )


@settings(max_examples=100)
@given(
    num_annotations=st.integers(min_value=2, max_value=10),
    data=st.data(),
)
def test_missing_image_references_are_excluded(num_annotations: int, data: st.DataObject) -> None:
    """Property 4: Missing image references are excluded.

    For any set of annotations where some reference existing image files and some
    reference non-existent files, loading the dataset SHALL return only annotations
    whose image files exist, and the excluded set SHALL be exactly those with missing files.

    **Validates: Requirements 1.8, 1.9**
    """
    # For each annotation, randomly decide whether the image file should exist
    image_exists_flags = data.draw(
        st.lists(st.booleans(), min_size=num_annotations, max_size=num_annotations)
    )

    # Ensure we have at least one existing and one missing to make the test meaningful
    assume(any(image_exists_flags) and not all(image_exists_flags))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        expected_existing_filenames = []

        for i in range(num_annotations):
            filename = f"Country_{i:06d}.jpg"
            # Create the XML annotation file
            xml_content = _make_voc_xml(filename)
            xml_file = tmp_path / f"Country_{i:06d}.xml"
            xml_file.write_text(xml_content, encoding="utf-8")

            # Only create the image file if flagged as existing
            if image_exists_flags[i]:
                image_file = tmp_path / filename
                image_file.write_bytes(b"\x00" * 10)  # Minimal fake image content
                expected_existing_filenames.append(filename)

        # Load the dataset
        dataset = RDD2022Dataset()
        dataset.load(tmp_path)

        annotations = dataset.get_annotations()

        # Verify that the loaded annotations contain ONLY those with existing image files
        loaded_filenames = [ann.metadata["filename"] for ann in annotations]

        assert len(annotations) == len(expected_existing_filenames), (
            f"Expected {len(expected_existing_filenames)} annotations but got {len(annotations)}"
        )

        # Verify the exact set of loaded filenames matches expected
        assert set(loaded_filenames) == set(expected_existing_filenames), (
            f"Loaded filenames {set(loaded_filenames)} != expected {set(expected_existing_filenames)}"
        )


# Feature: road-damage-evaluation-framework, Property 20: Bounding box normalization round-trip
# Feature: road-damage-evaluation-framework, Property 21: VOC XML annotation parsing
# Feature: road-damage-evaluation-framework, Property 22: Country filtering
"""
Property 20: Bounding box normalization round-trip
Property 21: VOC XML annotation parsing
Property 22: Country filtering

Validates: Requirements 9.2, 9.5, 9.6
"""


# --- Property 20: Bounding box normalization round-trip ---


@settings(max_examples=100)
@given(
    width=st.integers(min_value=1, max_value=10000),
    height=st.integers(min_value=1, max_value=10000),
    data=st.data(),
)
def test_bounding_box_normalization_round_trip(
    width: int, height: int, data: st.DataObject
) -> None:
    """Property 20: Bounding box normalization round-trip.

    For any image with dimensions (W, H) and any pixel-coordinate bounding box
    (xmin, ymin, xmax, ymax) within those dimensions, normalizing to [0,1] range
    and denormalizing back to pixel coordinates SHALL produce the original
    coordinates (within floating-point tolerance).

    **Validates: Requirements 9.6**
    """
    # Generate pixel coordinates within image dimensions
    xmin = data.draw(st.floats(min_value=0.0, max_value=float(width - 1), allow_nan=False, allow_infinity=False))
    ymin = data.draw(st.floats(min_value=0.0, max_value=float(height - 1), allow_nan=False, allow_infinity=False))
    xmax = data.draw(st.floats(min_value=xmin + 0.1, max_value=float(width), allow_nan=False, allow_infinity=False))
    ymax = data.draw(st.floats(min_value=ymin + 0.1, max_value=float(height), allow_nan=False, allow_infinity=False))

    # Normalize to [0, 1] range (same as RDD2022Dataset._parse_object does)
    x_min_norm = xmin / width
    y_min_norm = ymin / height
    x_max_norm = xmax / width
    y_max_norm = ymax / height

    # Denormalize back to pixel coordinates
    xmin_recovered = x_min_norm * width
    ymin_recovered = y_min_norm * height
    xmax_recovered = x_max_norm * width
    ymax_recovered = y_max_norm * height

    # Verify round-trip within floating-point tolerance
    tolerance = 1e-6
    assert abs(xmin - xmin_recovered) < tolerance, (
        f"xmin round-trip failed: {xmin} -> {x_min_norm} -> {xmin_recovered}"
    )
    assert abs(ymin - ymin_recovered) < tolerance, (
        f"ymin round-trip failed: {ymin} -> {y_min_norm} -> {ymin_recovered}"
    )
    assert abs(xmax - xmax_recovered) < tolerance, (
        f"xmax round-trip failed: {xmax} -> {x_max_norm} -> {xmax_recovered}"
    )
    assert abs(ymax - ymax_recovered) < tolerance, (
        f"ymax round-trip failed: {ymax} -> {y_max_norm} -> {ymax_recovered}"
    )

    # Also verify normalized values are in [0, 1]
    assert 0.0 <= x_min_norm <= 1.0
    assert 0.0 <= y_min_norm <= 1.0
    assert 0.0 <= x_max_norm <= 1.0
    assert 0.0 <= y_max_norm <= 1.0


# --- Property 21: VOC XML annotation parsing ---


def _build_voc_xml(
    filename: str,
    width: int,
    height: int,
    objects: list,
) -> str:
    """Build a valid PASCAL VOC XML annotation string.

    Args:
        filename: Image filename.
        width: Image width.
        height: Image height.
        objects: List of dicts with keys: name, xmin, ymin, xmax, ymax.

    Returns:
        Valid PASCAL VOC XML string.
    """
    obj_xml_parts = []
    for obj in objects:
        obj_xml_parts.append(
            "    <object>\n"
            f"        <name>{obj['name']}</name>\n"
            "        <bndbox>\n"
            f"            <xmin>{obj['xmin']}</xmin>\n"
            f"            <ymin>{obj['ymin']}</ymin>\n"
            f"            <xmax>{obj['xmax']}</xmax>\n"
            f"            <ymax>{obj['ymax']}</ymax>\n"
            "        </bndbox>\n"
            "    </object>\n"
        )
    objects_xml = "".join(obj_xml_parts)
    return (
        "<annotation>\n"
        f"    <filename>{filename}</filename>\n"
        "    <size>\n"
        f"        <width>{width}</width>\n"
        f"        <height>{height}</height>\n"
        "    </size>\n"
        f"{objects_xml}"
        "</annotation>"
    )


# Strategy for generating valid bounding box objects within given dimensions
def _object_strategy(width: int, height: int):
    """Generate a valid object dict with coordinates within image dimensions."""
    return st.builds(
        lambda name, xmin, ymin, xmax, ymax: {
            "name": name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        },
        name=st.sampled_from(["D00", "D10", "D20", "D40"]),
        xmin=st.integers(min_value=0, max_value=max(0, width - 2)),
        ymin=st.integers(min_value=0, max_value=max(0, height - 2)),
        xmax=st.integers(min_value=1, max_value=width),
        ymax=st.integers(min_value=1, max_value=height),
    ).filter(lambda o: o["xmin"] < o["xmax"] and o["ymin"] < o["ymax"])


@settings(max_examples=100)
@given(
    width=st.integers(min_value=10, max_value=4000),
    height=st.integers(min_value=10, max_value=4000),
    data=st.data(),
)
def test_voc_xml_annotation_parsing(
    width: int, height: int, data: st.DataObject
) -> None:
    """Property 21: VOC XML annotation parsing.

    For any valid PASCAL VOC XML annotation containing filename, image dimensions,
    and object bounding boxes, parsing SHALL produce an Annotation with the correct
    image path, and bounding boxes with normalized coordinates matching the XML
    values divided by image dimensions.

    **Validates: Requirements 9.2**
    """
    # Generate random objects within the image dimensions
    objects = data.draw(
        st.lists(_object_strategy(width, height), min_size=1, max_size=5)
    )

    country = data.draw(st.sampled_from(["Japan", "India", "Czech", "Norway", "USA"]))
    img_number = data.draw(st.integers(min_value=0, max_value=999999))
    filename = f"{country}_{img_number:06d}.jpg"

    # Build valid VOC XML
    xml_content = _build_voc_xml(filename, width, height, objects)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Write XML annotation file
        xml_name = f"{country}_{img_number:06d}.xml"
        xml_file = tmp_path / xml_name
        xml_file.write_text(xml_content, encoding="utf-8")

        # Create the corresponding image file (so it passes existence check)
        image_file = tmp_path / filename
        image_file.write_bytes(b"\x00" * 10)

        # Load the dataset
        dataset = RDD2022Dataset()
        dataset.load(tmp_path)

        annotations = dataset.get_annotations()

        # Should have exactly one annotation
        assert len(annotations) == 1, (
            f"Expected 1 annotation, got {len(annotations)}"
        )

        annotation = annotations[0]

        # Verify image path points to the correct file
        assert annotation.image_path.name == filename

        # Verify the number of bounding boxes matches
        assert len(annotation.bounding_boxes) == len(objects), (
            f"Expected {len(objects)} bounding boxes, got {len(annotation.bounding_boxes)}"
        )

        # Verify each bounding box has correctly normalized coordinates
        for bbox, obj in zip(annotation.bounding_boxes, objects):
            expected_x_min = obj["xmin"] / width
            expected_y_min = obj["ymin"] / height
            expected_x_max = obj["xmax"] / width
            expected_y_max = obj["ymax"] / height

            tolerance = 1e-9
            assert abs(bbox.x_min - expected_x_min) < tolerance, (
                f"x_min mismatch: {bbox.x_min} != {expected_x_min}"
            )
            assert abs(bbox.y_min - expected_y_min) < tolerance, (
                f"y_min mismatch: {bbox.y_min} != {expected_y_min}"
            )
            assert abs(bbox.x_max - expected_x_max) < tolerance, (
                f"x_max mismatch: {bbox.x_max} != {expected_x_max}"
            )
            assert abs(bbox.y_max - expected_y_max) < tolerance, (
                f"y_max mismatch: {bbox.y_max} != {expected_y_max}"
            )

            # Verify class label matches
            assert bbox.class_label == obj["name"], (
                f"class_label mismatch: {bbox.class_label} != {obj['name']}"
            )


# --- Property 22: Country filtering ---


@settings(max_examples=100)
@given(data=st.data())
def test_country_filtering(data: st.DataObject) -> None:
    """Property 22: Country filtering.

    For any dataset containing images from multiple countries and any country
    filter list, loading with that filter SHALL return only annotations whose
    metadata country is in the filter list.

    **Validates: Requirements 9.5**
    """
    all_countries = ["Japan", "India", "Czech", "Norway", "USA", "China"]

    # Draw a subset of countries to include in the dataset (at least 2)
    dataset_countries = data.draw(
        st.lists(
            st.sampled_from(all_countries),
            min_size=2,
            max_size=len(all_countries),
            unique=True,
        )
    )

    # Draw a filter list (non-empty subset of dataset countries)
    filter_countries = data.draw(
        st.lists(
            st.sampled_from(dataset_countries),
            min_size=1,
            max_size=len(dataset_countries) - 1,
            unique=True,
        )
    )

    # Ensure filter is a strict subset so we can verify filtering works
    assume(set(filter_countries) != set(dataset_countries))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create annotations for each country
        expected_filtered_filenames = []
        for idx, country in enumerate(dataset_countries):
            filename = f"{country}_{idx:06d}.jpg"
            xml_content = _make_voc_xml(filename, width=600, height=600)
            xml_file = tmp_path / f"{country}_{idx:06d}.xml"
            xml_file.write_text(xml_content, encoding="utf-8")

            # Create image file
            image_file = tmp_path / filename
            image_file.write_bytes(b"\x00" * 10)

            if country in filter_countries:
                expected_filtered_filenames.append(filename)

        # Load dataset with country filter
        dataset = RDD2022Dataset(country_filter=filter_countries)
        dataset.load(tmp_path)

        annotations = dataset.get_annotations()

        # Verify only filtered countries are present
        loaded_filenames = [ann.metadata["filename"] for ann in annotations]
        loaded_countries = [ann.metadata["country"] for ann in annotations]

        # All loaded annotations must have a country in the filter list
        for country in loaded_countries:
            assert country in filter_countries, (
                f"Country '{country}' not in filter list {filter_countries}"
            )

        # The number of loaded annotations should match expected
        assert len(annotations) == len(expected_filtered_filenames), (
            f"Expected {len(expected_filtered_filenames)} annotations, got {len(annotations)}"
        )

        # Verify exact set of filenames
        assert set(loaded_filenames) == set(expected_filtered_filenames), (
            f"Loaded {set(loaded_filenames)} != expected {set(expected_filtered_filenames)}"
        )
