"""RDD2022 dataset implementation supporting Supervisely JSON and PASCAL VOC XML formats.

Supports two annotation formats:
1. Supervisely/Dataset Ninja JSON (default for data from datasetninja.com)
   - Structure: {root}/train/img/*.jpg + {root}/train/ann/*.jpg.json
2. PASCAL VOC XML (original RDD2022 from sekilab/RoadDamageDetector)
   - Structure: {root}/*.xml + {root}/*.jpg (or images/ subdirectory)
"""

import json
import logging
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from model.datasets.base import Annotation, BaseDataset, BoundingBox
from model.exceptions import DatasetNotFoundError, ParseError

logger = logging.getLogger(__name__)


class RDD2022Dataset(BaseDataset):
    """RDD2022 road damage dataset implementation.

    Supports both Supervisely JSON format (from Dataset Ninja) and PASCAL VOC
    XML format (from the original RDD2022 release). The format is auto-detected
    based on directory structure.

    Dataset Ninja structure:
        {path}/train/img/Japan_000001.jpg
        {path}/train/ann/Japan_000001.jpg.json
        {path}/meta.json

    PASCAL VOC structure:
        {path}/Japan_000001.xml
        {path}/Japan_000001.jpg

    Args:
        country_filter: Optional list of country names to include.
            If provided, only annotations from matching countries are loaded.
        subset: Which subset to load ("train" or "test"). Only used for
            Supervisely format. Defaults to "train".
    """

    def __init__(
        self,
        country_filter: Optional[List[str]] = None,
        subset: str = "train",
    ):
        self._annotations: List[Annotation] = []
        self._country_filter = country_filter
        self._subset = subset

    def load(self, path: Path) -> None:
        """Load dataset from the given path.

        Auto-detects the annotation format (JSON or XML) and parses accordingly.

        Args:
            path: Root directory of the dataset.

        Raises:
            DatasetNotFoundError: If the path does not exist.
            ParseError: If an annotation file is malformed.
        """
        path = Path(path)
        if not path.exists():
            raise DatasetNotFoundError(path)

        # Auto-detect format based on directory structure
        if self._is_supervisely_format(path):
            self._load_supervisely(path)
        else:
            self._load_voc_xml(path)

    def _is_supervisely_format(self, path: Path) -> bool:
        """Check if the dataset uses Supervisely/Dataset Ninja JSON format.

        Detects by looking for the train/ann/ or train/img/ directory structure,
        or the presence of .json annotation files.
        """
        # Check for train/ann directory
        ann_dir = path / self._subset / "ann"
        if ann_dir.exists():
            return True

        # Check for meta.json (Supervisely project file)
        if (path / "meta.json").exists():
            return True

        # Check for any .json files that look like annotations
        json_files = list(path.rglob("*.json"))
        for jf in json_files[:5]:  # Check first few
            if jf.name.endswith(".jpg.json") or jf.name.endswith(".png.json"):
                return True

        return False

    # ------------------------------------------------------------------
    # Supervisely JSON format loading
    # ------------------------------------------------------------------

    def _load_supervisely(self, path: Path) -> None:
        """Load dataset in Supervisely/Dataset Ninja JSON format.

        Expected structure:
            {path}/{subset}/img/*.jpg
            {path}/{subset}/ann/*.jpg.json
            {path}/meta.json (optional, for class definitions)
        """
        ann_dir = path / self._subset / "ann"
        img_dir = path / self._subset / "img"

        if not ann_dir.exists():
            logger.warning("Annotation directory not found: %s", ann_dir)
            return

        # Find all JSON annotation files
        json_files = sorted(ann_dir.glob("*.json"))

        annotations: List[Annotation] = []
        for json_file in json_files:
            annotation = self._parse_supervisely_json(json_file, img_dir)
            if annotation is None:
                continue

            # Apply country filter
            if self._country_filter is not None:
                country = annotation.metadata.get("country", "")
                if country not in self._country_filter:
                    continue

            # Validate image file existence
            if not annotation.image_path.exists():
                logger.warning(
                    "Image file not found, excluding annotation: %s",
                    annotation.image_path,
                )
                continue

            annotations.append(annotation)

        self._annotations = annotations
        logger.info(
            "Loaded %d annotations from Supervisely format (%s subset)",
            len(annotations),
            self._subset,
        )

    def _parse_supervisely_json(
        self, json_path: Path, img_dir: Path
    ) -> Optional[Annotation]:
        """Parse a Supervisely/Dataset Ninja JSON annotation file.

        Args:
            json_path: Path to the JSON annotation file.
            img_dir: Directory containing the image files.

        Returns:
            Annotation object or None if parsing fails.

        Raises:
            ParseError: If the JSON is malformed or missing required fields.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ParseError(
                file_path=json_path,
                line_number=e.lineno or 1,
                description=f"Invalid JSON: {e}",
            )

        # Extract image dimensions
        size = data.get("size")
        if not size or "width" not in size or "height" not in size:
            raise ParseError(
                file_path=json_path,
                line_number=1,
                description="Missing required field: 'size' with 'width' and 'height'",
            )

        width = size["width"]
        height = size["height"]

        if width <= 0 or height <= 0:
            raise ParseError(
                file_path=json_path,
                line_number=1,
                description=f"Invalid image dimensions: width={width}, height={height}",
            )

        # Derive image filename from annotation filename
        # Annotation files are named like "Japan_000001.jpg.json"
        # Image filename is "Japan_000001.jpg"
        ann_name = json_path.name
        if ann_name.endswith(".json"):
            image_filename = ann_name[: -len(".json")]
        else:
            image_filename = ann_name

        image_path = img_dir / image_filename

        # Extract country from image-level tags
        country = self._extract_country_from_tags(data.get("tags", []))
        if not country:
            # Fallback: extract from filename
            country = self._extract_country(image_filename)

        # Parse objects (bounding boxes)
        bounding_boxes: List[BoundingBox] = []
        for obj in data.get("objects", []):
            bbox = self._parse_supervisely_object(obj, width, height, json_path)
            if bbox is not None:
                bounding_boxes.append(bbox)

        metadata = {
            "country": country,
            "source": "rdd2022",
            "filename": image_filename,
            "format": "supervisely",
        }

        return Annotation(
            image_path=image_path,
            bounding_boxes=bounding_boxes,
            metadata=metadata,
        )

    def _parse_supervisely_object(
        self, obj: dict, width: int, height: int, json_path: Path
    ) -> Optional[BoundingBox]:
        """Parse a single object from Supervisely JSON into a BoundingBox.

        Args:
            obj: Object dict from the JSON annotation.
            width: Image width for coordinate normalization.
            height: Image height for coordinate normalization.
            json_path: Path to the JSON file for error reporting.

        Returns:
            BoundingBox with normalized coordinates, or None if invalid.
        """
        # Only handle rectangle geometry
        geometry_type = obj.get("geometryType", "")
        if geometry_type != "rectangle":
            logger.debug(
                "Skipping non-rectangle object (type=%s) in %s",
                geometry_type,
                json_path,
            )
            return None

        class_title = obj.get("classTitle", "")
        if not class_title:
            logger.warning("Object missing classTitle in %s", json_path)
            return None

        points = obj.get("points", {})
        exterior = points.get("exterior", [])

        if len(exterior) != 2:
            logger.warning(
                "Object has invalid exterior points (expected 2, got %d) in %s",
                len(exterior),
                json_path,
            )
            return None

        # Supervisely format: exterior = [[x_min, y_min], [x_max, y_max]]
        x_min_px = exterior[0][0]
        y_min_px = exterior[0][1]
        x_max_px = exterior[1][0]
        y_max_px = exterior[1][1]

        # Normalize to [0, 1]
        x_min_norm = x_min_px / width
        y_min_norm = y_min_px / height
        x_max_norm = x_max_px / width
        y_max_norm = y_max_px / height

        # Ensure valid box
        if x_min_norm >= x_max_norm or y_min_norm >= y_max_norm:
            logger.warning(
                "Invalid bounding box coordinates in %s: [%f, %f, %f, %f]",
                json_path,
                x_min_norm,
                y_min_norm,
                x_max_norm,
                y_max_norm,
            )
            return None

        return BoundingBox(
            x_min=x_min_norm,
            y_min=y_min_norm,
            x_max=x_max_norm,
            y_max=y_max_norm,
            class_label=class_title,
            confidence=1.0,
        )

    def _extract_country_from_tags(self, tags: List[dict]) -> str:
        """Extract country name from Supervisely image-level tags.

        Tags with country names (Japan, India, Czech, Norway, United States,
        China_Drone, China_MotorBike) are used to identify the source country.

        Args:
            tags: List of tag dicts from the annotation.

        Returns:
            Country name or empty string if not found.
        """
        country_tags = {
            "Japan", "India", "Czech", "Norway", "United States",
            "China_Drone", "China_MotorBike",
        }
        for tag in tags:
            tag_name = tag.get("name", "")
            if tag_name in country_tags:
                return tag_name
        return ""

    # ------------------------------------------------------------------
    # PASCAL VOC XML format loading (backward compatibility)
    # ------------------------------------------------------------------

    def _load_voc_xml(self, path: Path) -> None:
        """Load dataset in PASCAL VOC XML format.

        This is the original RDD2022 format from sekilab/RoadDamageDetector.
        """
        xml_files = sorted(path.rglob("*.xml"))

        annotations: List[Annotation] = []
        for xml_file in xml_files:
            annotation = self._parse_xml(xml_file, path)
            if annotation is None:
                continue

            if self._country_filter is not None:
                country = annotation.metadata.get("country", "")
                if country not in self._country_filter:
                    continue

            if not annotation.image_path.exists():
                logger.warning(
                    "Image file not found, excluding annotation: %s",
                    annotation.image_path,
                )
                continue

            annotations.append(annotation)

        self._annotations = annotations

    def _parse_xml(self, xml_path: Path, dataset_root: Path) -> Optional[Annotation]:
        """Parse a PASCAL VOC XML annotation file."""
        try:
            tree = ET.parse(str(xml_path))
        except ET.ParseError as e:
            line_number = e.position[0] if e.position else 0
            raise ParseError(
                file_path=xml_path,
                line_number=line_number,
                description=f"Invalid XML: {e}",
            )

        root = tree.getroot()

        filename_elem = root.find("filename")
        if filename_elem is None or not filename_elem.text:
            raise ParseError(
                file_path=xml_path,
                line_number=1,
                description="Missing required element: <filename>",
            )
        filename = filename_elem.text.strip()

        size_elem = root.find("size")
        if size_elem is None:
            raise ParseError(
                file_path=xml_path,
                line_number=1,
                description="Missing required element: <size>",
            )

        width = self._get_dimension(size_elem, "width", xml_path)
        height = self._get_dimension(size_elem, "height", xml_path)

        if width <= 0 or height <= 0:
            raise ParseError(
                file_path=xml_path,
                line_number=1,
                description=f"Invalid image dimensions: width={width}, height={height}",
            )

        image_path = self._resolve_image_path(xml_path, dataset_root, filename)
        country = self._extract_country(filename)

        bounding_boxes: List[BoundingBox] = []
        for obj_elem in root.findall("object"):
            bbox = self._parse_object(obj_elem, width, height, xml_path)
            if bbox is not None:
                bounding_boxes.append(bbox)

        metadata = {
            "country": country,
            "source": "rdd2022",
            "filename": filename,
            "format": "voc_xml",
        }

        return Annotation(
            image_path=image_path,
            bounding_boxes=bounding_boxes,
            metadata=metadata,
        )

    def _get_dimension(self, size_elem: ET.Element, name: str, xml_path: Path) -> int:
        """Extract a dimension value from the size element."""
        elem = size_elem.find(name)
        if elem is None or not elem.text:
            raise ParseError(
                file_path=xml_path,
                line_number=1,
                description=f"Missing required element: <size>/<{name}>",
            )
        try:
            return int(elem.text.strip())
        except ValueError:
            raise ParseError(
                file_path=xml_path,
                line_number=1,
                description=f"Invalid {name} value: '{elem.text.strip()}'",
            )

    def _parse_object(
        self, obj_elem: ET.Element, width: int, height: int, xml_path: Path
    ) -> Optional[BoundingBox]:
        """Parse an <object> element into a BoundingBox."""
        name_elem = obj_elem.find("name")
        if name_elem is None or not name_elem.text:
            raise ParseError(
                file_path=xml_path,
                line_number=1,
                description="Missing required element: <object>/<name>",
            )
        class_label = name_elem.text.strip()

        bndbox_elem = obj_elem.find("bndbox")
        if bndbox_elem is None:
            raise ParseError(
                file_path=xml_path,
                line_number=1,
                description="Missing required element: <object>/<bndbox>",
            )

        xmin = self._get_coord(bndbox_elem, "xmin", xml_path)
        ymin = self._get_coord(bndbox_elem, "ymin", xml_path)
        xmax = self._get_coord(bndbox_elem, "xmax", xml_path)
        ymax = self._get_coord(bndbox_elem, "ymax", xml_path)

        x_min_norm = xmin / width
        y_min_norm = ymin / height
        x_max_norm = xmax / width
        y_max_norm = ymax / height

        return BoundingBox(
            x_min=x_min_norm,
            y_min=y_min_norm,
            x_max=x_max_norm,
            y_max=y_max_norm,
            class_label=class_label,
            confidence=1.0,
        )

    def _get_coord(
        self, bndbox_elem: ET.Element, name: str, xml_path: Path
    ) -> float:
        """Extract a coordinate value from the bndbox element."""
        elem = bndbox_elem.find(name)
        if elem is None or not elem.text:
            raise ParseError(
                file_path=xml_path,
                line_number=1,
                description=f"Missing required element: <bndbox>/<{name}>",
            )
        try:
            return float(elem.text.strip())
        except ValueError:
            raise ParseError(
                file_path=xml_path,
                line_number=1,
                description=f"Invalid coordinate value for <{name}>: '{elem.text.strip()}'",
            )

    def _resolve_image_path(
        self, xml_path: Path, dataset_root: Path, filename: str
    ) -> Path:
        """Resolve the image path from the annotation filename."""
        same_dir = xml_path.parent / filename
        if same_dir.exists():
            return same_dir

        sibling_images = xml_path.parent.parent / "images" / filename
        if sibling_images.exists():
            return sibling_images

        root_images = dataset_root / "images" / filename
        if root_images.exists():
            return root_images

        return same_dir

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _extract_country(self, filename: str) -> str:
        """Extract country name from filename prefix.

        Handles patterns like:
        - "Japan_000001.jpg" -> "Japan"
        - "United_States_000001.jpg" -> "United_States"
        - "China_Drone_000001.jpg" -> "China_Drone"
        - "China_MotorBike_000001.jpg" -> "China_MotorBike"
        """
        # Known multi-word prefixes
        known_prefixes = [
            "United_States",
            "China_Drone",
            "China_MotorBike",
        ]
        for prefix in known_prefixes:
            if filename.startswith(prefix + "_"):
                return prefix

        # Single-word prefix: split on last underscore before the number
        parts = filename.rsplit("_", 1)
        if len(parts) == 2:
            return parts[0]
        return ""

    def get_annotations(self) -> List[Annotation]:
        """Return all loaded annotations."""
        return list(self._annotations)

    def split(
        self,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int = 42,
    ) -> Tuple["RDD2022Dataset", "RDD2022Dataset", "RDD2022Dataset"]:
        """Split dataset into train/validation/test subsets."""
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.10f}")

        indices = list(range(len(self._annotations)))
        rng = random.Random(seed)
        rng.shuffle(indices)

        n = len(indices)
        train_end = int(round(n * train_ratio))
        val_end = train_end + int(round(n * val_ratio))

        train_dataset = RDD2022Dataset(
            country_filter=self._country_filter, subset=self._subset
        )
        train_dataset._annotations = [self._annotations[i] for i in indices[:train_end]]

        val_dataset = RDD2022Dataset(
            country_filter=self._country_filter, subset=self._subset
        )
        val_dataset._annotations = [
            self._annotations[i] for i in indices[train_end:val_end]
        ]

        test_dataset = RDD2022Dataset(
            country_filter=self._country_filter, subset=self._subset
        )
        test_dataset._annotations = [self._annotations[i] for i in indices[val_end:]]

        return train_dataset, val_dataset, test_dataset

    def __iter__(self) -> Iterator[Annotation]:
        """Iterate over dataset annotations."""
        return iter(self._annotations)

    def __len__(self) -> int:
        """Return number of annotations in the dataset."""
        return len(self._annotations)

    def get_class_names(self) -> List[str]:
        """Return sorted unique class labels from all annotations."""
        class_names = set()
        for annotation in self._annotations:
            for bbox in annotation.bounding_boxes:
                class_names.add(bbox.class_label)
        return sorted(class_names)
