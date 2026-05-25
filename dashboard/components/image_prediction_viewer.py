"""Image prediction viewer component for the Streamlit Results Dashboard.

Displays individual images with ground truth annotations and model predictions
side by side, allowing visual comparison of detection quality. Provides
confidence threshold filtering, class filtering, and image navigation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image, ImageDraw, ImageFont


# --- Data Models ---


@dataclass
class BoundingBox:
    """A single bounding box annotation or prediction."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    class_name: str
    confidence: Optional[float] = None  # None for ground truth boxes


@dataclass
class ImageAnnotation:
    """Ground truth and prediction data for a single image."""

    image_id: str
    image_path: str
    ground_truth_boxes: list[BoundingBox] = field(default_factory=list)
    prediction_boxes: list[BoundingBox] = field(default_factory=list)


# --- Constants ---

# Distinct colors per damage class (RGB tuples)
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "alligator crack": (255, 0, 0),       # Red
    "longitudinal crack": (0, 255, 0),     # Green
    "other corruption": (0, 0, 255),       # Blue
    "pothole": (255, 165, 0),              # Orange
    "transverse crack": (128, 0, 128),     # Purple
}

# Fallback color for unknown classes
DEFAULT_COLOR: tuple[int, int, int] = (200, 200, 200)


# --- Filtering Functions ---


def filter_by_confidence(
    boxes: list[BoundingBox], threshold: float
) -> list[BoundingBox]:
    """Filter prediction boxes by confidence threshold.

    Returns only boxes with confidence >= threshold.
    Ground truth boxes (confidence=None) are always included.

    Args:
        boxes: List of bounding boxes to filter.
        threshold: Minimum confidence score (0.0 to 1.0).

    Returns:
        Filtered list of bounding boxes.
    """
    return [
        box
        for box in boxes
        if box.confidence is None or box.confidence >= threshold
    ]


def filter_by_class(
    boxes: list[BoundingBox], selected_classes: list[str]
) -> list[BoundingBox]:
    """Filter bounding boxes by class name.

    Returns only boxes whose class_name is in the selected_classes list.

    Args:
        boxes: List of bounding boxes to filter.
        selected_classes: List of class names to include.

    Returns:
        Filtered list of bounding boxes.
    """
    if not selected_classes:
        return boxes
    return [box for box in boxes if box.class_name in selected_classes]


# --- Drawing Functions ---


def get_class_color(class_name: str) -> tuple[int, int, int]:
    """Get the color for a given damage class.

    Args:
        class_name: The damage class name.

    Returns:
        RGB color tuple for the class.
    """
    return CLASS_COLORS.get(class_name, DEFAULT_COLOR)


def _denormalize_boxes(
    boxes: list[BoundingBox], img_width: int, img_height: int
) -> list[BoundingBox]:
    """Scale bounding boxes from normalized [0,1] to pixel coordinates.

    If coordinates are already in pixel range (any coord > 1.0), returns as-is.

    Args:
        boxes: List of bounding boxes (possibly normalized).
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        List of BoundingBox with pixel coordinates.
    """
    if not boxes:
        return boxes

    # Check if coordinates are normalized (all <= 1.0)
    all_normalized = all(
        box.x_min <= 1.0 and box.y_min <= 1.0 and
        box.x_max <= 1.0 and box.y_max <= 1.0
        for box in boxes
    )

    if not all_normalized:
        return boxes  # Already in pixel coordinates

    return [
        BoundingBox(
            x_min=box.x_min * img_width,
            y_min=box.y_min * img_height,
            x_max=box.x_max * img_width,
            y_max=box.y_max * img_height,
            class_name=box.class_name,
            confidence=box.confidence,
        )
        for box in boxes
    ]


def draw_bounding_boxes(
    image: Image.Image,
    boxes: list[BoundingBox],
    show_confidence: bool = False,
) -> Image.Image:
    """Draw bounding boxes on an image with class labels and optional confidence.

    Args:
        image: PIL Image to draw on (will be copied, not modified in place).
        boxes: List of bounding boxes to draw.
        show_confidence: Whether to display confidence scores on labels.

    Returns:
        New PIL Image with bounding boxes drawn.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    # Try to use a basic font; fall back to default if unavailable
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for box in boxes:
        color = get_class_color(box.class_name)
        # Draw rectangle
        draw.rectangle(
            [box.x_min, box.y_min, box.x_max, box.y_max],
            outline=color,
            width=2,
        )

        # Build label text
        label = box.class_name
        if show_confidence and box.confidence is not None:
            label = f"{box.class_name} ({box.confidence:.2f})"

        # Draw label background
        bbox = draw.textbbox((box.x_min, box.y_min), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        label_y = max(box.y_min - text_h - 4, 0)
        draw.rectangle(
            [box.x_min, label_y, box.x_min + text_w + 4, label_y + text_h + 4],
            fill=color,
        )
        draw.text(
            (box.x_min + 2, label_y + 2),
            label,
            fill=(255, 255, 255),
            font=font,
        )

    return annotated


# --- Image Loading ---


def load_image(image_path: str) -> Optional[Image.Image]:
    """Load an image from disk.

    Args:
        image_path: Path to the image file.

    Returns:
        PIL Image if found, None otherwise.
    """
    path = Path(image_path)
    if path.exists():
        try:
            return Image.open(path).convert("RGB")
        except (OSError, IOError):
            return None
    return None


def create_placeholder_image(width: int = 640, height: int = 480) -> Image.Image:
    """Create a placeholder image when the actual image is not available.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        A gray placeholder PIL Image with centered text.
    """
    img = Image.new("RGB", (width, height), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except (OSError, IOError):
        font = ImageFont.load_default()
    text = "Image not found"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (width - text_w) // 2
    y = (height - text_h) // 2
    draw.text((x, y), text, fill=(100, 100, 100), font=font)
    return img


# --- Main Component ---


def render_image_prediction_viewer(
    annotations: list[ImageAnnotation],
    class_names: Optional[list[str]] = None,
    key_prefix: str = "",
) -> None:
    """Render the image prediction viewer component.

    Displays side-by-side panels with ground truth and prediction overlays,
    along with filtering controls and image navigation.

    Args:
        annotations: List of ImageAnnotation objects for available images.
        class_names: List of damage class names for the class filter.
            Defaults to the keys of CLASS_COLORS if not provided.
        key_prefix: Prefix for Streamlit widget keys to avoid conflicts
            when rendering multiple viewers on the same page.
    """
    st.header("Image Prediction Viewer")

    if not annotations:
        st.info(
            "No image annotations available. "
            "Run model evaluation with prediction output to view results here."
        )
        return

    if class_names is None:
        class_names = list(CLASS_COLORS.keys())

    # --- Controls ---

    # Image selection
    image_ids = [ann.image_id for ann in annotations]

    # Initialize session state for current image index
    idx_key = f"{key_prefix}_prediction_viewer_index"
    if idx_key not in st.session_state:
        st.session_state[idx_key] = 0

    # Clamp index to valid range
    current_index = st.session_state[idx_key]
    current_index = max(0, min(current_index, len(annotations) - 1))
    st.session_state[idx_key] = current_index

    # Image selection dropdown
    selected_id = st.selectbox(
        "Select Image",
        options=image_ids,
        index=current_index,
        key=f"{key_prefix}_prediction_viewer_select",
    )

    # Update index from selectbox
    if selected_id in image_ids:
        st.session_state[idx_key] = image_ids.index(selected_id)
        current_index = st.session_state[idx_key]

    # Navigation buttons
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col1:
        if st.button("← Previous", disabled=(current_index == 0), key=f"{key_prefix}_prev"):
            st.session_state[idx_key] = current_index - 1
            st.rerun()
    with nav_col3:
        if st.button(
            "Next →", disabled=(current_index >= len(annotations) - 1), key=f"{key_prefix}_next"
        ):
            st.session_state[idx_key] = current_index + 1
            st.rerun()
    with nav_col2:
        st.caption(f"Image {current_index + 1} of {len(annotations)}")

    # Confidence threshold slider
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        key=f"{key_prefix}_prediction_viewer_confidence",
        help="Filter predictions by minimum confidence score",
    )

    # Class filter multi-select
    selected_classes = st.multiselect(
        "Filter by Class",
        options=class_names,
        default=class_names,
        key=f"{key_prefix}_prediction_viewer_classes",
        help="Select damage classes to display",
    )

    # --- Get current annotation ---
    current_annotation = annotations[current_index]

    # --- Apply filters ---
    gt_boxes = filter_by_class(
        current_annotation.ground_truth_boxes, selected_classes
    )
    pred_boxes_filtered = filter_by_confidence(
        current_annotation.prediction_boxes, confidence_threshold
    )
    pred_boxes_filtered = filter_by_class(pred_boxes_filtered, selected_classes)

    # --- Display counts ---
    count_col1, count_col2 = st.columns(2)
    with count_col1:
        st.metric("Ground Truth Boxes", len(gt_boxes))
    with count_col2:
        st.metric("Predicted Boxes (above threshold)", len(pred_boxes_filtered))

    # --- Handle no predictions ---
    if not pred_boxes_filtered and current_annotation.prediction_boxes:
        st.warning(
            "No detections at this confidence level. "
            "Try lowering the confidence threshold."
        )
    elif not current_annotation.prediction_boxes:
        st.info("No predictions available for this image.")

    # --- Load and render images ---
    image = load_image(current_annotation.image_path)
    if image is None:
        image = create_placeholder_image()

    # Denormalize box coordinates if they are in [0, 1] range
    img_w, img_h = image.size
    gt_boxes_scaled = _denormalize_boxes(gt_boxes, img_w, img_h)
    pred_boxes_scaled = _denormalize_boxes(pred_boxes_filtered, img_w, img_h)

    # Draw overlays
    gt_image = draw_bounding_boxes(image, gt_boxes_scaled, show_confidence=False)
    pred_image = draw_bounding_boxes(
        image, pred_boxes_scaled, show_confidence=True
    )

    # Side-by-side display
    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Ground Truth")
        st.image(gt_image, width="stretch")
    with right_col:
        st.subheader("Predictions")
        st.image(pred_image, width="stretch")

    # --- Color legend ---
    st.markdown("**Class Colors:**")
    legend_cols = st.columns(len(class_names))
    for i, cls_name in enumerate(class_names):
        color = get_class_color(cls_name)
        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        with legend_cols[i]:
            st.markdown(
                f'<span style="color:{hex_color}; font-weight:bold;">■</span> '
                f"{cls_name}",
                unsafe_allow_html=True,
            )
