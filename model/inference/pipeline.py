"""Inference pipeline for running predictions on images using trained models."""

import logging
from pathlib import Path
from typing import List, Tuple

from model.datasets.base import BoundingBox
from model.models.registry import BaseDetector

logger = logging.getLogger(__name__)

# Supported image extensions for directory scanning
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def compute_iou(box_a: Tuple[float, float, float, float],
                box_b: Tuple[float, float, float, float]) -> float:
    """Compute Intersection over Union between two bounding boxes.

    Args:
        box_a: Tuple of (x_min, y_min, x_max, y_max).
        box_b: Tuple of (x_min, y_min, x_max, y_max).

    Returns:
        IoU value in [0, 1].
    """
    x_min = max(box_a[0], box_b[0])
    y_min = max(box_a[1], box_b[1])
    x_max = min(box_a[2], box_b[2])
    y_max = min(box_a[3], box_b[3])

    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    if intersection == 0.0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


def filter_by_confidence(predictions: List[BoundingBox],
                         threshold: float) -> List[BoundingBox]:
    """Filter predictions by confidence threshold.

    Args:
        predictions: List of BoundingBox predictions.
        threshold: Minimum confidence score to keep a prediction.

    Returns:
        List of BoundingBox predictions with confidence >= threshold.
    """
    return [p for p in predictions if p.confidence >= threshold]


def apply_nms(boxes: List[Tuple[float, float, float, float]],
              scores: List[float],
              iou_threshold: float) -> List[int]:
    """Apply Non-Maximum Suppression to a set of boxes.

    For a single class, removes overlapping boxes with IoU > iou_threshold,
    keeping the highest confidence box.

    Args:
        boxes: List of (x_min, y_min, x_max, y_max) tuples.
        scores: List of confidence scores corresponding to each box.
        iou_threshold: IoU threshold above which overlapping boxes are suppressed.

    Returns:
        List of indices of boxes to keep, sorted by descending score.
    """
    if not boxes:
        return []

    # Create list of (index, score) and sort by score descending
    indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    keep = []
    suppressed = set()

    for idx, _score in indexed_scores:
        if idx in suppressed:
            continue
        keep.append(idx)

        # Suppress all remaining boxes with IoU > threshold
        for other_idx, _other_score in indexed_scores:
            if other_idx in suppressed or other_idx == idx:
                continue
            iou = compute_iou(boxes[idx], boxes[other_idx])
            if iou > iou_threshold:
                suppressed.add(other_idx)

    return keep


def apply_nms_to_predictions(predictions: List[BoundingBox],
                             iou_threshold: float) -> List[BoundingBox]:
    """Apply per-class NMS to a list of BoundingBox predictions.

    For each class, applies NMS independently and returns the combined result.

    Args:
        predictions: List of BoundingBox predictions.
        iou_threshold: IoU threshold for NMS suppression.

    Returns:
        Filtered list of BoundingBox predictions after NMS.
    """
    if not predictions:
        return []

    # Group predictions by class
    class_groups: dict = {}
    for pred in predictions:
        if pred.class_label not in class_groups:
            class_groups[pred.class_label] = []
        class_groups[pred.class_label].append(pred)

    result = []
    for _class_label, class_preds in class_groups.items():
        boxes = [(p.x_min, p.y_min, p.x_max, p.y_max) for p in class_preds]
        scores = [p.confidence for p in class_preds]
        keep_indices = apply_nms(boxes, scores, iou_threshold)
        result.extend(class_preds[i] for i in keep_indices)

    return result


class InferencePipeline:
    """Runs inference on images using trained models.

    Provides methods for single-image prediction, batch directory prediction,
    and saving annotated images with drawn bounding boxes.
    """

    def __init__(self, model: BaseDetector, confidence_threshold: float = 0.5,
                 nms_iou_threshold: float = 0.45, batch_size: int = 8):
        """Initialize the inference pipeline.

        Args:
            model: A trained BaseDetector model instance.
            confidence_threshold: Minimum confidence to keep predictions.
            nms_iou_threshold: IoU threshold for Non-Maximum Suppression.
            batch_size: Number of images to process in a batch.
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.batch_size = batch_size

    def predict_image(self, image_path: Path) -> List[BoundingBox]:
        """Run inference on a single image.

        Loads the image, runs the model forward pass, applies confidence
        filtering and NMS post-processing.

        Args:
            image_path: Path to the image file.

        Returns:
            List of BoundingBox predictions after filtering and NMS.

        Raises:
            FileNotFoundError: If the image file does not exist.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image as tensor
        image_tensor = self._load_image(image_path)

        # Run model forward pass
        outputs = self.model.forward(image_tensor)

        # Convert model output to BoundingBox list
        predictions = self._parse_model_output(outputs)

        # Apply confidence threshold filtering
        predictions = filter_by_confidence(predictions, self.confidence_threshold)

        # Apply NMS per class
        predictions = apply_nms_to_predictions(predictions, self.nms_iou_threshold)

        return predictions

    def predict_directory(self, dir_path: Path) -> dict:
        """Run batch inference on all images in a directory.

        Finds all image files in the directory and runs predict_image on each.

        Args:
            dir_path: Path to directory containing images.

        Returns:
            Dict mapping filename (str) to list of BoundingBox predictions.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        # Find all image files
        image_files = sorted(
            f for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )

        results = {}
        for image_file in image_files:
            try:
                predictions = self.predict_image(image_file)
                results[image_file.name] = predictions
            except Exception as e:
                logger.warning(f"Failed to process {image_file.name}: {e}")
                results[image_file.name] = []

        return results

    def save_annotated(self, image_path: Path, predictions: List[BoundingBox],
                       output_path: Path) -> None:
        """Save image with drawn bounding boxes to output path.

        Draws each prediction as a colored rectangle with class label and
        confidence score.

        Args:
            image_path: Path to the original image.
            predictions: List of BoundingBox predictions to draw.
            output_path: Path where the annotated image will be saved.
        """
        image_path = Path(image_path)
        output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            logger.error(
                "Pillow is required for save_annotated. "
                "Install it with: pip install Pillow"
            )
            raise ImportError(
                "Pillow is required for saving annotated images. "
                "Install with: pip install Pillow"
            )

        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        width, height = image.size

        # Color palette for different classes
        colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00",
            "#FF00FF", "#00FFFF", "#FFA500", "#800080",
        ]
        class_colors: dict = {}
        color_idx = 0

        for pred in predictions:
            # Assign color per class
            if pred.class_label not in class_colors:
                class_colors[pred.class_label] = colors[color_idx % len(colors)]
                color_idx += 1
            color = class_colors[pred.class_label]

            # Convert normalized coordinates to pixel coordinates
            x_min = int(pred.x_min * width)
            y_min = int(pred.y_min * height)
            x_max = int(pred.x_max * width)
            y_max = int(pred.y_max * height)

            # Draw bounding box
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)

            # Draw label
            label = f"{pred.class_label} {pred.confidence:.2f}"
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            draw.text((x_min, max(0, y_min - 12)), label, fill=color, font=font)

        image.save(output_path)
        logger.info(f"Saved annotated image to {output_path}")

    def _load_image(self, image_path: Path):
        """Load an image file and convert to tensor format for model input.

        Attempts to use torchvision transforms if available, otherwise
        falls back to PIL + numpy conversion.

        Args:
            image_path: Path to the image file.

        Returns:
            Image tensor suitable for model.forward().
        """
        try:
            import torch
            from torchvision import transforms
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            tensor = transform(image)
            # Add batch dimension
            return tensor.unsqueeze(0)
        except ImportError:
            # Fallback: try PIL + numpy
            try:
                from PIL import Image
                import numpy as np

                image = Image.open(image_path).convert("RGB")
                arr = np.array(image, dtype=np.float32) / 255.0
                # Transpose to CHW format and add batch dimension
                arr = arr.transpose(2, 0, 1)
                arr = arr[np.newaxis, ...]
                return arr
            except ImportError:
                raise ImportError(
                    "Either torch+torchvision or PIL+numpy is required "
                    "for image loading. Install with: pip install torch torchvision "
                    "or pip install Pillow numpy"
                )

    def _parse_model_output(self, outputs: List[dict]) -> List[BoundingBox]:
        """Parse model forward pass output into BoundingBox list.

        The model returns a list of dicts per image, each containing:
            - boxes: Tensor/array of shape (N, 4) with coordinates
            - labels: Tensor/array of shape (N,) with class indices
            - scores: Tensor/array of shape (N,) with confidence scores

        Args:
            outputs: List of output dicts from model.forward().

        Returns:
            List of BoundingBox predictions.
        """
        predictions = []

        for output in outputs:
            boxes = output.get("boxes", [])
            labels = output.get("labels", [])
            scores = output.get("scores", [])

            # Handle torch tensors or numpy arrays
            try:
                boxes = boxes.detach().cpu().numpy() if hasattr(boxes, 'detach') else boxes
                labels = labels.detach().cpu().numpy() if hasattr(labels, 'detach') else labels
                scores = scores.detach().cpu().numpy() if hasattr(scores, 'detach') else scores
            except Exception:
                pass

            # Convert to list if needed
            try:
                num_boxes = len(boxes)
            except TypeError:
                num_boxes = 0

            for i in range(num_boxes):
                try:
                    box = boxes[i]
                    label = labels[i]
                    score = scores[i]

                    # Convert label index to string if it's numeric
                    class_label = str(int(label)) if not isinstance(label, str) else label

                    predictions.append(BoundingBox(
                        x_min=float(box[0]),
                        y_min=float(box[1]),
                        x_max=float(box[2]),
                        y_max=float(box[3]),
                        class_label=class_label,
                        confidence=float(score),
                    ))
                except (IndexError, TypeError, ValueError) as e:
                    logger.warning(f"Failed to parse prediction {i}: {e}")
                    continue

        return predictions
