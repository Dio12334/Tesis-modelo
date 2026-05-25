"""YOLOv6 model wrapper for the Road Damage Evaluation Framework."""

from pathlib import Path
from typing import TYPE_CHECKING, List

from model.exceptions import ConfigurationError
from model.models.registry import BaseDetector, ModelRegistry

if TYPE_CHECKING:
    import torch

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

VALID_BACKBONE_SIZES = ("nano", "small", "medium", "large")


@ModelRegistry.register("yolov6")
class YOLOv6Detector(BaseDetector):
    """YOLOv6 object detection model wrapper.

    Supports configurable backbone sizes: nano, small, medium, large.
    This is an adapter/wrapper that would integrate with the actual YOLOv6
    package when available. Currently provides a placeholder implementation
    for the forward pass.
    """

    def __init__(self, config: dict):
        """Initialize YOLOv6 detector with configuration.

        Args:
            config: Configuration dict. Must contain:
                - backbone_size (str): One of nano, small, medium, large
                - num_classes (int): Number of detection classes

        Raises:
            ConfigurationError: If backbone_size is not valid.
        """
        self.config = config
        self.backbone_size = config.get("backbone_size", "")
        self.num_classes = config.get("num_classes", 4)

        if self.backbone_size not in VALID_BACKBONE_SIZES:
            raise ConfigurationError(
                [
                    f"Invalid backbone_size '{self.backbone_size}'. "
                    f"Must be one of: {list(VALID_BACKBONE_SIZES)}"
                ]
            )

        # Placeholder for actual YOLOv6 model
        # In a real implementation, this would load the YOLOv6 model
        # from the meituan/YOLOv6 package with the specified backbone size.
        self._state_dict: dict = {}
        self._model_initialized = True

    def forward(self, images: "torch.Tensor") -> List[dict]:
        """Run forward pass on a batch of images.

        This is a placeholder implementation. The actual YOLOv6 integration
        would require the yolov6 package to be installed.

        Args:
            images: Batch of images as a torch.Tensor with shape (B, C, H, W).

        Returns:
            List of dicts per image, each containing:
                - boxes: Empty tensor of shape (0, 4)
                - labels: Empty tensor of shape (0,)
                - scores: Empty tensor of shape (0,)
        """
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for forward pass")

        batch_size = images.shape[0] if hasattr(images, "shape") else 1
        predictions = []
        for _ in range(batch_size):
            predictions.append(
                {
                    "boxes": torch.zeros((0, 4)),
                    "labels": torch.zeros((0,), dtype=torch.long),
                    "scores": torch.zeros((0,)),
                }
            )
        return predictions

    def get_config_schema(self) -> dict:
        """Return required configuration parameters for YOLOv6.

        Returns:
            Dict describing required config params.
        """
        return {
            "backbone_size": {"type": "str", "required": True},
            "num_classes": {"type": "int", "required": True},
        }

    def load_checkpoint(self, path: Path) -> None:
        """Load model weights from a checkpoint file.

        Args:
            path: Path to the checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required to load checkpoints")

        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self._state_dict = torch.load(checkpoint_path, map_location="cpu")

    def save_checkpoint(self, path: Path) -> None:
        """Save model weights to a checkpoint file.

        Args:
            path: Path where the checkpoint will be saved.
        """
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required to save checkpoints")

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._state_dict, checkpoint_path)
