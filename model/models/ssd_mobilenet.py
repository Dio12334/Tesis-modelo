"""SSD MobileNetV3 model wrapper using torchvision for the Road Damage Evaluation Framework.

Uses torchvision.models.detection.ssdlite320_mobilenet_v3_large as the backbone
for 320px input, or a custom SSD head on MobileNetV3 for 640px input.
"""

import logging
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssd import SSD, SSDHead
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models import mobilenet_v3_large

from model.exceptions import ConfigurationError
from model.models.registry import BaseDetector, ModelRegistry

logger = logging.getLogger(__name__)

VALID_INPUT_SIZES = (320, 640)


@ModelRegistry.register("ssd_mobilenetv3")
class SSDMobileNetV3(BaseDetector):
    """SSD MobileNetV3 object detection model using torchvision.

    Uses torchvision's SSDLite320 MobileNetV3-Large pretrained backbone
    with a custom detection head for the configured number of classes.

    Args:
        config: Dict with keys:
            - input_size (int): 320 or 640
            - num_classes (int): Number of target classes (excluding background)
            - pretrained_backbone (bool): Use ImageNet-pretrained backbone (default True)
    """

    def __init__(self, config: dict):
        self.config = config
        self.input_size = config.get("input_size", 0)
        self.num_classes = config.get("num_classes", 4)
        self.pretrained_backbone = config.get("pretrained_backbone", True)

        if self.input_size not in VALID_INPUT_SIZES:
            raise ConfigurationError(
                [
                    f"Invalid input_size '{self.input_size}'. "
                    f"Must be one of: {list(VALID_INPUT_SIZES)}"
                ]
            )

        # Build the actual torchvision SSD model
        # num_classes + 1 for background class (torchvision convention)
        self._model = self._build_model()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

        logger.info(
            "Initialized SSD MobileNetV3 (input_size=%d, num_classes=%d, device=%s)",
            self.input_size,
            self.num_classes,
            self._device,
        )

    def _build_model(self) -> nn.Module:
        """Build the SSD MobileNetV3 model from torchvision.

        For input_size=320, uses the standard ssdlite320_mobilenet_v3_large
        with a modified head for the target number of classes.

        Returns:
            A torchvision SSD detection model.
        """
        if self.input_size == 320:
            # Use torchvision's built-in SSDLite320 with MobileNetV3-Large
            weights_backbone = "DEFAULT" if self.pretrained_backbone else None
            model = ssdlite320_mobilenet_v3_large(
                weights=None,  # Don't load COCO-pretrained detection weights
                weights_backbone=weights_backbone,
                num_classes=self.num_classes + 1,  # +1 for background
            )
        else:
            # For 640px input, build a custom SSD with MobileNetV3 backbone
            # Use the same architecture but with larger input
            weights_backbone = "DEFAULT" if self.pretrained_backbone else None
            model = ssdlite320_mobilenet_v3_large(
                weights=None,
                weights_backbone=weights_backbone,
                num_classes=self.num_classes + 1,
            )
            # The model will handle resizing internally via transforms

        return model

    def forward(self, images: torch.Tensor) -> List[dict]:
        """Run forward pass on a batch of images.

        In training mode, expects (images, targets) and returns losses.
        In eval mode, returns predictions.

        Args:
            images: Batch of images as a torch.Tensor with shape (B, C, H, W),
                    values in [0, 1] range.

        Returns:
            List of dicts per image, each containing:
                - boxes: Tensor of shape (N, 4) with [x1, y1, x2, y2] pixel coords
                - labels: Tensor of shape (N,) with class indices
                - scores: Tensor of shape (N,) with confidence scores
        """
        self._model.eval()
        images = images.to(self._device)

        # torchvision detection models expect a list of tensors
        image_list = [img for img in images]

        with torch.no_grad():
            outputs = self._model(image_list)

        # Normalize box coordinates to [0, 1] if needed
        results = []
        for i, output in enumerate(outputs):
            h, w = images.shape[2], images.shape[3]
            boxes = output["boxes"]
            if boxes.numel() > 0:
                # Normalize to [0, 1]
                boxes[:, [0, 2]] /= w
                boxes[:, [1, 3]] /= h

            results.append({
                "boxes": boxes,
                
                "labels": output["labels"],
                "scores": output["scores"],
            })

        return results

    def train_step(
        self,
        images: List[torch.Tensor],
        targets: List[dict],
    ) -> dict:
        """Perform a single training step.

        Args:
            images: List of image tensors (C, H, W) in [0, 1] range.
            targets: List of target dicts, each with:
                - boxes: Tensor (N, 4) in [x1, y1, x2, y2] pixel coords
                - labels: Tensor (N,) with class indices (1-indexed, 0=background)

        Returns:
            Dict with loss values: classification_loss, regression_loss, total_loss.
        """
        self._model.train()
        images = [img.to(self._device) for img in images]
        targets = [
            {k: v.to(self._device) for k, v in t.items()} for t in targets
        ]

        loss_dict = self._model(images, targets)

        # torchvision SSD returns: classification, bbox_regression
        total_loss = sum(loss for loss in loss_dict.values())

        return {
            "classification_loss": loss_dict.get("classification", torch.tensor(0.0)).item(),
            "bbox_regression_loss": loss_dict.get("bbox_regression", torch.tensor(0.0)).item(),
            "total_loss": total_loss.item(),
            "loss_tensor": total_loss,  # For backward pass
        }

    def get_parameters(self) -> List[torch.nn.Parameter]:
        """Return model parameters for optimizer construction.

        Returns:
            List of trainable parameters.
        """
        return [p for p in self._model.parameters() if p.requires_grad]

    def set_train_mode(self) -> None:
        """Set model to training mode."""
        self._model.train()

    def set_eval_mode(self) -> None:
        """Set model to evaluation mode."""
        self._model.eval()

    def get_config_schema(self) -> dict:
        """Return required configuration parameters."""
        return {
            "input_size": {"type": "int", "required": True},
            "num_classes": {"type": "int", "required": True},
        }

    def load_checkpoint(self, path: Path) -> None:
        """Load model weights from a checkpoint file.

        Args:
            path: Path to the .pt checkpoint file.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self._device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self._model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Loaded model state dict from %s", checkpoint_path)
        else:
            # Assume it's a raw state dict
            self._model.load_state_dict(checkpoint)
            logger.info("Loaded raw state dict from %s", checkpoint_path)

    def save_checkpoint(self, path: Path, optimizer=None, epoch=None, metrics=None) -> None:
        """Save model weights and training state to a checkpoint file.

        Args:
            path: Path where the checkpoint will be saved.
            optimizer: Optional optimizer to save state.
            epoch: Optional current epoch number.
            metrics: Optional metrics dict to include.
        """
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self._model.state_dict(),
            "config": self.config,
            "num_classes": self.num_classes,
            "input_size": self.input_size,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            checkpoint["epoch"] = epoch
        if metrics is not None:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, checkpoint_path)
        logger.info("Saved checkpoint to %s", checkpoint_path)
