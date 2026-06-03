"""MobileNetV4 SSD object detection model for the Road Damage Evaluation Framework.

Uses timm's MobileNetV4 as the feature extraction backbone with a custom SSD
detection head for multi-scale object detection. Registered as "mobilenetv4_ssd"
in the ModelRegistry.
"""

import logging
from typing import List, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import box_iou, sigmoid_focal_loss

from model.exceptions import ConfigurationError
from model.models.registry import BaseDetector, ModelRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIANT_ALIASES = {
    "small": "mobilenetv4_conv_small.e2400_r224_in1k",
    "medium": "mobilenetv4_conv_medium.e500_r256_in1k",
    "large": "mobilenetv4_hybrid_large.e600_r384_in1k",
}

VALID_BACKBONE_VARIANTS = (
    "mobilenetv4_conv_small.e2400_r224_in1k",
    "mobilenetv4_conv_medium.e500_r256_in1k",
    "mobilenetv4_hybrid_large.e600_r384_in1k",
)


# ---------------------------------------------------------------------------
# MobileNetV4Backbone
# ---------------------------------------------------------------------------


class MobileNetV4Backbone(nn.Module):
    """MobileNetV4 feature extraction backbone using timm.

    Extracts multi-scale feature maps from a MobileNetV4 network loaded via
    ``timm.create_model(..., features_only=True)``. Produces 4 backbone-level
    outputs plus 2 additional downsampled feature maps (via extra convolutional
    layers) for a total of 6 multi-scale feature maps suitable for SSD-style
    detection heads.

    For a 640×640 input the spatial resolutions are:
        Level 0: 80×80   (timm stage at stride 8)
        Level 1: 40×40   (timm stage at stride 16)
        Level 2: 20×20   (timm stage at stride 32)
        Level 3: 10×10   (extra downsample from stride-32 stage)
        Level 4: 5×5     (extra conv layer 1)
        Level 5: 3×3     (extra conv layer 2)

    Args:
        backbone_variant: Full timm model name or alias ("small", "medium", "large").
        pretrained: Whether to load ImageNet-pretrained weights.
    """

    def __init__(self, backbone_variant: str = "small", pretrained: bool = True):
        super().__init__()

        # Resolve alias to full timm model name
        variant = VARIANT_ALIASES.get(backbone_variant, backbone_variant)
        if variant not in VALID_BACKBONE_VARIANTS:
            raise ValueError(
                f"Invalid backbone_variant '{backbone_variant}'. "
                f"Valid options: {list(VARIANT_ALIASES.keys())} or "
                f"{list(VALID_BACKBONE_VARIANTS)}"
            )

        # Create timm backbone with multi-scale feature extraction
        self.backbone = timm.create_model(
            variant,
            pretrained=pretrained,
            features_only=True,
        )

        # Get feature info from timm (channels at each stage)
        # timm MobileNetV4 models produce 5 stages with reductions [2, 4, 8, 16, 32].
        # For 640×640 input: 320×320, 160×160, 80×80, 40×40, 20×20.
        # We select the last 3 stages (stride 8, 16, 32) for 80×80, 40×40, 20×20
        # and add an extra stride-2 conv to produce 10×10 (the 4th backbone level).
        feature_info = self.backbone.feature_info
        self._feature_channels = feature_info.channels()
        self._feature_reductions = feature_info.reduction()

        # Select stages with reduction >= 8 (i.e., 80×80, 40×40, 20×20 for 640 input)
        # These correspond to the last 3 timm stages for MobileNetV4.
        num_stages = len(self._feature_channels)
        self._stage_indices = [
            i for i in range(num_stages) if self._feature_reductions[i] >= 8
        ]

        # Channel count of the last selected backbone stage (stride 32, 20×20)
        last_backbone_channels = self._feature_channels[self._stage_indices[-1]]

        # Downsample layer to produce 10×10 from 20×20 (4th backbone-level output)
        downsample_channels = 256
        self.downsample = nn.Sequential(
            nn.Conv2d(last_backbone_channels, downsample_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(downsample_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(
                downsample_channels, downsample_channels,
                kernel_size=3, stride=2, padding=1, bias=False,
            ),
            nn.BatchNorm2d(downsample_channels),
            nn.ReLU6(inplace=True),
        )

        # Extra convolutional layer 1: 10×10 → 5×5
        extra1_channels = 256
        self.extra1 = nn.Sequential(
            nn.Conv2d(downsample_channels, extra1_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(extra1_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(
                extra1_channels, extra1_channels,
                kernel_size=3, stride=2, padding=1, bias=False,
            ),
            nn.BatchNorm2d(extra1_channels),
            nn.ReLU6(inplace=True),
        )

        # Extra convolutional layer 2: 5×5 → 3×3
        extra2_channels = 256
        self.extra2 = nn.Sequential(
            nn.Conv2d(extra1_channels, extra2_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(extra2_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(
                extra2_channels, extra2_channels,
                kernel_size=3, stride=2, padding=1, bias=False,
            ),
            nn.BatchNorm2d(extra2_channels),
            nn.ReLU6(inplace=True),
        )

        # Store output channel counts for downstream use (e.g., SSD head)
        # 4 backbone-level outputs + 2 extra layers = 6 total
        selected_channels = [self._feature_channels[i] for i in self._stage_indices]
        self.out_channels = selected_channels + [
            downsample_channels, extra1_channels, extra2_channels,
        ]

        logger.info(
            "MobileNetV4Backbone initialized: variant=%s, pretrained=%s, "
            "output_channels=%s",
            variant, pretrained, self.out_channels,
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale feature maps from the input tensor.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            List of 6 feature map tensors at decreasing spatial resolutions:
            [~80×80, ~40×40, ~20×20, ~10×10, ~5×5, ~3×3] for 640×640 input.
        """
        # Extract all backbone stage features
        all_features = self.backbone(x)

        # Select the relevant stages (stride 8, 16, 32)
        features = [all_features[i] for i in self._stage_indices]

        # Generate 10×10 from the last backbone stage (20×20)
        ds_out = self.downsample(features[-1])
        features.append(ds_out)

        # Generate extra downsampled feature maps (5×5, 3×3)
        extra1_out = self.extra1(ds_out)
        features.append(extra1_out)

        extra2_out = self.extra2(extra1_out)
        features.append(extra2_out)

        return features


# ---------------------------------------------------------------------------
# SSDHead
# ---------------------------------------------------------------------------


class SSDHead(nn.Module):
    """SSD prediction head for classification and bounding box regression.

    Applies per-level convolutional predictors on multi-scale feature maps.
    Each feature map level gets its own classification and box regression
    convolution based on the number of anchors at that level.

    Excludes degenerate feature map stages (1×1 or smaller) from prediction.

    Args:
        in_channels: List of channel counts for each feature map level.
        num_anchors_per_location: List of anchor counts per spatial location
            at each level.
        num_classes: Number of object classes (excluding background).
    """

    def __init__(
        self,
        in_channels: List[int],
        num_anchors_per_location: List[int],
        num_classes: int,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Per-level classification predictors
        cls_heads = []
        for channels, num_anchors in zip(in_channels, num_anchors_per_location):
            cls_heads.append(
                nn.Conv2d(channels, num_anchors * num_classes, kernel_size=3, padding=1)
            )
        self.cls_heads = nn.ModuleList(cls_heads)

        # Per-level box regression predictors (4 coords per anchor)
        bbox_heads = []
        for channels, num_anchors in zip(in_channels, num_anchors_per_location):
            bbox_heads.append(
                nn.Conv2d(channels, num_anchors * 4, kernel_size=3, padding=1)
            )
        self.bbox_heads = nn.ModuleList(bbox_heads)

        logger.info(
            "SSDHead initialized: num_classes=%d, levels=%d",
            num_classes, len(in_channels),
        )

    def forward(
        self, features: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Produce classification and box regression predictions.

        Excludes feature maps with spatial resolution 1×1 or smaller.

        Args:
            features: List of feature map tensors from the backbone,
                each of shape (B, C, H, W).

        Returns:
            Tuple of:
                - cls_logits: (B, total_anchors, num_classes)
                - bbox_preds: (B, total_anchors, 4)
        """
        cls_logits_list = []
        bbox_preds_list = []

        for feat, cls_head, bbox_head in zip(features, self.cls_heads, self.bbox_heads):
            # Skip degenerate feature maps (1×1 or smaller)
            if feat.shape[2] <= 1 or feat.shape[3] <= 1:
                continue

            batch_size = feat.shape[0]

            # Classification: (B, num_anchors * num_classes, H, W) -> (B, H*W*num_anchors, num_classes)
            cls_out = cls_head(feat)
            cls_out = cls_out.permute(0, 2, 3, 1).contiguous()
            cls_out = cls_out.view(batch_size, -1, self.num_classes)
            cls_logits_list.append(cls_out)

            # Box regression: (B, num_anchors * 4, H, W) -> (B, H*W*num_anchors, 4)
            bbox_out = bbox_head(feat)
            bbox_out = bbox_out.permute(0, 2, 3, 1).contiguous()
            bbox_out = bbox_out.view(batch_size, -1, 4)
            bbox_preds_list.append(bbox_out)

        # Concatenate across all levels
        cls_logits = torch.cat(cls_logits_list, dim=1)
        bbox_preds = torch.cat(bbox_preds_list, dim=1)

        return cls_logits, bbox_preds


# ---------------------------------------------------------------------------
# MobileNetV4SSD
# ---------------------------------------------------------------------------


class MobileNetV4SSD(nn.Module):
    """Complete MobileNetV4 SSD detection model.

    Composes the MobileNetV4 backbone, SSD detection head, and anchor generator
    into a single ``nn.Module`` for clean state_dict management and device
    handling.

    The anchor generator uses torchvision's ``DefaultBoxGenerator`` configured
    with aspect ratios [1.0, 2.0, 0.5] at each of the 6 feature map levels,
    with anchor sizes scaled linearly from 20px to 500px across levels
    (relative to a 640×640 input image).

    Args:
        num_classes: Number of object classes (excluding background).
        backbone_variant: MobileNetV4 variant name or alias.
        pretrained_backbone: Whether to load ImageNet-pretrained weights.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_variant: str = "small",
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Backbone: multi-scale feature extractor
        self.backbone = MobileNetV4Backbone(
            backbone_variant=backbone_variant,
            pretrained=pretrained_backbone,
        )

        # Anchor generator configuration:
        # 6 feature map levels with aspect ratios [1.0, 2.0, 0.5]
        # Sizes linearly spaced from 20px to 500px across levels
        # DefaultBoxGenerator expects:
        #   aspect_ratios: list of list of floats (per level)
        #   min_ratio / max_ratio or sizes
        aspect_ratios = [[1.0, 2.0, 0.5]] * 6

        # Linearly space anchor sizes from 20px to 500px across 6 levels.
        # DefaultBoxGenerator uses min_ratio and max_ratio as percentages of
        # the image size (640). We specify them relative to 640.
        # 20/640 = 0.03125, 500/640 = 0.78125
        # We pass these as min_ratio and max_ratio percentages (×100 for the API).
        min_ratio = 20.0 / 640.0  # ~0.03125
        max_ratio = 500.0 / 640.0  # ~0.78125

        self.anchor_generator = DefaultBoxGenerator(
            aspect_ratios=aspect_ratios,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
        )

        # Determine number of anchors per location at each level
        # DefaultBoxGenerator produces 2*len(aspect_ratios) anchors per level
        # (each aspect_ratio gives 1 anchor, plus 1 extra scale anchor per ratio pair)
        # For aspect_ratios=[1.0, 2.0, 0.5] -> 2*3 = 6 anchors per location
        # Actually, DefaultBoxGenerator: for k aspect ratios, produces 2 + 2*(k-1) = 2*k
        # but the exact count depends on internal logic. We query it.
        num_anchors_per_location = self.anchor_generator.num_anchors_per_location()

        # SSD Head: per-level classification + box regression
        self.head = SSDHead(
            in_channels=self.backbone.out_channels,
            num_anchors_per_location=num_anchors_per_location,
            num_classes=num_classes,
        )

        logger.info(
            "MobileNetV4SSD initialized: num_classes=%d, backbone=%s, "
            "anchors_per_location=%s",
            num_classes, backbone_variant, num_anchors_per_location,
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through backbone and detection head.

        Args:
            images: Input tensor of shape (B, 3, H, W).

        Returns:
            Tuple of:
                - cls_logits: (B, total_anchors, num_classes) raw classification logits
                - bbox_preds: (B, total_anchors, 4) raw box regression predictions
        """
        features = self.backbone(images)
        cls_logits, bbox_preds = self.head(features)
        return cls_logits, bbox_preds


# ---------------------------------------------------------------------------
# MobileNetV4Detector (BaseDetector adapter)
# ---------------------------------------------------------------------------

CONFIG_SCHEMA = {
    "num_classes": {"type": "int", "required": True},
    "input_size": {"type": "int", "required": True},
    "backbone_variant": {"type": "str", "required": False},
    "pretrained_backbone": {"type": "bool", "required": False},
    "confidence_threshold": {"type": "float", "required": False},
    "iou_threshold": {"type": "float", "required": False},
}


@ModelRegistry.register("mobilenetv4_ssd")
class MobileNetV4Detector(BaseDetector):
    """MobileNetV4 SSD object detection model implementing BaseDetector.

    Uses MobileNetV4 backbone (from timm) with a custom SSD detection head
    for multi-scale object detection. Supports configurable backbone variants,
    confidence thresholds, and NMS IoU thresholds.

    Args:
        config: Configuration dict containing:
            - num_classes (int, required): Number of target classes [1, 1000]
            - input_size (int, required): Input spatial dimension (640)
            - backbone_variant (str, optional): Backbone variant name or alias.
                Default: "mobilenetv4_conv_small.e2400_r224_in1k"
            - pretrained_backbone (bool, optional): Load ImageNet weights. Default: True
            - confidence_threshold (float, optional): Min score threshold [0.0, 1.0].
                Default: 0.25
            - iou_threshold (float, optional): NMS IoU threshold [0.0, 1.0].
                Default: 0.5
    """

    CONFIG_SCHEMA = CONFIG_SCHEMA

    VALID_BACKBONE_VARIANTS = VALID_BACKBONE_VARIANTS
    VARIANT_ALIASES = VARIANT_ALIASES

    def __init__(self, config: dict) -> None:
        """Initialize MobileNetV4Detector with configuration validation.

        Args:
            config: Configuration dict (see class docstring for parameters).

        Raises:
            ConfigurationError: If required parameters are missing or values
                are out of valid ranges.
        """
        self._validate_config(config)

        self.config = config
        self.num_classes: int = config["num_classes"]
        self.input_size: int = config["input_size"]

        # Apply defaults for optional parameters
        raw_variant = config.get(
            "backbone_variant", "mobilenetv4_conv_small.e2400_r224_in1k"
        )
        # Resolve alias to full timm model name
        self.backbone_variant: str = VARIANT_ALIASES.get(raw_variant, raw_variant)
        self.pretrained_backbone: bool = config.get("pretrained_backbone", True)
        self.confidence_threshold: float = config.get("confidence_threshold", 0.25)
        self.iou_threshold: float = config.get("iou_threshold", 0.5)

        # Build the underlying nn.Module
        self._model = MobileNetV4SSD(
            num_classes=self.num_classes,
            backbone_variant=self.backbone_variant,
            pretrained_backbone=self.pretrained_backbone,
        )

        self._device = torch.device("cpu")

        logger.info(
            "MobileNetV4Detector initialized: num_classes=%d, input_size=%d, "
            "backbone=%s, pretrained=%s, conf_thresh=%.2f, iou_thresh=%.2f",
            self.num_classes,
            self.input_size,
            self.backbone_variant,
            self.pretrained_backbone,
            self.confidence_threshold,
            self.iou_threshold,
        )

    def _validate_config(self, config: dict) -> None:
        """Validate configuration parameters, collecting all violations.

        Args:
            config: Configuration dict to validate.

        Raises:
            ConfigurationError: If any validation rules are violated.
        """
        violations: List[str] = []

        # Check required parameters
        if "num_classes" not in config:
            violations.append("Missing required parameter: num_classes")
        if "input_size" not in config:
            violations.append("Missing required parameter: input_size")

        # Validate backbone_variant (only if present)
        if "backbone_variant" in config:
            variant = config["backbone_variant"]
            resolved = VARIANT_ALIASES.get(variant, variant)
            if resolved not in VALID_BACKBONE_VARIANTS:
                valid_options = list(VARIANT_ALIASES.keys()) + list(
                    VALID_BACKBONE_VARIANTS
                )
                violations.append(
                    f"Invalid backbone_variant '{variant}'. "
                    f"Must be one of: {valid_options}"
                )

        # Validate num_classes range (only if present)
        if "num_classes" in config:
            num_classes = config["num_classes"]
            if (
                not isinstance(num_classes, int)
                or isinstance(num_classes, bool)
                or num_classes < 1
                or num_classes > 1000
            ):
                violations.append(
                    f"Invalid num_classes '{num_classes}'. "
                    f"Must be an integer in range [1, 1000]"
                )

        # Validate confidence_threshold range (only if present)
        if "confidence_threshold" in config:
            conf = config["confidence_threshold"]
            if not isinstance(conf, (int, float)) or conf < 0.0 or conf > 1.0:
                violations.append(
                    f"Invalid confidence_threshold '{conf}'. "
                    f"Must be a float in range [0.0, 1.0]"
                )

        # Validate iou_threshold range (only if present)
        if "iou_threshold" in config:
            iou = config["iou_threshold"]
            if not isinstance(iou, (int, float)) or iou < 0.0 or iou > 1.0:
                violations.append(
                    f"Invalid iou_threshold '{iou}'. "
                    f"Must be a float in range [0.0, 1.0]"
                )

        if violations:
            raise ConfigurationError(violations)

    # ------------------------------------------------------------------
    # BaseDetector interface methods
    # ------------------------------------------------------------------

    def get_config_schema(self) -> dict:
        """Return the configuration schema for MobileNetV4 SSD.

        Returns:
            Dict mapping parameter names to dicts with 'type' and 'required' fields.
        """
        return self.CONFIG_SCHEMA

    def get_parameters(self) -> List[torch.nn.Parameter]:
        """Return all trainable parameters from the model.

        Returns parameters with ``requires_grad=True`` from both the backbone
        and the detection head submodules.

        Returns:
            List of trainable parameters suitable for optimizer construction.
        """
        return [p for p in self._model.parameters() if p.requires_grad]

    def set_train_mode(self) -> None:
        """Set the model to training mode.

        Toggles both the backbone and detection head (and all submodules)
        to training mode so that batch norm layers use batch statistics and
        dropout is active.
        """
        self._model.train()

    def set_eval_mode(self) -> None:
        """Set the model to evaluation mode.

        Toggles both the backbone and detection head (and all submodules)
        to evaluation mode so that batch norm layers use running statistics
        and dropout is deactivated.
        """
        self._model.eval()

    def to_device(self, device) -> None:
        """Move all model parameters and buffers to the target device.

        Moves all parameters, buffers (including batch norm running statistics
        and anchor generator tensors) to the specified device.

        Args:
            device: Target device (e.g., ``"cpu"``, ``"cuda"``, ``"cuda:0"``,
                or a ``torch.device`` instance).

        Raises:
            RuntimeError: If the specified device is not available on the host
                machine (e.g., requesting CUDA when no GPU is present).
        """
        # Normalize device to torch.device
        if not isinstance(device, torch.device):
            device = torch.device(device)

        # Validate device availability
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"Device '{device}' is not available. "
                    f"No CUDA-capable GPU detected on this machine."
                )
            # Check specific GPU index if provided
            if device.index is not None:
                gpu_count = torch.cuda.device_count()
                if device.index >= gpu_count:
                    raise RuntimeError(
                        f"Device '{device}' is not available. "
                        f"Only {gpu_count} CUDA device(s) detected "
                        f"(valid indices: 0-{gpu_count - 1})."
                    )

        self._model.to(device)
        self._device = device

    def forward(self, images: torch.Tensor) -> List[dict]:
        """Run forward pass on a batch of images.

        Args:
            images: Batch of images as a torch.Tensor with shape (B, 3, 640, 640).

        Returns:
            List of dicts per image, each containing:
                - boxes: Tensor of shape (N, 4) in xyxy pixel coords
                - labels: Tensor of shape (N,) with class indices
                - scores: Tensor of shape (N,) with confidence scores

        Raises:
            ValueError: If spatial dimensions are not 640×640.
        """
        # Validate input spatial dimensions
        if images.ndim != 4 or images.shape[2] != 640 or images.shape[3] != 640:
            h, w = images.shape[2] if images.ndim >= 3 else 0, images.shape[3] if images.ndim >= 4 else 0
            raise ValueError(
                f"Expected input spatial dimensions (640, 640), got ({h}, {w})"
            )

        device = images.device
        batch_size = images.shape[0]

        # Get raw predictions from the model
        cls_logits, bbox_preds = self._model(images)
        # cls_logits: (B, total_anchors, num_classes)
        # bbox_preds: (B, total_anchors, 4)

        # Generate anchor boxes using DefaultBoxGenerator
        # It requires an ImageList and feature_maps to determine grid sizes
        feature_maps = self._model.backbone(images)
        image_sizes = [(640, 640)] * batch_size
        image_list = ImageList(images, image_sizes)
        # anchors_per_image: list of B tensors, each (total_anchors, 4) in xyxy format
        anchors_per_image = self._model.anchor_generator(image_list, feature_maps)

        # Apply sigmoid to get confidence scores from logits
        scores_all = torch.sigmoid(cls_logits)

        # Decode box predictions and apply post-processing per image
        # Box decoding uses SSD-style encoding with weights (10, 10, 5, 5):
        #   dx = rel_codes[:, 0] / wx  -> pred_cx = dx * anchor_w + anchor_cx
        #   dy = rel_codes[:, 1] / wy  -> pred_cy = dy * anchor_h + anchor_cy
        #   dw = rel_codes[:, 2] / ww  -> pred_w = exp(dw) * anchor_w
        #   dh = rel_codes[:, 3] / wh  -> pred_h = exp(dh) * anchor_h
        center_weights = torch.tensor([10.0, 10.0, 5.0, 5.0], device=device)

        results: List[dict] = []

        for i in range(batch_size):
            # Anchors for this image: (total_anchors, 4) in xyxy
            anchors = anchors_per_image[i]  # (A, 4) xyxy

            # Convert anchors from xyxy to (cx, cy, w, h) for decoding
            anchor_widths = anchors[:, 2] - anchors[:, 0]
            anchor_heights = anchors[:, 3] - anchors[:, 1]
            anchor_cx = anchors[:, 0] + 0.5 * anchor_widths
            anchor_cy = anchors[:, 1] + 0.5 * anchor_heights

            # Decode box predictions
            preds = bbox_preds[i]  # (A, 4)
            dx = preds[:, 0] / center_weights[0]
            dy = preds[:, 1] / center_weights[1]
            dw = preds[:, 2] / center_weights[2]
            dh = preds[:, 3] / center_weights[3]

            # Clamp dw, dh to prevent overflow in exp
            dw = torch.clamp(dw, max=4.135)  # log(1000/6)
            dh = torch.clamp(dh, max=4.135)

            pred_cx = dx * anchor_widths + anchor_cx
            pred_cy = dy * anchor_heights + anchor_cy
            pred_w = torch.exp(dw) * anchor_widths
            pred_h = torch.exp(dh) * anchor_heights

            # Convert back to xyxy
            pred_boxes = torch.stack([
                pred_cx - 0.5 * pred_w,
                pred_cy - 0.5 * pred_h,
                pred_cx + 0.5 * pred_w,
                pred_cy + 0.5 * pred_h,
            ], dim=-1)  # (A, 4)

            # Clip boxes to image bounds [0, 640]
            pred_boxes = pred_boxes.clamp(min=0.0, max=640.0)

            # Per-class confidence filtering and NMS
            image_scores = scores_all[i]  # (A, num_classes)

            all_boxes = []
            all_scores = []
            all_labels = []

            for class_idx in range(self.num_classes):
                class_scores = image_scores[:, class_idx]

                # Filter below confidence threshold
                keep_mask = class_scores >= self.confidence_threshold
                if not keep_mask.any():
                    continue

                filtered_scores = class_scores[keep_mask]
                filtered_boxes = pred_boxes[keep_mask]

                # Apply NMS for this class
                nms_keep = torchvision.ops.nms(
                    filtered_boxes, filtered_scores, self.iou_threshold
                )

                all_boxes.append(filtered_boxes[nms_keep])
                all_scores.append(filtered_scores[nms_keep])
                all_labels.append(
                    torch.full(
                        (nms_keep.shape[0],),
                        class_idx,
                        dtype=torch.int64,
                        device=device,
                    )
                )

            if all_boxes:
                final_boxes = torch.cat(all_boxes, dim=0)
                final_scores = torch.cat(all_scores, dim=0)
                final_labels = torch.cat(all_labels, dim=0)

                # Cap at 200 detections, keeping highest-scoring
                if final_scores.shape[0] > 200:
                    _, topk_indices = final_scores.topk(200)
                    final_boxes = final_boxes[topk_indices]
                    final_scores = final_scores[topk_indices]
                    final_labels = final_labels[topk_indices]

                results.append({
                    "boxes": final_boxes,
                    "labels": final_labels,
                    "scores": final_scores,
                })
            else:
                # No detections: return empty tensors with correct shapes
                results.append({
                    "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                    "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                    "scores": torch.zeros((0,), dtype=torch.float32, device=device),
                })

        return results

    def train_step(
        self, images: List[torch.Tensor], targets: List[dict]
    ) -> dict:
        """Perform a single training step computing classification and bbox losses.

        Computes focal loss for classification and smooth L1 loss for bounding
        box regression. Ground truth targets are matched to anchor boxes using
        IoU-based assignment.

        Args:
            images: List of image tensors, each of shape (C, H, W).
            targets: List of target dicts, each with:
                - boxes: Tensor (N, 4) in xyxy pixel format
                - labels: Tensor (N,) with class indices

        Returns:
            Dict with:
                - loss_tensor: Scalar tensor with grad_fn for backpropagation
                - classification_loss: Float value of the focal loss
                - bbox_regression_loss: Float value of the smooth L1 loss
        """
        # Handle empty image list
        if not images or len(images) == 0:
            return {
                "loss_tensor": torch.tensor(0.0),
                "classification_loss": 0.0,
                "bbox_regression_loss": 0.0,
            }

        # Ensure model is in training mode
        self._model.train()

        # Stack images into a batch tensor and move to device
        batch = torch.stack(images).to(self._device)
        targets = [{k: v.to(self._device) for k, v in t.items()} for t in targets]

        # Forward pass to get raw predictions
        cls_logits, bbox_preds = self._model(batch)
        # cls_logits: (B, total_anchors, num_classes)
        # bbox_preds: (B, total_anchors, 4)

        # Generate anchor boxes using DefaultBoxGenerator
        features = self._model.backbone(batch)
        image_sizes = [(batch.shape[2], batch.shape[3])] * batch.shape[0]
        image_list = ImageList(batch, image_sizes)
        anchors_per_image = self._model.anchor_generator(image_list, features)
        # anchors_per_image: list of B tensors, each (total_anchors, 4) in xyxy format

        batch_size = batch.shape[0]
        total_cls_loss = torch.tensor(0.0, device=self._device, dtype=torch.float32)
        total_bbox_loss = torch.tensor(0.0, device=self._device, dtype=torch.float32)
        total_positives = 0

        for i in range(batch_size):
            gt_boxes = targets[i]["boxes"]  # (num_gt, 4) xyxy
            gt_labels = targets[i]["labels"]  # (num_gt,)
            anchors = anchors_per_image[i]  # (total_anchors, 4) xyxy

            num_anchors = anchors.shape[0]
            num_gt = gt_boxes.shape[0]

            if num_gt == 0:
                # No ground truth: all anchors are background
                cls_targets = torch.zeros(
                    num_anchors, self.num_classes,
                    device=self._device, dtype=torch.float32,
                )
                cls_loss = sigmoid_focal_loss(
                    cls_logits[i], cls_targets,
                    alpha=0.25, gamma=2.0, reduction="sum",
                )
                total_cls_loss = total_cls_loss + cls_loss
                continue

            # Compute IoU between anchors and GT boxes
            iou_matrix = box_iou(anchors, gt_boxes)  # (num_anchors, num_gt)

            # Assign each anchor to a GT box using IoU matching
            best_gt_iou, best_gt_idx = iou_matrix.max(dim=1)  # (num_anchors,)

            # Positive: IoU >= 0.5, Negative: IoU < 0.4
            positive_mask = best_gt_iou >= 0.5
            negative_mask = best_gt_iou < 0.4

            # Ensure each GT box has at least one positive anchor
            best_anchor_per_gt = iou_matrix.argmax(dim=0)  # (num_gt,)
            positive_mask[best_anchor_per_gt] = True
            negative_mask[best_anchor_per_gt] = False

            num_pos = positive_mask.sum().item()
            total_positives += num_pos

            # Build classification targets (one-hot encoded)
            cls_targets = torch.zeros(
                num_anchors, self.num_classes,
                device=self._device, dtype=torch.float32,
            )
            pos_labels = gt_labels[best_gt_idx[positive_mask]]
            cls_targets[positive_mask] = F.one_hot(
                pos_labels.long(), num_classes=self.num_classes
            ).float()

            # Classification focal loss on positive + negative anchors
            valid_mask = positive_mask | negative_mask
            cls_loss = sigmoid_focal_loss(
                cls_logits[i][valid_mask],
                cls_targets[valid_mask],
                alpha=0.25, gamma=2.0, reduction="sum",
            )
            total_cls_loss = total_cls_loss + cls_loss

            # Bbox regression loss (only on positive anchors)
            if num_pos > 0:
                matched_gt_boxes = gt_boxes[best_gt_idx[positive_mask]]  # (num_pos, 4)
                pred_boxes = bbox_preds[i][positive_mask]  # (num_pos, 4)
                pos_anchors = anchors[positive_mask]  # (num_pos, 4)

                # Encode GT relative to anchors (offset encoding)
                anchor_w = pos_anchors[:, 2] - pos_anchors[:, 0]
                anchor_h = pos_anchors[:, 3] - pos_anchors[:, 1]
                anchor_cx = pos_anchors[:, 0] + 0.5 * anchor_w
                anchor_cy = pos_anchors[:, 1] + 0.5 * anchor_h

                gt_cx = (matched_gt_boxes[:, 0] + matched_gt_boxes[:, 2]) / 2
                gt_cy = (matched_gt_boxes[:, 1] + matched_gt_boxes[:, 3]) / 2
                gt_w = matched_gt_boxes[:, 2] - matched_gt_boxes[:, 0]
                gt_h = matched_gt_boxes[:, 3] - matched_gt_boxes[:, 1]

                target_dx = (gt_cx - anchor_cx) / anchor_w
                target_dy = (gt_cy - anchor_cy) / anchor_h
                target_dw = torch.log(gt_w / anchor_w.clamp(min=1e-6))
                target_dh = torch.log(gt_h / anchor_h.clamp(min=1e-6))
                encoded_targets = torch.stack(
                    [target_dx, target_dy, target_dw, target_dh], dim=1
                )

                bbox_loss = F.smooth_l1_loss(
                    pred_boxes, encoded_targets, reduction="sum", beta=1.0
                )
                total_bbox_loss = total_bbox_loss + bbox_loss

        # Normalize by total number of positive anchors
        num_positives = max(total_positives, 1)
        cls_loss_normalized = total_cls_loss / num_positives
        bbox_loss_normalized = total_bbox_loss / num_positives
        total_loss = cls_loss_normalized + bbox_loss_normalized

        return {
            "loss_tensor": total_loss,
            "classification_loss": cls_loss_normalized.item(),
            "bbox_regression_loss": bbox_loss_normalized.item(),
        }

    def load_checkpoint(self, path) -> None:
        """Load model weights from a checkpoint file.

        Args:
            path: Path to the checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint path does not exist.
            RuntimeError: If the file cannot be deserialized as a valid checkpoint.
        """
        from pathlib import Path

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        try:
            checkpoint = torch.load(path, map_location=self._device, weights_only=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint from '{path}': {e}"
            ) from e

        try:
            self._model.load_state_dict(checkpoint["model_state_dict"])
        except (KeyError, RuntimeError) as e:
            raise RuntimeError(
                f"Failed to load checkpoint from '{path}': {e}"
            ) from e

    def save_checkpoint(self, path) -> None:
        """Save model weights to a checkpoint file.

        Saves the model state dictionary along with configuration metadata.
        Creates parent directories as needed.

        Args:
            path: Path where the checkpoint will be saved.
        """
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self._model.state_dict(),
            "config": self.config,
            "num_classes": self.num_classes,
            "input_size": self.input_size,
        }
        torch.save(checkpoint, path)
