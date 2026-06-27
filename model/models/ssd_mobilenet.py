"""SSDLite + MobileNetV3-Large detector wrapper (torchvision-based).

Implements a single ``BaseDetector`` adapter that builds an SSDLite detector
with a MobileNetV3-Large backbone via torchvision primitives. Supports both
the standard 320x320 input (matching ``ssdlite320_mobilenet_v3_large``) and a
640x640 variant where the anchor generator and detection head are rebuilt for
the larger spatial resolution.

Design contract (matches ``model/models/registry.BaseDetector``):

* ``forward(images)`` returns ``boxes`` in **xyxy pixel** coordinates of the
  input tensor (not normalized). The evaluation engine normalizes by image
  size when needed (``evaluation/engine.py``).
* ``train_step(images, targets)`` receives ``targets["labels"]`` as
  **0-indexed** class ids (the dataset adapter convention used by YOLO26 /
  RT-DETR). This wrapper internally shifts them by +1 because torchvision
  reserves class 0 for the background class.
* ``get_parameters()`` returns a flat parameter list by default, or a list of
  param-group dicts (backbone / head) when ``backbone_lr`` and ``head_lr`` are
  set in the config, enabling discriminative learning rates.
* ``on_epoch_start(epoch)`` (extension to BaseDetector, default no-op) is
  consumed by the trainer to drive the ``freeze_backbone_epochs`` schedule.

The class is registered as ``ssd_mobilenetv3``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torchvision
from torchvision.models import mobilenet_v3_large
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.ssdlite import (
    SSDLiteHead,
    _mobilenet_extractor,
)
from torchvision.models.detection import _utils as det_utils

from model.exceptions import ConfigurationError
from model.models.registry import BaseDetector, ModelRegistry

logger = logging.getLogger(__name__)

VALID_INPUT_SIZES = (320, 640)

# Background-aware class count helper. torchvision reserves index 0 for
# background, so the head is sized for ``num_classes + 1`` outputs and our
# 0-indexed dataset labels are shifted by +1 in ``train_step``.
_BACKGROUND_OFFSET = 1


CONFIG_SCHEMA = {
    "input_size": {"type": "int", "required": True},
    "num_classes": {"type": "int", "required": True},
    "pretrained_backbone": {"type": "bool", "required": False},
    "trainable_backbone_layers": {"type": "int", "required": False},
    "confidence_threshold": {"type": "float", "required": False},
    "iou_threshold": {"type": "float", "required": False},
    "nms_threshold": {"type": "float", "required": False},
    "detections_per_image": {"type": "int", "required": False},
    "score_threshold": {"type": "float", "required": False},
    "topk_candidates": {"type": "int", "required": False},
    "backbone_lr": {"type": "float", "required": False},
    "head_lr": {"type": "float", "required": False},
    "freeze_backbone_epochs": {"type": "int", "required": False},
    "load_coco_weights": {"type": "bool", "required": False},
}


@ModelRegistry.register("ssd_mobilenetv3")
class SSDMobileNetV3(BaseDetector):
    """SSDLite + MobileNetV3-Large detector built from torchvision primitives.

    Args:
        config: Dict containing:
            input_size (int, required):
                Spatial input size; one of ``{320, 640}``.
            num_classes (int, required):
                Number of damage classes excluding background.
            pretrained_backbone (bool, optional):
                Load ImageNet weights for MobileNetV3-Large. Default ``True``.
            trainable_backbone_layers (int, optional):
                Number of trainable backbone layers (0..6). Default ``6`` so the
                full backbone is trainable; combine with ``freeze_backbone_epochs``
                to implement a two-stage schedule that toggles ``requires_grad``
                via :meth:`on_epoch_start`.
            confidence_threshold (float, optional):
                Operating-point score threshold used by **downstream consumers**
                (inference CLI, P/R/F1 reporting). It is **not** applied inside
                :meth:`forward` because doing so would discard predictions in
                ``[score_threshold, confidence_threshold)`` that the mAP integral
                and the F1 threshold sweep need. Default ``0.25``.
            iou_threshold / nms_threshold (float, optional):
                IoU threshold for NMS inside the SSD postprocessor. Both names
                are accepted (``iou_threshold`` is the cross-wrapper convention,
                ``nms_threshold`` matches torchvision). Default ``0.5``.
            detections_per_image (int, optional):
                Cap on detections returned per image. Default ``300``.
            score_threshold (float, optional):
                Low score-cut applied inside torchvision's SSD postprocessor
                (used for mAP@0.001 evaluation). Default ``0.001``.
            topk_candidates (int, optional):
                Top-K filter before NMS. Default ``300``.
            backbone_lr / head_lr (float, optional):
                When both are set, :meth:`get_parameters` returns two parameter
                groups with the respective learning rates (discriminative LR).
            freeze_backbone_epochs (int, optional):
                If > 0, the backbone is frozen for the first N epochs. The
                trainer calls :meth:`on_epoch_start` once per epoch to drive
                the schedule. Default ``0`` (no freeze).
    """

    CONFIG_SCHEMA = CONFIG_SCHEMA

    def __init__(self, config: dict):
        self._validate_config(config)
        self.config = config

        self.input_size: int = int(config["input_size"])
        self.num_classes: int = int(config["num_classes"])
        self.pretrained_backbone: bool = bool(
            config.get("pretrained_backbone", True)
        )
        self.trainable_backbone_layers: int = int(
            config.get("trainable_backbone_layers", 6)
        )

        # Inference / postprocess parameters
        self.confidence_threshold: float = float(
            config.get("confidence_threshold", 0.25)
        )
        # Accept both ``iou_threshold`` and ``nms_threshold`` for cross-wrapper
        # consistency. ``nms_threshold`` wins when both are present because it
        # matches torchvision's own parameter name.
        nms = config.get("nms_threshold", config.get("iou_threshold", 0.5))
        self.nms_threshold: float = float(nms)
        self.detections_per_image: int = int(
            config.get("detections_per_image", 300)
        )
        # Use a low default score threshold to enable proper mAP integration.
        # The trainer / evaluator can still apply a stricter
        # ``confidence_threshold`` filter on top.
        self.score_threshold: float = float(
            config.get("score_threshold", 0.001)
        )
        self.topk_candidates: int = int(config.get("topk_candidates", 300))

        # Discriminative LR (optional)
        self.backbone_lr: Optional[float] = (
            float(config["backbone_lr"]) if "backbone_lr" in config else None
        )
        self.head_lr: Optional[float] = (
            float(config["head_lr"]) if "head_lr" in config else None
        )

        # Freeze schedule (optional)
        self.freeze_backbone_epochs: int = int(
            config.get("freeze_backbone_epochs", 0)
        )
        # Whether to initialise the detector from torchvision's COCO-pretrained
        # SSDLite320 weights via selective state-dict transfer (backbone +
        # bbox-regression head copy across; classification head is
        # shape-incompatible and left at random init). Only meaningful at
        # ``input_size == 320`` because torchvision ships official pretrained
        # detector weights only for that variant.
        self.load_coco_weights: bool = bool(
            config.get("load_coco_weights", False)
        )
        # Internal state: are backbone params currently frozen?
        self._backbone_frozen: bool = False
        # Flag the trainer reads to know it must rebuild the optimizer because
        # ``requires_grad`` toggled this epoch.
        self._needs_optimizer_rebuild: bool = False

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model: SSD = self._build_model()
        self._model.to(self._device)

        # If freezing is requested, freeze immediately so the first
        # ``get_parameters()`` call (used to build the optimizer) doesn't
        # include backbone params.
        if self.freeze_backbone_epochs > 0:
            self._set_backbone_trainable(False)

        logger.info(
            "Initialized SSDMobileNetV3 (input_size=%d, num_classes=%d, "
            "trainable_backbone_layers=%d, pretrained_backbone=%s, "
            "load_coco_weights=%s, freeze_backbone_epochs=%d, device=%s)",
            self.input_size,
            self.num_classes,
            self.trainable_backbone_layers,
            self.pretrained_backbone,
            self.load_coco_weights,
            self.freeze_backbone_epochs,
            self._device,
        )

    # ------------------------------------------------------------------
    # Configuration validation
    # ------------------------------------------------------------------

    def _validate_config(self, config: dict) -> None:
        """Validate config; collect all violations into one ConfigurationError."""
        errors: List[str] = []

        if "input_size" not in config:
            errors.append("Missing required parameter: input_size")
        else:
            size = config["input_size"]
            if size not in VALID_INPUT_SIZES:
                errors.append(
                    f"Invalid input_size {size!r}. Must be one of "
                    f"{list(VALID_INPUT_SIZES)}."
                )

        if "num_classes" not in config:
            errors.append("Missing required parameter: num_classes")
        else:
            nc = config["num_classes"]
            if not isinstance(nc, int) or nc < 1 or nc > 1000:
                errors.append(
                    f"num_classes must be an int in [1, 1000], got {nc!r}."
                )

        tbl = config.get("trainable_backbone_layers", 6)
        if not isinstance(tbl, int) or tbl < 0 or tbl > 6:
            errors.append(
                f"trainable_backbone_layers must be an int in [0, 6], got "
                f"{tbl!r}."
            )

        for name in ("confidence_threshold", "score_threshold"):
            if name in config:
                v = config[name]
                if not isinstance(v, (int, float)) or not (0.0 <= v <= 1.0):
                    errors.append(f"{name} must be in [0, 1], got {v!r}.")

        for name in ("iou_threshold", "nms_threshold"):
            if name in config:
                v = config[name]
                if not isinstance(v, (int, float)) or not (0.0 < v < 1.0):
                    errors.append(f"{name} must be in (0, 1), got {v!r}.")

        if "freeze_backbone_epochs" in config:
            fbe = config["freeze_backbone_epochs"]
            if not isinstance(fbe, int) or fbe < 0:
                errors.append(
                    f"freeze_backbone_epochs must be a non-negative int, got "
                    f"{fbe!r}."
                )

        if ("backbone_lr" in config) ^ ("head_lr" in config):
            errors.append(
                "backbone_lr and head_lr must be set together (or not at all)."
            )

        if errors:
            raise ConfigurationError(errors)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self) -> SSD:
        """Build the underlying torchvision SSD module.

        For ``input_size == 320`` we rely on torchvision's ``ssdlite320_*``
        factory so the architecture matches the published SSDLite-320 setup
        (defaults: ``score_thresh=0.001``, ``nms_thresh=0.55``,
        ``image_mean=image_std=[0.5,0.5,0.5]``).

        For ``input_size == 640`` we assemble the SSD manually so the anchor
        generator scales to the larger resolution and the detection head is
        sized correctly. Default anchor ratios mirror torchvision's SSDLite
        configuration (``[[2, 3]] * 6``).
        """
        num_classes_with_bg = self.num_classes + _BACKGROUND_OFFSET
        weights_backbone = "DEFAULT" if self.pretrained_backbone else None

        if self.input_size == 320:
            if self.load_coco_weights:
                # Build target with random head sized for our num_classes; the
                # COCO backbone + bbox-regression head weights are loaded
                # selectively below. The classification head is
                # shape-incompatible (COCO has 91 classes, we have
                # num_classes + background) and is left at random init.
                model = ssdlite320_mobilenet_v3_large(
                    weights=None,
                    weights_backbone=None,
                    num_classes=num_classes_with_bg,
                    trainable_backbone_layers=self.trainable_backbone_layers,
                )
                # Reference COCO model with the canonical 91-class detection
                # head. torchvision ships official pretrained weights only at
                # 320x320; this is the only variant for which a true
                # detector-level transfer is possible.
                coco_model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")
                self._partial_load_state_dict(model, coco_model.state_dict())
                del coco_model
            else:
                model = ssdlite320_mobilenet_v3_large(
                    weights=None,
                    weights_backbone=weights_backbone,
                    num_classes=num_classes_with_bg,
                    trainable_backbone_layers=self.trainable_backbone_layers,
                )
        else:
            if self.load_coco_weights:
                logger.warning(
                    "load_coco_weights=True is only supported at input_size=320 "
                    "(torchvision ships SSDLite COCO weights only for the 320 "
                    "variant). Falling back to ImageNet backbone-only "
                    "initialisation for input_size=%d.",
                    self.input_size,
                )
            model = self._build_ssdlite_640(num_classes_with_bg, weights_backbone)

        # Override postprocessing parameters that the factory locks down. The
        # torchvision SSD constructor stores these as attributes consumed in
        # ``postprocess_detections`` (see torchvision's ssd.py).
        model.score_thresh = self.score_threshold
        model.nms_thresh = self.nms_threshold
        model.detections_per_img = self.detections_per_image
        model.topk_candidates = self.topk_candidates

        return model

    @staticmethod
    def _partial_load_state_dict(target_model, source_state_dict) -> None:
        """Copy parameters from a source state-dict into a target model where
        shapes match.

        Used to transfer torchvision's COCO-pretrained SSDLite320 weights into
        our target detector when ``num_classes`` differs (COCO has 91 classes,
        RDD has 5 + 1 background). The backbone and bbox-regression head
        transfer cleanly (identical shapes because both use the same
        ``num_anchors_per_location`` pattern). The classification head is
        shape-incompatible and is therefore skipped, leaving the target's
        random initialisation in place. BatchNorm running statistics in the
        backbone are also carried over.

        Logs per-call statistics (transferred / shape-mismatched / missing)
        and emits a WARNING if very few parameters transferred, which would
        indicate an unexpected torchvision-version key-name change.
        """
        target_state = target_model.state_dict()
        transferred = 0
        skipped_shape = 0
        missing = 0
        sample_mismatches: list = []
        for key, value in source_state_dict.items():
            if key not in target_state:
                missing += 1
                continue
            if target_state[key].shape != value.shape:
                skipped_shape += 1
                if len(sample_mismatches) < 5:
                    sample_mismatches.append(
                        f"{key} (target={tuple(target_state[key].shape)} "
                        f"source={tuple(value.shape)})"
                    )
                continue
            target_state[key] = value
            transferred += 1
        target_model.load_state_dict(target_state, strict=False)
        logger.info(
            "COCO transfer: %d params transferred, %d skipped (shape "
            "mismatch), %d missing in target",
            transferred,
            skipped_shape,
            missing,
        )
        if sample_mismatches:
            logger.info(
                "Sample shape-mismatch keys (expected: classification head):"
                "\n  %s",
                "\n  ".join(sample_mismatches),
            )
        if transferred < 100:
            logger.warning(
                "Only %d parameters transferred from COCO state dict -- "
                "verify torchvision version compatibility (torchvision=%s).",
                transferred,
                torchvision.__version__,
            )

    def _build_ssdlite_640(
        self,
        num_classes_with_bg: int,
        weights_backbone,
    ) -> SSD:
        """Construct an SSDLite detector with MobileNetV3-Large at 640x640.

        Mirrors the assembly inside ``torchvision.models.detection.ssdlite``
        but with ``size=(640, 640)``. Anchor sizes are derived from the size
        at runtime by ``DefaultBoxGenerator`` (using ratios).
        """
        # Norm-layer: SSDLite uses BatchNorm2d with eps=0.001, momentum=0.03
        # (mirroring the official MobileNet/SSDLite recipe).
        def norm_layer(channels):
            return nn.BatchNorm2d(channels, eps=0.001, momentum=0.03)

        # Reduced-tail variant matches the standard SSDLite-320 build.
        backbone = mobilenet_v3_large(
            weights=weights_backbone,
            norm_layer=norm_layer,
            reduced_tail=False,
        )
        backbone = _mobilenet_extractor(
            backbone,
            trainable_layers=self.trainable_backbone_layers,
            norm_layer=norm_layer,
        )

        size = (self.input_size, self.input_size)
        anchor_generator = DefaultBoxGenerator(
            aspect_ratios=[[2, 3] for _ in range(6)],
            min_ratio=0.2,
            max_ratio=0.95,
        )

        out_channels = det_utils.retrieve_out_channels(backbone, size)
        num_anchors = anchor_generator.num_anchors_per_location()
        if len(out_channels) != len(num_anchors):
            raise RuntimeError(
                f"SSDLite-{self.input_size}: backbone produced {len(out_channels)} "
                f"feature maps but anchor generator expects {len(num_anchors)}. "
                f"Check torchvision version compatibility."
            )

        head = SSDLiteHead(
            in_channels=out_channels,
            num_anchors=num_anchors,
            num_classes=num_classes_with_bg,
            norm_layer=norm_layer,
        )

        # ImageNet-style normalization is more appropriate when the backbone
        # is ImageNet-pretrained; the default SSDLite [0.5, 0.5, 0.5] only
        # makes sense after COCO fine-tuning (which we don't have at 640).
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]

        model = SSD(
            backbone=backbone,
            anchor_generator=anchor_generator,
            size=size,
            num_classes=num_classes_with_bg,
            head=head,
            image_mean=image_mean,
            image_std=image_std,
            score_thresh=self.score_threshold,
            nms_thresh=self.nms_threshold,
            detections_per_img=self.detections_per_image,
            topk_candidates=self.topk_candidates,
        )
        return model

    # ------------------------------------------------------------------
    # Inference / training contract
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor) -> List[dict]:
        """Run inference and return per-image dicts of pixel-xyxy boxes.

        Accepts either a ``(B, C, H, W)`` tensor or a list of ``(C, H, W)``
        tensors so callers can pass either an evaluation batch or the
        per-image lists that the trainer uses elsewhere.

        Returned detections are filtered only by the SSD postprocessor's
        ``score_threshold`` (low default 0.001 to support proper mAP
        integration). ``confidence_threshold`` is **not** applied here: it
        is the operating-point used by downstream P/R/F1 reporting and by
        inference CLIs that post-filter detections; applying it inside
        ``forward()`` would discard predictions in the
        ``[score_threshold, confidence_threshold)`` band that the mAP
        integral and the F1-threshold-sweep need.

        Args:
            images: Image tensor(s) with values already in ``[0, 1]``.

        Returns:
            List of dicts ``{"boxes", "labels", "scores"}`` per image. Boxes
            are in **xyxy pixel** coordinates of the input tensor. Labels are
            shifted back to 0-indexed class ids (matching the dataset adapter).
        """
        # Ensure eval mode for inference (torchvision SSD returns losses in
        # train mode and detections in eval mode).
        self._model.eval()

        if isinstance(images, torch.Tensor):
            if images.dim() == 3:
                image_list = [images]
            elif images.dim() == 4:
                image_list = [img for img in images]
            else:
                raise ValueError(
                    f"forward expected a (C,H,W) or (B,C,H,W) tensor, got "
                    f"shape {tuple(images.shape)}"
                )
        else:
            image_list = list(images)

        image_list = [img.to(self._device) for img in image_list]

        with torch.no_grad():
            outputs = self._model(image_list)

        results: List[dict] = []
        for output in outputs:
            boxes = output["boxes"]
            labels = output["labels"]
            scores = output["scores"]

            # Shift labels back to 0-indexed (torchvision returns 1..C+1; we
            # exported as 0..C from the dataset). Background (0) should never
            # appear in postprocessed outputs, but guard against it.
            if labels.numel() > 0:
                labels = labels - _BACKGROUND_OFFSET
                valid = labels >= 0
                if not bool(valid.all()):
                    boxes = boxes[valid]
                    labels = labels[valid]
                    scores = scores[valid]

            results.append({
                "boxes": boxes,
                "labels": labels,
                "scores": scores,
            })

        return results

    def train_step(
        self,
        images: List[torch.Tensor],
        targets: List[dict],
    ) -> dict:
        """Compute a single training step loss.

        Args:
            images: List of ``(C, H, W)`` tensors in ``[0, 1]``.
            targets: List of dicts with ``boxes`` (xyxy pixels) and ``labels``
                (0-indexed class ids; this method shifts them to 1-indexed
                before calling the torchvision SSD).

        Returns:
            Dict containing component losses plus ``loss_tensor`` for backward.
            Returns a zero-loss dict for empty batches.
        """
        self._model.train()

        if not images:
            zero = torch.tensor(0.0, device=self._device, requires_grad=True)
            return {
                "classification_loss": 0.0,
                "bbox_regression_loss": 0.0,
                "total_loss": 0.0,
                "loss_tensor": zero,
            }

        images = [img.to(self._device) for img in images]

        prepared_targets: List[dict] = []
        for t in targets:
            boxes = t["boxes"].to(self._device)
            labels = t["labels"].to(self._device).long()

            if boxes.shape[0] == 0:
                # torchvision SSD requires shape (0, 4) for empty boxes; the
                # framework already provides this, but guarantee dtype/device.
                prepared_targets.append({
                    "boxes": boxes.float(),
                    "labels": labels,
                })
                continue

            # Filter degenerate boxes (zero width/height) which would cause
            # torchvision to raise ValueError("All bounding boxes should
            # have positive height and width.").
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            valid = (w > 0) & (h > 0)
            if not bool(valid.all()):
                boxes = boxes[valid]
                labels = labels[valid]

            # Shift labels by +1 (torchvision reserves 0 for background).
            labels = labels + _BACKGROUND_OFFSET

            prepared_targets.append({
                "boxes": boxes.float(),
                "labels": labels,
            })

        loss_dict = self._model(images, prepared_targets)

        # torchvision SSD returns {"classification": ..., "bbox_regression": ...}
        cls_loss = loss_dict.get("classification")
        bbox_loss = loss_dict.get("bbox_regression")
        total_loss = sum(loss_dict.values())

        return {
            "classification_loss": float(cls_loss.detach()) if cls_loss is not None else 0.0,
            "bbox_regression_loss": float(bbox_loss.detach()) if bbox_loss is not None else 0.0,
            "total_loss": float(total_loss.detach()),
            "loss_tensor": total_loss,
        }

    # ------------------------------------------------------------------
    # Optimizer & schedule helpers
    # ------------------------------------------------------------------

    def get_parameters(self):
        """Return parameters for the optimizer.

        Returns a flat list of trainable parameters by default. When both
        ``backbone_lr`` and ``head_lr`` are set in the config, returns a list
        of param-group dicts so the optimizer applies discriminative LRs.
        """
        if self.backbone_lr is None or self.head_lr is None:
            return [p for p in self._model.parameters() if p.requires_grad]

        backbone_params = [
            p for p in self._model.backbone.parameters() if p.requires_grad
        ]
        head_params = [
            p for p in self._model.head.parameters() if p.requires_grad
        ]
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": self.backbone_lr})
        if head_params:
            groups.append({"params": head_params, "lr": self.head_lr})
        if not groups:
            # All params frozen; return an empty param group so the optimizer
            # construction doesn't raise.
            return [{"params": [], "lr": self.head_lr}]
        return groups

    def on_epoch_start(self, epoch: int) -> None:
        """Hook called by the trainer at the start of each epoch (0-indexed).

        Implements the ``freeze_backbone_epochs`` schedule: backbone is frozen
        for ``epoch < freeze_backbone_epochs`` and unfrozen afterwards. The
        toggle is idempotent so calling it every epoch is safe.

        When the backbone is unfrozen (or frozen) mid-training, this method
        sets the ``requires_optimizer_rebuild`` flag so the trainer rebuilds
        the optimizer with the updated parameter set; PyTorch optimizers do
        not pick up newly-enabled parameters once constructed.
        """
        if self.freeze_backbone_epochs <= 0:
            return

        should_freeze = epoch < self.freeze_backbone_epochs
        if should_freeze != self._backbone_frozen:
            self._set_backbone_trainable(not should_freeze)
            self._needs_optimizer_rebuild = True
            logger.info(
                "Epoch %d: backbone %s (freeze_backbone_epochs=%d). "
                "Trainer will rebuild optimizer.",
                epoch,
                "frozen" if should_freeze else "unfrozen",
                self.freeze_backbone_epochs,
            )

    def requires_optimizer_rebuild(self) -> bool:
        return self._needs_optimizer_rebuild

    def acknowledge_optimizer_rebuild(self) -> None:
        self._needs_optimizer_rebuild = False

    def _set_backbone_trainable(self, trainable: bool) -> None:
        """Toggle ``requires_grad`` on every backbone parameter."""
        for p in self._model.backbone.parameters():
            p.requires_grad = trainable
        self._backbone_frozen = not trainable

    # ------------------------------------------------------------------
    # BaseDetector boilerplate
    # ------------------------------------------------------------------

    def set_train_mode(self) -> None:
        self._model.train()

    def set_eval_mode(self) -> None:
        self._model.eval()

    def to_device(self, device) -> None:
        self._model.to(device)
        self._device = device if isinstance(device, torch.device) else torch.device(device)

    def supports_torch_compile(self) -> bool:
        """torchvision SSD has Python control flow that breaks ``torch.compile``.

        Its training-mode forward calls ``random.choice`` over ``min_size`` and
        invokes ``.item()`` inside the GeneralizedRCNNTransform, both of which
        trigger graph breaks. The YAML key ``training.use_torch_compile`` can
        still force compilation on if you want to override this default.
        """
        return False

    def get_config_schema(self) -> dict:
        return CONFIG_SCHEMA

    def load_checkpoint(self, path: Path) -> None:
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self._device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self._model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Loaded model state dict from %s", checkpoint_path)
        else:
            self._model.load_state_dict(checkpoint)
            logger.info("Loaded raw state dict from %s", checkpoint_path)

    def save_checkpoint(
        self,
        path: Path,
        optimizer=None,
        epoch=None,
        metrics=None,
    ) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self._model.state_dict(),
            "config": self.config,
            "num_classes": self.num_classes,
            "input_size": self.input_size,
            "torchvision_version": torchvision.__version__,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            checkpoint["epoch"] = epoch
        if metrics is not None:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, checkpoint_path)
        logger.info("Saved checkpoint to %s", checkpoint_path)
