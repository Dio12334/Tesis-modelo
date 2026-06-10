"""YOLO26 model wrapper for the Road Damage Evaluation Framework.

Integrates the Ultralytics YOLO26 model using the adapter pattern,
implementing the BaseDetector interface for seamless use within the
framework's training and evaluation pipelines.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional

logger = logging.getLogger(__name__)

from model.exceptions import ConfigurationError
from model.models.registry import BaseDetector, ModelRegistry

if TYPE_CHECKING:
    import torch

try:
    import ultralytics
except ImportError:
    ultralytics = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@ModelRegistry.register("yolo26")
class YOLO26Detector(BaseDetector):
    """YOLO26 detection model wrapper using the Ultralytics package.

    Supports configurable model sizes (n, s, m, l, x) and dual-head
    inference modes (end2end NMS-free or one-to-many with NMS).
    """

    VALID_MODEL_SIZES = ("n", "s", "m", "l", "x")
    MODEL_FILE_MAP = {
        "n": "yolo26n.pt",
        "s": "yolo26s.pt",
        "m": "yolo26m.pt",
        "l": "yolo26l.pt",
        "x": "yolo26x.pt",
    }

    def __init__(self, config: dict) -> None:
        """Initialize YOLO26 detector with configuration.

        Args:
            config: Configuration dict. Must contain:
                - model_size (str, required): one of "n", "s", "m", "l", "x"
                - num_classes (int, required): 1..1000
                - end2end (bool, optional): default True
                - confidence_threshold (float, optional): default 0.25, range [0.0, 1.0]
                - iou_threshold (float, optional): default 0.7, range [0.0, 1.0]
                - pretrained_weights (str, optional): path to .pt file

        Raises:
            ImportError: If ultralytics is not installed.
            ConfigurationError: If config is invalid.
            FileNotFoundError: If pretrained_weights path does not exist.
        """
        if ultralytics is None:
            raise ImportError(
                "The 'ultralytics' package is required for YOLO26Detector. "
                "Install it with: pip install ultralytics>=8.3.0"
            )

        self._validate_config(config)

        self.config = config
        self.model_size: str = config["model_size"]
        self.num_classes: int = config["num_classes"]
        self.end2end: bool = config.get("end2end", True)
        self.confidence_threshold: float = config.get("confidence_threshold", 0.25)
        self.iou_threshold: float = config.get("iou_threshold", 0.7)

        # Initialize the model
        self._device: Optional[Any] = None
        self._loss_fn: Optional[Callable] = None

        # Load model from pretrained weights or default model file
        pretrained_weights = config.get("pretrained_weights")
        if pretrained_weights:
            weights_path = Path(pretrained_weights)
            # Only check existence for paths that look like local files
            # (contain directory separators). Bare model names like "yolo26m.pt"
            # are handled by Ultralytics which downloads them automatically.
            if weights_path.parent != Path(".") and not weights_path.exists():
                raise FileNotFoundError(
                    f"Pretrained weights not found: {weights_path}"
                )
            self._model = ultralytics.YOLO(str(pretrained_weights))
        else:
            model_file = self.MODEL_FILE_MAP[self.model_size]
            self._model = ultralytics.YOLO(model_file)

        # Reshape detection head if pretrained model has different num_classes
        self._reshape_head_if_needed()

        # Configure parameter freezing for transfer learning
        freeze_backbone = config.get("freeze_backbone", False)
        freeze_layers = config.get("freeze_layers", None)

        # Validate freeze_layers does not exceed total layer count (requires loaded model)
        if freeze_layers is not None and isinstance(freeze_layers, int) and freeze_layers >= 0:
            model_module = self._model.model
            if hasattr(model_module, "model") and hasattr(model_module.model, "__len__"):
                total_layers = len(model_module.model)
            else:
                total_layers = self._DEFAULT_BACKBONE_LAYERS
            if freeze_layers > total_layers:
                raise ConfigurationError(
                    [f"Invalid freeze_layers '{freeze_layers}'. "
                     f"Must not exceed total layer count ({total_layers})"]
                )

        if freeze_backbone:
            # Freeze all backbone layers, keep head trainable
            backbone_end = self._get_backbone_layer_count()
            frozen_count = 0
            trainable_count = 0
            for name, param in self._model.model.named_parameters():
                if self._is_backbone_param(name, backbone_end):
                    param.requires_grad = False
                    frozen_count += 1
                else:
                    param.requires_grad = True
                    trainable_count += 1
            logger.info(
                f"Backbone frozen (layers 0-{backbone_end - 1}): "
                f"{frozen_count} params frozen, {trainable_count} params trainable"
            )
        elif freeze_layers is not None:
            # Freeze first N layers
            frozen_names = self._get_first_n_layer_params(freeze_layers)
            trainable_count = 0
            for name, param in self._model.model.named_parameters():
                param.requires_grad = name not in frozen_names
                if param.requires_grad:
                    trainable_count += 1
            logger.info(
                f"First {freeze_layers} layers frozen: "
                f"{len(frozen_names)} params frozen, {trainable_count} params trainable"
            )
        else:
            # Default: all params trainable (current behavior)
            for param in self._model.model.parameters():
                param.requires_grad = True
            total = sum(1 for _ in self._model.model.parameters())
            logger.info(f"All {total} params trainable (no freeze config)")

        # Build the loss function for training
        self._build_loss_fn()

    def _validate_config(self, config: dict) -> None:
        """Validate configuration parameters, collecting all violations.

        Args:
            config: Configuration dict to validate.

        Raises:
            ConfigurationError: If any validation rules are violated.
        """
        violations: List[str] = []

        # Check required parameters
        if "model_size" not in config:
            violations.append("Missing required parameter: model_size")
        if "num_classes" not in config:
            violations.append("Missing required parameter: num_classes")

        # Validate model_size value (only if present)
        if "model_size" in config:
            model_size = config["model_size"]
            if model_size not in self.VALID_MODEL_SIZES:
                violations.append(
                    f"Invalid model_size '{model_size}'. "
                    f"Must be one of: {list(self.VALID_MODEL_SIZES)}"
                )

        # Validate num_classes range (only if present)
        if "num_classes" in config:
            num_classes = config["num_classes"]
            if not isinstance(num_classes, int) or num_classes < 1 or num_classes > 1000:
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

        # Validate freeze_backbone type (only if present)
        if "freeze_backbone" in config:
            freeze_backbone = config["freeze_backbone"]
            if not isinstance(freeze_backbone, bool):
                violations.append(
                    f"Invalid freeze_backbone '{freeze_backbone}'. "
                    f"Must be a boolean"
                )

        # Validate freeze_layers type and range (only if present)
        if "freeze_layers" in config:
            freeze_layers = config["freeze_layers"]
            if not isinstance(freeze_layers, int) or isinstance(freeze_layers, bool):
                violations.append(
                    f"Invalid freeze_layers '{freeze_layers}'. "
                    f"Must be a non-negative integer"
                )
            elif freeze_layers < 0:
                violations.append(
                    f"Invalid freeze_layers '{freeze_layers}'. "
                    f"Must be a non-negative integer"
                )

        if violations:
            raise ConfigurationError(violations)

    def _reshape_head_if_needed(self) -> None:
        """Reshape the detection head to match the configured num_classes.

        Pretrained YOLO weights (e.g., COCO with 80 classes) have classification
        layers sized for their original class count. This method rebuilds the
        classification convolution layers (cv3 and one2one_cv3) in the Detect
        head to output predictions for self.num_classes instead.

        The box regression layers (cv2) are left unchanged since they are
        class-agnostic.
        """
        model_module = self._model.model
        model_nc = getattr(model_module, "nc", None)

        # Skip if nc is not a real integer (e.g., mocked model) or already matches
        if not isinstance(model_nc, int) or model_nc == self.num_classes:
            return

        logger.info(
            f"Reshaping detection head: {model_nc} classes -> {self.num_classes} classes"
        )

        # Update the model-level nc attribute
        model_module.nc = self.num_classes

        # Get the detection head (last layer in the sequential model)
        head = model_module.model[-1]
        head.nc = self.num_classes
        head.no = self.num_classes + head.reg_max * 4

        # Determine input channels per scale from existing cv3 layers
        # Each cv3[i] is a Sequential; the first sub-module's first layer
        # reveals the input channel count for that scale.
        ch = []
        for cv3_scale in head.cv3:
            # Navigate into the Sequential to find the first Conv/DWConv input channels
            first_module = cv3_scale[0]
            if hasattr(first_module, '__getitem__'):
                # Nested Sequential (DWConv path): first_module[0] is DWConv
                inp_ch = self._get_input_channels(first_module[0])
            else:
                inp_ch = self._get_input_channels(first_module)
            ch.append(inp_ch)

        # Rebuild classification layers (cv3) for the new num_classes
        # Use the same architecture as the Detect head in Ultralytics
        from ultralytics.nn.modules.conv import Conv, DWConv

        c3 = max(ch[0], min(self.num_classes, 100))
        legacy = getattr(head, "legacy", False)

        if legacy:
            new_cv3 = torch.nn.ModuleList(
                torch.nn.Sequential(
                    Conv(x, c3, 3), Conv(c3, c3, 3),
                    torch.nn.Conv2d(c3, self.num_classes, 1)
                ) for x in ch
            )
        else:
            new_cv3 = torch.nn.ModuleList(
                torch.nn.Sequential(
                    torch.nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    torch.nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    torch.nn.Conv2d(c3, self.num_classes, 1),
                )
                for x in ch
            )

        head.cv3 = new_cv3

        # Also rebuild one2one_cv3 if end2end mode is available
        if hasattr(head, "one2one_cv3") and head.one2one_cv3 is not None:
            import copy
            head.one2one_cv3 = copy.deepcopy(new_cv3)

        # Re-initialize detection head biases to stabilize early training.
        # Without this, the freshly created cv3 layers have zero bias, causing
        # all anchors to predict ~50% confidence and producing enormous loss.
        if hasattr(head, "bias_init") and hasattr(head, "stride") and head.stride is not None:
            try:
                # stride must be populated for bias_init to work
                if head.stride.sum() > 0:
                    head.bias_init()
                    logger.info("Detection head biases re-initialized")
            except Exception as e:
                logger.debug("Could not re-initialize head biases: %s", e)

        logger.info(
            f"Detection head reshaped successfully: "
            f"cv3 rebuilt for {self.num_classes} classes with ch={ch}"
        )

    @staticmethod
    def _get_input_channels(module) -> int:
        """Extract input channel count from a Conv/DWConv/nn.Conv2d module.

        Args:
            module: A convolutional module.

        Returns:
            Number of input channels.
        """
        if hasattr(module, "conv"):
            # Ultralytics Conv wrapper stores the actual nn.Conv2d as .conv
            return module.conv.in_channels
        elif hasattr(module, "in_channels"):
            return module.in_channels
        else:
            # Fallback: try first child
            for child in module.children():
                return YOLO26Detector._get_input_channels(child)
            raise RuntimeError(f"Cannot determine input channels from {type(module)}")

    def _build_loss_fn(self) -> None:
        """Set up the loss computation from the Ultralytics model.

        The Ultralytics YOLO model requires init_criterion() to be called
        to set up the loss function. The criterion also needs the model's
        hyperparameters (args) to include loss weights (box, cls, dfl).

        If the config contains a ``loss`` dict with ``label_smoothing`` or
        ``focal_gamma`` > 0, uses EnhancedDetectionLoss (subclass of
        v8DetectionLoss) which adds label smoothing and focal loss support.
        """
        if self._model is not None and hasattr(self._model, "model"):
            model_module = self._model.model

            # Ensure model.args has loss hyperparameters and is a namespace
            from types import SimpleNamespace
            if hasattr(model_module, "args"):
                args = model_module.args
                if isinstance(args, dict):
                    args.setdefault("box", 7.5)
                    args.setdefault("cls", 0.5)
                    args.setdefault("dfl", 1.5)
                    model_module.args = SimpleNamespace(**args)
                elif isinstance(args, SimpleNamespace):
                    if not hasattr(args, "box"):
                        args.box = 7.5
                    if not hasattr(args, "cls"):
                        args.cls = 0.5
                    if not hasattr(args, "dfl"):
                        args.dfl = 1.5
            else:
                model_module.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

            # Inject enhanced loss parameters from config into model.args
            loss_config = self.config.get("loss", {})
            label_smoothing = float(loss_config.get("label_smoothing", 0.0))
            focal_gamma = float(loss_config.get("focal_gamma", 0.0))
            focal_alpha = float(loss_config.get("focal_alpha", -1.0))
            # Enable focal_gamma from focal_loss: true shorthand
            if loss_config.get("focal_loss", False) and focal_gamma == 0.0:
                focal_gamma = 1.5  # default gamma when focal_loss: true

            model_module.args.label_smoothing = label_smoothing
            model_module.args.focal_gamma = focal_gamma
            model_module.args.focal_alpha = focal_alpha

            # Use EnhancedDetectionLoss if any enhanced feature is active;
            # otherwise fall back to stock criterion for zero-overhead path.
            use_enhanced = label_smoothing > 0 or focal_gamma > 0
            if use_enhanced:
                try:
                    from model.training.loss import (
                        EnhancedDetectionLoss,
                        EnhancedE2EDetectLoss,
                    )
                    # Detect if model uses E2E loss (one2many + one2one branches)
                    stock_criterion = None
                    if hasattr(model_module, "init_criterion"):
                        try:
                            stock_criterion = model_module.init_criterion()
                        except Exception:
                            pass
                    is_e2e = (
                        stock_criterion is not None
                        and hasattr(stock_criterion, "one2many")
                    )
                    if is_e2e:
                        self._loss_fn = EnhancedE2EDetectLoss(model_module)
                    else:
                        self._loss_fn = EnhancedDetectionLoss(model_module)
                    model_module.criterion = self._loss_fn
                    logger.info(
                        "Using %s (label_smoothing=%.3f, "
                        "focal_gamma=%.2f, focal_alpha=%.2f)",
                        type(self._loss_fn).__name__,
                        label_smoothing, focal_gamma, focal_alpha,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to build EnhancedDetectionLoss: %s. "
                        "Falling back to stock criterion.", e
                    )
                    use_enhanced = False

            if not use_enhanced:
                # Stock criterion path (backward compatible)
                if hasattr(model_module, "init_criterion"):
                    try:
                        self._loss_fn = model_module.init_criterion()
                        model_module.criterion = self._loss_fn
                    except Exception:
                        self._loss_fn = None
                elif hasattr(model_module, "criterion") and model_module.criterion is not None:
                    self._loss_fn = model_module.criterion
                else:
                    self._loss_fn = None

    # ------------------------------------------------------------------
    # Backbone / head layer identification helpers
    # ------------------------------------------------------------------

    # Default backbone layer count for YOLO26 architectures.
    # In Ultralytics YOLO models, the first 10 layers (indices 0-9) of the
    # nn.Sequential at model.model.model form the backbone (feature extractor),
    # while layers 10+ are the detection neck/head.
    _DEFAULT_BACKBONE_LAYERS = 10

    def _get_backbone_layer_count(self) -> int:
        """Determine the number of backbone layers from the model structure.

        Inspects the Ultralytics model's sequential layer list
        (``self._model.model.model``) to find how many layers constitute the
        backbone. If the internal structure is not available (e.g. in mocked
        models), falls back to the class default of 10.

        Returns:
            Number of backbone layers (int).
        """
        model_module = self._model.model
        # Ultralytics exposes the layer list at model.model.model (nn.Sequential)
        if hasattr(model_module, "model") and hasattr(model_module.model, "__len__"):
            total_layers = len(model_module.model)
            # Backbone is the first ~10 layers; never exceed total
            return min(self._DEFAULT_BACKBONE_LAYERS, total_layers)
        # Fallback for mock/test structures where model.model is the Sequential
        return self._DEFAULT_BACKBONE_LAYERS

    @staticmethod
    def _is_backbone_param(name: str, backbone_end: int) -> bool:
        """Check if a named parameter belongs to backbone layers.

        Parameters follow the naming pattern ``model.<layer_idx>.<sublayer>.<kind>``
        (e.g. ``model.0.conv.weight``, ``model.9.bn.bias``). A parameter belongs
        to the backbone when its layer index is strictly less than *backbone_end*.

        Args:
            name: Fully-qualified parameter name from ``named_parameters()``.
            backbone_end: Layer index where backbone ends (exclusive).

        Returns:
            True if the parameter belongs to a backbone layer, False otherwise.
        """
        parts = name.split(".")
        if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
            layer_idx = int(parts[1])
            return layer_idx < backbone_end
        return False

    def _get_first_n_layer_params(self, n: int) -> set:
        """Return parameter names for the first N layers of the model.

        Iterates over all named parameters and collects those whose layer
        index (extracted from the name prefix) is in ``[0, n)``.

        Args:
            n: Number of initial layers whose parameters to collect.

        Returns:
            A set of parameter name strings belonging to the first N layers.
        """
        frozen_names: set = set()
        for name, _ in self._model.model.named_parameters():
            if self._is_backbone_param(name, backbone_end=n):
                frozen_names.add(name)
        return frozen_names

    def get_parameters(self) -> List["torch.nn.Parameter"]:
        """Return trainable model parameters for optimizer construction.

        Returns:
            List of torch.nn.Parameter objects with requires_grad=True.
        """
        return [p for p in self._model.model.parameters() if p.requires_grad]

    def train_step(
        self,
        images: List["torch.Tensor"],
        targets: List[dict],
    ) -> dict:
        """Perform a single training step computing loss.

        Args:
            images: List of image tensors, each of shape (C, H, W).
            targets: List of target dicts, each with:
                - boxes: Tensor (N, 4) in xyxy format
                - labels: Tensor (N,) with class indices

        Returns:
            Dict with "loss_tensor" key containing a scalar tensor with grad_fn
            suitable for backpropagation. Returns torch.tensor(0.0) for empty batches.
        """
        # Handle empty batch case
        if not images or len(images) == 0:
            return {"loss_tensor": torch.tensor(0.0)}

        # Stack images into a batch tensor if they are a list
        if isinstance(images, list):
            batch = torch.stack(images)
        else:
            batch = images

        # Ensure model is in training mode
        model_module = self._model.model
        model_module.train()

        # Get model device
        device = next(model_module.parameters()).device
        batch = batch.to(device)

        img_h, img_w = batch.shape[2], batch.shape[3]

        # Build target tensor in Ultralytics format: [batch_idx, cls, x_c, y_c, w, h]
        target_list = []
        for batch_idx, target in enumerate(targets):
            boxes = target["boxes"].to(device)  # (N, 4) xyxy format
            labels = target["labels"].to(device)  # (N,)

            if boxes.shape[0] == 0:
                continue

            # Convert xyxy to xywh normalized
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x_center = ((x1 + x2) / 2.0) / img_w
            y_center = ((y1 + y2) / 2.0) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h

            n = boxes.shape[0]
            batch_indices = torch.full((n, 1), batch_idx, dtype=boxes.dtype, device=device)
            cls_col = labels.float().unsqueeze(1)

            # Each row: [batch_idx, class, x_center, y_center, width, height]
            row = torch.cat([batch_indices, cls_col, x_center.unsqueeze(1),
                           y_center.unsqueeze(1), width.unsqueeze(1),
                           height.unsqueeze(1)], dim=1)
            target_list.append(row)

        if len(target_list) == 0:
            return {"loss_tensor": torch.tensor(0.0)}

        batch_targets = torch.cat(target_list, dim=0)

        # Use model.loss() which handles forward + criterion + device internally
        batch_dict = {
            "img": batch,
            "batch_idx": batch_targets[:, 0],
            "cls": batch_targets[:, 1],
            "bboxes": batch_targets[:, 2:],
        }
        loss = model_module.loss(batch_dict)
        if isinstance(loss, tuple):
            # Returns (loss_components_tensor, detached_losses)
            loss = loss[0].sum()
        elif isinstance(loss, dict):
            loss = sum(loss.values())

        # Ensure loss is a scalar
        if loss.dim() > 0:
            loss = loss.sum()

        return {"loss_tensor": loss}

    def set_train_mode(self) -> None:
        """Set the underlying model to training mode."""
        self._model.model.train()

    def set_eval_mode(self) -> None:
        """Set the underlying model to evaluation mode."""
        self._model.model.eval()

    def forward(self, images: "torch.Tensor") -> List[dict]:
        """Run forward pass on a batch of images.

        Args:
            images: Batch of images as a torch.Tensor with shape (B, C, H, W).

        Returns:
            List of dicts per image, each containing:
                - boxes: Tensor of shape (N, 4) in xyxy format
                - labels: Tensor of shape (N,) with dtype int64
                - scores: Tensor of shape (N,) with values in [0.0, 1.0]
        """
        device = images.device

        # Build predict kwargs based on end2end mode
        predict_kwargs = {
            "conf": self.confidence_threshold,
            "verbose": False,
        }
        if not self.end2end:
            predict_kwargs["iou"] = self.iou_threshold

        results = self._model.predict(images, **predict_kwargs)

        return self._convert_results(results, device)

    def _convert_results(
        self, results: list, device: "torch.device"
    ) -> List[dict]:
        """Convert Ultralytics Results objects to framework prediction dicts.

        Args:
            results: List of Ultralytics Results objects from model.predict().
            device: Target device for all output tensors.

        Returns:
            List of prediction dicts, one per image, each containing:
                - boxes: Tensor of shape (N, 4) in xyxy format on device
                - labels: Tensor of shape (N,) with dtype int64 on device
                - scores: Tensor of shape (N,) with values in [0.0, 1.0] on device
        """
        predictions: List[dict] = []

        for result in results:
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                predictions.append({
                    "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                    "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                    "scores": torch.zeros((0,), dtype=torch.float32, device=device),
                })
                continue

            xyxy = boxes.xyxy.to(device)
            cls = boxes.cls.to(device)
            conf = boxes.conf.to(device)

            # Apply confidence filtering
            mask = conf >= self.confidence_threshold
            xyxy = xyxy[mask]
            cls = cls[mask]
            conf = conf[mask]

            if xyxy.shape[0] == 0:
                predictions.append({
                    "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                    "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                    "scores": torch.zeros((0,), dtype=torch.float32, device=device),
                })
            else:
                predictions.append({
                    "boxes": xyxy.float(),
                    "labels": cls.long(),
                    "scores": conf.float(),
                })

        return predictions

    def get_config_schema(self) -> dict:
        """Return the configuration schema for YOLO26.

        Returns:
            Dict describing config parameters with type and required fields.
        """
        return {
            "model_size": {"type": "str", "required": True},
            "num_classes": {"type": "int", "required": True},
            "end2end": {"type": "bool", "required": False},
            "confidence_threshold": {"type": "float", "required": False},
            "iou_threshold": {"type": "float", "required": False},
        }

    def load_checkpoint(self, path: Path) -> None:
        """Load model weights from a checkpoint file.

        Supports both Ultralytics native .pt files and framework-saved .pt files.
        After loading, the model is ready for inference or continued training.

        Args:
            path: Path to the checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: If the checkpoint is corrupted or invalid format.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        # Try loading as a framework-saved state dict first (preferred path).
        # This preserves the current model architecture (including reshaped head).
        try:
            state_dict = torch.load(
                str(checkpoint_path), map_location="cpu"
            )
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                saved_state = state_dict["model_state_dict"]

                # Handle key prefix mismatches caused by torch.compile()
                # or different model wrapping.
                model_keys = set(self._model.model.state_dict().keys())
                saved_keys = set(saved_state.keys())

                if model_keys and saved_keys and not (model_keys & saved_keys):
                    sample_saved_key = next(iter(saved_keys))

                    if sample_saved_key.startswith("_orig_mod."):
                        saved_state = {
                            k.removeprefix("_orig_mod."): v
                            for k, v in saved_state.items()
                        }
                    else:
                        sample_model_key = next(iter(model_keys))
                        if sample_model_key.startswith("model.") and not sample_saved_key.startswith("model."):
                            saved_state = {f"model.{k}": v for k, v in saved_state.items()}
                        elif not sample_model_key.startswith("model.") and sample_saved_key.startswith("model."):
                            saved_state = {k.removeprefix("model."): v for k, v in saved_state.items()}

                # Check for shape mismatches before loading
                model_state = self._model.model.state_dict()
                mismatched = []
                for key in saved_state:
                    if key in model_state and saved_state[key].shape != model_state[key].shape:
                        mismatched.append(
                            f"  {key}: checkpoint={list(saved_state[key].shape)} "
                            f"vs model={list(model_state[key].shape)}"
                        )

                if mismatched:
                    logger.warning(
                        "Shape mismatches during checkpoint loading "
                        "(these layers will use random weights):\n%s",
                        "\n".join(mismatched),
                    )

                result = self._model.model.load_state_dict(saved_state, strict=False)

                # Report missing and unexpected keys
                if result.missing_keys:
                    logger.warning(
                        "Checkpoint missing %d keys (using initialized weights): %s",
                        len(result.missing_keys),
                        result.missing_keys[:10],
                    )
                if result.unexpected_keys:
                    logger.warning(
                        "Checkpoint has %d unexpected keys (ignored): %s",
                        len(result.unexpected_keys),
                        result.unexpected_keys[:10],
                    )

                logger.info("Loaded framework checkpoint: %s", checkpoint_path)
                return
            else:
                logger.debug(
                    "Checkpoint is not framework format (no 'model_state_dict' key), "
                    "trying Ultralytics native format..."
                )
        except Exception as e:
            logger.warning(
                "Failed to load checkpoint as framework state_dict: %s. "
                "Trying Ultralytics native format...", e
            )

        # Fallback: try loading as an Ultralytics native checkpoint.
        # This replaces the model entirely, so we need to reshape again.
        YOLO = ultralytics.YOLO
        try:
            self._model = YOLO(str(checkpoint_path))
            # Re-apply head reshape if the loaded model has wrong nc
            self._reshape_head_if_needed()
            logger.info("Loaded Ultralytics checkpoint: %s", checkpoint_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load checkpoint '{checkpoint_path}': "
                f"file is corrupted or not a valid checkpoint format"
            ) from exc

    def save_checkpoint(self, path: Path) -> None:
        """Save model weights to a checkpoint file.

        Creates parent directories as needed and overwrites any existing file.

        Args:
            path: Path where the checkpoint will be saved.
        """
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        if self._model is not None and hasattr(self._model, "model"):
            # Save the underlying PyTorch model state dict in a format
            # that can be reloaded by both load_checkpoint and torch.load.
            # Strip "_orig_mod." prefix that torch.compile() adds, so
            # checkpoints are always saved with clean key names.
            raw_state = self._model.model.state_dict()
            clean_state = {
                k.removeprefix("_orig_mod."): v for k, v in raw_state.items()
            }
            torch.save(
                {"model_state_dict": clean_state},
                str(checkpoint_path),
            )
        elif self._model is not None:
            # Fallback: save whatever the model object holds
            torch.save(self._model, str(checkpoint_path))
        else:
            # No model loaded yet — save an empty state
            torch.save({}, str(checkpoint_path))
