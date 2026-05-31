"""YOLO26 model wrapper for the Road Damage Evaluation Framework.

Integrates the Ultralytics YOLO26 model using the adapter pattern,
implementing the BaseDetector interface for seamless use within the
framework's training and evaluation pipelines.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional

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

        # Unfreeze all parameters for fine-tuning
        for param in self._model.model.parameters():
            param.requires_grad = True

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

        if violations:
            raise ConfigurationError(violations)

    def _build_loss_fn(self) -> None:
        """Set up the loss computation from the Ultralytics model.

        The Ultralytics YOLO model exposes a `criterion` attribute on the
        underlying nn.Module (model.model) that can be used to compute
        training loss. This method stores a reference to that loss function.
        If the model does not have a criterion attribute, the loss will be
        computed via the model's forward pass in training mode.
        """
        if self._model is not None and hasattr(self._model, "model"):
            model_module = self._model.model
            # Ultralytics models expose criterion for loss computation
            if hasattr(model_module, "criterion") and model_module.criterion is not None:
                self._loss_fn = model_module.criterion
            else:
                # Fallback: loss will be computed via model forward in train mode
                self._loss_fn = None

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
        self._model.model.train()

        # Prepare targets in the format expected by Ultralytics loss:
        # Ultralytics expects a batch tensor of shape (N_total, 6) where each row is
        # [batch_idx, class_id, x_center, y_center, width, height] (normalized)
        # However, the exact format depends on the model's loss function.
        # We use the model's forward pass in training mode which handles loss internally.
        img_h, img_w = batch.shape[2], batch.shape[3]

        # Build target tensor in Ultralytics format: [batch_idx, cls, x_c, y_c, w, h]
        target_list = []
        for batch_idx, target in enumerate(targets):
            boxes = target["boxes"]  # (N, 4) xyxy format
            labels = target["labels"]  # (N,)

            if boxes.shape[0] == 0:
                continue

            # Convert xyxy to xywh normalized
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x_center = ((x1 + x2) / 2.0) / img_w
            y_center = ((y1 + y2) / 2.0) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h

            n = boxes.shape[0]
            batch_indices = torch.full((n, 1), batch_idx, dtype=boxes.dtype, device=boxes.device)
            cls_col = labels.float().unsqueeze(1)

            # Each row: [batch_idx, class, x_center, y_center, width, height]
            row = torch.cat([batch_indices, cls_col, x_center.unsqueeze(1),
                           y_center.unsqueeze(1), width.unsqueeze(1),
                           height.unsqueeze(1)], dim=1)
            target_list.append(row)

        if len(target_list) == 0:
            return {"loss_tensor": torch.tensor(0.0)}

        batch_targets = torch.cat(target_list, dim=0)

        # Compute loss using the model's forward pass in training mode
        # Ultralytics DetectionModel.forward() returns loss when targets are provided
        # The model's loss method expects: preds from model forward, batch dict
        model_module = self._model.model

        # Run forward pass to get predictions (feature maps)
        preds = model_module(batch)

        # Compute loss using the model's criterion/loss function
        if self._loss_fn is not None:
            # Use the criterion directly
            batch_dict = {"batch_idx": batch_targets[:, 0],
                         "cls": batch_targets[:, 1],
                         "bboxes": batch_targets[:, 2:]}
            loss = self._loss_fn(preds, batch_dict)
            if isinstance(loss, tuple):
                # criterion returns (total_loss, individual_losses)
                loss = loss[0]
            elif isinstance(loss, dict):
                loss = sum(loss.values())
        else:
            # Fallback: use model's built-in loss computation
            # Ultralytics models compute loss when called with targets in train mode
            batch_dict = {"img": batch, "batch_idx": batch_targets[:, 0],
                         "cls": batch_targets[:, 1],
                         "bboxes": batch_targets[:, 2:]}
            loss = model_module.loss(batch, batch_dict)
            if isinstance(loss, tuple):
                loss = loss[0]
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
                    "boxes": torch.zeros((0, 4), device=device),
                    "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                    "scores": torch.zeros((0,), device=device),
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
                    "boxes": torch.zeros((0, 4), device=device),
                    "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                    "scores": torch.zeros((0,), device=device),
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

        YOLO = ultralytics.YOLO

        try:
            # Try loading as an Ultralytics native checkpoint first.
            # YOLO(path) handles both native Ultralytics .pt files and
            # standard PyTorch state dicts saved by the framework.
            self._model = YOLO(str(checkpoint_path))
        except Exception as exc:
            # If direct YOLO loading fails, try loading as a framework-saved
            # state dict into the existing model.
            try:
                if self._model is None:
                    # Initialize a base model from the configured size
                    model_file = self.MODEL_FILE_MAP[self.model_size]
                    self._model = YOLO(model_file)
                state_dict = torch.load(
                    str(checkpoint_path), map_location="cpu"
                )
                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    self._model.model.load_state_dict(
                        state_dict["model_state_dict"]
                    )
                else:
                    # Re-raise the original error if it's not a recognized format
                    raise
            except (FileNotFoundError, RuntimeError):
                raise
            except Exception:
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
            # that can be reloaded by both load_checkpoint and torch.load
            torch.save(
                {"model_state_dict": self._model.model.state_dict()},
                str(checkpoint_path),
            )
        elif self._model is not None:
            # Fallback: save whatever the model object holds
            torch.save(self._model, str(checkpoint_path))
        else:
            # No model loaded yet — save an empty state
            torch.save({}, str(checkpoint_path))
