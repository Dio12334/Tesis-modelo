"""RT-DETR model wrapper for the Road Damage Evaluation Framework.

Integrates the Ultralytics RT-DETR model using the adapter pattern,
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
    from ultralytics import RTDETR
except ImportError:
    RTDETR = None  # type: ignore[assignment,misc]

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@ModelRegistry.register("rt_detr")
class RT_DETR_Detector(BaseDetector):
    """RT-DETR detection model wrapper using the Ultralytics package.

    Supports configurable model sizes (l, x) and transformer-based
    end-to-end detection without NMS post-processing.
    """

    VALID_MODEL_SIZES = ("l", "x")
    MODEL_FILE_MAP = {
        "l": "rtdetr-l.pt",
        "x": "rtdetr-x.pt",
    }

    def __init__(self, config: dict) -> None:
        """Initialize RT-DETR detector with configuration.

        Args:
            config: Configuration dict. Must contain:
                - model_size (str, required): one of "l", "x"
                - num_classes (int, required): 1..1000
                - confidence_threshold (float, optional): default 0.25, range [0.0, 1.0]
                - iou_threshold (float, optional): default 0.7, range [0.0, 1.0]
                - pretrained_weights (str, optional): path to .pt file

        Raises:
            ImportError: If ultralytics is not installed.
            ConfigurationError: If config is invalid.
            FileNotFoundError: If pretrained_weights path does not exist.
        """
        if RTDETR is None:
            raise ImportError(
                "The 'ultralytics' package is required for RT_DETR_Detector. "
                "Install it with: pip install ultralytics>=8.3.0"
            )

        self._validate_config(config)

        self.config = config
        self.model_size: str = config["model_size"]
        self.num_classes: int = config["num_classes"]
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
            # (contain directory separators). Bare model names like "rtdetr-l.pt"
            # are handled by Ultralytics which downloads them automatically.
            if weights_path.parent != Path(".") and not weights_path.exists():
                raise FileNotFoundError(
                    f"Pretrained weights not found: {weights_path}"
                )
            self._model = RTDETR(str(pretrained_weights))
        else:
            model_file = self.MODEL_FILE_MAP[self.model_size]
            self._model = RTDETR(model_file)

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

        RT-DETR uses RTDETRDetectionLoss (Hungarian matching + set prediction loss),
        which is different from YOLO's v8DetectionLoss. When loading from a .pt file,
        Ultralytics creates a DetectionModel instead of RTDETRDetectionModel, so we
        patch the model class to enable the correct loss computation path.
        """
        if self._model is not None and hasattr(self._model, "model"):
            model_module = self._model.model

            # Patch the model class to RTDETRDetectionModel if needed.
            # When loading from .pt, Ultralytics creates a DetectionModel but
            # RTDETRDetectionModel has the correct init_criterion() and loss() methods.
            # Only attempt this when model_module has the expected Ultralytics structure.
            try:
                from ultralytics.models.rtdetr.model import RTDETRDetectionModel

                if (
                    not isinstance(model_module, RTDETRDetectionModel)
                    and hasattr(model_module, "model")
                    and hasattr(model_module.model, "__getitem__")
                ):
                    decoder = model_module.model[-1]
                    if hasattr(decoder, "nc"):
                        model_module.nc = decoder.nc
                    else:
                        model_module.nc = 80  # COCO default
                    model_module.__class__ = RTDETRDetectionModel
            except (ImportError, AttributeError, IndexError, TypeError):
                pass

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

            # Initialize criterion via the model's init_criterion method
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

    def get_config_schema(self) -> dict:
        """Return the configuration schema for RT-DETR.

        Returns:
            Dict describing config parameters with type and required fields.
        """
        return {
            "model_size": {"type": "str", "required": True},
            "num_classes": {"type": "int", "required": True},
            "confidence_threshold": {"type": "float", "required": False},
            "iou_threshold": {"type": "float", "required": False},
            "pretrained_weights": {"type": "str", "required": False},
        }

    def set_train_mode(self) -> None:
        """Set the underlying model to training mode."""
        self._model.model.train()

    def set_eval_mode(self) -> None:
        """Set the underlying model to evaluation mode."""
        self._model.model.eval()

    def get_parameters(self) -> List["torch.nn.Parameter"]:
        """Return trainable model parameters for optimizer construction.

        Returns:
            List of torch.nn.Parameter objects with requires_grad=True.
        """
        return [p for p in self._model.model.parameters() if p.requires_grad]

    def to_device(self, device) -> None:
        """Move the model to the specified device.

        Args:
            device: The target device (e.g., 'cpu', 'cuda', torch.device).
        """
        self._model.model.to(device)
        self._device = device

    def train_step(
        self,
        images: List["torch.Tensor"],
        targets: List[dict],
    ) -> dict:
        """Perform a single training step computing loss.

        Args:
            images: List of image tensors, each of shape (C, H, W).
            targets: List of target dicts, each with:
                - boxes: Tensor (N, 4) in xyxy pixel format
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

        # Build target tensor in Ultralytics format
        batch_indices_list = []
        cls_list = []
        bboxes_list = []

        for batch_idx, target in enumerate(targets):
            boxes = target["boxes"].to(device)  # (N, 4) xyxy format
            labels = target["labels"].to(device)  # (N,)

            if boxes.shape[0] == 0:
                continue

            # Convert xyxy to normalized xywh
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x_center = ((x1 + x2) / 2.0) / img_w
            y_center = ((y1 + y2) / 2.0) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h

            n = boxes.shape[0]
            batch_indices_list.append(
                torch.full((n,), batch_idx, dtype=torch.float32, device=device)
            )
            cls_list.append(labels.float())
            bboxes_list.append(
                torch.stack([x_center, y_center, width, height], dim=1)
            )

        # Handle all-empty targets case
        if len(bboxes_list) == 0:
            return {"loss_tensor": torch.tensor(0.0)}

        all_batch_idx = torch.cat(batch_indices_list, dim=0)
        all_cls = torch.cat(cls_list, dim=0)
        all_bboxes = torch.cat(bboxes_list, dim=0)

        # Build Ultralytics internal batch dict
        batch_dict = {
            "img": batch,
            "batch_idx": all_batch_idx,
            "cls": all_cls,
            "bboxes": all_bboxes,
        }

        # Compute loss via model.loss() which handles forward + criterion internally
        # RTDETRDetectionModel.loss() returns (total_loss, loss_items_tensor)
        loss_result = model_module.loss(batch_dict)
        if isinstance(loss_result, tuple):
            loss = loss_result[0]
        elif isinstance(loss_result, dict):
            loss = sum(loss_result.values())
        else:
            loss = loss_result

        # Ensure loss is a scalar
        if loss.dim() > 0:
            loss = loss.sum()

        return {"loss_tensor": loss}

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

        results = self._model.predict(
            images,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

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

        try:
            state_dict = torch.load(str(checkpoint_path), map_location="cpu")
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                self._model.model.load_state_dict(state_dict["model_state_dict"])
            else:
                self._model.model.load_state_dict(state_dict)
        except (FileNotFoundError, RuntimeError):
            raise
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
            torch.save(
                {"model_state_dict": self._model.model.state_dict()},
                str(checkpoint_path),
            )
        elif self._model is not None:
            torch.save(self._model, str(checkpoint_path))
        else:
            torch.save({}, str(checkpoint_path))
