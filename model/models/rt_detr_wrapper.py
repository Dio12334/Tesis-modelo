"""RT-DETR model wrapper for the Road Damage Evaluation Framework.

Integrates the Ultralytics RT-DETR model using the adapter pattern,
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

        # Reshape classification head if num_classes differs from pretrained nc.
        # Must run BEFORE freeze logic (so freeze operates on final parameter graph)
        # and BEFORE _build_loss_fn (so init_criterion reads the corrected nc).
        self._reshape_head_if_needed()

        # Apply freeze layers logic (or default: all trainable)
        freeze_layers = config.get("freeze_layers", None)

        if freeze_layers is not None:
            # Validate against actual model layer count
            total_layers = self._get_total_layer_count()
            if freeze_layers > total_layers:
                raise ConfigurationError(
                    [f"Invalid freeze_layers '{freeze_layers}'. "
                     f"Must not exceed total layer count ({total_layers})"]
                )
            # Freeze first N layers
            frozen_names = self._get_first_n_layer_params(freeze_layers)
            for name, param in self._model.model.named_parameters():
                param.requires_grad = name not in frozen_names
            total_params = sum(1 for _ in self._model.model.parameters())
            frozen_count = len(frozen_names)
            trainable_count = total_params - frozen_count
            logger.info(
                f"Freeze layers: {freeze_layers}/{total_layers} layers frozen "
                f"({frozen_count} params frozen, {trainable_count} trainable)"
            )
        else:
            # Default: all params trainable (existing behavior)
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

        # Validate freeze_layers type and value (only if present)
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

    @staticmethod
    def _is_backbone_param(name: str, backbone_end: int) -> bool:
        """Determine if a named parameter belongs to a layer below the freeze threshold.

        Parses the ``model.<layer_idx>.<sublayer>.<kind>`` naming pattern used by
        the Ultralytics sequential layer structure.

        Args:
            name: The parameter name from ``model.named_parameters()``.
            backbone_end: The number of initial layers to consider as backbone
                (i.e., the freeze_layers value).

        Returns:
            True if the parameter belongs to a layer with index < backbone_end.
        """
        parts = name.split(".")
        if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
            layer_idx = int(parts[1])
            return layer_idx < backbone_end
        return False

    def _get_first_n_layer_params(self, n: int) -> set:
        """Collect all parameter names belonging to the first N layers.

        Iterates over the model's named parameters and identifies those
        belonging to layers with index < n using ``_is_backbone_param``.

        Args:
            n: Number of initial layers whose parameters should be collected.

        Returns:
            A set of parameter name strings for O(1) membership lookup.
        """
        frozen_names: set = set()
        for name, _ in self._model.model.named_parameters():
            if self._is_backbone_param(name, backbone_end=n):
                frozen_names.add(name)
        return frozen_names

    def _reshape_head_if_needed(self) -> None:
        """Rebuild RT-DETR classification heads when num_classes differs from pretrained nc.

        Pretrained RT-DETR weights (rtdetr-l.pt, rtdetr-x.pt) ship with ``nc=80``
        (COCO). When the project's configured ``num_classes`` differs (e.g., 5 for
        RDD2022), the decoder's classification heads must be replaced before
        training, otherwise:
            - The model emits 80-way logits instead of ``num_classes``-way.
            - The criterion is built against the wrong ``nc``.
            - Predicted label indices fall outside ``[0, num_classes - 1]``,
              breaking the metrics pipeline silently.

        Modules rebuilt (per Ultralytics RTDETRDecoder structure):
            - ``decoder.dec_score_head`` : ModuleList of ``Linear(hidden_dim, nc)``
              (one per decoder layer, default 6 layers).
            - ``decoder.enc_score_head`` : ``Linear(hidden_dim, nc)`` for encoder
              query selection.
            - ``decoder.denoising_class_embed`` : ``Embedding(nc, hidden_dim)``
              used by contrastive denoising training. Note this is ``nc``, not
              ``nc + 1`` (verified against ultralytics source).

        After rebuilding, ``decoder._reset_parameters()`` is invoked to set the
        focal-loss bias prior on the new classification layers, which is critical
        for stable early training.

        No-op when ``num_classes == decoder.nc``.

        Side effects:
            - Sets ``decoder.nc = self.num_classes``.
            - Sets ``model_module.nc = self.num_classes``.
            - All newly created module parameters start with ``requires_grad=True``,
              so subsequent freeze logic (which targets backbone-layer indices)
              leaves them trainable.
        """
        if self._model is None or not hasattr(self._model, "model"):
            return

        model_module = self._model.model
        if not hasattr(model_module, "model") or not hasattr(
            model_module.model, "__getitem__"
        ):
            return

        try:
            decoder = model_module.model[-1]
        except (IndexError, TypeError):
            return

        pretrained_nc = getattr(decoder, "nc", None)
        if not isinstance(pretrained_nc, int):
            return

        if pretrained_nc == self.num_classes:
            logger.debug(
                "Classification head already matches num_classes=%d; skipping reshape",
                self.num_classes,
            )
            return

        hidden_dim = getattr(decoder, "hidden_dim", None)
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            logger.warning(
                "Cannot reshape RT-DETR head: decoder.hidden_dim missing or invalid"
            )
            return

        n = self.num_classes

        logger.info(
            "Reshaping RT-DETR classification head: %d classes -> %d classes "
            "(hidden_dim=%d)",
            pretrained_nc,
            n,
            hidden_dim,
        )

        # Rebuild dec_score_head: ModuleList of Linear(hidden_dim, n), one per
        # decoder layer. Preserve the original list length.
        old_dec = decoder.dec_score_head
        try:
            num_dec_layers = len(old_dec)
        except TypeError:
            num_dec_layers = 6  # RT-DETR default
        decoder.dec_score_head = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, n) for _ in range(num_dec_layers)]
        )

        # Rebuild enc_score_head: single Linear(hidden_dim, n).
        decoder.enc_score_head = torch.nn.Linear(hidden_dim, n)

        # Rebuild denoising_class_embed if present: Embedding(n, hidden_dim).
        # Per ultralytics source, this is sized exactly nc (not nc+1).
        if hasattr(decoder, "denoising_class_embed"):
            decoder.denoising_class_embed = torch.nn.Embedding(n, hidden_dim)

        # Update nc on decoder and outer model module.
        decoder.nc = n
        model_module.nc = n

        # Re-initialize all decoder parameters via Ultralytics' canonical method.
        # This sets focal-loss bias priors on the new classification heads,
        # which is essential for numerical stability in the first training steps.
        if hasattr(decoder, "_reset_parameters"):
            try:
                decoder._reset_parameters()
                logger.info("RT-DETR decoder parameters re-initialized after head reshape")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "decoder._reset_parameters() failed after head reshape: %s",
                    exc,
                )

    def _build_loss_fn(self) -> None:
        """Set up the loss computation from the Ultralytics model.

        RT-DETR uses RTDETRDetectionLoss (Hungarian matching + set prediction loss).
        When loading from a .pt file, Ultralytics may create a generic DetectionModel
        instead of RTDETRDetectionModel, so we patch the model's class to enable the
        correct loss path.

        Critically, ``model_module.nc`` is set to ``self.num_classes`` BEFORE
        ``init_criterion()`` is called, so the criterion is built against the
        correct class count. This complements ``_reshape_head_if_needed()``,
        which has already updated ``decoder.nc`` and rebuilt the classification
        modules.

        We do NOT pre-populate ``model_module.args`` with YOLO's ``box``/``cls``/``dfl``
        loss-weight defaults: ``RTDETRDetectionLoss.__init__`` does not read those
        keys (verified against ultralytics.models.utils.loss). It uses its own
        ``loss_gain`` defaults: ``{"class": 1, "bbox": 5, "giou": 2, "no_object": 0.1}``.
        Setting the YOLO keys is dead state that obscures intent.

        Raises:
            RuntimeError: If the constructed criterion's nc attribute does not
                equal self.num_classes (defensive assertion).
        """
        if self._model is None or not hasattr(self._model, "model"):
            return

        model_module = self._model.model

        # Patch the model class to RTDETRDetectionModel if needed so that the
        # correct ``init_criterion()`` and ``loss()`` methods are used.
        try:
            from ultralytics.models.rtdetr.model import RTDETRDetectionModel

            if (
                not isinstance(model_module, RTDETRDetectionModel)
                and hasattr(model_module, "model")
                and hasattr(model_module.model, "__getitem__")
            ):
                model_module.__class__ = RTDETRDetectionModel
        except (ImportError, AttributeError, IndexError, TypeError):
            pass

        # Force model_module.nc to the configured num_classes BEFORE init_criterion.
        # This is the single most important line for correct loss adaptation.
        model_module.nc = self.num_classes

        # Initialize criterion via the model's init_criterion method.
        # RTDETRDetectionLoss is built with nc=model_module.nc, so this picks up
        # our corrected value.
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

        # Defensive assertion: criterion's nc must match self.num_classes.
        # Hard failure here is preferable to silent training against wrong nc.
        if self._loss_fn is not None and hasattr(self._loss_fn, "nc"):
            criterion_nc = self._loss_fn.nc
            # Only enforce when nc is a concrete int (skip Mocks in unit tests).
            if isinstance(criterion_nc, int) and criterion_nc != self.num_classes:
                raise RuntimeError(
                    f"RT-DETR criterion nc mismatch: criterion was initialized "
                    f"with nc={criterion_nc} but configuration requires "
                    f"num_classes={self.num_classes}. This indicates the head "
                    f"reshape or model.nc assignment did not take effect. "
                    f"Aborting to prevent silent training against the wrong "
                    f"output space."
                )

    def _get_total_layer_count(self) -> int:
        """Return the total number of freezable sequential layers.

        Inspects `_model.model.model` for a sequence with `__len__`.
        Falls back to 0 if the model structure is unexpected.

        Returns:
            Integer count of top-level sequential modules, or 0 if unavailable.
        """
        model_module = self._model.model
        if hasattr(model_module, "model") and hasattr(model_module.model, "__len__"):
            return len(model_module.model)
        return 0

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

        Supports both Ultralytics native ``.pt`` files and framework-saved ``.pt``
        files. The state dict undergoes the following pre-processing before
        ``load_state_dict``:

        1. ``_orig_mod.`` key prefixes (added by ``torch.compile()``) are stripped.
        2. Tensor shapes are validated against the current model's ``state_dict()``.
           Any shape mismatch raises ``RuntimeError`` with a diagnostic listing
           the first mismatching keys; we do NOT silently fall back to
           ``strict=False`` partial loading, since that masks ``num_classes``
           mismatches and other structural changes that should fail loud.

        Args:
            path: Path to the checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: On shape mismatch (with diagnostic), or if the file is
                corrupted or otherwise unloadable.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        try:
            raw = torch.load(str(checkpoint_path), map_location="cpu")
        except (FileNotFoundError, RuntimeError):
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load checkpoint '{checkpoint_path}': "
                f"file is corrupted or not a valid checkpoint format"
            ) from exc

        # Extract state dict from common wrappers
        if isinstance(raw, dict) and "model_state_dict" in raw:
            state = raw["model_state_dict"]
        elif isinstance(raw, dict) and "state_dict" in raw:
            state = raw["state_dict"]
        elif isinstance(raw, dict):
            state = raw
        else:
            # Probably a full pickled model object; cannot validate shapes here.
            try:
                self._model.model.load_state_dict(raw)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load checkpoint '{checkpoint_path}': "
                    f"unrecognized checkpoint format ({type(raw).__name__})"
                ) from exc
            return

        # Strip torch.compile prefix from keys (does nothing if absent)
        state = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in state.items()
        }

        # Validate shapes against the current model's state dict.
        # Hard fail on any mismatch, with a diagnostic message.
        current = self._model.model.state_dict()
        mismatches: List[tuple] = []
        for k, v in state.items():
            if k in current:
                expected = tuple(current[k].shape)
                actual = tuple(v.shape) if hasattr(v, "shape") else None
                if actual is not None and actual != expected:
                    mismatches.append((k, expected, actual))
                    if len(mismatches) >= 5:
                        break

        if mismatches:
            details = "; ".join(
                f"{k}: expected {exp}, got {act}" for k, exp, act in mismatches
            )
            raise RuntimeError(
                f"Checkpoint shape mismatch when loading '{checkpoint_path}'. "
                f"This typically means num_classes (or another structural "
                f"parameter) differs between the checkpoint and the current "
                f"configuration. Mismatches (first {len(mismatches)}): {details}"
            )

        try:
            self._model.model.load_state_dict(state)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load checkpoint '{checkpoint_path}' "
                f"(load_state_dict raised after shape validation): {exc}"
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
