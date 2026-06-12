"""MMR-DETR model wrapper for the Road Damage Evaluation Framework.

Integrates a custom RT-DETR variant (MMR-DETR) using the Ultralytics RTDETR
class with a user-supplied architecture YAML, implementing the BaseDetector
interface for seamless use within the framework's training and evaluation
pipelines.
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

# ---------------------------------------------------------------------------
# Backward compatibility with ultralytics 8.0.x checkpoints
# MMR-DETR was trained with ultralytics==8.0.201 which had classes removed
# in later versions (ConvNormLayer, Blocks, BasicBlock, and several
# transformer modules under ultralytics.nn.extra_modules.transformer).
# These patches are NO-OPs for the current architecture (they only exist
# so pickle.load can deserialize the old checkpoint format).
# ---------------------------------------------------------------------------
import types as _types

try:
    import ultralytics.nn.modules.block as _block

    if not hasattr(_block, "ConvNormLayer"):

        class _ConvNormLayer(torch.nn.Module):
            def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
                super().__init__()
                p = (kernel_size - 1) // 2 if padding is None else padding
                self.conv = torch.nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding=p, bias=bias)
                self.norm = torch.nn.BatchNorm2d(ch_out)
                self.act = torch.nn.Identity() if act is None else act

            def forward(self, x):
                return self.act(self.norm(self.conv(x)))

        _block.ConvNormLayer = _ConvNormLayer

        class _Blocks(torch.nn.Module):
            def __init__(self, ch_in, ch_out, n, block=_ConvNormLayer, **kwargs):
                super().__init__()
                self.blocks = torch.nn.ModuleList([block(ch_in, ch_out, **kwargs) for _ in range(n)])

            def forward(self, x):
                for b in self.blocks:
                    x = b(x)
                return x

        _block.Blocks = _Blocks

        class _BasicBlock(torch.nn.Module):
            """Old ultralytics BasicBlock (ResNet-style) used by RT-DETR.

            The pickle stores these attributes (not set by __init__):
                branch2a : ConvNormLayer
                branch2b : ConvNormLayer
                act      : nn.ReLU
                short    : ConvNormLayer or Sequential (optional shortcut)
            """
            def __init__(self, ch_in, ch_out, stride=1, shortcut=True):
                super().__init__()
                self.shortcut = shortcut
                self.branch2a = _ConvNormLayer(ch_in, ch_out, 3, stride)
                self.branch2b = _ConvNormLayer(ch_out, ch_out, 3, 1)
                self.act = torch.nn.ReLU()
                if shortcut and (stride != 1 or ch_in != ch_out):
                    self.short = _ConvNormLayer(ch_in, ch_out, 1, stride)

            def forward(self, x):
                identity = x
                out = self.branch2a(x)
                out = self.branch2b(out)
                if hasattr(self, "short"):
                    identity = self.short(identity)
                out += identity
                return self.act(out)

        _block.BasicBlock = _BasicBlock

    # Create ultralytics.nn.extra_modules.transformer placeholder
    _extra_mod = _types.ModuleType("ultralytics.nn.extra_modules")
    _extra_mod.__path__ = []
    import sys as _sys
    _sys.modules.setdefault("ultralytics.nn.extra_modules", _extra_mod)

    _trans_mod = _types.ModuleType("ultralytics.nn.extra_modules.transformer")
    _extra_mod.transformer = _trans_mod
    _sys.modules.setdefault("ultralytics.nn.extra_modules.transformer", _trans_mod)

    # Patch all known missing transformer classes as placeholders
    for _name in [
        "MLP",
        "TransformerEncoderLayer_MSMHSA",
        "Mutilscal_MHSA",
        "MutilScal",
        "LayerNorm",
        "Adaptive2DPositionalEncoding",
        "PatchEmbed",
        "DeformableTransformerDecoderLayer",
        "DeformableTransformerDecoder",
        "MSDeformableAttention",
    ]:
        if not hasattr(_trans_mod, _name):
            _dummy = type(_name, (torch.nn.Module,), {
                "__init__": lambda self, *a, **kw: torch.nn.Module.__init__(self),
                "forward": lambda self, x, *a, **kw: x,
            })
            setattr(_trans_mod, _name, _dummy)

except ImportError:
    pass


@ModelRegistry.register("mmr_detr")
class MMR_DETR_Detector(BaseDetector):
    """MMR-DETR detection model wrapper using a custom Ultralytics YAML.

    Loads an RT-DETR architecture from a user-supplied YAML file instead of
    a predefined model size. All train_step, forward, save/load checkpoint
    logic is identical to RT-DETR since MMR-DETR uses the same underlying
    ``ultralytics.RTDETR`` class.
    """

    DEFAULT_MODEL_YAML = str(
        Path(__file__).resolve().parent.parent / "configs" / "models" / "mmr_detr.yaml"
    )

    def __init__(self, config: dict) -> None:
        """Initialize MMR-DETR detector with configuration.

        Args:
            config: Configuration dict. Must contain:
                - num_classes (int, required): 1..1000
                - model_yaml (str, optional): path to architecture YAML.
                  Defaults to model/configs/models/mmr_detr.yaml
                - confidence_threshold (float, optional): default 0.25
                - iou_threshold (float, optional): default 0.7
                - pretrained_weights (str, optional): path to .pt file
                - freeze_layers (int, optional): freeze first N layers

        Raises:
            ImportError: If ultralytics is not installed.
            ConfigurationError: If config is invalid.
            FileNotFoundError: If model_yaml or pretrained_weights not found.
        """
        if RTDETR is None:
            raise ImportError(
                "The 'ultralytics' package is required for MMR_DETR_Detector. "
                "Install it with: pip install ultralytics>=8.3.0"
            )

        self._validate_config(config)

        self.config = config
        self.num_classes: int = config["num_classes"]
        self.confidence_threshold: float = config.get("confidence_threshold", 0.25)
        self.iou_threshold: float = config.get("iou_threshold", 0.7)
        self._device: Optional[Any] = None
        self._loss_fn: Optional[Callable] = None

        # Load model from custom YAML (architecture only) or pretrained weights
        model_yaml = config.get("model_yaml", self.DEFAULT_MODEL_YAML)
        pretrained_weights = config.get("pretrained_weights")

        if pretrained_weights:
            weights_path = Path(pretrained_weights)
            if weights_path.parent != Path(".") and not weights_path.exists():
                raise FileNotFoundError(
                    f"Pretrained weights not found: {weights_path}"
                )
            self._model = RTDETR(str(pretrained_weights))
        else:
            yaml_path = Path(model_yaml)
            if not yaml_path.exists():
                raise FileNotFoundError(
                    f"Model YAML not found: {yaml_path}"
                )
            self._model = RTDETR(str(yaml_path))

        # Reshape classification head if num_classes differs from YAML nc
        self._reshape_head_if_needed()

        # Apply freeze layers logic (or default: all trainable)
        freeze_layers = config.get("freeze_layers", None)

        if freeze_layers is not None:
            total_layers = self._get_total_layer_count()
            if freeze_layers > total_layers:
                raise ConfigurationError(
                    [f"Invalid freeze_layers '{freeze_layers}'. "
                     f"Must not exceed total layer count ({total_layers})"]
                )
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
            for param in self._model.model.parameters():
                param.requires_grad = True

        # Build the loss function for training
        self._build_loss_fn()

    def _validate_config(self, config: dict) -> None:
        """Validate configuration parameters.

        Args:
            config: Configuration dict to validate.

        Raises:
            ConfigurationError: If any validation rules are violated.
        """
        violations: List[str] = []

        if "num_classes" not in config:
            violations.append("Missing required parameter: num_classes")

        if "num_classes" in config:
            num_classes = config["num_classes"]
            if not isinstance(num_classes, int) or num_classes < 1 or num_classes > 1000:
                violations.append(
                    f"Invalid num_classes '{num_classes}'. "
                    f"Must be an integer in range [1, 1000]"
                )

        if "model_yaml" in config:
            yaml_path = Path(config["model_yaml"])
            if not yaml_path.exists():
                violations.append(f"model_yaml not found: {yaml_path}")

        if "confidence_threshold" in config:
            conf = config["confidence_threshold"]
            if not isinstance(conf, (int, float)) or conf < 0.0 or conf > 1.0:
                violations.append(
                    f"Invalid confidence_threshold '{conf}'. "
                    f"Must be a float in range [0.0, 1.0]"
                )

        if "iou_threshold" in config:
            iou = config["iou_threshold"]
            if not isinstance(iou, (int, float)) or iou < 0.0 or iou > 1.0:
                violations.append(
                    f"Invalid iou_threshold '{iou}'. "
                    f"Must be a float in range [0.0, 1.0]"
                )

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

        Args:
            name: The parameter name from ``model.named_parameters()``.
            backbone_end: The number of initial layers to consider frozen.

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
        """Rebuild RT-DETR classification heads when num_classes differs from YAML nc.

        Same logic as RT_DETR_Detector._reshape_head_if_needed: replaces the
        decoder's score heads, encoder score head, and denoising embeddings
        to match the configured num_classes.
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
                "Cannot reshape MMR-DETR head: decoder.hidden_dim missing or invalid"
            )
            return

        n = self.num_classes

        logger.info(
            "Reshaping MMR-DETR classification head: %d classes -> %d classes "
            "(hidden_dim=%d)",
            pretrained_nc,
            n,
            hidden_dim,
        )

        old_dec = decoder.dec_score_head
        try:
            num_dec_layers = len(old_dec)
        except TypeError:
            num_dec_layers = 6
        decoder.dec_score_head = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, n) for _ in range(num_dec_layers)]
        )

        decoder.enc_score_head = torch.nn.Linear(hidden_dim, n)

        if hasattr(decoder, "denoising_class_embed"):
            decoder.denoising_class_embed = torch.nn.Embedding(n, hidden_dim)

        decoder.nc = n
        model_module.nc = n

        if hasattr(decoder, "_reset_parameters"):
            try:
                decoder._reset_parameters()
                logger.info("MMR-DETR decoder parameters re-initialized after head reshape")
            except Exception as exc:
                logger.warning(
                    "decoder._reset_parameters() failed after head reshape: %s",
                    exc,
                )

    def _build_loss_fn(self) -> None:
        """Set up the loss computation from the Ultralytics model.

        Identical to RT_DETR_Detector._build_loss_fn: ensures
        model_module.nc is correct and initializes the RTDETRDetectionLoss
        criterion.
        """
        if self._model is None or not hasattr(self._model, "model"):
            return

        model_module = self._model.model

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

        model_module.nc = self.num_classes

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

        if self._loss_fn is not None and hasattr(self._loss_fn, "nc"):
            criterion_nc = self._loss_fn.nc
            if isinstance(criterion_nc, int) and criterion_nc != self.num_classes:
                raise RuntimeError(
                    f"MMR-DETR criterion nc mismatch: criterion was initialized "
                    f"with nc={criterion_nc} but configuration requires "
                    f"num_classes={self.num_classes}. Aborting to prevent "
                    f"silent training against the wrong output space."
                )

    def _get_total_layer_count(self) -> int:
        """Return the total number of freezable sequential layers.

        Returns:
            Integer count of top-level sequential modules, or 0 if unavailable.
        """
        model_module = self._model.model
        if hasattr(model_module, "model") and hasattr(model_module.model, "__len__"):
            return len(model_module.model)
        return 0

    def get_config_schema(self) -> dict:
        """Return the configuration schema for MMR-DETR.

        Returns:
            Dict describing config parameters with type and required fields.
        """
        return {
            "num_classes": {"type": "int", "required": True},
            "model_yaml": {"type": "str", "required": False},
            "confidence_threshold": {"type": "float", "required": False},
            "iou_threshold": {"type": "float", "required": False},
            "pretrained_weights": {"type": "str", "required": False},
            "freeze_layers": {"type": "int", "required": False},
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

    def get_parameter_groups(self, backbone_lr: float, head_lr: float,
                             backbone_layers: int = 10) -> List[dict]:
        """Return optimizer parameter groups with discriminative learning rates.

        Splits parameters into a backbone group (low LR, to preserve pretrained
        features) and a head/encoder/decoder group (normal LR, to adapt to the
        target task). The split is by top-level sequential layer index: layers
        with index < ``backbone_layers`` are treated as backbone.

        This replaces binary ``freeze_layers`` (which fails on MMR-DETR because
        its layer numbering differs from RT-DETR-L).

        Args:
            backbone_lr: Learning rate for backbone parameters.
            head_lr: Learning rate for head/encoder/decoder parameters.
            backbone_layers: Number of initial layers considered backbone.

        Returns:
            List of dicts suitable for torch optimizers, e.g.
            ``[{"params": [...], "lr": backbone_lr}, {"params": [...], "lr": head_lr}]``.
        """
        backbone_params = []
        head_params = []
        for name, param in self._model.model.named_parameters():
            if not param.requires_grad:
                continue
            if self._is_backbone_param(name, backbone_end=backbone_layers):
                backbone_params.append(param)
            else:
                head_params.append(param)

        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        if head_params:
            groups.append({"params": head_params, "lr": head_lr})
        logger.info(
            "MMR-DETR discriminative LR: %d backbone params @ lr=%.2e, "
            "%d head params @ lr=%.2e (backbone_layers=%d)",
            len(backbone_params), backbone_lr, len(head_params), head_lr,
            backbone_layers,
        )
        return groups

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
            Dict with "loss_tensor" key containing a scalar tensor with grad_fn.
        """
        if not images or len(images) == 0:
            device = next(self._model.model.parameters()).device
            return {"loss_tensor": torch.tensor(0.0, requires_grad=True, device=device)}

        if isinstance(images, list):
            batch = torch.stack(images)
        else:
            batch = images

        model_module = self._model.model
        model_module.train()

        device = next(model_module.parameters()).device
        batch = batch.to(device)

        img_h, img_w = batch.shape[2], batch.shape[3]

        batch_indices_list = []
        cls_list = []
        bboxes_list = []

        for batch_idx, target in enumerate(targets):
            boxes = target["boxes"].to(device)
            labels = target["labels"].to(device)

            if boxes.shape[0] == 0:
                continue

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

        if len(bboxes_list) == 0:
            return {"loss_tensor": torch.tensor(0.0, requires_grad=True, device=device)}

        all_batch_idx = torch.cat(batch_indices_list, dim=0)
        all_cls = torch.cat(cls_list, dim=0)
        all_bboxes = torch.cat(bboxes_list, dim=0)

        batch_dict = {
            "img": batch,
            "batch_idx": all_batch_idx,
            "cls": all_cls,
            "bboxes": all_bboxes,
        }

        loss_result = model_module.loss(batch_dict)
        if isinstance(loss_result, tuple):
            loss = loss_result[0]
        elif isinstance(loss_result, dict):
            loss = sum(loss_result.values())
        else:
            loss = loss_result

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
            List of prediction dicts, one per image.
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

        Supports both Ultralytics native .pt files and framework-saved .pt
        files. Strips ``_orig_mod.`` prefixes from torch.compile and
        validates tensor shapes before loading.

        Args:
            path: Path to the checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: On shape mismatch (with diagnostic), or if the file
                is corrupted or otherwise unloadable.
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

        if isinstance(raw, dict) and "model_state_dict" in raw:
            state = raw["model_state_dict"]
        elif isinstance(raw, dict) and "state_dict" in raw:
            state = raw["state_dict"]
        elif isinstance(raw, dict):
            state = raw
        else:
            try:
                self._model.model.load_state_dict(raw)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load checkpoint '{checkpoint_path}': "
                    f"unrecognized checkpoint format ({type(raw).__name__})"
                ) from exc
            return

        state = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in state.items()
        }

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
                f"Mismatches (first {len(mismatches)}): {details}"
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
