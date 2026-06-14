"""Two-stage detection pipeline: binary classifier → object detector.

The first stage (BinaryDamageClassifier) filters out background images so
the expensive second-stage detector only runs on images predicted to contain
road damage.  Compatible with the full inference and evaluation CLI.
"""

import logging
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

from model.models.binary_classifier import BinaryDamageClassifier
from model.models.registry import BaseDetector, ModelRegistry

# ImageNet stats used when training BinaryDamageClassifier
_CLS_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_CLS_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

logger = logging.getLogger(__name__)


@ModelRegistry.register("two_stage")
class TwoStageDetector(BaseDetector):
    """Cascade detector: BinaryDamageClassifier → any registered detector.

    Stage 1 — ``BinaryDamageClassifier``:
        Given a batch of images, computes ``p_damage ∈ [0, 1]`` per image.
        Images with ``p_damage < classifier_threshold`` are labelled *background*
        and receive empty prediction dicts without touching the detector.

    Stage 2 — registered detector (e.g. ``rt_detr``, ``mmr_detr``):
        Runs only on the subset of images flagged as damaged.  Detections are
        re-inserted at their original batch positions.

    Config keys (all under ``model.config`` in the YAML):
        classifier_checkpoint (str, required):
            Path to a ``BinaryDamageClassifier`` ``.pt`` file.
        classifier_architecture (str):
            timm model name used when the checkpoint was created.
            Default: ``"efficientnet_b0"``.
        classifier_threshold (float):
            Sigmoid probability above which an image is considered damaged.
            Default: ``0.5``.  Lower values increase recall at the cost of
            more images reaching the detector.
        detector_type (str, required):
            Registered model identifier (e.g. ``"rt_detr"``).
        detector_checkpoint (str, required):
            Path to the detector ``.pt`` checkpoint.
        detector_config (dict, required):
            Config dict forwarded to ``ModelRegistry.create(detector_type, ...)``.
    """

    def __init__(self, config: dict) -> None:
        # ------------------------------------------------------------------ #
        # Stage 1: Binary classifier                                           #
        # ------------------------------------------------------------------ #
        classifier_ckpt = config["classifier_checkpoint"]
        classifier_cfg = {
            "architecture": config.get("classifier_architecture", "efficientnet_b0"),
            "pretrained": False,
            "threshold": config.get("classifier_threshold", 0.5),
        }
        self._classifier = BinaryDamageClassifier(classifier_cfg)
        self._classifier.load_checkpoint(Path(classifier_ckpt))
        self._classifier_input_size: int = int(config.get("classifier_input_size", 224))
        logger.info("TwoStageDetector — classifier loaded: %s", classifier_ckpt)

        # ------------------------------------------------------------------ #
        # Stage 2: Object detector                                             #
        # ------------------------------------------------------------------ #
        detector_type: str = config["detector_type"]
        detector_cfg: dict = config["detector_config"]
        self._detector: BaseDetector = ModelRegistry.create(detector_type, detector_cfg)
        detector_ckpt: str = config["detector_checkpoint"]
        self._detector.load_checkpoint(Path(detector_ckpt))
        logger.info(
            "TwoStageDetector — detector '%s' loaded: %s", detector_type, detector_ckpt
        )

        self._device: Optional[torch.device] = None

    # ---------------------------------------------------------------------- #
    # BaseDetector interface                                                   #
    # ---------------------------------------------------------------------- #

    def get_config_schema(self) -> dict:
        return {
            "classifier_checkpoint": {"type": "str", "required": True},
            "detector_type": {"type": "str", "required": True},
            "detector_checkpoint": {"type": "str", "required": True},
            "detector_config": {"type": "dict", "required": True},
        }

    def forward(self, images: torch.Tensor) -> List[dict]:
        """Run two-stage inference on a batch of images.

        Args:
            images: Float tensor ``(N, C, H, W)``.

        Returns:
            List of ``N`` prediction dicts, each with keys
            ``"boxes"`` (N,4), ``"labels"`` (N,), ``"scores"`` (N,).
            Background images return zero-row tensors.
        """
        self._classifier.set_eval_mode()
        self._detector.set_eval_mode()

        n = images.shape[0]
        device = images.device

        # Prepare classifier input: resize to training resolution + ImageNet normalization.
        # The evaluation pipeline delivers images at detector resolution (e.g. 640x640)
        # without channel normalization; the classifier was trained at 224x224 with
        # ImageNet mean/std, so we must re-preprocess before Stage 1.
        cls_size = self._classifier_input_size
        cls_images = F.interpolate(
            images, size=(cls_size, cls_size), mode="bilinear", align_corners=False
        )
        mean = _CLS_MEAN.to(device)
        std = _CLS_STD.to(device)
        cls_images = (cls_images - mean) / std

        # Stage 1
        _, damage_mask = self._classifier.predict(cls_images)
        damage_idx = damage_mask.nonzero(as_tuple=True)[0].tolist()
        bg_idx = (~damage_mask).nonzero(as_tuple=True)[0].tolist()

        logger.info(
            "TwoStageDetector: %d/%d images predicted as damaged (%.1f%% filtered as background)",
            len(damage_idx),
            n,
            len(bg_idx) / n * 100 if n > 0 else 0.0,
        )

        def _empty() -> dict:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                "scores": torch.zeros((0,), dtype=torch.float32, device=device),
            }

        results: List[Optional[dict]] = [None] * n
        for i in bg_idx:
            results[i] = _empty()

        # Stage 2 — only on damaged subset
        if damage_idx:
            damaged_batch = images[damage_idx]
            detections = self._detector.forward(damaged_batch)
            for slot, det in zip(damage_idx, detections):
                results[slot] = det

        return results  # type: ignore[return-value]

    def set_train_mode(self) -> None:
        self._classifier.set_train_mode()
        self._detector.set_train_mode()

    def set_eval_mode(self) -> None:
        self._classifier.set_eval_mode()
        self._detector.set_eval_mode()

    def to_device(self, device) -> None:
        self._classifier.to_device(device)
        self._detector.to_device(device)
        self._device = device

    def save_checkpoint(self, path: Path) -> None:
        """Delegates to the inner detector checkpoint."""
        self._detector.save_checkpoint(path)

    def load_checkpoint(self, path: Path) -> None:
        """No-op: both components are loaded from config in ``__init__``.

        To swap checkpoints, update the config and re-instantiate.
        """
        logger.warning(
            "TwoStageDetector.load_checkpoint() is a no-op — "
            "components are configured in __init__."
        )

    def train_step(self, images, targets) -> dict:
        raise NotImplementedError(
            "TwoStageDetector is inference-only. "
            "Train the classifier with train_classifier.py and the "
            "detector with train_detection.py separately."
        )
