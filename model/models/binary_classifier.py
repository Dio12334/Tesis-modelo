"""Binary damage classifier for the Road Damage Evaluation Framework.

Predicts whether an image contains road damage (1) or is background (0).
Used as the first stage in TwoStageDetector.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import timm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BinaryDamageClassifier:
    """Binary image classifier: background (0) vs. road damage (1).

    Uses a timm EfficientNet-B0 backbone with a single-output linear head.
    The original 1000-class head is replaced by ``Linear(feat_dim, 1)`` so
    ``forward()`` returns a scalar logit per image; ``predict()`` applies
    sigmoid and compares against the configured threshold.

    Args:
        config: Configuration dict. Recognized keys:
            - architecture (str): timm model name, default ``"efficientnet_b0"``
            - pretrained (bool): load ImageNet weights, default ``True``
            - threshold (float): sigmoid threshold for ``predict()``, default ``0.5``
    """

    def __init__(self, config: dict) -> None:
        architecture: str = config.get("architecture", "efficientnet_b0")
        pretrained: bool = bool(config.get("pretrained", True))
        self._threshold: float = float(config.get("threshold", 0.5))
        self._device: Optional[torch.device] = None

        self._model: nn.Module = timm.create_model(
            architecture, pretrained=pretrained, num_classes=1
        )
        logger.info(
            "BinaryDamageClassifier: %s (pretrained=%s, threshold=%.2f)",
            architecture,
            pretrained,
            self._threshold,
        )

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Compute raw logits for a batch of images.

        Args:
            images: Float tensor ``(N, C, H, W)`` in ``[0, 1]``.

        Returns:
            Logit tensor of shape ``(N,)``.
        """
        return self._model(images).squeeze(1)

    def predict(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute damage probabilities and binary labels.

        Args:
            images: Float tensor ``(N, C, H, W)`` in ``[0, 1]``.

        Returns:
            Tuple ``(probs, damage_mask)``:
                - ``probs``: Float tensor ``(N,)`` in ``[0, 1]``
                - ``damage_mask``: BoolTensor ``(N,)`` — ``True`` means damage
        """
        with torch.no_grad():
            logits = self.forward(images)
            probs = torch.sigmoid(logits)
            damage_mask = probs >= self._threshold
        return probs, damage_mask

    # ------------------------------------------------------------------
    # Mode helpers
    # ------------------------------------------------------------------

    def set_train_mode(self) -> None:
        self._model.train()

    def set_eval_mode(self) -> None:
        self._model.eval()

    def to_device(self, device) -> None:
        self._model.to(device)
        self._device = device

    def get_parameters(self):
        return [p for p in self._model.parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path) -> None:
        """Save model weights and threshold to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "threshold": self._threshold,
            },
            str(path),
        )
        logger.info("BinaryDamageClassifier saved → %s", path)

    def load_checkpoint(self, path: Path) -> None:
        """Load model weights (and optionally threshold) from ``path``."""
        path = Path(path)
        state = torch.load(str(path), map_location=self._device or "cpu")
        self._model.load_state_dict(state["model_state_dict"])
        if "threshold" in state:
            self._threshold = float(state["threshold"])
        logger.info("BinaryDamageClassifier loaded ← %s (threshold=%.2f)", path, self._threshold)
