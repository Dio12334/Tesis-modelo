"""YOLO-RD attention modules (CSAF, LGECA) and in-place surgery for YOLO26.

Ports the two architectural contributions of YOLO-RD (Wang et al.,
*Neurocomputing* 662 (2026)) onto an Ultralytics YOLO26 model:

  - **CSAF** (Convolution Spatial-to-Depth Attention Fusion): a stride-2
    downsampling block fusing a standard conv branch with a space-to-depth
    branch via ESE channel attention + softmax gating. Replaces the stem conv.
  - **LGECA** (Local-Global Enhanced Context Attention): an ECA-style channel
    attention combining a global branch (GAP+GMP → Conv1d) and a local branch
    (adaptive-pool → Conv1d) with a learnable fusion weight. Wrapped around each
    detection-head input block.

Integration uses **in-place module replacement** rather than a custom model
YAML: we build the standard pretrained YOLO26, then swap the stem for CSAF and
wrap the three detect-input blocks with LGECA at the *same* layer indices. This
preserves Ultralytics' forward routing (each module's ``.i``/``.f``/``.type``)
and lets every unchanged layer keep its pretrained weights — only CSAF and the
LGECA additions start from random init.
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

logger = logging.getLogger(__name__)


class ESE(nn.Module):
    """Effective Squeeze-and-Excitation channel attention (channel-preserving).

    Single GAP + 1x1 conv + sigmoid gate (the "effective" SE variant from
    CenterMask/VoVNetV2 that drops the dimensionality-reducing FC).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean((2, 3), keepdim=True)  # global average pool
        w = torch.sigmoid(self.fc(w))
        return x * w


class CSAF(nn.Module):
    """Convolution Spatial-to-Depth Attention Fusion (YOLO-RD, Eqs. 1-7).

    Drop-in replacement for a stride-2 stem ``Conv(c1, c2, 3, 2)``. Both branches
    downsample by 2 and emit ``c2`` channels; an ESE-gated softmax dynamically
    weights conv (semantic) vs. space-to-depth (fine detail) features.

    Args:
        c1: Input channels.
        c2: Output channels.
        k: Conv-branch kernel size (default 3).
        s: Stride (default 2).
        e: Attention bottleneck ratio (default 0.5).
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2, e: float = 0.5):
        super().__init__()
        self.conv_branch = Conv(c1, c2, k, s)          # stride-2 conv branch
        self.spd_conv = Conv(4 * c1, c2, 1, 1)         # 1x1 after space-to-depth
        cm = max(8, int(c2 * e))
        self.compress_conv = Conv(c2, cm, 1, 1)
        self.compress_spd = Conv(c2, cm, 1, 1)
        self.ese = ESE(2 * cm)
        self.weight_conv = nn.Conv2d(2 * cm, 2, kernel_size=3, stride=1, padding=1)
        self.proj = Conv(c2, c2, 1, 1)

    @staticmethod
    def _space_to_depth(x: torch.Tensor) -> torch.Tensor:
        """Interleave 2x2 spatial blocks into the channel dim (no information loss)."""
        return torch.cat(
            [x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_conv = self.conv_branch(x)                   # (B, c2, H/2, W/2)
        x_spd = self.spd_conv(self._space_to_depth(x))  # (B, c2, H/2, W/2)
        # Attention weights over the two branches.
        feats = torch.cat([self.compress_conv(x_conv), self.compress_spd(x_spd)], dim=1)
        w = self.weight_conv(self.ese(feats))           # (B, 2, H/2, W/2)
        w = torch.softmax(w, dim=1)
        fused = w[:, 0:1] * x_conv + w[:, 1:2] * x_spd
        return self.proj(fused)


class LGECA(nn.Module):
    """Local-Global Enhanced Context Attention (YOLO-RD, Eqs. 8-10).

    Channel-preserving and channel-count-agnostic (ECA-style Conv1d over the
    channel descriptor, so it needs no channel argument):

      - Global branch: (GAP + GMP) → Conv1d(k_global) → sigmoid → (B, C, 1, 1)
      - Local branch:  (LAP + LMP to s×s) → Conv1d(k_local) per cell → sigmoid
      - Fuse: ``alpha * up(global) + (1-alpha) * local`` (learnable alpha), then
        upsample to the input resolution and gate the input.
    """

    def __init__(self, k_global: int = 3, k_local: int = 7, local_size: int = 4):
        super().__init__()
        self.local_size = local_size
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.lap = nn.AdaptiveAvgPool2d(local_size)
        self.lmp = nn.AdaptiveMaxPool2d(local_size)
        self.conv_g = nn.Conv1d(1, 1, k_global, padding=k_global // 2, bias=False)
        self.conv_l = nn.Conv1d(1, 1, k_local, padding=k_local // 2, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # Global branch: descriptor (B, C, 1, 1) -> (B, 1, C) -> Conv1d -> back.
        g = self.gap(x) + self.gmp(x)                      # (B, C, 1, 1)
        g = g.view(b, 1, c)                                # (B, 1, C)
        g = self.conv_g(g).view(b, c, 1, 1)
        w_global = torch.sigmoid(g)                        # (B, C, 1, 1)

        # Local branch: per-cell ECA over channels at s×s resolution.
        s = self.local_size
        loc = self.lap(x) + self.lmp(x)                    # (B, C, s, s)
        loc = loc.permute(0, 2, 3, 1).reshape(b * s * s, 1, c)  # (B*s*s, 1, C)
        loc = self.conv_l(loc).reshape(b, s, s, c).permute(0, 3, 1, 2)  # (B, C, s, s)
        w_local = torch.sigmoid(loc)

        # Adaptive fusion + upsample to input resolution, then gate.
        a = self.alpha
        w_global_up = F.interpolate(w_global, size=(s, s), mode="nearest")
        fused = a * w_global_up + (1.0 - a) * w_local      # (B, C, s, s)
        attn = F.interpolate(fused, size=(h, w), mode="nearest")
        return x * attn


class LGECABlock(nn.Module):
    """Wrap an existing layer so its output is refined by LGECA (same channels)."""

    def __init__(self, block: nn.Module, lgeca: LGECA):
        super().__init__()
        self.block = block
        self.lgeca = lgeca

    def forward(self, x):
        return self.lgeca(self.block(x))


def _copy_routing_attrs(dst: nn.Module, src: nn.Module, type_name: str) -> None:
    """Copy Ultralytics forward-routing attributes (.i/.f/.type) onto ``dst``."""
    dst.i = src.i
    dst.f = src.f
    dst.type = type_name


def apply_yolo_rd(
    ultra_model: nn.Module,
    use_csaf: bool = True,
    use_lgeca: bool = True,
) -> nn.Module:
    """Surgically insert CSAF (stem) and LGECA (detect inputs) in place.

    Args:
        ultra_model: The Ultralytics detection ``nn.Module`` (has ``.model``
            ``nn.Sequential`` and a Detect head as the last layer).
        use_csaf: Replace the stem conv (layer 0) with CSAF.
        use_lgeca: Wrap each detection-head input block with LGECA.

    Returns:
        The same model, modified in place.
    """
    seq = ultra_model.model
    device = next(ultra_model.parameters()).device

    if use_csaf:
        stem = seq[0]
        c1 = stem.conv.in_channels
        c2 = stem.conv.out_channels
        csaf = CSAF(c1, c2, k=3, s=2).to(device)
        _copy_routing_attrs(csaf, stem, "yolo_rd.CSAF")
        seq[0] = csaf
        logger.info("CSAF inserted at stem (layer 0): %d -> %d channels, stride 2", c1, c2)

    if use_lgeca:
        detect = seq[-1]
        det_inputs = detect.f  # e.g. [16, 19, 22]
        if not isinstance(det_inputs, (list, tuple)):
            logger.warning("Detect head 'from' is not a list (%s); skipping LGECA.", det_inputs)
            return ultra_model
        for idx in det_inputs:
            block = seq[idx]
            wrapped = LGECABlock(block, LGECA()).to(device)
            _copy_routing_attrs(wrapped, block, "yolo_rd.LGECA")
            seq[idx] = wrapped
        logger.info("LGECA wrapped around detect-input layers %s", list(det_inputs))

    return ultra_model
