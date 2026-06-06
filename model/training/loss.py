"""Enhanced detection loss with label smoothing and focal loss support.

Subclasses Ultralytics v8DetectionLoss to inject:
  - Label smoothing: softens classification targets to reduce overconfidence.
  - Focal loss: down-weights easy negatives via (1-p_t)^gamma modulation.

Both features are controlled via hyperparameters on model.args (SimpleNamespace):
  - label_smoothing (float): 0.0 = disabled, 0.01 typical.
  - focal_gamma (float): 0.0 = plain BCE, 1.5 typical.
  - focal_alpha (float): per-class balance factor. -1.0 = disabled, 0.25 typical.

When both are disabled (defaults), the loss is numerically identical to stock
v8DetectionLoss — full backward compatibility.
"""

from typing import Any

import torch
import torch.nn as nn

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


class EnhancedDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss with label smoothing and optional focal loss.

    Drop-in replacement: when label_smoothing=0 and focal_gamma=0, produces
    numerically identical output to the stock v8DetectionLoss.
    """

    def __init__(self, model: "torch.nn.Module", tal_topk: int = 10, tal_topk2=None):
        """Initialize with same signature as v8DetectionLoss.

        Reads additional hyperparameters from model.args:
            - label_smoothing (float)
            - focal_gamma (float)
            - focal_alpha (float)
        """
        super().__init__(model, tal_topk, tal_topk2)
        # Pull enhanced loss params from model hyperparameters
        self.label_smoothing = getattr(self.hyp, "label_smoothing", 0.0)
        self.focal_gamma = getattr(self.hyp, "focal_gamma", 0.0)
        self.focal_alpha = getattr(self.hyp, "focal_alpha", -1.0)

    def get_assigned_targets_and_loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, Any]
    ) -> tuple:
        """Calculate detection loss with label smoothing and focal modulation.

        This is a modified copy of v8DetectionLoss.get_assigned_targets_and_loss
        with two surgical insertions:
          1. Label smoothing applied to target_scores after assignment.
          2. Focal loss modulation applied to bce_loss before summation.

        Returns same tuple format as parent: (assignments, loss, loss_detach).
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # --- MODIFICATION 1: Label smoothing ---
        # Soften targets: [0, 1] -> [ls/2, 1 - ls/2]
        # This reduces overconfidence and improves calibration.
        if self.label_smoothing > 0:
            target_scores = (
                target_scores * (1.0 - self.label_smoothing)
                + 0.5 * self.label_smoothing
            )

        # Cls loss with optional class weighting
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # (bs, num_anchors, nc)
        if self.class_weights is not None:
            bce_loss *= self.class_weights

        # --- MODIFICATION 2: Focal loss modulation ---
        # Down-weight easy examples via (1 - p_t)^gamma factor.
        if self.focal_gamma > 0:
            p = pred_scores.detach().sigmoid()
            # p_t: probability of the "correct" class (target)
            p_t = p * target_scores.to(dtype) + (1.0 - p) * (1.0 - target_scores.to(dtype))
            focal_weight = (1.0 - p_t) ** self.focal_gamma
            # Optional alpha weighting (balances positive vs negative)
            if self.focal_alpha >= 0:
                alpha_t = (
                    self.focal_alpha * target_scores.to(dtype)
                    + (1.0 - self.focal_alpha) * (1.0 - target_scores.to(dtype))
                )
                focal_weight = focal_weight * alpha_t
            bce_loss = bce_loss * focal_weight

        loss[1] = bce_loss.sum() / target_scores_sum  # BCE (with focal if enabled)

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )  # loss(box, cls, dfl)


class EnhancedE2EDetectLoss:
    """Drop-in replacement for E2EDetectLoss using EnhancedDetectionLoss.

    E2EDetectLoss wraps two v8DetectionLoss instances (one-to-many with topk=10,
    one-to-one with topk=1). This class does the same but uses our enhanced
    subclass so label smoothing and focal loss are applied to both branches.
    """

    def __init__(self, model: "torch.nn.Module"):
        """Initialize with EnhancedDetectionLoss for both branches."""
        self.one2many = EnhancedDetectionLoss(model, tal_topk=10)
        self.one2one = EnhancedDetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        """Calculate loss for both one2many and one2one branches."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]
