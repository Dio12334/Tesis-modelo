# RT-DETR Loss Configuration

## Overview

RT-DETR uses `RTDETRDetectionLoss` (from `ultralytics.models.utils.loss`), which combines:

- **Hungarian matching** — assigns each prediction to the closest ground-truth box
- **Focal loss** on class logits — downweights easy examples to focus on hard ones
- **L1 + GIoU loss** on bounding box regression
- **No-object loss** — penalizes predicting boxes where there is nothing

All parameters are configurable from `training.loss` in `train_rt_detr.yaml` and are
applied inside `RT_DETR_Detector._build_loss_fn()` in [model/models/rt_detr_wrapper.py](../model/models/rt_detr_wrapper.py).

---

## Configuration Reference

```yaml
training:
  loss:
    focal_loss: true          # enable focal loss on classification head
    focal_gamma: 2.0          # focusing exponent  (paper default: 2.0)
    focal_alpha: 0.25         # class balance factor (paper default: 0.25)
    no_object_weight: 0.1     # penalty weight for predicting boxes on empty regions
    class_weight: 1.0         # classification loss weight
    bbox_weight: 5.0          # L1 bounding box regression weight
    giou_weight: 2.0          # GIoU loss weight
```

### Parameter effects

| Parameter | Default | Effect when increased | Effect when decreased |
|-----------|---------|----------------------|----------------------|
| `focal_gamma` | 1.5 | More focus on hard, misclassified examples | Closer to standard cross-entropy |
| `focal_alpha` | 0.25 | More recall (fewer missed GT boxes) | More precision (fewer false positives) |
| `no_object_weight` | 0.1 | Model penalized more for predicting boxes in empty regions → fewer FPs | Model more permissive about empty regions → more FPs |
| `class_weight` | 1.0 | Stronger class discrimination signal | — |
| `bbox_weight` | 5.0 | Tighter box regression | — |
| `giou_weight` | 2.0 | Better IoU alignment between pred and GT | — |

### `focal_gamma` in detail

```
focal_loss = -(1 - p)^gamma * log(p)
```

| gamma | Behavior |
|-------|----------|
| 0 | Standard cross-entropy (no focusing) |
| 1.5 | Moderate focus on hard examples (previous default) |
| 2.0 | Original RetinaNet/focal loss paper value — recommended |
| 3.0+ | Aggressive — most loss concentrated on hard examples |

### `no_object_weight` in detail

This weight controls how much the model is penalized when it predicts a bounding box
in a region where there is no ground-truth object. It is the key lever for
**reducing false positives** during training.

| `no_object_weight` | FP tendency | FN tendency |
|-------------------|-------------|-------------|
| 0.1 (default) | High — model is cheap to predict extra boxes | Low |
| 0.3 | Moderate — good starting point for RDD2022 | Moderate |
| 0.5 | Low — model is conservative about predictions | Higher |
| 1.0 | Very low — model rarely predicts without strong evidence | High |

**Current setting:** `0.3` — raises the penalty 3x over default while still allowing
the model to detect subtle damage patterns.

---

## What does NOT work

`label_smoothing` is present in the YAML for historical reasons but is **not a parameter
of `RTDETRDetectionLoss`** — it is silently ignored. The comment in the YAML documents this.

YOLO-style loss weight keys (`box`, `cls`, `dfl`) are also ignored — they apply to YOLO's
`DetectionLoss`, not RT-DETR's Hungarian matching loss.

---

## How it is applied

`train_detection.py` reads `training.loss` from the YAML and forwards it as
`model_cfg["loss"]` to `RT_DETR_Detector.__init__()`, which stores it in `self.config`.
`_build_loss_fn()` reads `self.config["loss"]` and instantiates `RTDETRDetectionLoss`
with the configured parameters:

```python
# rt_detr_wrapper.py — _build_loss_fn()
loss_cfg = self.config.get("loss", {})
loss_gain = dict(self._DEFAULT_LOSS_GAIN)
loss_gain["no_object"] = float(loss_cfg.get("no_object_weight", 0.1))
# ... other weights ...
self._loss_fn = RTDETRDetectionLoss(
    nc=self.num_classes,
    loss_gain=loss_gain,
    gamma=float(loss_cfg.get("focal_gamma", 1.5)),
    alpha=float(loss_cfg.get("focal_alpha", 0.25)),
    use_fl=bool(loss_cfg.get("focal_loss", True)),
)
```

The actual parameter values are logged at `INFO` level at the start of training:
```
INFO: RTDETRDetectionLoss: no_object=0.30 class=1.0 bbox=5.0 giou=2.0 focal_gamma=2.00 focal_alpha=0.20 use_fl=True
```

---

## Tuning strategy

To reduce false positives without retraining from scratch, raise `no_object_weight`
and `focal_gamma` together. Start with:

```yaml
loss:
  focal_gamma: 2.0
  focal_alpha: 0.20
  no_object_weight: 0.3
```

If FPs are still too high after a training run, increase `no_object_weight` to `0.5`.
If recall drops too much (too many GT boxes missed), lower back to `0.2`.

For deployment without retraining, raise `confidence_threshold` in the model config —
this filters low-confidence predictions at inference time and does not require retraining.
