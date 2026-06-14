# Two-Stage Detection Pipeline — Technical Reference

> For high-level motivation and usage overview see [`docs/bimodelo.md`](bimodelo.md).
> This document covers the internal mechanics: data flow, tensor shapes, preprocessing
> contracts, and configuration.

---

## Architecture at a Glance

```
Input batch (N images)
        │
        ▼
┌─────────────────────────────────────────────┐
│  Stage 1: BinaryDamageClassifier            │
│                                             │
│  1. Resize  →  (N, 3, H_cls, W_cls)         │
│  2. Normalize  (ImageNet mean/std)          │
│  3. EfficientNet-B* → logit (N,)            │
│  4. Sigmoid → p_damage ∈ [0, 1]            │
│  5. p >= threshold  →  damage_mask (N,)     │
└─────────────────────────────────────────────┘
        │
        ├─── bg_idx  →  empty predictions { boxes:(0,4), labels:(0,), scores:(0,) }
        │
        └─── damage_idx (subset)
                  │
                  ▼
        ┌──────────────────────────────────────┐
        │  Stage 2: Object Detector (RT-DETR)  │
        │                                      │
        │  images[damage_idx] → detector.forward()
        │  → List[dict] of bboxes             │
        └──────────────────────────────────────┘
                  │
                  ▼
        Re-insert at original batch positions
        │
        ▼
Output: List[dict] of length N
        (background slots have empty tensors)
```

---

## Data Flow in Detail

### Input contract

`evaluate_detection.py` preprocesses each image before calling `forward()`:

```python
transform = T.Compose([
    T.Resize((input_size, input_size)),   # default: 640×640 for RT-DETR
    T.ToTensor(),                          # PIL → float32 [0, 1], shape (3, H, W)
])
```

Images arrive at `TwoStageDetector.forward(images)` as:
```
images: torch.Tensor  shape (N, 3, 640, 640)  dtype float32  range [0, 1]
```

No channel normalization is applied by the evaluation pipeline — the detector
receives raw [0, 1] tensors and handles its own internal normalization.

### Stage 1 preprocessing (inside `TwoStageDetector.forward`)

The classifier was trained at a different resolution with ImageNet normalization.
This mismatch is corrected inline before calling the classifier:

```python
# Resize to classifier's training resolution
cls_images = F.interpolate(
    images,
    size=(self._classifier_input_size, self._classifier_input_size),
    mode="bilinear", align_corners=False,
)

# ImageNet normalization  (same constants as BinaryClassificationDataset)
mean = _CLS_MEAN.to(device)   # [[[0.485]], [[0.456]], [[0.406]]]  shape (1,3,1,1)
std  = _CLS_STD.to(device)    # [[[0.229]], [[0.224]], [[0.225]]]  shape (1,3,1,1)
cls_images = (cls_images - mean) / std
```

**Why this is correct:** `ToTensor()` already scales pixel values to [0, 1], so
`(pixel_value - mean) / std` is the standard ImageNet preprocessing formula.

Tensor shape after preprocessing:
```
cls_images: (N, 3, classifier_input_size, classifier_input_size)   normalized
```

### Stage 1 decision

```python
_, damage_mask = self._classifier.predict(cls_images)
# damage_mask: BoolTensor (N,)
# True  → image contains road damage  →  goes to detector
# False → background image            →  gets empty prediction
```

Indices are split:
```python
damage_idx = damage_mask.nonzero(as_tuple=True)[0].tolist()   # subset for detector
bg_idx     = (~damage_mask).nonzero(as_tuple=True)[0].tolist() # gets empty preds
```

**Edge cases handled correctly:**
- All background (`damage_idx == []`): `if damage_idx:` check skips detector entirely.
- All damage (`bg_idx == []`): empty-dict loop runs zero times; detector processes full batch.

### Stage 2 inference

```python
damaged_batch = images[damage_idx]          # original 640×640 [0,1] tensors — NOT cls_images
detections = self._detector.forward(damaged_batch)
```

The detector receives the **original** (unmodified) images, not the classifier-preprocessed
ones. Results are re-inserted at their correct positions:

```python
for slot, det in zip(damage_idx, detections):
    results[slot] = det
```

### Output format

Each element of the returned list is a dict:

```python
{
    "boxes":  torch.Tensor  # shape (K, 4)  normalized [x1,y1,x2,y2] in [0,1]
    "labels": torch.Tensor  # shape (K,)    int64 class indices
    "scores": torch.Tensor  # shape (K,)    float32 confidence scores
}
```

For background images K=0 (zero-row tensors, not None).

---

## Preprocessing Contract

**Critical invariant:** `classifier_input_size` in `two_stage.yaml` **must match**
the `input_size` used when the classifier checkpoint was trained.

| Checkpoint | Architecture | Training input_size | two_stage.yaml value | Status |
|------------|-------------|--------------------|-----------------------|--------|
| `9b750f78` | EfficientNet-B0 | 224 | `classifier_input_size: 224` | superseded |
| `8fa192b0` | EfficientNet-B2 | 260 | `classifier_input_size: 260` | **active** |

Mismatching these values causes silent distribution shift and degrades classifier performance.

---

## Configuration Reference (`two_stage.yaml`)

```yaml
model:
  type: two_stage          # registers TwoStageDetector via ModelRegistry

  config:
    # ── Stage 1 ──────────────────────────────────────────────────────────
    classifier_checkpoint:  ./checkpoints/binary_classifier/<uuid>/best_model.pt
    classifier_architecture: efficientnet_b2    # timm model name — must match checkpoint
    classifier_input_size:  260                 # MUST match training input_size
    classifier_threshold:   0.15               # sigmoid threshold for damage decision
                                                # lower → more recall, more detector calls
                                                # higher → more precision, fewer calls

    # ── Stage 2 ──────────────────────────────────────────────────────────
    detector_type:       rt_detr               # any ModelRegistry-registered model
    detector_checkpoint: ./checkpoints/rt_detr/<uuid>/best_model.pt
    num_classes:         5                     # required by evaluate_detection validation
    detector_config:                           # forwarded verbatim to ModelRegistry.create()
      model_size: "l"
      num_classes: 5
      confidence_threshold: 0.25
      iou_threshold: 0.7

dataset:
  type: rdd2022
  path: model/data/rdd2022/
  class_mapping: model/configs/rdd2022_classes.yaml

evaluation:
  split: val
  confidence_threshold: 0.25
  iou_thresholds: [0.25, 0.5, 0.75]
```

### Threshold tuning guide

| `classifier_threshold` | Stage-1 recall | Stage-1 FN risk | Detector load |
|------------------------|---------------|-----------------|---------------|
| 0.05 – 0.15 | Very high (>91%) | Very low | ~90% of images |
| 0.20 – 0.35 | High (88–91%) | Low | ~75% of images |
| 0.40 – 0.50 | Moderate (87%) | Moderate | ~65% of images |
| 0.60 – 0.80 | Lower (<85%) | High | ~50% of images |

For road monitoring (no missed damage acceptable): use **0.10–0.20**.
For throughput-critical deployment: use **0.40–0.50**.

---

## Training Each Component

### Train the binary classifier

```bash
conda activate tesis-modelo
python -u -m model.cli --verbose train-classifier \
    --config model/configs/train_binary_classifier.yaml
```

Key training details:
- Labels derived automatically: images with ≥1 bbox → damage (1), else background (0)
- `WeightedRandomSampler` balances the dataset to 50/50 per batch
- `BCEWithLogitsLoss(pos_weight=2.0)` biases the loss further toward recall
- Primary metric: AUC-ROC (not loss) — best checkpoint saved on AUC improvement
- Early stopping via `early_stopping_patience` epochs without AUC improvement

Checkpoint output:
```
checkpoints/binary_classifier/<uuid>/best_model.pt    ← used in two_stage.yaml
checkpoints/binary_classifier/<uuid>/last_model.pt
checkpoints/binary_classifier/<uuid>.json             ← ExperimentTracker run file
```

### Train the detector (independently)

```bash
python -u -m model.cli --verbose train \
    --config model/configs/train_rt_detr.yaml
```

Checkpoint output:
```
checkpoints/rt_detr/<uuid>/best_model.pt
checkpoints/rt_detr/<uuid>.json
```

---

## Evaluation

```bash
python -m model.training.evaluate_detection \
    --config model/configs/two_stage.yaml \
    --checkpoint checkpoints/rt_detr/<uuid>/best_model.pt
```

The `--checkpoint` argument determines where the evaluation report is saved
(`checkpoints/rt_detr/<uuid>/val_evaluation_report.json`).  
`TwoStageDetector.load_checkpoint()` is a documented no-op — both checkpoints
are already loaded from the config in `__init__`.

The report is then visible in the dashboard under the corresponding RT-DETR run.

---

## Current Metrics

### Stage 1 — BinaryDamageClassifier `9b750f78` (EfficientNet-B0, threshold=0.15) — superseded

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.9346 |
| Precision | 0.8861 |
| Recall (damage) | 0.9157 |
| F1 | 0.9007 |
| False Negatives (val) | 663 / 5174 damage images |
| False Positives (val) | 399 / 2503 background images |

### Stage 1 — BinaryDamageClassifier `8fa192b0` (EfficientNet-B2, threshold=0.15) — **active**

> Pending evaluation. Run `evaluate-classifier` to populate this table.

### Two-Stage System `8fa192b0` + `2b46b55e` (val split, 7677 images) — **active**

#### Detection metrics (B2 classifier, preprocessing fixed)

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.5557 |
| mAP@0.5:0.95 | — |
| Precision | — |
| Recall | — |
| F1 | 0.4474 |

#### Filter statistics (from `val_inference.json`)

| Stat | Value |
|------|-------|
| Total val images | 7677 |
| Background images (no GT) | 2503 (32.6%) |
| Damage images (has GT) | 5174 (67.4%) |
| Images filtered (0 preds) | 2272 (29.6%) |
| Background correctly filtered | 1973 / 2503 (78.8%) |
| Damage images missed entirely | 299 / 5174 (5.8%) |

The small mAP improvement over standalone RT-DETR is expected: correctly filtering
background images removes no GT boxes from the metric computation (they had none),
so the gain comes only from eliminating detector hallucinations on background images.

#### Historical (B0 + `2b46b55e`, with preprocessing bug — classifier received 640×640 non-normalized):

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.5454 |
| mAP@0.5:0.95 | 0.2843 |
| Precision | 0.3230 |
| Recall | 0.7020 |
| F1 | 0.4424 |

---

## Updating Checkpoints

When a better classifier or detector is trained, update `two_stage.yaml`:

```yaml
model:
  config:
    # Update classifier
    classifier_checkpoint:   ./checkpoints/binary_classifier/<new_uuid>/best_model.pt
    classifier_architecture: efficientnet_b2    # if architecture changed
    classifier_input_size:   260                # must match new training input_size

    # Update detector
    detector_checkpoint: ./checkpoints/rt_detr/<new_uuid>/best_model.pt
```

Then re-run evaluation to get updated metrics in the dashboard.

---

## Implementation Files

| File | Role |
|------|------|
| `model/models/binary_classifier.py` | `BinaryDamageClassifier` — EfficientNet wrapper |
| `model/models/two_stage_detector.py` | `TwoStageDetector` — cascade orchestration |
| `model/training/train_classifier.py` | Classifier training loop + `evaluate_classifier()` |
| `model/configs/train_binary_classifier.yaml` | Classifier hyperparameters |
| `model/configs/two_stage.yaml` | Inference config for the full pipeline |
| `model/training/evaluate_detection.py` | Generic evaluation (supports `two_stage` via ModelRegistry) |
