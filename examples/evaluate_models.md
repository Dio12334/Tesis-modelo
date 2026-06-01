# Generic Model Evaluation Guide

The `evaluate_detection.py` script now supports **any model** registered in the ModelRegistry (YOLO26, SSD MobileNetV3, YOLOv6, etc.).

## Quick Start (Recommended)

### Evaluate YOLO26 Model

```bash
python -m model.training.evaluate_detection \
    --config model/configs/train_yolo26.yaml \
    --checkpoint checkpoints/yolo26/best_model.pt \
    --split val
```

### Evaluate on Test Set

```bash
python -m model.training.evaluate_detection \
    --config model/configs/train_yolo26.yaml \
    --checkpoint checkpoints/yolo26/best_model.pt \
    --split test
```

### Evaluate SSD MobileNetV3 Model

```bash
python -m model.training.evaluate_detection \
    --config model/configs/train_ssd.yaml \
    --checkpoint checkpoints/ssd_mobilenetv3/best_model.pt \
    --split val
```

## Usage Patterns

### 1. Using Config File (Recommended)

The config file contains all model settings:

```bash
python -m model.training.evaluate_detection \
    --config model/configs/train_yolo26.yaml \
    --checkpoint checkpoints/yolo26/best_model.pt
```

**Benefits:**
- Automatically loads model type, input size, num_classes, thresholds
- Ensures consistency with training configuration
- Less error-prone

### 2. Using Run ID

If you saved checkpoints with a run ID:

```bash
python -m model.training.evaluate_detection \
    --config model/configs/train_yolo26.yaml \
    --run-id a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

The script will look for: `checkpoints/yolo26/<run-id>/best_model.pt`

### 3. Override Thresholds

Override confidence or IoU thresholds from the config:

```bash
python -m model.training.evaluate_detection \
    --config model/configs/train_yolo26.yaml \
    --checkpoint checkpoints/yolo26/best_model.pt \
    --confidence 0.3 \
    --iou 0.6
```

### 4. Legacy Mode (Without Config File)

For backward compatibility, you can specify everything manually:

```bash
python -m model.training.evaluate_detection \
    --checkpoint checkpoints/yolo26/best_model.pt \
    --model-type yolo26 \
    --input-size 640 \
    --num-classes 5 \
    --confidence 0.25 \
    --iou 0.5
```

## Evaluation Splits

### Validation Split (default)
Evaluates on the validation portion of the training data:

```bash
--split val
```

### Training Split
Evaluates on the training data (useful for checking overfitting):

```bash
--split train
```

### Test Split
Evaluates on the separate test set (if available):

```bash
--split test
```

**Note:** The test split loads from the `test/` folder in your dataset, not from splitting the training data.

## Output Files

The script generates two files in the checkpoint directory (or `--output-dir`):

### 1. Evaluation Report (`evaluation_report.json`)

```json
{
  "checkpoint": "checkpoints/yolo26/best_model.pt",
  "model_type": "yolo26",
  "split": "val",
  "num_images": 200,
  "metrics": {
    "mAP@0.5": 0.7234,
    "mAP@0.5:0.95": 0.5123,
    "precision": 0.8012,
    "recall": 0.7456,
    "f1_score": 0.7723,
    "per_class_ap": {
      "D00": 0.8123,
      "D10": 0.7234,
      ...
    }
  },
  "confusion_matrix": [[...], ...]
}
```

### 2. Per-Image Predictions (`validation_inference.json`)

Contains predictions and ground truth for each image, useful for visualization and debugging.

## Supported Models

The script works with any model registered in `ModelRegistry`:

- **yolo26** - YOLO v26 (Ultralytics)
- **yolov6** - YOLO v6
- **ssd_mobilenet** - SSD MobileNetV3

To add support for a new model, simply register it in the ModelRegistry and ensure it implements the `BaseDetector` interface.

## Examples

### Compare Multiple Models

```bash
# Evaluate YOLO26
python -m model.training.evaluate_detection \
    --config model/configs/train_yolo26.yaml \
    --checkpoint checkpoints/yolo26/best_model.pt \
    --output-dir results/yolo26

# Evaluate SSD
python -m model.training.evaluate_detection \
    --config model/configs/train_ssd.yaml \
    --checkpoint checkpoints/ssd_mobilenetv3/best_model.pt \
    --output-dir results/ssd
```

### Evaluate at Different Confidence Thresholds

```bash
for conf in 0.1 0.25 0.5 0.75; do
    python -m model.training.evaluate_detection \
        --config model/configs/train_yolo26.yaml \
        --checkpoint checkpoints/yolo26/best_model.pt \
        --confidence $conf \
        --output-dir results/yolo26_conf_$conf
done
```

## Troubleshooting

### "model_type must be provided"
- Use `--config` to load model type from config file, OR
- Use `--model-type` to specify manually

### "Checkpoint not found"
- Check the checkpoint path exists
- If using `--run-id`, ensure the checkpoint directory structure is correct

### "Missing required parameter"
- Ensure your config file has all required model parameters
- Or provide them manually with `--input-size`, `--num-classes`, etc.

### Boxes not normalized correctly
- The script auto-detects if boxes are in pixel or normalized coordinates
- YOLO models return normalized boxes [0, 1]
- SSD models may return pixel coordinates

## Key Features

✅ **Model-agnostic**: Works with any ModelRegistry model  
✅ **Config-driven**: Load all settings from training config  
✅ **Test set support**: Evaluate on separate test data  
✅ **Flexible thresholds**: Override confidence and IoU  
✅ **Comprehensive metrics**: mAP, precision, recall, F1, confusion matrix  
✅ **Per-image output**: Detailed predictions for visualization
