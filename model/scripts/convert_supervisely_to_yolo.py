"""Convert Supervisely-format RDD2022 annotations to YOLO format.

Creates a YOLO-compatible dataset structure with:
- images/train/, images/val/
- labels/train/, labels/val/

Uses the existing train/test split from the source dataset:
- train/ folder → images/train + labels/train
- test/ folder  → images/val + labels/val

Each label file contains one line per object:
    class_id x_center y_center width height  (all normalized to [0, 1])

Usage:
    python -m model.scripts.convert_supervisely_to_yolo \
        --src model/data/rdd2022/sample \
        --dst model/data/rdd2022/sample_yolo
"""

import argparse
import json
import shutil
from pathlib import Path

# Class mapping: only the 5 classes used in training
CLASS_NAMES = [
    "alligator crack",
    "longitudinal crack",
    "other corruption",
    "pothole",
    "transverse crack",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def convert_annotation(ann_path: Path, img_w: int, img_h: int) -> list:
    """Convert a single Supervisely JSON annotation to YOLO format lines."""
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = []
    for obj in data.get("objects", []):
        class_title = obj.get("classTitle", "")
        if class_title not in CLASS_TO_ID:
            continue  # Skip classes not in our 5-class set

        class_id = CLASS_TO_ID[class_title]
        exterior = obj["points"]["exterior"]
        x1, y1 = exterior[0]
        x2, y2 = exterior[1]

        # Convert to YOLO format: x_center, y_center, width, height (normalized)
        x_center = ((x1 + x2) / 2.0) / img_w
        y_center = ((y1 + y2) / 2.0) / img_h
        width = abs(x2 - x1) / img_w
        height = abs(y2 - y1) / img_h

        # Clamp to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return lines


def convert_split(src_dir: Path, dst: Path, split_name: str):
    """Convert one split (train or test) from Supervisely to YOLO format."""
    ann_dir = src_dir / "ann"
    img_dir = src_dir / "img"

    if not ann_dir.exists():
        print(f"  Skipping {split_name}: {ann_dir} not found")
        return 0

    ann_files = sorted(ann_dir.glob("*.json"))
    (dst / "images" / split_name).mkdir(parents=True, exist_ok=True)
    (dst / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    for ann_path in ann_files:
        img_name = ann_path.stem  # e.g., "China_Drone_000035.jpg"
        img_path = img_dir / img_name

        if not img_path.exists():
            skipped += 1
            continue

        # Read annotation to get image size
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_w = data["size"]["width"]
        img_h = data["size"]["height"]

        # Convert annotations
        yolo_lines = convert_annotation(ann_path, img_w, img_h)

        # Copy image
        dst_img = dst / "images" / split_name / img_name
        shutil.copy2(img_path, dst_img)

        # Write label file
        label_name = img_name.rsplit(".", 1)[0] + ".txt"
        dst_label = dst / "labels" / split_name / label_name
        with open(dst_label, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))
            if yolo_lines:
                f.write("\n")

        converted += 1

    print(f"  {split_name}: {converted} converted, {skipped} skipped (missing images)")
    return converted


def convert_dataset(src: Path, dst: Path):
    """Convert the full Supervisely dataset to YOLO format.

    Uses train/ as training data and test/ as validation data.
    """
    print(f"Source: {src}")
    print(f"Destination: {dst}")

    # Convert train/ → images/train + labels/train
    train_count = convert_split(src / "train", dst, "train")

    # Convert test/ → images/val + labels/val
    val_count = convert_split(src / "test", dst, "val")

    # Create data.yaml
    data_yaml = dst / "data.yaml"
    yaml_content = f"""# RDD2022 Road Damage Detection Dataset (YOLO format)
# Converted from Supervisely format
# train/ folder used as training, test/ folder used as validation

path: {dst.resolve()}
train: images/train
val: images/val

# Classes
names:
  0: alligator crack
  1: longitudinal crack
  2: other corruption
  3: pothole
  4: transverse crack

nc: 5
"""
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\nDone! Dataset saved to: {dst}")
    print(f"  Train images: {train_count}")
    print(f"  Val images: {val_count}")
    print(f"  data.yaml: {data_yaml}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Supervisely to YOLO format")
    parser.add_argument("--src", type=str, required=True, help="Source Supervisely dataset path")
    parser.add_argument("--dst", type=str, required=True, help="Destination YOLO dataset path")
    args = parser.parse_args()

    convert_dataset(Path(args.src), Path(args.dst))
