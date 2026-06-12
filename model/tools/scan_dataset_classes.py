"""Read-only diagnostic: enumerate raw classTitle labels across RDD2022 splits.

Walks every Supervisely JSON annotation under
``model/data/rdd2022/complete/{train,test}/ann/`` (and ``val/ann`` if present),
counts each distinct raw ``classTitle`` value, and reports per-split breakdown
plus a cross-check against ``model/configs/rdd2022_classes.yaml``.

This tool only reads files; it never writes, fixes, or modifies anything. It
exists so we can confidently flip the silent ``_class_to_idx.get(label, 0)``
fallback in ``RDD2022TorchDataset`` to a hard error without surprises.

Usage:
    python -m model.tools.scan_dataset_classes
    python -m model.tools.scan_dataset_classes --root path/to/rdd2022/complete
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import yaml


DEFAULT_ROOT = Path("model/data/rdd2022/complete")
CLASS_MAPPING_YAML = Path("model/configs/rdd2022_classes.yaml")


def _scan_split(ann_dir: Path) -> Counter:
    """Count distinct ``classTitle`` values in every JSON under ``ann_dir``.

    Malformed JSON files are recorded under the sentinel key ``"<malformed>"``
    so the caller can see how many we skipped.
    """
    counts: Counter = Counter()
    json_files = sorted(ann_dir.glob("*.json"))
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            counts["<malformed>"] += 1
            continue

        objects = data.get("objects", [])
        if not objects:
            counts["<empty_annotation>"] += 1
            continue

        for obj in objects:
            geom = obj.get("geometryType", "")
            if geom != "rectangle":
                counts[f"<non_rect:{geom or 'unspecified'}>"] += 1
                continue
            title = obj.get("classTitle", "")
            if not title:
                counts["<missing_classTitle>"] += 1
                continue
            counts[title] += 1
    return counts


def _load_taxonomy(yaml_path: Path) -> Dict[str, object]:
    if not yaml_path.exists():
        return {}
    with open(yaml_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _format_table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out_lines = []
    for ri, row in enumerate(rows):
        line = "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        out_lines.append(line)
        if ri == 0:
            out_lines.append("  ".join("-" * widths[i] for i in range(len(row))))
    return "\n".join(out_lines)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Dataset root containing train/ann etc. (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=CLASS_MAPPING_YAML,
        help=f"Class mapping YAML (default: {CLASS_MAPPING_YAML})",
    )
    args = parser.parse_args(argv)

    root: Path = args.root
    if not root.exists():
        print(f"ERROR: dataset root does not exist: {root}", file=sys.stderr)
        return 2

    splits = ["train", "val", "test"]
    per_split: Dict[str, Counter] = {}
    for split in splits:
        ann_dir = root / split / "ann"
        if not ann_dir.exists():
            continue
        per_split[split] = _scan_split(ann_dir)

    if not per_split:
        print(f"ERROR: no Supervisely ann/ subdirectory found under {root}", file=sys.stderr)
        return 2

    # --- Aggregate ---------------------------------------------------------
    union_keys = sorted({k for c in per_split.values() for k in c})
    real_labels = [k for k in union_keys if not k.startswith("<")]
    sentinel_keys = [k for k in union_keys if k.startswith("<")]

    print("=" * 72)
    print("RDD2022 raw-classTitle diagnostic scan (READ-ONLY)")
    print("=" * 72)
    print(f"Root: {root.resolve()}")
    print(f"Splits scanned: {', '.join(per_split.keys())}")
    print()

    # Real-label table
    header = ["raw classTitle"] + list(per_split.keys()) + ["TOTAL"]
    rows: List[List[str]] = [header]
    totals_per_split = {s: 0 for s in per_split}
    for label in real_labels:
        cells = [label]
        row_total = 0
        for s in per_split:
            v = per_split[s].get(label, 0)
            cells.append(str(v))
            totals_per_split[s] += v
            row_total += v
        cells.append(str(row_total))
        rows.append(cells)
    rows.append(
        ["TOTAL_OBJECTS"]
        + [str(totals_per_split[s]) for s in per_split]
        + [str(sum(totals_per_split.values()))]
    )
    print("Distinct raw classTitle values (only real labels):")
    print(_format_table(rows))
    print()

    # Sentinels
    if sentinel_keys:
        print("Anomalies:")
        srows: List[List[str]] = [["sentinel"] + list(per_split.keys()) + ["TOTAL"]]
        for label in sentinel_keys:
            cells = [label]
            row_total = 0
            for s in per_split:
                v = per_split[s].get(label, 0)
                cells.append(str(v))
                row_total += v
            cells.append(str(row_total))
            srows.append(cells)
        print(_format_table(srows))
        print()

    # --- Cross-check vs YAML ----------------------------------------------
    cfg = _load_taxonomy(args.mapping)
    taxonomy: List[str] = list(cfg.get("taxonomy", []) or [])
    mappings: Dict[str, str] = dict(cfg.get("mappings", {}) or {})
    default_class = cfg.get("default_class", None)

    print(f"Class mapping YAML: {args.mapping}")
    print(f"  taxonomy ({len(taxonomy)} entries): {taxonomy}")
    print(f"  mappings ({len(mappings)} entries) keys: {sorted(mappings.keys())}")
    print(f"  default_class: {default_class!r}")
    print()

    # raw labels NOT covered by mappings
    unmapped = [r for r in real_labels if r not in mappings]
    # mapping keys not seen in data
    unused_keys = [k for k in mappings if k not in real_labels]
    # mapping targets not in taxonomy
    bad_targets = sorted({v for v in mappings.values() if v not in taxonomy})

    print("Cross-check:")
    print(f"  raw labels seen in data:     {len(real_labels)}")
    print(f"  raw labels NOT in mappings:  {len(unmapped)} -> {unmapped}")
    print(f"  mapping keys NOT seen:       {len(unused_keys)} -> {unused_keys}")
    print(f"  mapping targets NOT in taxonomy: {len(bad_targets)} -> {bad_targets}")
    print()

    # What RDD2022Dataset.get_class_names() will produce in train_detection.py
    # (sorted distinct raw classTitle observed across the chosen subset)
    train_real_labels = sorted(
        k for k in per_split.get("train", Counter()) if not k.startswith("<")
    )
    print("RDD2022TorchDataset._class_to_idx (TRAIN split, sorted distinct raw labels):")
    for i, name in enumerate(train_real_labels):
        print(f"  {i:>2}: {name}")
    print()
    print(
        f"=> If train configs declare num_classes=5 but len(distinct raw labels)="
        f"{len(train_real_labels)}, indices >= 5 fall back to 0 silently."
    )
    print()

    # --- Final verdict -----------------------------------------------------
    issues: List[str] = []
    if unmapped:
        issues.append(
            f"{len(unmapped)} raw label(s) have no mapping entry: {unmapped}"
        )
    if bad_targets:
        issues.append(
            f"mapping targets reference unknown taxonomy entries: {bad_targets}"
        )
    if len(train_real_labels) != len(taxonomy):
        issues.append(
            f"distinct raw labels in train ({len(train_real_labels)}) != "
            f"taxonomy size ({len(taxonomy)})"
        )

    if issues:
        print("VERDICT: ISSUES FOUND")
        for it in issues:
            print(f"  - {it}")
        return 1
    print("VERDICT: dataset and YAML are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
