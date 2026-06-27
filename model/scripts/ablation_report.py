"""Aggregate evaluation reports into a markdown ablation table.

Reads the ``val_evaluation_report.json`` produced by ``evaluate_detection`` for a
set of runs (declared in a manifest), and emits a thesis-ready markdown report:
overall metrics (with Δ within each base architecture), per-class AP@0.5, and
per-region mAP@0.5. Optionally counts each model's parameters.

Usage:
    python -m model.scripts.ablation_report --manifest model/configs/ablation_manifest.yaml
    python -m model.scripts.ablation_report --manifest <m>.yaml --out report.md --with-params

Manifest format (YAML):
    checkpoint_dir: ./checkpoints        # optional, default ./checkpoints
    runs:
      - label: "YOLO26-S baseline"
        base: YOLO26-S                   # runs are grouped by base; first per
        run_id: f77e9617-...             # group is treated as that group's baseline
      - label: "YOLO26-S + full"
        base: YOLO26-S
        run_id: cbdea0cd-...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def _report_path(checkpoint_dir: Path, run_id: str, model_type: str = "yolo26") -> Path:
    return checkpoint_dir / model_type / run_id / "val_evaluation_report.json"


def _load_report(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _fmt(x: Optional[float], nd: int = 4) -> str:
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def _signed(x: Optional[float], nd: int = 4) -> str:
    if not isinstance(x, (int, float)):
        return "—"
    return f"{x:+.{nd}f}"


def _collect(run: dict, checkpoint_dir: Path) -> dict:
    """Resolve a manifest run to its report + extracted metrics."""
    out = {
        "label": run.get("label", run.get("run_id", "?")),
        "base": run.get("base", ""),
        "run_id": run.get("run_id"),
        "report": None,
    }
    rid = run.get("run_id")
    if not rid:
        return out
    path = _report_path(checkpoint_dir, str(rid))
    report = _load_report(path)
    if report is None:
        print(f"WARNING: report not found for '{out['label']}' at {path}", file=sys.stderr)
        return out
    m = report.get("metrics", {})
    out["report"] = report
    out["map50"] = m.get("map_50")
    out["map5095"] = m.get("map_50_95")
    out["precision"] = m.get("precision")
    out["recall"] = m.get("recall")
    out["f1"] = m.get("f1_score")
    out["per_class_ap"] = m.get("per_class_ap", {})
    out["per_region"] = {
        r: d.get("metrics", {}).get("map_50")
        for r, d in (report.get("per_region", {}) or {}).items()
    }
    out["model_config"] = report.get("model_config", {})
    return out


def _baseline_map(rows: List[dict]) -> Dict[str, float]:
    """First run of each base group is its baseline; return base -> baseline mAP."""
    base_map: Dict[str, float] = {}
    for r in rows:
        b = r.get("base", "")
        if b not in base_map and isinstance(r.get("map50"), (int, float)):
            base_map[b] = r["map50"]
    return base_map


def _main_table(rows: List[dict], with_params: bool) -> str:
    base_map = _baseline_map(rows)
    header = ["Model", "Base", "mAP@0.5", "Δ mAP@0.5", "mAP@0.5:0.95", "P", "R", "F1"]
    if with_params:
        header.append("Params (M)")
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        base_b = base_map.get(r.get("base", ""))
        delta = (r["map50"] - base_b) if isinstance(r.get("map50"), (int, float)) and isinstance(base_b, (int, float)) else None
        cells = [
            r["label"], r.get("base", ""),
            _fmt(r.get("map50")), _signed(delta),
            _fmt(r.get("map5095")), _fmt(r.get("precision"), 3),
            _fmt(r.get("recall"), 3), _fmt(r.get("f1"), 3),
        ]
        if with_params:
            cells.append(_fmt(r.get("params_m"), 2))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _per_class_table(rows: List[dict]) -> str:
    classes: List[str] = []
    for r in rows:
        for c in (r.get("per_class_ap") or {}):
            if c not in classes:
                classes.append(c)
    if not classes:
        return "_(no per-class data)_"
    header = ["Class"] + [r["label"] for r in rows]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for c in classes:
        cells = [c] + [_fmt((r.get("per_class_ap") or {}).get(c), 3) for r in rows]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _per_region_table(rows: List[dict]) -> str:
    regions: List[str] = []
    for r in rows:
        for reg in (r.get("per_region") or {}):
            if reg not in regions:
                regions.append(reg)
    if not regions:
        return "_(no per-region data)_"
    header = ["Region"] + [r["label"] for r in rows]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for reg in regions:
        cells = [reg] + [_fmt((r.get("per_region") or {}).get(reg), 3) for r in rows]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _add_params(rows: List[dict]) -> None:
    """Build each model from its config and count parameters (heavy: loads weights)."""
    from model.models.yolo26_wrapper import YOLO26Detector
    for r in rows:
        cfg = r.get("model_config")
        if not cfg:
            continue
        try:
            m = YOLO26Detector(dict(cfg))
            r["params_m"] = sum(p.numel() for p in m._model.model.parameters()) / 1e6
        except Exception as e:  # pragma: no cover
            print(f"WARNING: could not count params for '{r['label']}': {e}", file=sys.stderr)


def build_report(manifest: dict, with_params: bool = False) -> str:
    checkpoint_dir = Path(manifest.get("checkpoint_dir", "./checkpoints"))
    rows = [_collect(run, checkpoint_dir) for run in manifest.get("runs", [])]
    rows = [r for r in rows if r.get("report") is not None]
    if not rows:
        return "No reports found. Check run IDs / checkpoint_dir in the manifest."
    if with_params:
        _add_params(rows)

    out = ["# YOLO-RD ablation report", "",
           "_4-country val split. mAP@0.5 best checkpoint (EMA weights), conf 0.25._",
           "", "## Overall metrics", "",
           _main_table(rows, with_params),
           "", "_Δ mAP@0.5 is vs the first run of each base architecture._",
           "", "## Per-class AP@0.5", "",
           _per_class_table(rows),
           "", "## Per-region mAP@0.5", "",
           _per_region_table(rows), ""]
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path, help="Path to the manifest YAML.")
    ap.add_argument("--out", type=Path, default=None, help="Write markdown here (else stdout).")
    ap.add_argument("--with-params", action="store_true",
                    help="Build each model to count parameters (downloads weights; slow).")
    args = ap.parse_args(argv)

    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh) or {}

    md = build_report(manifest, with_params=args.with_params)
    if args.out:
        args.out.write_text(md, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
