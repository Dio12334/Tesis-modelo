"""Backfill the ``per_region`` field in existing evaluation reports.

For every ``<split>_evaluation_report.json`` under a checkpoints directory
that has a sibling ``<split>_inference.json``, recompute the per-region
metric blocks (using :func:`model.evaluation.region.compute_per_region_metrics`)
and patch the report file in place by adding a top-level ``per_region`` key.

Existing reports are upgraded so the dashboard can read per-region data
without re-running evaluation. The CLI is idempotent: running it again on
an already-upgraded report is a no-op unless ``--force`` is supplied.

Usage::

    python -m model.scripts.backfill_per_region --checkpoints-dir checkpoints
    python -m model.scripts.backfill_per_region --run-id <uuid>
    python -m model.scripts.backfill_per_region --split val --dry-run
    python -m model.scripts.backfill_per_region --force

The script reads the inference dump, converts it to the (predictions,
ground_truths) shape expected by the metric helpers, and writes the patched
report back atomically (temp-file + rename).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from model.evaluation.region import compute_per_region_metrics

logger = logging.getLogger(__name__)


SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Inference-JSON -> evaluator input shape
# ---------------------------------------------------------------------------


def _predictions_from_inference(inference_data: dict) -> Tuple[List[dict], List[dict]]:
    """Convert an ``<split>_inference.json`` payload to evaluator inputs.

    The on-disk shape is::

        {
            "images": [
                {
                    "image_id": str,
                    "ground_truth": {"boxes": [...], "labels": [...]},
                    "predictions": {"boxes": [...], "labels": [...], "scores": [...]},
                },
                ...
            ],
            ...
        }

    We split each entry into two aligned lists matching the ``predictions``
    and ``ground_truths`` arguments of
    :mod:`model.evaluation.metrics` collaborators.
    """
    predictions: List[dict] = []
    ground_truths: List[dict] = []
    for img in inference_data.get("images") or []:
        image_id = img.get("image_id", "")
        gt = img.get("ground_truth") or {}
        pred = img.get("predictions") or {}
        predictions.append(
            {
                "image_id": image_id,
                "boxes": list(pred.get("boxes") or []),
                "labels": list(pred.get("labels") or []),
                "scores": list(pred.get("scores") or []),
            }
        )
        ground_truths.append(
            {
                "image_id": image_id,
                "boxes": list(gt.get("boxes") or []),
                "labels": list(gt.get("labels") or []),
            }
        )
    return predictions, ground_truths


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` to ``path`` atomically (temp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _discover_report_pairs(
    checkpoints_dir: Path,
    *,
    run_id: Optional[str] = None,
    split: Optional[str] = None,
) -> List[Tuple[Path, Path, str]]:
    """Find ``(report_path, inference_path, split)`` triples to backfill.

    Walks ``checkpoints_dir`` looking for ``<split>_evaluation_report.json``
    files that have a sibling ``<split>_inference.json``. Optionally filters
    by ``run_id`` (the parent directory name, expected to be a UUID) and/or
    ``split``.
    """
    if not checkpoints_dir.exists():
        return []

    splits: Iterable[str] = (split,) if split else SPLITS
    pairs: List[Tuple[Path, Path, str]] = []
    for s in splits:
        report_glob = f"{s}_evaluation_report.json"
        inference_name = f"{s}_inference.json"
        for report_path in checkpoints_dir.rglob(report_glob):
            if run_id is not None and report_path.parent.name != run_id:
                continue
            inference_path = report_path.parent / inference_name
            if not inference_path.exists():
                logger.warning(
                    "Skipping %s: sibling %s not found",
                    report_path,
                    inference_name,
                )
                continue
            pairs.append((report_path, inference_path, s))
    return pairs


# ---------------------------------------------------------------------------
# Per-file backfill
# ---------------------------------------------------------------------------


def backfill_one(
    report_path: Path,
    inference_path: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> str:
    """Backfill ``per_region`` for a single report.

    Args:
        report_path: ``<split>_evaluation_report.json`` to patch.
        inference_path: Sibling ``<split>_inference.json`` providing per-image
            predictions and ground truths.
        force: Recompute even when ``per_region`` already exists.
        dry_run: When ``True``, no file is written; the function returns the
            action it would have taken.

    Returns:
        One of ``"skipped"``, ``"updated"``, or ``"would-update"`` (dry run).
    """
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    if "per_region" in report and not force:
        logger.info(
            "Skipping %s (per_region already present; use --force to recompute)",
            report_path,
        )
        return "skipped"

    with open(inference_path, "r", encoding="utf-8") as f:
        inference_data = json.load(f)

    predictions, ground_truths = _predictions_from_inference(inference_data)

    class_names = report.get("class_names") or inference_data.get("class_names") or []
    confidence_threshold = float(report.get("confidence_threshold", 0.25))
    iou_threshold = float(report.get("iou_threshold", 0.5))

    # Source the sweep thresholds from the existing report so backfilled
    # blocks carry the same f1_sweep granularity as the global metrics they
    # sit alongside. When the report has no sweep (legacy pre-Phase-2 file)
    # we omit the sweep fields entirely.
    sweep_thresholds: Optional[List[float]] = None
    metrics = report.get("metrics") or {}
    f1_sweep = metrics.get("f1_sweep")
    if isinstance(f1_sweep, list) and f1_sweep:
        sweep_thresholds = [float(entry["confidence"]) for entry in f1_sweep]

    per_region = compute_per_region_metrics(
        predictions=predictions,
        ground_truths=ground_truths,
        class_names=list(class_names),
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        confidence_thresholds_sweep=sweep_thresholds,
    )

    if not per_region:
        logger.warning(
            "%s: no images matched the RDD2022 region pattern; "
            "nothing to backfill",
            report_path,
        )
        return "skipped"

    report["per_region"] = per_region

    if dry_run:
        logger.info(
            "[dry-run] %s: would add per_region with %d regions: %s",
            report_path,
            len(per_region),
            ", ".join(sorted(per_region.keys())),
        )
        return "would-update"

    _atomic_write_json(report_path, report)
    logger.info(
        "Updated %s: per_region with %d region(s): %s",
        report_path,
        len(per_region),
        ", ".join(sorted(per_region.keys())),
    )
    return "updated"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the backfill script."""
    parser = argparse.ArgumentParser(
        description=(
            "Backfill the per_region field in existing evaluation reports "
            "by recomputing per-region metric blocks from the sibling "
            "*_inference.json files."
        )
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Root checkpoints directory to scan (default: checkpoints).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Restrict the backfill to a specific run UUID (matches the "
            "report's parent-directory name)."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=list(SPLITS),
        help=(
            "Restrict the backfill to a specific split. By default all "
            "splits (train, val, test) are processed."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recompute and overwrite per_region even when the field is "
            "already present in a report."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without writing any files.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    pairs = _discover_report_pairs(
        args.checkpoints_dir,
        run_id=args.run_id,
        split=args.split,
    )
    if not pairs:
        logger.warning(
            "No (report, inference) pairs discovered under %s "
            "(run_id=%s, split=%s).",
            args.checkpoints_dir,
            args.run_id,
            args.split,
        )
        return 0

    counts = {"updated": 0, "would-update": 0, "skipped": 0, "errors": 0}
    for report_path, inference_path, _split in pairs:
        try:
            outcome = backfill_one(
                report_path,
                inference_path,
                force=args.force,
                dry_run=args.dry_run,
            )
        except Exception:
            logger.exception("Failed to backfill %s", report_path)
            counts["errors"] += 1
            continue
        counts[outcome] = counts.get(outcome, 0) + 1

    logger.info(
        "Backfill complete: %d updated, %d would-update, %d skipped, %d errors",
        counts["updated"],
        counts["would-update"],
        counts["skipped"],
        counts["errors"],
    )
    return 1 if counts["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
