"""Region-aware helpers for the dashboard.

The dashboard reads per-region metrics straight off the evaluation report
(field ``per_region`` populated by the evaluator or the
``model.scripts.backfill_per_region`` CLI). This module provides:

1. :func:`extract_region` / :func:`discover_regions` — pull the RDD2022 region
   prefix from an inference dump's ``image_id`` strings (used by the
   Predictions gallery to filter per-image annotations).
2. :func:`filter_predictions_by_region` — slice an inference-JSON dump so the
   image gallery shows only one region.
3. :func:`regions_from_report` — list the regions present on a report's
   ``per_region`` field, sorted alphabetically.
4. :func:`region_report_view` — return an :class:`EvaluationReport` view
   over a single region by swapping in that region's metric block /
   confusion matrix. The original report is not mutated.

This module deliberately has *no* Streamlit imports so it stays
unit-testable without an app context.
"""

from __future__ import annotations

import re
from copy import copy
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Optional

try:
    from data_loader import EvaluationReport  # type: ignore  # noqa: E402
except ImportError:  # pragma: no cover - exercised only under pytest
    from dashboard.data_loader import EvaluationReport  # noqa: E402


# ---------------------------------------------------------------------------
# Region extraction
# ---------------------------------------------------------------------------

# Match ``<region>_<digits>.<ext>`` where ``<region>`` may itself contain
# underscores (e.g. ``China_Drone``, ``United_States``). The non-greedy
# ``(.+?)`` combined with the strict ``_\d+\.[A-Za-z0-9]+$`` suffix anchors
# the split at the *last* ``_<digits>.<ext>`` group while keeping the
# region prefix intact.
_REGION_RE = re.compile(r"^(.+?)_\d+\.[A-Za-z0-9]+$")

ALL_REGIONS_LABEL = "All"


def extract_region(image_id: str) -> Optional[str]:
    """Return the region prefix from an image identifier, or ``None``.

    Args:
        image_id: Path-like image identifier from an inference JSON file.

    Returns:
        Region string (e.g. ``"United_States"``, ``"China_Drone"``) when the
        filename matches the expected ``<region>_<digits>.<ext>`` pattern,
        otherwise ``None``.
    """
    if not image_id:
        return None
    name = Path(image_id).name
    match = _REGION_RE.match(name)
    if match is None:
        return None
    return match.group(1)


def discover_regions(predictions_data: Optional[dict]) -> list[str]:
    """Return the sorted unique regions present in an inference dump.

    Args:
        predictions_data: Loaded inference JSON (``val_inference.json`` etc.).
            May be ``None`` when no predictions are available for the run.

    Returns:
        Sorted list of unique non-``None`` region strings. Empty list when
        ``predictions_data`` is missing or contains no recognisable regions.
    """
    if not predictions_data:
        return []
    images = predictions_data.get("images") or []
    regions: set[str] = set()
    for img in images:
        region = extract_region(img.get("image_id", ""))
        if region is not None:
            regions.add(region)
    return sorted(regions)


def filter_predictions_by_region(
    predictions_data: Optional[dict], region: str
) -> Optional[dict]:
    """Return a shallow copy of ``predictions_data`` keeping only ``region``.

    Top-level keys (``class_names``, ``confidence_threshold``, ...) are
    preserved verbatim; only the ``images`` list is filtered.

    Args:
        predictions_data: Loaded inference JSON. ``None`` is propagated.
        region: Region label as returned by :func:`extract_region`. The
            sentinel :data:`ALL_REGIONS_LABEL` returns the input unchanged.

    Returns:
        New dict with the filtered ``images`` list. The original dict is not
        mutated. Returns ``None`` when ``predictions_data`` is ``None``.
    """
    if predictions_data is None:
        return None
    if region == ALL_REGIONS_LABEL:
        return predictions_data
    filtered = copy(predictions_data)
    filtered["images"] = [
        img
        for img in predictions_data.get("images", [])
        if extract_region(img.get("image_id", "")) == region
    ]
    return filtered


# ---------------------------------------------------------------------------
# Per-region report view (reads from EvaluationReport.per_region)
# ---------------------------------------------------------------------------


def regions_from_report(report: Optional[EvaluationReport]) -> list[str]:
    """Return the sorted region labels present on a report's per_region.

    Args:
        report: :class:`EvaluationReport` whose ``per_region`` field carries
            region-keyed metric blocks. ``None`` returns ``[]``.

    Returns:
        Sorted list of region labels, or ``[]`` when no per-region data is
        available (legacy reports that pre-date the feature).
    """
    if report is None:
        return []
    per_region = getattr(report, "per_region", None) or {}
    return sorted(per_region.keys())


def region_report_view(
    report: EvaluationReport, region: str
) -> EvaluationReport:
    """Return an :class:`EvaluationReport` view restricted to ``region``.

    Swaps the report's ``metrics`` and ``confusion_matrix`` for the region's
    block (read from ``report.per_region[region]``). When ``region`` is
    :data:`ALL_REGIONS_LABEL`, the input report is returned unchanged. When
    the requested region is missing from ``per_region`` the input is also
    returned unchanged so callers can keep rendering the global view.

    The returned dataclass is a *new* instance (via
    :func:`dataclasses.replace`); the input report is never mutated.

    Args:
        report: The full evaluation report.
        region: The region label, or :data:`ALL_REGIONS_LABEL`.

    Returns:
        A region-restricted :class:`EvaluationReport`, or the input when no
        substitution is needed.
    """
    if region == ALL_REGIONS_LABEL:
        return report
    per_region = getattr(report, "per_region", None) or {}
    block = per_region.get(region)
    if not block:
        return report
    metrics = dict(block.get("metrics") or {})
    confusion_matrix = block.get("confusion_matrix") or report.confusion_matrix
    num_images = int(block.get("num_images", report.num_val_images))
    return dataclass_replace(
        report,
        metrics=metrics,
        confusion_matrix=confusion_matrix,
        num_val_images=num_images,
    )


# ---------------------------------------------------------------------------
# Compatibility re-export (kept for ``replace`` import path consumers)
# ---------------------------------------------------------------------------

# ``replace`` is intentionally re-exported so older imports of the dashboard
# helpers don't break. The dataclass ``replace`` is the public contract.
__all__ = [
    "ALL_REGIONS_LABEL",
    "extract_region",
    "discover_regions",
    "filter_predictions_by_region",
    "regions_from_report",
    "region_report_view",
]
