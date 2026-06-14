"""Outer region-tab navigation for the dashboard.

The dashboard wraps every metric tab inside an outer set of region tabs:
``All`` followed by every region present on the selected run's
``per_region`` field. Each tab is a Streamlit context manager so the
caller can render the existing six inner tabs inside it; widget keys
should be suffixed with the active region label to avoid collisions.

When no per-region data is available (legacy reports that pre-date the
per_region field or non-RDD runs) the function returns a single ``All``
tab and an :func:`st.info` notice recommending the backfill CLI.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

try:
    from region_metrics import ALL_REGIONS_LABEL, regions_from_report  # type: ignore
    from data_loader import EvaluationReport  # type: ignore
except ImportError:  # pragma: no cover - executed only under pytest
    from dashboard.region_metrics import (
        ALL_REGIONS_LABEL,
        regions_from_report,
    )
    from dashboard.data_loader import EvaluationReport


def render_region_tabs(
    report: Optional[EvaluationReport],
    *,
    show_legacy_notice: bool = True,
) -> tuple[list[str], list]:
    """Render the outer region-tab navigation.

    Args:
        report: The active run's evaluation report (used to source the list
            of available regions from its ``per_region`` field). When
            ``None`` or when the report has no ``per_region`` data, only
            an ``All`` tab is returned.
        show_legacy_notice: When ``True`` (the default), an :func:`st.info`
            notice is displayed for legacy reports recommending the
            ``model.scripts.backfill_per_region`` CLI.

    Returns:
        A ``(labels, tabs)`` pair where ``labels[i]`` corresponds to
        ``tabs[i]``. The first label is always :data:`ALL_REGIONS_LABEL`.
        Each ``tabs[i]`` is the Streamlit tab container produced by
        :func:`st.tabs` and can be used as a context manager.
    """
    regions = regions_from_report(report)
    labels = [ALL_REGIONS_LABEL] + regions
    tabs = st.tabs(labels)

    if not regions and show_legacy_notice and report is not None:
        with tabs[0]:
            st.info(
                "Per-region metrics are not available for this run. "
                "Re-run evaluation, or backfill existing reports with:\n\n"
                "```\n"
                "python -m model.scripts.backfill_per_region "
                "--checkpoints-dir checkpoints\n"
                "```"
            )

    return labels, tabs
