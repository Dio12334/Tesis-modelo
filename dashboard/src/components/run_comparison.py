"""Run comparison component for the Streamlit Results Dashboard.

Allows selection of two or more runs for side-by-side comparison,
displaying an enriched metrics table, F1-vs-confidence and parametric
Precision-vs-Recall overlays, per-class grouped bars (best F1 / AP),
and overlaid loss curves.

Phase-2 sweep keys consumed (when present in ``EvaluationReport.metrics``):
``mAP@0.5:0.95``, ``best_f1``, ``f1_sweep``, ``per_class_best_f1``,
``per_class_ap``, ``default_confidence_threshold``. Each is read defensively
so legacy reports without these keys continue to render with ``"N/A"``
placeholders and skipped traces.
"""

from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import DashboardData, EvaluationReport, ExperimentRun

# Distinct color palette for overlaid runs
RUN_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

# Mapping: highlight-radio label -> (metric_path, display_short_label)
HIGHLIGHT_METRIC_OPTIONS = ("mAP@0.5", "mAP@0.5:0.95", "Best F1")

# Default value used when a numeric metric is missing.
NA = "N/A"


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def _resolve_report(
    run: ExperimentRun,
    evaluation_report: Optional[EvaluationReport],
    evaluation_reports: Optional[dict] = None,
) -> Optional[EvaluationReport]:
    """Return the per-run report when available, else the global one."""
    if evaluation_reports:
        report = evaluation_reports.get(run.run_id)
        if report is not None:
            return report
    return evaluation_report


def _get_metric(
    run: ExperimentRun,
    key: str,
    evaluation_report: Optional[EvaluationReport],
    evaluation_reports: Optional[dict] = None,
) -> Optional[Any]:
    """Fetch ``metrics[key]`` for the run's resolved report.

    Returns None when the report or the key is absent. Generalizes the
    legacy ``_get_map_for_run`` / ``_get_f1_for_run`` helpers.
    """
    report = _resolve_report(run, evaluation_report, evaluation_reports)
    if report is None:
        return None
    return report.metrics.get(key)


def _get_map_for_run(
    run: ExperimentRun,
    evaluation_report: Optional[EvaluationReport],
    evaluation_reports: Optional[dict] = None,
) -> Optional[float]:
    """Get mAP@0.5 for a run (legacy helper kept for backwards compatibility)."""
    return _get_metric(run, "mAP@0.5", evaluation_report, evaluation_reports)


def _get_f1_for_run(
    run: ExperimentRun,
    evaluation_report: Optional[EvaluationReport],
    evaluation_reports: Optional[dict] = None,
) -> Optional[float]:
    """Get the deployment-confidence F1 (legacy helper kept for backwards compat)."""
    return _get_metric(run, "f1_score", evaluation_report, evaluation_reports)


def _get_best_f1(
    run: ExperimentRun,
    evaluation_report: Optional[EvaluationReport],
    evaluation_reports: Optional[dict] = None,
) -> Optional[dict]:
    """Return the ``best_f1`` dict (confidence/precision/recall/f1) or None."""
    return _get_metric(run, "best_f1", evaluation_report, evaluation_reports)


def _get_f1_sweep(
    run: ExperimentRun,
    evaluation_report: Optional[EvaluationReport],
    evaluation_reports: Optional[dict] = None,
) -> Optional[list]:
    """Return the ``f1_sweep`` list of dicts or None."""
    return _get_metric(run, "f1_sweep", evaluation_report, evaluation_reports)


def _get_per_class_best_f1(
    run: ExperimentRun,
    evaluation_report: Optional[EvaluationReport],
    evaluation_reports: Optional[dict] = None,
) -> Optional[dict]:
    """Return the per-class best-F1 dict or None."""
    return _get_metric(run, "per_class_best_f1", evaluation_report, evaluation_reports)


def _get_per_class_ap(
    run: ExperimentRun,
    evaluation_report: Optional[EvaluationReport],
    evaluation_reports: Optional[dict] = None,
) -> Optional[dict]:
    """Return the per-class AP dict or None."""
    return _get_metric(run, "per_class_ap", evaluation_report, evaluation_reports)


def _get_class_display_names(report: Optional[EvaluationReport]) -> list[str]:
    """Return Spanish display names if available, otherwise canonical class names.

    Returns an empty list when the report is None.
    """
    if report is None:
        return []
    display = getattr(report, "display_class_names", None)
    if display:
        return list(display)
    return list(report.class_names)


def _get_run_label(run: ExperimentRun) -> str:
    """Create a short label for a run using model name and run_id prefix."""
    short_id = run.run_id[:8]
    return f"{run.model_name} ({short_id})"


# ---------------------------------------------------------------------------
# Tier 1: comparison-table builder
# ---------------------------------------------------------------------------


def _fmt_float(value: Optional[float], digits: int = 4) -> str:
    """Format a float to ``digits`` decimals, or ``"N/A"`` if None."""
    if value is None:
        return NA
    return f"{value:.{digits}f}"


def _build_comparison_dataframe(
    selected_runs: list[ExperimentRun],
    evaluation_report: Optional[EvaluationReport],
    evaluation_reports: Optional[dict] = None,
) -> pd.DataFrame:
    """Build a comparison DataFrame with key metrics for each run.

    Columns (in order):
      Run ID, Model, Final Train Loss, Final Val Loss, Best Val Loss,
      Best Epoch, Total Epochs, mAP@0.5, mAP@0.5:0.95, Best F1,
      Best F1 conf, P @ best F1, R @ best F1, Default conf, F1-Score.

    Sweep-derived columns hold ``"N/A"`` when the underlying key is absent.
    Numeric metric columns use 4-decimal precision; confidence columns use
    2-decimal precision.
    """
    rows = []
    for run in selected_runs:
        row = {
            "Run ID": run.run_id[:8],
            "Model": run.model_name,
            "Final Train Loss": (
                f"{run.final_results.final_train_loss:.4f}"
                if run.final_results
                else NA
            ),
            "Final Val Loss": (
                f"{run.final_results.final_val_loss:.4f}"
                if run.final_results
                else NA
            ),
            "Best Val Loss": (
                f"{run.final_results.best_val_loss:.4f}"
                if run.final_results
                else NA
            ),
            "Best Epoch": (
                str(run.final_results.best_epoch) if run.final_results else NA
            ),
            "Total Epochs": (
                str(run.final_results.total_epochs) if run.final_results else NA
            ),
        }

        row["mAP@0.5"] = _fmt_float(
            _get_metric(run, "mAP@0.5", evaluation_report, evaluation_reports)
        )
        row["mAP@0.5:0.95"] = _fmt_float(
            _get_metric(run, "mAP@0.5:0.95", evaluation_report, evaluation_reports)
        )

        best_f1 = _get_best_f1(run, evaluation_report, evaluation_reports)
        if isinstance(best_f1, dict):
            row["Best F1"] = _fmt_float(best_f1.get("f1"))
            row["Best F1 conf"] = _fmt_float(best_f1.get("confidence"), digits=2)
            row["P @ best F1"] = _fmt_float(best_f1.get("precision"))
            row["R @ best F1"] = _fmt_float(best_f1.get("recall"))
        else:
            row["Best F1"] = NA
            row["Best F1 conf"] = NA
            row["P @ best F1"] = NA
            row["R @ best F1"] = NA

        row["Default conf"] = _fmt_float(
            _get_metric(
                run, "default_confidence_threshold", evaluation_report, evaluation_reports
            ),
            digits=2,
        )

        row["F1-Score"] = _fmt_float(
            _get_metric(run, "f1_score", evaluation_report, evaluation_reports)
        )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Highlight metric: best-run index resolver
# ---------------------------------------------------------------------------


def _extract_highlight_value(
    run: ExperimentRun,
    metric_key: str,
    evaluation_report: Optional[EvaluationReport],
    evaluation_reports: Optional[dict],
) -> Optional[float]:
    """Resolve the numeric value used by the highlight metric.

    Supports three labels: ``"mAP@0.5"``, ``"mAP@0.5:0.95"``, ``"Best F1"``.
    Returns None when the metric cannot be resolved.
    """
    if metric_key in ("mAP@0.5", "mAP@0.5:0.95"):
        value = _get_metric(run, metric_key, evaluation_report, evaluation_reports)
        return value if isinstance(value, (int, float)) else None
    if metric_key == "Best F1":
        best = _get_best_f1(run, evaluation_report, evaluation_reports)
        if isinstance(best, dict):
            f1 = best.get("f1")
            return f1 if isinstance(f1, (int, float)) else None
        return None
    return None


def _find_best_run_index(
    selected_runs: list[ExperimentRun],
    evaluation_report: Optional[EvaluationReport],
    evaluation_reports: Optional[dict] = None,
    metric_key: str = "mAP@0.5",
) -> Optional[int]:
    """Find the index of the run with the highest value for ``metric_key``.

    ``metric_key`` must be one of :data:`HIGHLIGHT_METRIC_OPTIONS`. The
    legacy three-argument call (no ``metric_key``) preserves the previous
    behaviour of selecting the run with the highest mAP@0.5.

    Returns None when no run has a resolvable value for the chosen metric.
    """
    if evaluation_report is None and not evaluation_reports:
        return None

    best_idx: Optional[int] = None
    best_value: float = float("-inf")

    for idx, run in enumerate(selected_runs):
        value = _extract_highlight_value(
            run, metric_key, evaluation_report, evaluation_reports
        )
        if value is not None and value > best_value:
            best_value = value
            best_idx = idx

    return best_idx


# ---------------------------------------------------------------------------
# Tier 2 figure builders
# ---------------------------------------------------------------------------


def _build_f1_vs_conf_figure(
    selected_runs: list[ExperimentRun],
    evaluation_report: Optional[EvaluationReport],
    evaluation_reports: Optional[dict] = None,
) -> go.Figure:
    """Overlay F1 vs confidence for each run that exposes ``f1_sweep``.

    Adds an additional star marker at each run's ``best_f1`` confidence.
    """
    fig = go.Figure()

    for idx, run in enumerate(selected_runs):
        sweep = _get_f1_sweep(run, evaluation_report, evaluation_reports)
        if not sweep:
            continue

        color = RUN_COLORS[idx % len(RUN_COLORS)]
        label = _get_run_label(run)
        confidences = [pt["confidence"] for pt in sweep]
        f1s = [pt["f1"] for pt in sweep]

        fig.add_trace(
            go.Scatter(
                x=confidences,
                y=f1s,
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
                hovertemplate=(
                    f"{label}<br>Confidence: %{{x:.2f}}<br>F1: %{{y:.4f}}<extra></extra>"
                ),
            )
        )

        best_f1 = _get_best_f1(run, evaluation_report, evaluation_reports)
        if isinstance(best_f1, dict) and best_f1.get("confidence") is not None:
            fig.add_trace(
                go.Scatter(
                    x=[best_f1["confidence"]],
                    y=[best_f1.get("f1", 0.0)],
                    mode="markers",
                    marker=dict(color=color, size=12, symbol="star"),
                    name=f"{label} - Best F1",
                    showlegend=False,
                    hovertemplate=(
                        f"{label} - Best F1<br>"
                        "Confidence: %{x:.2f}<br>"
                        "F1: %{y:.4f}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title="F1 vs Confidence",
        xaxis_title="Confidence threshold",
        yaxis_title="F1",
        hovermode="x unified",
        template="plotly_white",
        height=420,
    )
    return fig


def _build_pr_parametric_figure(
    selected_runs: list[ExperimentRun],
    evaluation_report: Optional[EvaluationReport],
    evaluation_reports: Optional[dict] = None,
) -> go.Figure:
    """Overlay parametric Precision-vs-Recall (parameter = confidence).

    For each run with ``f1_sweep``, plots one curve where the parameter
    along the curve is the confidence threshold. Hover exposes the
    underlying confidence so the user can read off the operating point.
    """
    fig = go.Figure()

    for idx, run in enumerate(selected_runs):
        sweep = _get_f1_sweep(run, evaluation_report, evaluation_reports)
        if not sweep:
            continue

        color = RUN_COLORS[idx % len(RUN_COLORS)]
        label = _get_run_label(run)
        recalls = [pt["recall"] for pt in sweep]
        precisions = [pt["precision"] for pt in sweep]
        confidences = [pt["confidence"] for pt in sweep]

        fig.add_trace(
            go.Scatter(
                x=recalls,
                y=precisions,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                customdata=confidences,
                hovertemplate=(
                    f"{label}<br>"
                    "Recall: %{x:.4f}<br>"
                    "Precision: %{y:.4f}<br>"
                    "Confidence: %{customdata:.2f}<extra></extra>"
                ),
            )
        )

        best_f1 = _get_best_f1(run, evaluation_report, evaluation_reports)
        if isinstance(best_f1, dict):
            r = best_f1.get("recall")
            p = best_f1.get("precision")
            if r is not None and p is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[r],
                        y=[p],
                        mode="markers",
                        marker=dict(color=color, size=14, symbol="star"),
                        name=f"{label} - Best F1",
                        showlegend=False,
                        hovertemplate=(
                            f"{label} - Best F1<br>"
                            "Recall: %{x:.4f}<br>"
                            "Precision: %{y:.4f}<extra></extra>"
                        ),
                    )
                )

    fig.update_layout(
        title="Precision vs Recall (paramétrico en confianza)",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
        height=420,
    )
    return fig


def _extract_per_class_values(
    metric_key: str,
    per_class_data: dict,
    class_names: list[str],
) -> list[Optional[float]]:
    """Pick the bar-height value per class for the given metric.

    For ``per_class_best_f1`` each entry is a dict with an ``f1`` key.
    For ``per_class_ap`` each entry is a scalar AP value.
    Missing classes are filled with None.
    """
    values: list[Optional[float]] = []
    for name in class_names:
        entry = per_class_data.get(name)
        if entry is None:
            values.append(None)
            continue
        if metric_key == "per_class_best_f1":
            if isinstance(entry, dict):
                v = entry.get("f1")
                values.append(v if isinstance(v, (int, float)) else None)
            else:
                values.append(None)
        else:  # per_class_ap (scalar)
            values.append(entry if isinstance(entry, (int, float)) else None)
    return values


def _build_per_class_grouped_bars(
    selected_runs: list[ExperimentRun],
    evaluation_report: Optional[EvaluationReport],
    evaluation_reports: Optional[dict] = None,
    *,
    metric_key: str = "per_class_best_f1",
) -> go.Figure:
    """Build a grouped-bar figure with one bar per (run, class).

    The x-axis uses Spanish class names from ``display_class_names`` when the
    resolved report exposes them; otherwise falls back to canonical names.

    Runs whose resolved report does not contain ``metric_key`` are skipped.
    """
    if metric_key not in ("per_class_best_f1", "per_class_ap"):
        raise ValueError(
            f"metric_key must be 'per_class_best_f1' or 'per_class_ap', got {metric_key!r}"
        )

    fig = go.Figure()

    # Determine canonical class order and Spanish labels from the first run
    # that has a resolvable report (and the requested metric).
    canonical_class_names: Optional[list[str]] = None
    display_names: Optional[list[str]] = None
    for run in selected_runs:
        report = _resolve_report(run, evaluation_report, evaluation_reports)
        if report is None:
            continue
        if metric_key == "per_class_best_f1":
            data = _get_per_class_best_f1(run, evaluation_report, evaluation_reports)
        else:
            data = _get_per_class_ap(run, evaluation_report, evaluation_reports)
        if not data:
            continue
        canonical_class_names = list(report.class_names)
        display_names = _get_class_display_names(report)
        break

    if canonical_class_names is None or display_names is None:
        # No usable run found; return an empty figure with an annotation.
        fig.update_layout(
            title=(
                "F1 por clase (mejor por clase)"
                if metric_key == "per_class_best_f1"
                else "AP por clase"
            ),
            template="plotly_white",
            height=420,
        )
        return fig

    for idx, run in enumerate(selected_runs):
        if metric_key == "per_class_best_f1":
            data = _get_per_class_best_f1(run, evaluation_report, evaluation_reports)
        else:
            data = _get_per_class_ap(run, evaluation_report, evaluation_reports)
        if not data:
            continue

        values = _extract_per_class_values(metric_key, data, canonical_class_names)
        color = RUN_COLORS[idx % len(RUN_COLORS)]
        label = _get_run_label(run)

        fig.add_trace(
            go.Bar(
                x=display_names,
                y=values,
                name=label,
                marker_color=color,
                hovertemplate=(
                    f"{label}<br>Clase: %{{x}}<br>Valor: %{{y:.4f}}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=(
            "F1 por clase (mejor por clase)"
            if metric_key == "per_class_best_f1"
            else "AP por clase"
        ),
        xaxis_title="Clase",
        yaxis_title=("F1" if metric_key == "per_class_best_f1" else "AP"),
        barmode="group",
        template="plotly_white",
        height=420,
    )
    return fig


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _render_comparison_table(
    selected_runs: list[ExperimentRun],
    evaluation_report: Optional[EvaluationReport],
    evaluation_reports: Optional[dict] = None,
    highlight_metric: str = "mAP@0.5",
) -> None:
    """Render the comparison table with highlighting for the best run."""
    df = _build_comparison_dataframe(
        selected_runs, evaluation_report, evaluation_reports
    )
    best_idx = _find_best_run_index(
        selected_runs, evaluation_report, evaluation_reports, metric_key=highlight_metric
    )

    if best_idx is not None:
        # Highlight the best run row using pandas Styler
        def highlight_best(row: pd.Series) -> list[str]:
            if row.name == best_idx:
                return ["background-color: #d4edda; font-weight: bold"] * len(row)
            return [""] * len(row)

        styled = df.style.apply(highlight_best, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_overlay_loss_chart(selected_runs: list[ExperimentRun]) -> None:
    """Render overlaid loss curves from all selected runs on a single chart."""
    fig = go.Figure()

    for idx, run in enumerate(selected_runs):
        if not run.metrics_history:
            continue

        color = RUN_COLORS[idx % len(RUN_COLORS)]
        label = _get_run_label(run)
        epochs = [entry.step for entry in run.metrics_history]
        train_losses = [entry.train_loss for entry in run.metrics_history]
        val_losses = [entry.val_loss for entry in run.metrics_history]

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=train_losses,
                mode="lines",
                name=f"{label} - Train",
                line=dict(color=color, width=2, dash="solid"),
                hovertemplate=(
                    f"{label}<br>"
                    "Epoch: %{x}<br>"
                    "Train Loss: %{y:.4f}"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=val_losses,
                mode="lines",
                name=f"{label} - Val",
                line=dict(color=color, width=2, dash="dash"),
                hovertemplate=(
                    f"{label}<br>"
                    "Epoch: %{x}<br>"
                    "Val Loss: %{y:.4f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Loss Curves Comparison",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)


def _any_run_has_sweep(
    selected_runs: list[ExperimentRun],
    evaluation_report: Optional[EvaluationReport],
    evaluation_reports: Optional[dict],
) -> bool:
    """True if at least one selected run exposes ``f1_sweep``."""
    return any(
        _get_f1_sweep(run, evaluation_report, evaluation_reports)
        for run in selected_runs
    )


def _any_run_has_per_class(
    selected_runs: list[ExperimentRun],
    evaluation_report: Optional[EvaluationReport],
    evaluation_reports: Optional[dict],
    metric_key: str,
) -> bool:
    """True if at least one selected run exposes ``metric_key``."""
    fetch = (
        _get_per_class_best_f1
        if metric_key == "per_class_best_f1"
        else _get_per_class_ap
    )
    return any(fetch(run, evaluation_report, evaluation_reports) for run in selected_runs)


def render_run_comparison(
    data: DashboardData, selected_runs: list[ExperimentRun]
) -> None:
    """Render run comparison view: table, sweep plots, per-class bars, loss curves.

    Args:
        data: The full dashboard data containing all runs and evaluation reports.
        selected_runs: Pre-selected runs to compare (can be empty for fresh selection).
    """
    st.header("Run Comparison")

    if len(data.runs) < 2:
        st.info("At least two runs are required for comparison.")
        return

    run_options = {_get_run_label(run): run for run in data.runs}
    default_labels = [_get_run_label(run) for run in selected_runs]

    chosen_labels = st.multiselect(
        "Select runs to compare",
        options=list(run_options.keys()),
        default=default_labels,
        help="Select two or more runs to compare their metrics and loss curves.",
    )

    chosen_runs = [run_options[label] for label in chosen_labels]

    if len(chosen_runs) < 2:
        st.warning("Please select at least two runs to compare.")
        return

    highlight_metric = st.radio(
        "Highlight best run by",
        options=HIGHLIGHT_METRIC_OPTIONS,
        index=0,
        horizontal=True,
        help="Choose which metric drives the green highlight in the table below.",
    )

    st.subheader("Metrics Comparison")
    _render_comparison_table(
        chosen_runs, data.evaluation_report, data.evaluation_reports, highlight_metric
    )

    if data.evaluation_report is not None or data.evaluation_reports:
        best_idx = _find_best_run_index(
            chosen_runs,
            data.evaluation_report,
            data.evaluation_reports,
            metric_key=highlight_metric,
        )
        if best_idx is not None:
            best_label = _get_run_label(chosen_runs[best_idx])
            st.success(
                f"🏆 Best run by {highlight_metric}: **{best_label}**"
            )

    if _any_run_has_sweep(chosen_runs, data.evaluation_report, data.evaluation_reports):
        st.subheader("F1 vs Confidence")
        st.plotly_chart(
            _build_f1_vs_conf_figure(
                chosen_runs, data.evaluation_report, data.evaluation_reports
            ),
            use_container_width=True,
        )

        st.subheader("Precision vs Recall (paramétrico en confianza)")
        st.plotly_chart(
            _build_pr_parametric_figure(
                chosen_runs, data.evaluation_report, data.evaluation_reports
            ),
            use_container_width=True,
        )
    else:
        st.info("F1 sweep no disponible para los runs seleccionados.")

    if _any_run_has_per_class(
        chosen_runs,
        data.evaluation_report,
        data.evaluation_reports,
        "per_class_best_f1",
    ):
        st.subheader("F1 por clase (mejor por clase)")
        st.plotly_chart(
            _build_per_class_grouped_bars(
                chosen_runs,
                data.evaluation_report,
                data.evaluation_reports,
                metric_key="per_class_best_f1",
            ),
            use_container_width=True,
        )

    if _any_run_has_per_class(
        chosen_runs,
        data.evaluation_report,
        data.evaluation_reports,
        "per_class_ap",
    ):
        st.subheader("AP por clase")
        st.plotly_chart(
            _build_per_class_grouped_bars(
                chosen_runs,
                data.evaluation_report,
                data.evaluation_reports,
                metric_key="per_class_ap",
            ),
            use_container_width=True,
        )

    st.subheader("Loss Curves Overlay")
    _render_overlay_loss_chart(chosen_runs)
