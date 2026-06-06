"""Run comparison component for the Streamlit Results Dashboard.

Allows selection of two or more runs for side-by-side comparison,
displaying a metrics table and overlaid loss curves.
"""

from typing import Optional

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


def _get_run_label(run: ExperimentRun) -> str:
    """Create a short label for a run using model name and run_id prefix."""
    short_id = run.run_id[:8]
    return f"{run.model_name} ({short_id})"


def _get_map_for_run(
    run: ExperimentRun, evaluation_report: Optional[EvaluationReport],
    evaluation_reports: Optional[dict] = None,
) -> Optional[float]:
    """Get mAP@0.5 for a run from its per-run evaluation report.

    Looks up the run-specific evaluation report first. Falls back to the
    global evaluation report if no per-run report is available.
    Returns the mAP value if a report exists, otherwise None.
    """
    report = None
    if evaluation_reports:
        report = evaluation_reports.get(run.run_id)
    if report is None:
        report = evaluation_report
    if report is None:
        return None
    return report.metrics.get("mAP@0.5")


def _get_f1_for_run(
    run: ExperimentRun, evaluation_report: Optional[EvaluationReport],
    evaluation_reports: Optional[dict] = None,
) -> Optional[float]:
    """Get F1-score for a run from its per-run evaluation report."""
    report = None
    if evaluation_reports:
        report = evaluation_reports.get(run.run_id)
    if report is None:
        report = evaluation_report
    if report is None:
        return None
    return report.metrics.get("f1_score")


def _build_comparison_dataframe(
    selected_runs: list[ExperimentRun],
    evaluation_report: Optional[EvaluationReport],
    evaluation_reports: Optional[dict] = None,
) -> pd.DataFrame:
    """Build a comparison DataFrame with key metrics for each run.

    Columns: Run ID, Model, Final Train Loss, Final Val Loss,
    Best Val Loss, Best Epoch, Total Epochs, mAP@0.5, F1-Score.
    """
    rows = []
    for run in selected_runs:
        row = {
            "Run ID": run.run_id[:8],
            "Model": run.model_name,
            "Final Train Loss": (
                f"{run.final_results.final_train_loss:.4f}"
                if run.final_results
                else "N/A"
            ),
            "Final Val Loss": (
                f"{run.final_results.final_val_loss:.4f}"
                if run.final_results
                else "N/A"
            ),
            "Best Val Loss": (
                f"{run.final_results.best_val_loss:.4f}"
                if run.final_results
                else "N/A"
            ),
            "Best Epoch": (
                str(run.final_results.best_epoch) if run.final_results else "N/A"
            ),
            "Total Epochs": (
                str(run.final_results.total_epochs) if run.final_results else "N/A"
            ),
        }

        map_val = _get_map_for_run(run, evaluation_report, evaluation_reports)
        row["mAP@0.5"] = f"{map_val:.2f}" if map_val is not None else "N/A"

        f1_val = _get_f1_for_run(run, evaluation_report, evaluation_reports)
        row["F1-Score"] = f"{f1_val:.2f}" if f1_val is not None else "N/A"

        rows.append(row)

    return pd.DataFrame(rows)


def _find_best_run_index(
    selected_runs: list[ExperimentRun],
    evaluation_report: Optional[EvaluationReport],
    evaluation_reports: Optional[dict] = None,
) -> Optional[int]:
    """Find the index of the run with the best (highest) mAP@0.5.

    Returns None if no evaluation report is available.
    """
    if evaluation_report is None and not evaluation_reports:
        return None

    best_idx: Optional[int] = None
    best_map: float = -1.0

    for idx, run in enumerate(selected_runs):
        map_val = _get_map_for_run(run, evaluation_report, evaluation_reports)
        if map_val is not None and map_val > best_map:
            best_map = map_val
            best_idx = idx

    return best_idx


def _render_comparison_table(
    selected_runs: list[ExperimentRun],
    evaluation_report: Optional[EvaluationReport],
    evaluation_reports: Optional[dict] = None,
) -> None:
    """Render the comparison table with highlighting for the best run."""
    df = _build_comparison_dataframe(selected_runs, evaluation_report, evaluation_reports)
    best_idx = _find_best_run_index(selected_runs, evaluation_report, evaluation_reports)

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
    """Render overlaid loss curves from all selected runs on a single chart.

    Each run gets a distinct color. Both train and val loss are shown
    with solid and dashed lines respectively.
    """
    fig = go.Figure()

    for idx, run in enumerate(selected_runs):
        if not run.metrics_history:
            continue

        color = RUN_COLORS[idx % len(RUN_COLORS)]
        label = _get_run_label(run)
        epochs = [entry.step for entry in run.metrics_history]
        train_losses = [entry.train_loss for entry in run.metrics_history]
        val_losses = [entry.val_loss for entry in run.metrics_history]

        # Training loss (solid line)
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

        # Validation loss (dashed line)
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


def render_run_comparison(
    data: DashboardData, selected_runs: list[ExperimentRun]
) -> None:
    """Render run comparison view with multi-select, table, and overlaid charts.

    Allows selection of two or more runs for comparison. Displays a metrics
    comparison table and overlaid loss curves with distinct colors per run.
    Highlights the run with the best mAP@0.5 value.

    Args:
        data: The full dashboard data containing all runs and evaluation report.
        selected_runs: Pre-selected runs to compare (can be empty for fresh selection).
    """
    st.header("Run Comparison")

    if len(data.runs) < 2:
        st.info("At least two runs are required for comparison.")
        return

    # Multi-select for runs
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

    # Comparison table
    st.subheader("Metrics Comparison")
    _render_comparison_table(
        chosen_runs, data.evaluation_report, data.evaluation_reports
    )

    if data.evaluation_report is not None or data.evaluation_reports:
        best_idx = _find_best_run_index(
            chosen_runs, data.evaluation_report, data.evaluation_reports
        )
        if best_idx is not None:
            best_label = _get_run_label(chosen_runs[best_idx])
            st.success(f"🏆 Best run (highest mAP@0.5): **{best_label}**")

    # Overlaid loss curves
    st.subheader("Loss Curves Overlay")
    _render_overlay_loss_chart(chosen_runs)
