"""Main application entry point for the Streamlit Results Dashboard.

Wires together all dashboard components: data loading, sidebar navigation,
and tabbed content views for metrics, charts, and comparisons.

Layout (Plan A): the dashboard wraps every metric tab inside an outer
``st.tabs`` of regions ("All" + every region present on the run's
``per_region`` field). Each region tab contains the full six inner tabs;
Loss Curves and Compare Runs ignore the region filter and display a
caption noting that.

Launch with: streamlit run dashboard/app.py
"""

import logging
from pathlib import Path
from typing import Optional

import streamlit as st

from data_loader import load_all_data, DashboardData, EvaluationReport
from components.sidebar import render_sidebar
from components.metrics_overview import render_metrics_overview
from components.loss_charts import render_loss_chart, render_learning_rate_chart
from components.class_performance import render_class_performance
from components.confusion_matrix import render_confusion_matrix
from components.run_comparison import render_run_comparison
from components.config_display import render_config
from components.image_prediction_viewer import (
    render_image_prediction_viewer,
    ImageAnnotation,
    BoundingBox,
)
from components.region_filter import render_region_tabs
from region_metrics import (
    ALL_REGIONS_LABEL,
    filter_predictions_by_region,
    region_report_view,
)

logger = logging.getLogger(__name__)


def _build_annotations(predictions_data: Optional[dict]) -> list[ImageAnnotation]:
    """Convert predictions_data from JSON into ImageAnnotation objects."""
    if predictions_data is None:
        return []

    images = predictions_data.get("images", [])
    annotations = []

    for img_data in images:
        image_path = img_data.get("image_id", "")
        gt_data = img_data.get("ground_truth", {})
        pred_data = img_data.get("predictions", {})

        gt_boxes = []
        for coords, label in zip(gt_data.get("boxes", []), gt_data.get("labels", [])):
            if len(coords) == 4:
                gt_boxes.append(BoundingBox(
                    x_min=coords[0], y_min=coords[1],
                    x_max=coords[2], y_max=coords[3],
                    class_name=label, confidence=None,
                ))

        pred_boxes = []
        for coords, label, score in zip(
            pred_data.get("boxes", []),
            pred_data.get("labels", []),
            pred_data.get("scores", []),
        ):
            if len(coords) == 4:
                pred_boxes.append(BoundingBox(
                    x_min=coords[0], y_min=coords[1],
                    x_max=coords[2], y_max=coords[3],
                    class_name=label, confidence=score,
                ))

        annotations.append(ImageAnnotation(
            image_id=Path(image_path).name,
            image_path=image_path,
            ground_truth_boxes=gt_boxes,
            prediction_boxes=pred_boxes,
        ))

    return annotations


def render_welcome() -> None:
    """Display a welcome message when no run is selected."""
    st.title("Road Damage Detection - Results Dashboard")
    st.markdown(
        """
        Welcome to the Road Damage Detection Results Dashboard.

        **Getting started:**
        1. Select an experiment run from the sidebar on the left
        2. Use the model filter to narrow down runs by architecture
        3. Once a run is selected, explore the tabs for detailed analysis

        **Available views:**
        - **Overview** — Key evaluation metrics and experiment configuration
        - **Loss Curves** — Training/validation loss and learning rate schedule
        - **Class Performance** — Per-class Average Precision breakdown
        - **Confusion Matrix** — Class-level detection confusion heatmap
        - **Compare Runs** — Side-by-side comparison of multiple runs
        - **Predictions** — Visual comparison of ground truth vs. model predictions

        **Region tabs:** once a run with per-region metrics is selected,
        the dashboard exposes outer tabs for **All** plus every RDD2022
        region present on the run (Japan, Czech, India, Norway,
        United_States, China_Drone, China_MotorBike). Overview, Class
        Performance, Confusion Matrix, and Predictions are restricted to
        the active region; Loss Curves and Compare Runs ignore it.
        """
    )


# ---------------------------------------------------------------------------
# Per-region report resolver
# ---------------------------------------------------------------------------


def _report_for_region(
    base_report: Optional[EvaluationReport], region: str
) -> Optional[EvaluationReport]:
    """Return ``base_report`` swapped to the active region's metric block."""
    if base_report is None:
        return None
    return region_report_view(base_report, region)


# ---------------------------------------------------------------------------
# Inner-tab rendering (one render per region tab)
# ---------------------------------------------------------------------------


def _render_region_inner_tabs(
    *,
    region: str,
    selected_run,
    data: DashboardData,
    run_eval_report: Optional[EvaluationReport],
    run_reports_by_split: dict,
    run_val_preds: Optional[dict],
    run_test_preds: Optional[dict],
    run_train_preds: Optional[dict] = None,
) -> None:
    """Render the full six-tab inner navigation for a single region.

    Widget keys are suffixed with ``region`` so each region tab keeps
    independent state (selected image, slider value, etc.).

    Loss Curves and Compare Runs ignore ``region`` (training history /
    cross-run comparisons are intrinsically region-independent); a caption
    flags this so the user knows the region filter is inactive there.
    """

    headline_report = _report_for_region(run_eval_report, region)
    split_predictions = {
        "val": run_val_preds,
        "test": run_test_preds,
        "train": run_train_preds,
    }
    # The single-split (headline) view assumes the evaluator's primary
    # split is validation; the data loader prefers val when populating
    # ``data.evaluation_report``.
    headline_predictions = run_val_preds

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview",
        "Loss Curves",
        "Class Performance",
        "Confusion Matrix",
        "Compare Runs",
        "Predictions",
    ])

    with tab1:
        if run_reports_by_split and len(run_reports_by_split) > 1:
            split_tabs = st.tabs([f"Metrics ({s})" for s in run_reports_by_split])
            for split_tab, (split_name, split_report) in zip(
                split_tabs, run_reports_by_split.items()
            ):
                with split_tab:
                    render_metrics_overview(_report_for_region(split_report, region))
        else:
            render_metrics_overview(headline_report)
        render_config(selected_run)

    with tab2:
        if region == ALL_REGIONS_LABEL:
            render_loss_chart(selected_run)
            render_learning_rate_chart(selected_run)
        else:
            st.caption(
                "Loss curves are region-independent — training history is "
                f"logged per epoch, not per image. See the **{ALL_REGIONS_LABEL}** "
                "outer tab for the global training curves."
            )

    with tab3:
        if run_reports_by_split and len(run_reports_by_split) > 1:
            split_tabs = st.tabs(
                [f"Class Performance ({s})" for s in run_reports_by_split]
            )
            for split_tab, (split_name, split_report) in zip(
                split_tabs, run_reports_by_split.items()
            ):
                with split_tab:
                    render_class_performance(
                        _report_for_region(split_report, region)
                    )
        else:
            render_class_performance(headline_report)

    with tab4:
        if run_reports_by_split and len(run_reports_by_split) > 1:
            split_tabs = st.tabs(
                [f"Confusion Matrix ({s})" for s in run_reports_by_split]
            )
            for split_tab, (split_name, split_report) in zip(
                split_tabs, run_reports_by_split.items()
            ):
                with split_tab:
                    render_confusion_matrix(
                        _report_for_region(split_report, region),
                        predictions_data=split_predictions.get(split_name),
                        region=region,
                        key_prefix=f"cm_{region}_{split_name}",
                    )
        else:
            render_confusion_matrix(
                headline_report,
                predictions_data=headline_predictions,
                region=region,
                key_prefix=f"cm_{region}",
            )

    with tab5:
        if region == ALL_REGIONS_LABEL:
            render_run_comparison(data, [selected_run])
        else:
            st.caption(
                "Cross-run comparisons use the global metrics for each run "
                f"and ignore the active region tab. See the **{ALL_REGIONS_LABEL}** "
                "outer tab for the comparison view."
            )

    with tab6:
        class_names = (
            run_eval_report.class_names if run_eval_report is not None else None
        )
        pred_tab_val, pred_tab_test = st.tabs(
            ["Predictions (Val)", "Predictions (Test)"]
        )

        with pred_tab_val:
            val_filtered = filter_predictions_by_region(run_val_preds, region)
            render_image_prediction_viewer(
                annotations=_build_annotations(val_filtered),
                class_names=class_names,
                key_prefix=f"val_{region}",
            )

        with pred_tab_test:
            test_filtered = filter_predictions_by_region(run_test_preds, region)
            render_image_prediction_viewer(
                annotations=_build_annotations(test_filtered),
                class_names=class_names,
                key_prefix=f"test_{region}",
            )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def main() -> None:
    """Main application entry point.

    Sets up page config, loads data, renders sidebar and tabbed content.
    Handles top-level exceptions with a generic error page.
    """
    try:
        st.set_page_config(
            page_title="Road Damage Detection - Results",
            layout="wide",
        )

        results_dir = Path(st.query_params.get("results_dir", "results"))
        checkpoints_dir = Path(
            st.query_params.get("checkpoints_dir", "checkpoints")
        )

        data = load_all_data(results_dir, checkpoints_dir)

        selected_run = render_sidebar(data)
        if selected_run is None:
            render_welcome()
            return

        short_id = selected_run.run_id[:8]
        st.title(
            f"Road Damage Detection - {selected_run.model_name} ({short_id})"
        )

        run_eval_report = data.evaluation_reports.get(
            selected_run.run_id, data.evaluation_report
        )
        run_val_preds = data.predictions_by_run.get(selected_run.run_id)
        run_train_preds = data.train_predictions_by_run.get(selected_run.run_id)
        run_test_preds = data.test_predictions_by_run.get(selected_run.run_id)
        run_reports_by_split = data.evaluation_reports_by_split.get(
            selected_run.run_id, {}
        )

        # Outer region-tab navigation (Plan A): "All" + every region on
        # the run's per_region field. Each tab below renders the full
        # inner six-tab layout, with widget keys suffixed by the active
        # region label so state stays isolated per tab.
        labels, region_tabs = render_region_tabs(run_eval_report)
        for label, tab in zip(labels, region_tabs):
            with tab:
                _render_region_inner_tabs(
                    region=label,
                    selected_run=selected_run,
                    data=data,
                    run_eval_report=run_eval_report,
                    run_reports_by_split=run_reports_by_split,
                    run_val_preds=run_val_preds,
                    run_test_preds=run_test_preds,
                    run_train_preds=run_train_preds,
                )

    except Exception as e:
        logger.exception("Unhandled exception in dashboard application")
        st.error(
            "An unexpected error occurred. Please check the application logs "
            "for details and try refreshing the page."
        )
        st.exception(e)


if __name__ == "__main__":
    main()
