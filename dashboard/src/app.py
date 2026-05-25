"""Main application entry point for the Streamlit Results Dashboard.

Wires together all dashboard components: data loading, sidebar navigation,
and tabbed content views for metrics, charts, and comparisons.

Launch with: streamlit run dashboard/app.py
"""

import logging
from pathlib import Path

import streamlit as st

from dashboard.data_loader import load_all_data
from dashboard.components.sidebar import render_sidebar
from dashboard.components.metrics_overview import render_metrics_overview
from dashboard.components.loss_charts import render_loss_chart, render_learning_rate_chart
from dashboard.components.class_performance import render_class_performance
from dashboard.components.confusion_matrix import render_confusion_matrix
from dashboard.components.run_comparison import render_run_comparison
from dashboard.components.config_display import render_config
from dashboard.components.image_prediction_viewer import (
    render_image_prediction_viewer,
    ImageAnnotation,
    BoundingBox,
)
from dashboard.data_loader import DashboardData
from typing import Optional

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

        # Build ground truth boxes
        gt_boxes = []
        gt_box_coords = gt_data.get("boxes", [])
        gt_labels = gt_data.get("labels", [])
        for coords, label in zip(gt_box_coords, gt_labels):
            if len(coords) == 4:
                gt_boxes.append(BoundingBox(
                    x_min=coords[0], y_min=coords[1],
                    x_max=coords[2], y_max=coords[3],
                    class_name=label, confidence=None,
                ))

        # Build prediction boxes
        pred_boxes = []
        pred_box_coords = pred_data.get("boxes", [])
        pred_labels = pred_data.get("labels", [])
        pred_scores = pred_data.get("scores", [])
        for coords, label, score in zip(pred_box_coords, pred_labels, pred_scores):
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
        """
    )


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

        # Read directory paths from query params with defaults
        results_dir = Path(st.query_params.get("results_dir", "results"))
        checkpoints_dir = Path(st.query_params.get("checkpoints_dir", "checkpoints"))

        # Load all experiment data
        data = load_all_data(results_dir, checkpoints_dir)

        # Render sidebar and get selected run
        selected_run = render_sidebar(data)

        # If no run is selected, show welcome message
        if selected_run is None:
            render_welcome()
            return

        # Display page title with selected run info
        short_id = selected_run.run_id[:8]
        st.title(f"Road Damage Detection - {selected_run.model_name} ({short_id})")

        # Get per-run evaluation report (fall back to global)
        run_eval_report = data.evaluation_reports.get(
            selected_run.run_id, data.evaluation_report
        )

        # Get per-run predictions
        run_val_preds = data.predictions_by_run.get(selected_run.run_id)
        run_train_preds = data.train_predictions_by_run.get(selected_run.run_id)

        # Main content organized into tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Overview",
            "Loss Curves",
            "Class Performance",
            "Confusion Matrix",
            "Compare Runs",
            "Predictions",
        ])

        with tab1:
            render_metrics_overview(run_eval_report)
            render_config(selected_run)

        with tab2:
            render_loss_chart(selected_run)
            render_learning_rate_chart(selected_run)

        with tab3:
            render_class_performance(run_eval_report)

        with tab4:
            render_confusion_matrix(run_eval_report)

        with tab5:
            render_run_comparison(data, [selected_run])

        with tab6:
            # Sub-tabs for validation vs training predictions
            class_names = None
            if run_eval_report:
                class_names = run_eval_report.class_names

            pred_tab_val, pred_tab_train = st.tabs(["Predictions (Val)", "Predictions (Train)"])

            with pred_tab_val:
                val_annotations = _build_annotations(run_val_preds)
                render_image_prediction_viewer(
                    annotations=val_annotations,
                    class_names=class_names,
                    key_prefix="val",
                )

            with pred_tab_train:
                train_annotations = _build_annotations(run_train_preds)
                render_image_prediction_viewer(
                    annotations=train_annotations,
                    class_names=class_names,
                    key_prefix="train",
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
