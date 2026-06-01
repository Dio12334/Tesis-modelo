"""Metrics overview component for the Streamlit Results Dashboard.

Displays key evaluation metrics (mAP, precision, recall, F1-score) as
prominent KPI cards, along with evaluation parameters.
"""

from typing import Optional

import streamlit as st

from data_loader import EvaluationReport


def render_metrics_overview(report: Optional[EvaluationReport]) -> None:
    """Render evaluation metrics as prominent KPI cards.

    Displays mAP@0.5, mAP@0.5:0.95, precision, recall, and F1-score
    as large metric cards. Also shows confidence threshold, IoU threshold,
    number of validation images, and number of classes.

    Args:
        report: The evaluation report to display, or None if evaluation
            has not been performed.
    """
    st.header("Evaluation Metrics")

    if report is None:
        st.info(
            "Evaluation has not been performed. "
            "Run model evaluation to see metrics here."
        )
        return

    metrics = report.metrics

    # Primary metrics row: mAP values
    st.subheader("Detection Performance")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        map_50 = metrics.get("mAP@0.5", 0.0)
        st.metric(label="mAP@0.5", value=f"{map_50:.2f}")

    with col2:
        map_50_95 = metrics.get("mAP@0.5:0.95", 0.0)
        st.metric(label="mAP@0.5:0.95", value=f"{map_50_95:.2f}")

    with col3:
        precision = metrics.get("precision", 0.0)
        st.metric(label="Precision", value=f"{precision:.2f}")

    with col4:
        recall = metrics.get("recall", 0.0)
        st.metric(label="Recall", value=f"{recall:.2f}")

    with col5:
        f1_score = metrics.get("f1_score", 0.0)
        st.metric(label="F1-Score", value=f"{f1_score:.2f}")

    # Evaluation parameters row
    st.subheader("Evaluation Parameters")
    param_col1, param_col2, param_col3, param_col4 = st.columns(4)

    with param_col1:
        st.metric(
            label="Confidence Threshold",
            value=f"{report.confidence_threshold:.2f}",
        )

    with param_col2:
        st.metric(label="IoU Threshold", value=f"{report.iou_threshold:.2f}")

    with param_col3:
        st.metric(label="Validation Images", value=str(report.num_val_images))

    with param_col4:
        st.metric(label="Number of Classes", value=str(report.num_classes))
