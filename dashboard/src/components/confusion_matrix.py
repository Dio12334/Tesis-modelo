"""Confusion matrix component for the Streamlit Results Dashboard.

Renders the confusion matrix as an annotated heatmap with class labels
on rows (ground truth) and columns (predicted), using a sequential color
scale where higher values have more intense color.
"""

from typing import Optional

import plotly.figure_factory as ff
import streamlit as st

from data_loader import EvaluationReport


def render_confusion_matrix(report: Optional[EvaluationReport]) -> None:
    """Render annotated heatmap of the confusion matrix.

    Displays the confusion matrix from the evaluation report as a
    color-coded heatmap with numeric counts in each cell. Rows represent
    ground truth classes and columns represent predicted classes.

    Args:
        report: The evaluation report containing the confusion matrix,
            or None if evaluation has not been performed.
    """
    st.header("Confusion Matrix")

    if report is None:
        st.info(
            "No evaluation report available. "
            "Run model evaluation to see the confusion matrix here."
        )
        return

    matrix = report.confusion_matrix
    class_names = report.class_names

    # If matrix is (C+1)x(C+1), the last row/column is "background"
    if len(matrix) == len(class_names) + 1:
        display_names = class_names + ["background"]
    else:
        display_names = class_names

    # Check if confusion matrix is all zeros
    if all(cell == 0 for row in matrix for cell in row):
        st.info(
            "Insufficient detections for confusion analysis. "
            "The confusion matrix contains all zeros."
        )
        return

    # Create annotated heatmap using Plotly figure_factory
    # Rows = ground truth, Columns = predicted
    fig = ff.create_annotated_heatmap(
        z=matrix,
        x=display_names,
        y=display_names,
        annotation_text=[[str(cell) for cell in row] for row in matrix],
        colorscale="Blues",
        showscale=True,
    )

    # Update layout for clarity
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="Ground Truth Class",
        xaxis=dict(side="bottom"),
        height=500,
    )

    # Ensure y-axis is not auto-reversed (ground truth top-to-bottom)
    fig.update_yaxes(autorange="reversed")

    st.plotly_chart(fig, use_container_width=True)
