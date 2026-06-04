"""Confusion matrix component for the Streamlit Results Dashboard.

Renders the confusion matrix as an annotated heatmap with class labels
on rows (ground truth) and columns (predicted), using a sequential color
scale where higher values have more intense color.
"""

from typing import Optional

import numpy as np
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

    # Display mode selector
    display_mode = st.radio(
        "Display mode",
        options=["Nominal", "Percentage (row-wise)"],
        horizontal=True,
        help="Nominal shows absolute counts. Percentage shows row-wise normalization (% of each ground truth class).",
    )

    # Convert matrix to numpy array for easier manipulation
    matrix_np = np.array(matrix, dtype=float)

    if display_mode == "Percentage (row-wise)":
        # Row-wise normalization: percentage of each ground truth class
        row_sums = matrix_np.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        matrix_percent = (matrix_np / row_sums) * 100
        z_values = matrix_percent.tolist()
        annotation_text = [[f"{val:.1f}%" for val in row] for row in matrix_percent]
        title = "Confusion Matrix (Percentage)"
    else:
        z_values = matrix
        annotation_text = [[str(cell) for cell in row] for row in matrix]
        title = "Confusion Matrix"

    # Create annotated heatmap using Plotly figure_factory
    # Rows = ground truth, Columns = predicted
    fig = ff.create_annotated_heatmap(
        z=z_values,
        x=display_names,
        y=display_names,
        annotation_text=annotation_text,
        colorscale="Blues",
        showscale=True,
    )

    # Update layout for clarity
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Class",
        yaxis_title="Ground Truth Class",
        xaxis=dict(side="bottom"),
        height=500,
    )

    # Ensure y-axis is not auto-reversed (ground truth top-to-bottom)
    fig.update_yaxes(autorange="reversed")

    st.plotly_chart(fig, use_container_width=True)
