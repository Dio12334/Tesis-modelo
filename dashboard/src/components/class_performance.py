"""Class performance component for per-class AP visualization.

Renders a horizontal bar chart showing Average Precision per class,
sorted in descending order, with visual distinction for zero-AP classes.
"""

from typing import Optional

import plotly.graph_objects as go
import streamlit as st

from data_loader import EvaluationReport


def render_class_performance(report: Optional[EvaluationReport]) -> None:
    """Render horizontal bar chart of per-class AP sorted descending.

    Displays a Plotly horizontal bar chart where each bar represents a
    damage class and its Average Precision value. Classes are sorted by
    AP in descending order. Classes with zero AP are visually distinguished
    using reduced opacity.

    Args:
        report: The evaluation report containing per-class AP metrics,
                or None if no evaluation has been performed.
    """
    if report is None:
        st.info("No evaluation report available. Run model evaluation to see class performance.")
        return

    per_class_ap = report.metrics.get("per_class_ap")
    if not per_class_ap:
        st.info("No per-class AP data available in the evaluation report.")
        return

    # Sort classes by AP descending
    sorted_classes = sorted(per_class_ap.items(), key=lambda x: x[1], reverse=True)
    class_names = [name for name, _ in sorted_classes]
    ap_values = [ap for _, ap in sorted_classes]

    # Assign colors: full opacity for non-zero AP, reduced opacity for zero AP
    colors = [
        "rgba(99, 110, 250, 0.3)" if ap == 0.0 else "rgba(99, 110, 250, 1.0)"
        for ap in ap_values
    ]

    # Build horizontal bar chart
    fig = go.Figure(
        go.Bar(
            x=ap_values,
            y=class_names,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{ap:.4f}" for ap in ap_values],
            textposition="outside",
            textfont=dict(size=12),
        )
    )

    fig.update_layout(
        title="Per-Class Average Precision",
        xaxis_title="Average Precision",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),  # Keep descending order top-to-bottom
        xaxis=dict(range=[0, max(ap_values) * 1.2 if max(ap_values) > 0 else 1.0]),
        height=max(300, len(class_names) * 60),
        margin=dict(l=20, r=80, t=50, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)
