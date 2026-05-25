"""Loss charts component for the Streamlit Results Dashboard.

Renders training/validation loss curves and learning rate schedule
using Plotly interactive charts.
"""

import streamlit as st
import plotly.graph_objects as go

from dashboard.data_loader import ExperimentRun


def render_loss_chart(run: ExperimentRun) -> None:
    """Render dual-line loss chart with best epoch marker.

    Displays training loss and validation loss over epochs on the same chart
    with distinct colors, a legend, and a marker at the best epoch (minimum
    val_loss). Hover tooltips show epoch number, loss value, and learning rate.

    Args:
        run: The selected ExperimentRun to visualize.
    """
    if not run.metrics_history:
        st.info("No training history available for this run.")
        return

    epochs = [entry.step for entry in run.metrics_history]
    train_losses = [entry.train_loss for entry in run.metrics_history]
    val_losses = [entry.val_loss for entry in run.metrics_history]
    learning_rates = [entry.learning_rate for entry in run.metrics_history]

    fig = go.Figure()

    # Training loss line
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=train_losses,
            mode="lines+markers",
            name="Train Loss",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=4),
            customdata=learning_rates,
            hovertemplate=(
                "Epoch: %{x}<br>"
                "Train Loss: %{y:.4f}<br>"
                "Learning Rate: %{customdata:.2e}"
                "<extra></extra>"
            ),
        )
    )

    # Validation loss line
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=val_losses,
            mode="lines+markers",
            name="Val Loss",
            line=dict(color="#ff7f0e", width=2),
            marker=dict(size=4),
            customdata=learning_rates,
            hovertemplate=(
                "Epoch: %{x}<br>"
                "Val Loss: %{y:.4f}<br>"
                "Learning Rate: %{customdata:.2e}"
                "<extra></extra>"
            ),
        )
    )

    # Best epoch marker (minimum val_loss)
    best_idx = val_losses.index(min(val_losses))
    best_epoch = epochs[best_idx]
    best_val_loss = val_losses[best_idx]
    best_lr = learning_rates[best_idx]

    fig.add_trace(
        go.Scatter(
            x=[best_epoch],
            y=[best_val_loss],
            mode="markers",
            name=f"Best Epoch ({best_epoch})",
            marker=dict(
                color="#2ca02c",
                size=14,
                symbol="star",
                line=dict(width=2, color="darkgreen"),
            ),
            customdata=[best_lr],
            hovertemplate=(
                "Best Epoch: %{x}<br>"
                "Val Loss: %{y:.4f}<br>"
                "Learning Rate: %{customdata:.2e}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Training & Validation Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        hovermode="x unified",
        template="plotly_white",
        height=450,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_learning_rate_chart(run: ExperimentRun) -> None:
    """Render learning rate schedule chart.

    Displays the learning rate over epochs with hover tooltips showing
    epoch number and learning rate value.

    Args:
        run: The selected ExperimentRun to visualize.
    """
    if not run.metrics_history:
        st.info("No training history available for this run.")
        return

    epochs = [entry.step for entry in run.metrics_history]
    learning_rates = [entry.learning_rate for entry in run.metrics_history]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=learning_rates,
            mode="lines+markers",
            name="Learning Rate",
            line=dict(color="#9467bd", width=2),
            marker=dict(size=4),
            hovertemplate=(
                "Epoch: %{x}<br>"
                "Learning Rate: %{y:.2e}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Learning Rate Schedule",
        xaxis_title="Epoch",
        yaxis_title="Learning Rate",
        hovermode="x unified",
        template="plotly_white",
        height=350,
    )

    st.plotly_chart(fig, use_container_width=True)
