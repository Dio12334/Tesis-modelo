"""Sidebar component for run selection and filtering.

Renders the sidebar with model name filtering and a selectable run list.
"""

from typing import Optional

import streamlit as st

from data_loader import DashboardData, ExperimentRun


def render_sidebar(data: DashboardData) -> Optional[ExperimentRun]:
    """Render sidebar controls and return the selected run (or None).

    Displays a model name filter dropdown and a list of runs filtered
    by the selected model. Returns the ExperimentRun selected by the user,
    or None if no run is selected.

    Args:
        data: The aggregated dashboard data containing all runs.

    Returns:
        The selected ExperimentRun, or None if no selection is made.
    """
    with st.sidebar:
        st.header("Experiment Runs")

        if not data.runs:
            st.info("No experiment runs available.")
            return None

        # Show warning count if there were loading errors
        if data.errors:
            st.warning(f"{len(data.errors)} file(s) failed to load.")

        # Model name filter
        selected_model = render_model_filter(data)

        # Filter runs by selected model
        if selected_model == "All":
            filtered_runs = data.runs
        else:
            filtered_runs = [
                run for run in data.runs if run.model_name == selected_model
            ]

        if not filtered_runs:
            st.info("No runs match the selected filter.")
            return None

        # Render selectable run list
        selected_run_id = render_run_list(filtered_runs)

        if selected_run_id is None:
            return None

        # Find and return the selected ExperimentRun
        for run in filtered_runs:
            if run.run_id == selected_run_id:
                return run

    return None


def render_model_filter(data: DashboardData) -> str:
    """Render model name filter dropdown, return selected model.

    Extracts unique model names from all runs and presents them
    in a selectbox with an "All" option.

    Args:
        data: The aggregated dashboard data containing all runs.

    Returns:
        The selected model name string, or "All" for no filtering.
    """
    model_names = sorted(set(run.model_name for run in data.runs))
    options = ["All"] + model_names
    selected = st.selectbox("Filter by model", options, key="model_filter")
    return selected


def render_run_list(runs: list[ExperimentRun]) -> Optional[str]:
    """Render the run list and return selected run_id.

    Displays a radio button list of runs showing key information:
    run_id (truncated), model_name, start_time, final_val_loss, and total_epochs.

    Args:
        runs: List of ExperimentRun objects to display.

    Returns:
        The run_id of the selected run, or None if no selection.
    """
    # Build display labels for each run
    run_options = []
    for run in runs:
        short_id = run.run_id[:8]

        label = f"{run.model_name} ({short_id})"
        run_options.append(label)

    if not run_options:
        return None

    selected_index = st.radio(
        "Select a run",
        range(len(run_options)),
        format_func=lambda i: run_options[i],
        key="run_selector",
    )

    if selected_index is not None:
        return runs[selected_index].run_id

    return None
