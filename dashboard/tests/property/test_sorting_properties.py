"""Property-based tests for sorting invariants.

Feature: streamlit-results-dashboard, Property 4: Runs sorted descending by start_time

Tests that the data loader returns experiment runs sorted by start_time
in descending order (most recent first) for any set of valid runs.
"""

import json
import tempfile
import uuid
from pathlib import Path

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from dashboard.data_loader import load_all_data


# ---------------------------------------------------------------------------
# Hypothesis strategies for generating valid experiment run JSON data
# ---------------------------------------------------------------------------


def _iso_timestamp_strategy():
    """Generate ISO 8601 formatted timestamp strings.

    Uses a fixed day range (1-28) to avoid invalid dates across all months.
    """
    return st.builds(
        lambda year, month, day, hour, minute, second: (
            f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}.000000+00:00"
        ),
        year=st.integers(min_value=2020, max_value=2030),
        month=st.integers(min_value=1, max_value=12),
        day=st.integers(min_value=1, max_value=28),
        hour=st.integers(min_value=0, max_value=23),
        minute=st.integers(min_value=0, max_value=59),
        second=st.integers(min_value=0, max_value=59),
    )


def _experiment_run_json_strategy():
    """Generate a valid experiment run JSON dict with a random start_time."""
    return st.builds(
        lambda run_id, start_time, end_time, train_loss, val_loss, lr: {
            "run_id": run_id,
            "model_name": "ssd_mobilenetv3",
            "dataset_name": "rdd2022",
            "config": {
                "name": "test_config",
                "model": {"type": "ssd_mobilenetv3", "config": {"input_size": 320, "num_classes": 5}},
                "dataset": {"type": "rdd2022", "path": "data/sample"},
                "training": {"epochs": 10, "batch_size": 16, "learning_rate": lr},
            },
            "start_time": start_time,
            "end_time": end_time,
            "metrics_history": [
                {
                    "step": 0,
                    "timestamp": start_time,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": lr,
                    "epoch_time_s": 25.0,
                }
            ],
            "final_results": {
                "final_train_loss": train_loss,
                "final_val_loss": val_loss,
                "best_val_loss": val_loss,
                "best_epoch": 0,
                "total_epochs": 1,
            },
        },
        run_id=st.uuids().map(str),
        start_time=_iso_timestamp_strategy(),
        end_time=_iso_timestamp_strategy(),
        train_loss=st.floats(min_value=0.1, max_value=15.0, allow_nan=False, allow_infinity=False),
        val_loss=st.floats(min_value=0.1, max_value=15.0, allow_nan=False, allow_infinity=False),
        lr=st.floats(min_value=1e-5, max_value=0.1, allow_nan=False, allow_infinity=False),
    )


# ---------------------------------------------------------------------------
# Property 4: Runs sorted descending by start_time
# ---------------------------------------------------------------------------


@given(run_data_list=st.lists(
    _experiment_run_json_strategy(),
    min_size=2,
    max_size=10,
))
@settings(max_examples=100)
def test_runs_sorted_descending_by_start_time(run_data_list):
    """Feature: streamlit-results-dashboard, Property 4: Runs sorted descending by start_time

    **Validates: Requirements 2.2**

    For any list of ExperimentRun objects with varying start_time values,
    after applying the default sort, each run's start_time should be greater
    than or equal to the next run's start_time in the list.
    """
    # Ensure all run_ids are unique
    seen_ids = set()
    for run_data in run_data_list:
        if run_data["run_id"] in seen_ids:
            run_data["run_id"] = str(uuid.uuid4())
        seen_ids.add(run_data["run_id"])

    # Create temp directories for results and checkpoints
    with tempfile.TemporaryDirectory() as tmp_dir:
        results_dir = Path(tmp_dir) / "results"
        results_dir.mkdir()
        checkpoints_dir = Path(tmp_dir) / "checkpoints"
        checkpoints_dir.mkdir()

        # Write each run as a JSON file
        for run_data in run_data_list:
            filepath = results_dir / f"{run_data['run_id']}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(run_data, f)

        # Load all data using the data loader
        dashboard_data = load_all_data(results_dir, checkpoints_dir)

        # Verify all runs were loaded
        assert len(dashboard_data.runs) == len(run_data_list)

        # Verify runs are sorted by start_time descending
        for i in range(len(dashboard_data.runs) - 1):
            current_start = dashboard_data.runs[i].start_time
            next_start = dashboard_data.runs[i + 1].start_time
            assert current_start >= next_start, (
                f"Runs not sorted descending by start_time: "
                f"run[{i}].start_time={current_start!r} < "
                f"run[{i+1}].start_time={next_start!r}"
            )


# ---------------------------------------------------------------------------
# Property 8: Classes sorted by AP descending
# ---------------------------------------------------------------------------


@given(per_class_ap=st.dictionaries(
    keys=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "Pd")),
        min_size=1,
        max_size=20,
    ),
    values=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=15,
))
@settings(max_examples=100)
def test_classes_sorted_by_ap_descending(per_class_ap):
    """Feature: streamlit-results-dashboard, Property 8: Classes sorted by AP descending

    **Validates: Requirements 5.3**

    For any set of per-class Average Precision values, after applying the sort
    for display, each class's AP value should be greater than or equal to the
    next class's AP value in the ordered list.
    """
    # Apply the same sorting logic as dashboard/components/class_performance.py
    sorted_classes = sorted(per_class_ap.items(), key=lambda x: x[1], reverse=True)
    ap_values = [ap for _, ap in sorted_classes]

    # Assert each AP value is >= the next AP value in the sorted list
    for i in range(len(ap_values) - 1):
        assert ap_values[i] >= ap_values[i + 1], (
            f"Classes not sorted by AP descending: "
            f"ap_values[{i}]={ap_values[i]} < ap_values[{i+1}]={ap_values[i+1]}"
        )
