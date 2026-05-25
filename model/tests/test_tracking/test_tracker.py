"""Property-based tests for ExperimentTracker."""

# Feature: road-damage-evaluation-framework, Property 17: Experiment IDs are unique

import tempfile
from pathlib import Path

from hypothesis import given, settings, strategies as st

from model.tracking.tracker import ExperimentTracker


# Strategies for generating random experiment inputs
config_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
    values=st.one_of(
        st.integers(min_value=0, max_value=1000),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz"),
        st.booleans(),
    ),
    min_size=0,
    max_size=5,
)

model_name_strategy = st.text(
    min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_0123456789"
)

dataset_name_strategy = st.text(
    min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_0123456789"
)


@settings(max_examples=100)
@given(
    n=st.integers(min_value=2, max_value=20),
    configs=st.lists(config_strategy, min_size=20, max_size=20),
    model_names=st.lists(model_name_strategy, min_size=20, max_size=20),
    dataset_names=st.lists(dataset_name_strategy, min_size=20, max_size=20),
)
def test_experiment_ids_are_unique(n, configs, model_names, dataset_names):
    """For any sequence of N experiment runs started in succession,
    all assigned run IDs SHALL be distinct.

    **Validates: Requirements 7.1**
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracker = ExperimentTracker(output_dir=Path(tmp_dir) / "experiments")

        run_ids = []
        for i in range(n):
            run_id = tracker.start_run(
                config=configs[i],
                model_name=model_names[i],
                dataset_name=dataset_names[i],
            )
            run_ids.append(run_id)

        # All IDs must be distinct
        assert len(set(run_ids)) == len(run_ids), (
            f"Expected {len(run_ids)} unique IDs but got {len(set(run_ids))} unique out of {run_ids}"
        )
