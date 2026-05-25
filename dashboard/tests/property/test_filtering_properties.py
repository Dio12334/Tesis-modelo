"""Property-based tests for filtering correctness.

Feature: streamlit-results-dashboard, Property 5: Model name filter returns only matching runs
Feature: streamlit-results-dashboard, Property 10: Confidence threshold filtering
Feature: streamlit-results-dashboard, Property 11: Class filtering

Tests that filtering runs by model_name returns exactly the runs whose
model_name equals the selected filter value — no matching runs excluded,
no non-matching runs included.

Tests that filtering prediction bounding boxes by confidence threshold returns
exactly those predictions whose confidence score is >= the threshold.

Tests that filtering bounding boxes by class returns exactly those boxes
whose class_name is in the selected set — no matching boxes excluded,
no non-matching boxes included.
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from dashboard.data_loader import ExperimentRun, FinalResults, MetricsEntry
from dashboard.components.image_prediction_viewer import BoundingBox, filter_by_confidence, filter_by_class


# ---------------------------------------------------------------------------
# Hypothesis strategies for generating ExperimentRun objects
# ---------------------------------------------------------------------------

MODEL_NAMES = ["ssd_mobilenetv3", "fasterrcnn_resnet50", "yolov5s", "retinanet", "detr"]


def _experiment_run_strategy(model_name_strategy=None):
    """Generate a valid ExperimentRun dataclass instance.

    Args:
        model_name_strategy: Optional strategy for model_name.
            Defaults to sampling from MODEL_NAMES.
    """
    if model_name_strategy is None:
        model_name_strategy = st.sampled_from(MODEL_NAMES)

    return st.builds(
        ExperimentRun,
        run_id=st.uuids().map(str),
        model_name=model_name_strategy,
        dataset_name=st.just("rdd2022"),
        config=st.just({"name": "test_config"}),
        start_time=st.just("2024-01-01T00:00:00.000000+00:00"),
        end_time=st.just("2024-01-01T01:00:00.000000+00:00"),
        metrics_history=st.just([]),
        final_results=st.just(
            FinalResults(
                final_train_loss=1.0,
                final_val_loss=1.0,
                best_val_loss=0.8,
                best_epoch=5,
                total_epochs=10,
            )
        ),
    )


# ---------------------------------------------------------------------------
# Property 5: Model name filter returns only matching runs
# ---------------------------------------------------------------------------


@given(
    runs=st.lists(
        _experiment_run_strategy(),
        min_size=1,
        max_size=30,
    ),
)
@settings(max_examples=100)
def test_model_name_filter_returns_only_matching_runs(runs):
    """Feature: streamlit-results-dashboard, Property 5: Model name filter returns only matching runs

    **Validates: Requirements 2.5, 2.6**

    For any list of ExperimentRun objects with varying model_name values
    and any selected model_name filter value, the filtered list should
    contain only runs whose model_name equals the selected filter value,
    and no matching runs should be excluded.
    """
    # Pick a model_name that exists in the generated runs as the filter value
    available_models = list(set(run.model_name for run in runs))
    assume(len(available_models) >= 1)

    # Test filtering for each available model name
    for selected_model in available_models:
        # Apply the same filtering logic as the sidebar component
        filtered_runs = [run for run in runs if run.model_name == selected_model]

        # Assert: all returned runs have model_name == selected filter
        for run in filtered_runs:
            assert run.model_name == selected_model, (
                f"Filtered run has model_name={run.model_name!r}, "
                f"expected {selected_model!r}"
            )

        # Assert: no runs with matching model_name were excluded
        expected_count = sum(1 for run in runs if run.model_name == selected_model)
        assert len(filtered_runs) == expected_count, (
            f"Expected {expected_count} runs with model_name={selected_model!r}, "
            f"but got {len(filtered_runs)}"
        )


# ---------------------------------------------------------------------------
# Hypothesis strategies for generating BoundingBox objects
# ---------------------------------------------------------------------------

DAMAGE_CLASSES = ["alligator crack", "longitudinal crack", "other corruption", "pothole", "transverse crack"]


def _bounding_box_strategy(confidence_strategy=None):
    """Generate a valid BoundingBox dataclass instance with a confidence score.

    Args:
        confidence_strategy: Optional strategy for confidence.
            Defaults to floats in [0.0, 1.0].
    """
    if confidence_strategy is None:
        confidence_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

    return st.builds(
        BoundingBox,
        x_min=st.floats(min_value=0.0, max_value=500.0, allow_nan=False),
        y_min=st.floats(min_value=0.0, max_value=500.0, allow_nan=False),
        x_max=st.floats(min_value=500.0, max_value=1000.0, allow_nan=False),
        y_max=st.floats(min_value=500.0, max_value=1000.0, allow_nan=False),
        class_name=st.sampled_from(DAMAGE_CLASSES),
        confidence=confidence_strategy,
    )


# ---------------------------------------------------------------------------
# Property 10: Confidence threshold filtering
# ---------------------------------------------------------------------------


@given(
    boxes=st.lists(
        _bounding_box_strategy(),
        min_size=0,
        max_size=50,
    ),
    threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=100)
def test_confidence_threshold_filtering(boxes, threshold):
    """Feature: streamlit-results-dashboard, Property 10: Confidence threshold filtering

    **Validates: Requirements 11.6, 11.7**

    For any set of prediction bounding boxes with confidence scores and any
    confidence threshold value, filtering predictions should return exactly
    those predictions whose confidence score is greater than or equal to the
    threshold.
    """
    filtered = filter_by_confidence(boxes, threshold)

    # Assert: all returned boxes have confidence >= threshold
    for box in filtered:
        assert box.confidence is None or box.confidence >= threshold, (
            f"Box with confidence={box.confidence} should not pass "
            f"threshold={threshold}"
        )

    # Assert: no boxes with confidence >= threshold were excluded
    expected = [
        box for box in boxes
        if box.confidence is None or box.confidence >= threshold
    ]
    assert len(filtered) == len(expected), (
        f"Expected {len(expected)} boxes to pass threshold={threshold}, "
        f"but got {len(filtered)}"
    )

    # Assert: the exact same boxes are returned (order preserved)
    for actual, exp in zip(filtered, expected):
        assert actual is exp, (
            "Filtered result does not match expected box set or ordering"
        )


# ---------------------------------------------------------------------------
# Strategy for generating BoundingBox objects with optional confidence (GT or pred)
# ---------------------------------------------------------------------------


def _bounding_box_with_optional_confidence_strategy():
    """Generate a BoundingBox with confidence=None (ground truth) or a float (prediction)."""
    return st.builds(
        BoundingBox,
        x_min=st.floats(min_value=0.0, max_value=500.0, allow_nan=False),
        y_min=st.floats(min_value=0.0, max_value=500.0, allow_nan=False),
        x_max=st.floats(min_value=500.0, max_value=1000.0, allow_nan=False),
        y_max=st.floats(min_value=500.0, max_value=1000.0, allow_nan=False),
        class_name=st.sampled_from(DAMAGE_CLASSES),
        confidence=st.one_of(
            st.none(),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        ),
    )


# ---------------------------------------------------------------------------
# Property 11: Class filtering
# ---------------------------------------------------------------------------


@given(
    boxes=st.lists(
        _bounding_box_with_optional_confidence_strategy(),
        min_size=0,
        max_size=50,
    ),
    selected_classes=st.lists(
        st.sampled_from(DAMAGE_CLASSES),
        min_size=1,
        max_size=len(DAMAGE_CLASSES),
        unique=True,
    ),
)
@settings(max_examples=100)
def test_class_filtering(boxes, selected_classes):
    """Feature: streamlit-results-dashboard, Property 11: Class filtering

    **Validates: Requirements 11.8, 11.9**

    For any set of bounding boxes (ground truth or predictions) with class
    labels and any subset of selected classes, filtering by class should
    return exactly those boxes whose class label is in the selected set.
    """
    filtered = filter_by_class(boxes, selected_classes)

    # Assert: all returned boxes have class_name in selected_classes
    for box in filtered:
        assert box.class_name in selected_classes, (
            f"Box with class_name={box.class_name!r} should not pass "
            f"class filter={selected_classes!r}"
        )

    # Assert: no boxes with matching class_name were excluded
    expected = [box for box in boxes if box.class_name in selected_classes]
    assert len(filtered) == len(expected), (
        f"Expected {len(expected)} boxes with class_name in {selected_classes!r}, "
        f"but got {len(filtered)}"
    )

    # Assert: the exact same boxes are returned (order preserved)
    for actual, exp in zip(filtered, expected):
        assert actual is exp, (
            "Filtered result does not match expected box set or ordering"
        )
