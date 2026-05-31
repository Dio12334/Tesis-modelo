"""Property-based tests for prediction count consistency.

Feature: streamlit-results-dashboard, Property 12: Prediction count matches filtered set size

Tests that for any set of prediction bounding boxes and any combination of
confidence threshold and class filter, the displayed count of predicted boxes
equals the number of boxes that pass both the confidence threshold and class filter.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from dashboard.components.image_prediction_viewer import (
    BoundingBox,
    filter_by_confidence,
    filter_by_class,
)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    "alligator crack",
    "longitudinal crack",
    "other corruption",
    "pothole",
    "transverse crack",
]


def _bounding_box_strategy():
    """Generate a BoundingBox with random confidence and class_name."""
    return st.builds(
        BoundingBox,
        x_min=st.floats(min_value=0.0, max_value=500.0),
        y_min=st.floats(min_value=0.0, max_value=500.0),
        x_max=st.floats(min_value=500.0, max_value=1000.0),
        y_max=st.floats(min_value=500.0, max_value=1000.0),
        class_name=st.sampled_from(CLASS_NAMES),
        confidence=st.floats(min_value=0.0, max_value=1.0),
    )


# ---------------------------------------------------------------------------
# Property 12: Prediction count matches filtered set size
# ---------------------------------------------------------------------------


@given(
    boxes=st.lists(_bounding_box_strategy(), min_size=0, max_size=50),
    threshold=st.floats(min_value=0.0, max_value=1.0),
    selected_classes=st.lists(
        st.sampled_from(CLASS_NAMES), min_size=0, max_size=5, unique=True
    ),
)
@settings(max_examples=200)
def test_prediction_count_matches_filtered_set_size(boxes, threshold, selected_classes):
    """Feature: streamlit-results-dashboard, Property 12: Prediction count matches filtered set size

    **Validates: Requirements 11.10**

    For any set of prediction bounding boxes and any combination of confidence
    threshold and class filter, the displayed count of predicted boxes should
    equal the number of boxes that pass both the confidence threshold and class
    filter.
    """
    # Apply filters in the same order as the component:
    # confidence first, then class
    after_confidence = filter_by_confidence(boxes, threshold)
    after_both = filter_by_class(after_confidence, selected_classes)

    # Independently count boxes that pass both filters
    expected_count = sum(
        1
        for box in boxes
        if (box.confidence is None or box.confidence >= threshold)
        and (not selected_classes or box.class_name in selected_classes)
    )

    # The displayed count (len of filtered list) must equal the expected count
    assert len(after_both) == expected_count, (
        f"Filtered list has {len(after_both)} boxes, "
        f"but expected {expected_count} boxes passing both filters "
        f"(threshold={threshold}, classes={selected_classes})"
    )
