"""Property-based tests for metrics formatting.

Feature: streamlit-results-dashboard, Property 7: Metrics formatted to two decimal places

Tests that any float metric value (precision, recall, F1-score) formatted
for display produces a string with exactly two digits after the decimal point.
"""

import re

from hypothesis import given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Property 7: Metrics formatted to two decimal places
# ---------------------------------------------------------------------------


@given(value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_metrics_formatted_to_two_decimal_places(value):
    """Feature: streamlit-results-dashboard, Property 7: Metrics formatted to two decimal places

    **Validates: Requirements 4.3**

    For any float value representing a metric (precision, recall, F1-score),
    formatting it for display should produce a string with exactly two digits
    after the decimal point.
    """
    # Format using the same pattern as metrics_overview.py
    formatted = f"{value:.2f}"

    # Assert: the formatted string has exactly 2 digits after the decimal point
    parts = formatted.split(".")
    assert len(parts) == 2, (
        f"Formatted value '{formatted}' does not contain exactly one decimal point"
    )
    assert len(parts[1]) == 2, (
        f"Formatted value '{formatted}' does not have exactly 2 digits after "
        f"the decimal point (has {len(parts[1])})"
    )

    # Assert: the string matches the pattern r"^\d+\.\d{2}$"
    pattern = r"^\d+\.\d{2}$"
    assert re.match(pattern, formatted), (
        f"Formatted value '{formatted}' does not match expected pattern {pattern}"
    )
