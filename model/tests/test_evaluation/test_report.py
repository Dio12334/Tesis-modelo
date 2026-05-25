# Feature: road-damage-evaluation-framework, Property 12: Evaluation report JSON round-trip
"""Property-based test for EvaluationReport JSON round-trip serialization.

Validates: Requirements 5.7

For any valid EvaluationReport, serializing to JSON and deserializing SHALL produce
an equivalent report with matching model_id, timestamp, and all metric values.
"""

import datetime
import string

import numpy as np
from hypothesis import given, settings, HealthCheck, strategies as st

from model.evaluation.report import EvaluationReport


# Strategy for generating unique class name lists (simple ASCII for speed)
class_names_strategy = st.lists(
    st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=10),
    min_size=1,
    max_size=5,
    unique=True,
)


@st.composite
def evaluation_report_strategy(draw):
    """Generate valid EvaluationReport instances."""
    model_id = draw(st.text(alphabet=string.ascii_letters + string.digits + "-_", min_size=1, max_size=30))

    # ISO format timestamps using simple integer components for speed
    dt = draw(st.datetimes(
        min_value=datetime.datetime(2020, 1, 1),
        max_value=datetime.datetime(2030, 12, 31),
    ))
    timestamp = dt.isoformat()

    # Metrics in [0, 1]
    map_50 = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    map_50_95 = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    precision = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    recall = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    f1_score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))

    # Class names (unique strings)
    class_names = draw(class_names_strategy)
    num_classes = len(class_names)

    # per_class_ap: dict of class_name -> float in [0, 1]
    per_class_ap_values = draw(st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=num_classes,
        max_size=num_classes,
    ))
    per_class_ap = dict(zip(class_names, per_class_ap_values))

    # confusion_matrix: numpy array of non-negative integers with shape (C, C)
    matrix_values = draw(st.lists(
        st.integers(min_value=0, max_value=1000),
        min_size=num_classes * num_classes,
        max_size=num_classes * num_classes,
    ))
    confusion_matrix = np.array(matrix_values, dtype=np.int64).reshape(num_classes, num_classes)

    # config: dict with string keys and simple values
    config = draw(st.dictionaries(
        keys=st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=8),
        values=st.one_of(
            st.integers(min_value=-1000, max_value=1000),
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            st.text(alphabet=string.ascii_letters, min_size=0, max_size=10),
            st.booleans(),
        ),
        min_size=0,
        max_size=4,
    ))

    return EvaluationReport(
        model_id=model_id,
        timestamp=timestamp,
        map_50=map_50,
        map_50_95=map_50_95,
        per_class_ap=per_class_ap,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        confusion_matrix=confusion_matrix,
        class_names=class_names,
        config=config,
    )


@given(report=evaluation_report_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_evaluation_report_json_round_trip(report: EvaluationReport):
    """Property 12: Evaluation report JSON round-trip.

    For any valid EvaluationReport, serializing to JSON and deserializing SHALL
    produce an equivalent report with matching model_id, timestamp, and all metric values.

    **Validates: Requirements 5.7**
    """
    # Serialize and deserialize
    json_str = report.to_json()
    restored = EvaluationReport.from_json(json_str)

    # 1. model_id matches
    assert restored.model_id == report.model_id

    # 2. timestamp matches
    assert restored.timestamp == report.timestamp

    # 3. All metric values match (within floating-point tolerance)
    assert abs(restored.map_50 - report.map_50) < 1e-10
    assert abs(restored.map_50_95 - report.map_50_95) < 1e-10
    assert abs(restored.precision - report.precision) < 1e-10
    assert abs(restored.recall - report.recall) < 1e-10
    assert abs(restored.f1_score - report.f1_score) < 1e-10

    # 4. confusion_matrix matches (numpy array equality)
    np.testing.assert_array_equal(restored.confusion_matrix, report.confusion_matrix)

    # 5. class_names match
    assert restored.class_names == report.class_names

    # 6. per_class_ap matches
    assert set(restored.per_class_ap.keys()) == set(report.per_class_ap.keys())
    for key in report.per_class_ap:
        assert abs(restored.per_class_ap[key] - report.per_class_ap[key]) < 1e-10

    # 7. config matches
    assert restored.config == report.config
