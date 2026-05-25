"""Property-based tests for data loader parsing round-trips.

Feature: streamlit-results-dashboard
"""

import json
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from dashboard.data_loader import (
    EvaluationReport,
    ExperimentRun,
    FinalResults,
    MetricsEntry,
    load_evaluation_report,
    load_experiment_run,
)


# --- Strategies for ExperimentRun (Property 1) ---

# Strategy for generating valid ISO 8601 timestamps
timestamps = st.datetimes().map(lambda dt: dt.isoformat())

# Strategy for generating valid MetricsEntry data
metrics_entry_strategy = st.builds(
    MetricsEntry,
    step=st.integers(min_value=0, max_value=10000),
    timestamp=timestamps,
    train_loss=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    val_loss=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    learning_rate=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    epoch_time_s=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
)

# Strategy for generating valid FinalResults data
final_results_strategy = st.builds(
    FinalResults,
    final_train_loss=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    final_val_loss=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    best_val_loss=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    best_epoch=st.integers(min_value=0, max_value=10000),
    total_epochs=st.integers(min_value=1, max_value=10000),
)

# Strategy for generating simple JSON-serializable config dicts
config_strategy = st.fixed_dictionaries(
    {
        "learning_rate": st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False),
        "batch_size": st.integers(min_value=1, max_value=256),
        "epochs": st.integers(min_value=1, max_value=10000),
        "optimizer": st.sampled_from(["SGD", "Adam", "AdamW"]),
    }
)

# Strategy for generating valid ExperimentRun instances
experiment_run_strategy = st.builds(
    ExperimentRun,
    run_id=st.uuids().map(str),
    model_name=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "Pd")),
        min_size=1,
        max_size=50,
    ),
    dataset_name=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "Pd")),
        min_size=1,
        max_size=50,
    ),
    config=config_strategy,
    start_time=timestamps,
    end_time=st.one_of(st.none(), timestamps),
    metrics_history=st.lists(metrics_entry_strategy, min_size=0, max_size=20),
    final_results=st.one_of(st.none(), final_results_strategy),
)


def _serialize_experiment_run(run: ExperimentRun) -> dict:
    """Serialize an ExperimentRun to a JSON-compatible dict."""
    data = {
        "run_id": run.run_id,
        "model_name": run.model_name,
        "dataset_name": run.dataset_name,
        "config": run.config,
        "start_time": run.start_time,
        "end_time": run.end_time,
        "metrics_history": [
            {
                "step": entry.step,
                "timestamp": entry.timestamp,
                "train_loss": entry.train_loss,
                "val_loss": entry.val_loss,
                "learning_rate": entry.learning_rate,
                "epoch_time_s": entry.epoch_time_s,
            }
            for entry in run.metrics_history
        ],
    }
    if run.final_results is not None:
        data["final_results"] = {
            "final_train_loss": run.final_results.final_train_loss,
            "final_val_loss": run.final_results.final_val_loss,
            "best_val_loss": run.final_results.best_val_loss,
            "best_epoch": run.final_results.best_epoch,
            "total_epochs": run.final_results.total_epochs,
        }
    else:
        data["final_results"] = None
    return data


# --- Strategies for EvaluationReport (Property 2) ---

# Strategy for generating valid class names (non-empty strings without control chars)
class_name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Z"), whitelist_characters="_- "),
    min_size=1,
    max_size=30,
)

# Strategy for generating valid metrics dictionaries
metrics_dict_strategy = st.fixed_dictionaries(
    {
        "mAP@0.5": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        "mAP@0.5:0.95": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        "precision": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        "recall": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        "f1_score": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    }
)


def evaluation_report_strategy():
    """Strategy for generating valid EvaluationReport data."""
    return st.integers(min_value=2, max_value=10).flatmap(
        lambda num_classes: st.fixed_dictionaries(
            {
                "checkpoint": st.text(min_size=1, max_size=100).filter(lambda s: s.strip() != ""),
                "dataset": st.text(min_size=1, max_size=100).filter(lambda s: s.strip() != ""),
                "num_val_images": st.integers(min_value=1, max_value=100000),
                "num_classes": st.just(num_classes),
                "class_names": st.lists(
                    class_name_strategy, min_size=num_classes, max_size=num_classes
                ),
                "confidence_threshold": st.floats(
                    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
                ),
                "iou_threshold": st.floats(
                    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
                ),
                "metrics": metrics_dict_strategy,
                "confusion_matrix": st.lists(
                    st.lists(
                        st.integers(min_value=0, max_value=10000),
                        min_size=num_classes,
                        max_size=num_classes,
                    ),
                    min_size=num_classes,
                    max_size=num_classes,
                ),
            }
        )
    )


# --- Property Tests ---


@given(run=experiment_run_strategy)
@settings(max_examples=100)
def test_experiment_run_parsing_round_trip(run: ExperimentRun) -> None:
    """Feature: streamlit-results-dashboard, Property 1: ExperimentRun parsing round-trip

    **Validates: Requirements 1.3**

    For any valid ExperimentRun data (with valid run_id, model_name, dataset_name,
    config, metrics_history, and final_results), serializing it to JSON and then
    parsing it with `load_experiment_run` should produce an ExperimentRun with all
    fields equal to the original.
    """
    # Serialize the ExperimentRun to a JSON dict
    data = _serialize_experiment_run(run)

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(data, f)
        tmp_path = Path(f.name)

    try:
        # Parse it back using load_experiment_run
        parsed = load_experiment_run(tmp_path)

        # Assert all scalar fields are equal
        assert parsed.run_id == run.run_id
        assert parsed.model_name == run.model_name
        assert parsed.dataset_name == run.dataset_name
        assert parsed.config == run.config
        assert parsed.start_time == run.start_time
        assert parsed.end_time == run.end_time

        # Compare metrics_history
        assert len(parsed.metrics_history) == len(run.metrics_history)
        for parsed_entry, original_entry in zip(
            parsed.metrics_history, run.metrics_history
        ):
            assert parsed_entry.step == original_entry.step
            assert parsed_entry.timestamp == original_entry.timestamp
            assert parsed_entry.train_loss == original_entry.train_loss
            assert parsed_entry.val_loss == original_entry.val_loss
            assert parsed_entry.learning_rate == original_entry.learning_rate
            assert parsed_entry.epoch_time_s == original_entry.epoch_time_s

        # Compare final_results
        if run.final_results is None:
            assert parsed.final_results is None
        else:
            assert parsed.final_results is not None
            assert (
                parsed.final_results.final_train_loss
                == run.final_results.final_train_loss
            )
            assert (
                parsed.final_results.final_val_loss
                == run.final_results.final_val_loss
            )
            assert (
                parsed.final_results.best_val_loss == run.final_results.best_val_loss
            )
            assert parsed.final_results.best_epoch == run.final_results.best_epoch
            assert parsed.final_results.total_epochs == run.final_results.total_epochs
    finally:
        tmp_path.unlink(missing_ok=True)


@given(data=evaluation_report_strategy())
@settings(max_examples=100)
def test_evaluation_report_parsing_round_trip(data: dict) -> None:
    """Feature: streamlit-results-dashboard, Property 2: EvaluationReport parsing round-trip

    **Validates: Requirements 1.4**

    For any valid EvaluationReport data (with valid metrics, class_names, and
    confusion_matrix), serializing it to JSON and then parsing it with
    `load_evaluation_report` should produce an EvaluationReport with all fields
    equal to the original.
    """
    # Serialize to JSON and write to a temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(data, f)
        temp_path = Path(f.name)

    try:
        # Parse with load_evaluation_report
        result = load_evaluation_report(temp_path)

        # Verify it returns an EvaluationReport instance
        assert isinstance(result, EvaluationReport)

        # Verify all fields match the original data
        assert result.checkpoint == data["checkpoint"]
        assert result.dataset == data["dataset"]
        assert result.num_val_images == data["num_val_images"]
        assert result.num_classes == data["num_classes"]
        assert result.class_names == data["class_names"]
        assert result.confidence_threshold == data["confidence_threshold"]
        assert result.iou_threshold == data["iou_threshold"]
        assert result.metrics == data["metrics"]
        assert result.confusion_matrix == data["confusion_matrix"]
    finally:
        temp_path.unlink(missing_ok=True)


# --- Additional imports for Property 3 ---
import os

from dashboard.data_loader import DashboardData, load_all_data


# --- Strategies for Property 3: Malformed JSON graceful handling ---

# Strategy for generating arbitrary byte strings that are NOT valid JSON
# or valid JSON that does not conform to the expected schema
malformed_content_strategy = st.one_of(
    # Arbitrary binary data (very unlikely to be valid JSON)
    st.binary(min_size=0, max_size=500),
    # Arbitrary text (may or may not be valid JSON, but won't conform to schema)
    st.text(min_size=0, max_size=500),
    # Strings that look like partial JSON
    st.from_regex(r'[\{\[\"\d][^\x00]{0,100}', fullmatch=False),
    # Valid JSON but wrong schema (arrays, numbers, strings, booleans, null)
    st.one_of(
        st.integers().map(lambda x: json.dumps(x).encode()),
        st.lists(st.integers(), max_size=10).map(lambda x: json.dumps(x).encode()),
        st.text(max_size=50).map(lambda x: json.dumps(x).encode()),
        st.booleans().map(lambda x: json.dumps(x).encode()),
        st.just(b"null"),
        # Valid JSON object but missing required fields
        st.fixed_dictionaries(
            {"random_key": st.text(max_size=20)}
        ).map(lambda x: json.dumps(x).encode()),
    ),
)


@given(content=malformed_content_strategy)
@settings(max_examples=100)
def test_malformed_json_graceful_handling(content) -> None:
    """Feature: streamlit-results-dashboard, Property 3: Malformed JSON graceful handling

    **Validates: Requirements 1.5**

    For any arbitrary byte string that is not valid JSON (or valid JSON that does
    not conform to the expected schema), calling the result loader should never
    raise an unhandled exception and should return a warning in the errors list.
    """
    # Ensure content is bytes for writing in binary mode
    if isinstance(content, str):
        content_bytes = content.encode("utf-8", errors="replace")
    else:
        content_bytes = content

    # Create a temp directory structured like results/ssd_mobilenetv3/
    with tempfile.TemporaryDirectory() as tmp_dir:
        results_dir = Path(tmp_dir) / "results" / "ssd_mobilenetv3"
        results_dir.mkdir(parents=True)

        # Write the malformed content to a .json file
        malformed_file = results_dir / "malformed_run.json"
        malformed_file.write_bytes(content_bytes)

        # Also create an empty checkpoints dir
        checkpoints_dir = Path(tmp_dir) / "checkpoints"
        checkpoints_dir.mkdir(parents=True)

        # Call load_all_data - this should NEVER raise an unhandled exception
        result = load_all_data(
            results_dir=Path(tmp_dir) / "results",
            checkpoints_dir=checkpoints_dir,
        )

        # Verify the result is a valid DashboardData instance
        assert isinstance(result, DashboardData)

        # The malformed file should NOT have been loaded as a valid run
        # (it's possible but extremely unlikely that random bytes form valid JSON
        # conforming to the exact schema - if it does, that's still acceptable)
        # The key property: errors list should be non-empty because the file was malformed
        assert len(result.errors) > 0, (
            f"Expected errors list to be non-empty for malformed content, "
            f"but got no errors. Content: {content_bytes[:100]!r}"
        )
