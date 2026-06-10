"""Data loading module for the Streamlit Results Dashboard.

Discovers and parses experiment result JSON files and evaluation reports
from the results/ and checkpoints/ directories.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricsEntry:
    """A single epoch's training metrics."""

    step: int
    timestamp: str
    train_loss: float
    val_loss: float
    learning_rate: float
    epoch_time_s: float


@dataclass
class FinalResults:
    """Summary results at the end of a training run."""

    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    best_epoch: int
    total_epochs: int


@dataclass
class ExperimentRun:
    """A single training run identified by a UUID."""

    run_id: str
    model_name: str
    dataset_name: str
    config: dict
    start_time: str
    end_time: Optional[str]
    metrics_history: list[MetricsEntry]
    final_results: Optional[FinalResults]


@dataclass
class EvaluationReport:
    """Evaluation metrics for a model checkpoint."""

    checkpoint: str
    dataset: str
    num_val_images: int
    num_classes: int
    class_names: list[str]
    confidence_threshold: float
    iou_threshold: float
    metrics: dict
    confusion_matrix: list[list[int]]
    display_class_names: Optional[list[str]] = None


@dataclass
class DashboardData:
    """Aggregated data for the dashboard."""

    runs: list[ExperimentRun] = field(default_factory=list)
    evaluation_report: Optional[EvaluationReport] = None
    evaluation_reports: dict = field(default_factory=dict)  # run_id -> EvaluationReport
    evaluation_reports_by_split: dict = field(default_factory=dict)  # run_id -> {split: EvaluationReport}
    best_model_metadata: Optional[dict] = None
    predictions_data: Optional[dict] = None
    train_predictions_data: Optional[dict] = None
    test_predictions_data: Optional[dict] = None
    predictions_by_run: dict = field(default_factory=dict)  # run_id -> predictions dict
    train_predictions_by_run: dict = field(default_factory=dict)  # run_id -> train predictions dict
    test_predictions_by_run: dict = field(default_factory=dict)  # run_id -> test predictions dict
    errors: list[str] = field(default_factory=list)


def load_experiment_run(filepath: Path) -> ExperimentRun:
    """Parse a single run result JSON file into an ExperimentRun.

    Args:
        filepath: Path to the experiment run JSON file.

    Returns:
        An ExperimentRun dataclass instance.

    Raises:
        json.JSONDecodeError: If the file contains invalid JSON.
        KeyError: If required fields are missing.
        TypeError: If field types don't match expectations.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics_history = [
        MetricsEntry(
            step=entry["step"],
            timestamp=entry["timestamp"],
            train_loss=entry["train_loss"],
            val_loss=entry["val_loss"],
            learning_rate=entry["learning_rate"],
            epoch_time_s=entry["epoch_time_s"],
        )
        for entry in data["metrics_history"]
    ]

    final_results: Optional[FinalResults] = None
    final = data.get("final_results")
    if final is not None:
        # Handle both 'total_epochs' and 'total_epochs_trained' field names
        total_epochs = final.get("total_epochs", final.get("total_epochs_trained", 0))
        final_results = FinalResults(
            final_train_loss=final["final_train_loss"],
            final_val_loss=final["final_val_loss"],
            best_val_loss=final["best_val_loss"],
            best_epoch=final["best_epoch"],
            total_epochs=total_epochs,
        )
    elif metrics_history:
        # Fallback: compute final_results from metrics_history when the
        # training process was interrupted before calling end_run().
        last = metrics_history[-1]
        best_entry = min(metrics_history, key=lambda e: e.val_loss)
        best_idx = metrics_history.index(best_entry)
        final_results = FinalResults(
            final_train_loss=last.train_loss,
            final_val_loss=last.val_loss,
            best_val_loss=best_entry.val_loss,
            best_epoch=best_idx + 1,  # 1-indexed
            total_epochs=len(metrics_history),
        )

    return ExperimentRun(
        run_id=data["run_id"],
        model_name=data["model_name"],
        dataset_name=data["dataset_name"],
        config=data["config"],
        start_time=data["start_time"],
        end_time=data.get("end_time"),
        metrics_history=metrics_history,
        final_results=final_results,
    )


def load_evaluation_report(filepath: Path) -> EvaluationReport:
    """Parse the evaluation report JSON file.

    Args:
        filepath: Path to the evaluation report JSON file.

    Returns:
        An EvaluationReport dataclass instance.

    Raises:
        json.JSONDecodeError: If the file contains invalid JSON.
        KeyError: If required fields are missing.
        TypeError: If field types don't match expectations.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return EvaluationReport(
        checkpoint=data["checkpoint"],
        dataset=data["dataset"],
        num_val_images=data.get("num_val_images", data.get("num_images", 0)),
        num_classes=data["num_classes"],
        class_names=data["class_names"],
        confidence_threshold=data["confidence_threshold"],
        iou_threshold=data["iou_threshold"],
        metrics=data["metrics"],
        confusion_matrix=data["confusion_matrix"],
        display_class_names=data.get("display_class_names"),
    )


def load_best_model_metadata(filepath: Path) -> dict:
    """Parse the best model metadata JSON file.

    Args:
        filepath: Path to the best model metadata JSON file.

    Returns:
        A dictionary containing the best model metadata.

    Raises:
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_split_from_filename(filename: str) -> str:
    """Infer the evaluation split from the report filename."""
    if filename.startswith("val"):
        return "val"
    elif filename.startswith("train"):
        return "train"
    elif filename.startswith("test"):
        return "test"
    return "val"  # default


def load_all_data(results_dir: Path, checkpoints_dir: Path) -> DashboardData:
    """Scan directories and load all experiment data.

    Discovers all JSON files in the results directory tree and loads them
    as ExperimentRun instances. Also loads the evaluation report and best
    model metadata from the checkpoints directory.

    Handles malformed JSON and missing directories gracefully by logging
    warnings and collecting errors in DashboardData.errors.

    Runs are sorted by start_time descending (most recent first).

    Args:
        results_dir: Path to the results directory (e.g., "results/").
        checkpoints_dir: Path to the checkpoints directory (e.g., "checkpoints/").

    Returns:
        A DashboardData instance with all loaded data and any errors encountered.
    """
    dashboard_data = DashboardData()

    # Load experiment runs
    if results_dir.exists() and results_dir.is_dir():
        json_files = sorted(results_dir.rglob("*.json"))
        for json_file in json_files:
            try:
                run = load_experiment_run(json_file)
                dashboard_data.runs.append(run)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                msg = f"Failed to load experiment run from {json_file}: {e}"
                logger.warning(msg)
                dashboard_data.errors.append(msg)
            except PermissionError as e:
                msg = f"Permission denied reading {json_file}: {e}"
                logger.warning(msg)
                dashboard_data.errors.append(msg)
    else:
        # Results directory doesn't exist — not an error if checkpoints has runs
        logger.info(f"Results directory not found: {results_dir}")

    # Sort runs by start_time descending (most recent first) — after all sources loaded
    # (sort is applied below, after checkpoints are also scanned)

    # Load from checkpoints directory
    if checkpoints_dir.exists() and checkpoints_dir.is_dir():
        # Also scan checkpoints for experiment run JSON files
        # (skip known non-run files)
        SKIP_FILENAMES = {
            "evaluation_report.json", "best.json", "best_model.json",
            "val_evaluation_report.json", "train_evaluation_report.json",
            "test_evaluation_report.json", "val_inference.json",
            "train_inference.json", "test_inference.json",
            "validation_inference.json", "predictions.json",
        }
        # Skip files inside run UUID subdirectories (those contain .pt files, not runs)
        checkpoint_json_files = sorted(checkpoints_dir.rglob("*.json"))
        existing_run_ids = {run.run_id for run in dashboard_data.runs}

        for json_file in checkpoint_json_files:
            if json_file.name in SKIP_FILENAMES:
                continue
            # Skip files inside the global/ or temp/ directories
            if "global" in json_file.parts or "temp" in json_file.parts:
                continue
            try:
                run = load_experiment_run(json_file)
                # Avoid duplicates if already loaded from results_dir
                if run.run_id not in existing_run_ids:
                    dashboard_data.runs.append(run)
                    existing_run_ids.add(run.run_id)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass  # Not a run file, skip silently
            except PermissionError:
                pass

        # Search for evaluation_report.json files and associate with runs
        eval_reports = list(checkpoints_dir.rglob("evaluation_report.json"))
        # Also search for split-tagged report files (val_evaluation_report.json, etc.)
        # Order: val first (preferred for dashboard display), then train, then test
        for split_name in ("val", "train", "test"):
            eval_reports.extend(
                checkpoints_dir.rglob(f"{split_name}_evaluation_report.json")
            )
        for eval_path in eval_reports:
            try:
                report = load_evaluation_report(eval_path)
                parent_name = eval_path.parent.name
                if len(parent_name) == 36 and parent_name.count("-") == 4:
                    # Store by split for multi-split display
                    split_name = _infer_split_from_filename(eval_path.name)
                    if parent_name not in dashboard_data.evaluation_reports_by_split:
                        dashboard_data.evaluation_reports_by_split[parent_name] = {}
                    dashboard_data.evaluation_reports_by_split[parent_name][split_name] = report

                    # Only overwrite if no report exists yet for this run,
                    # or if the existing report has empty metrics (e.g., test split
                    # with no ground truth) and the new one has actual data.
                    existing = dashboard_data.evaluation_reports.get(parent_name)
                    if existing is None:
                        dashboard_data.evaluation_reports[parent_name] = report
                    elif not existing.metrics.get("per_class_ap") and report.metrics.get("per_class_ap"):
                        dashboard_data.evaluation_reports[parent_name] = report
                if dashboard_data.evaluation_report is None:
                    dashboard_data.evaluation_report = report
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                msg = f"Failed to load evaluation report from {eval_path}: {e}"
                logger.warning(msg)
                dashboard_data.errors.append(msg)
            except PermissionError:
                pass

        # Load best model metadata
        best_files = list(checkpoints_dir.rglob("best.json"))
        if best_files:
            best_path = best_files[0]
            try:
                dashboard_data.best_model_metadata = load_best_model_metadata(best_path)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                msg = f"Failed to load best model metadata from {best_path}: {e}"
                logger.warning(msg)
                dashboard_data.errors.append(msg)
            except PermissionError as e:
                msg = f"Permission denied reading {best_path}: {e}"
                logger.warning(msg)
                dashboard_data.errors.append(msg)

        # Load per-image predictions (validation) per run
        val_pred_files = list(checkpoints_dir.rglob("validation_inference.json"))
        if not val_pred_files:
            val_pred_files = list(checkpoints_dir.rglob("predictions.json"))
        # Also search for split-tagged prediction files (val_inference.json)
        val_pred_files.extend(checkpoints_dir.rglob("val_inference.json"))
        for pred_path in val_pred_files:
            try:
                with open(pred_path, "r", encoding="utf-8") as f:
                    pred_data = json.load(f)
                parent_name = pred_path.parent.name
                if len(parent_name) == 36 and parent_name.count("-") == 4:
                    dashboard_data.predictions_by_run[parent_name] = pred_data
                if dashboard_data.predictions_data is None:
                    dashboard_data.predictions_data = pred_data
            except (json.JSONDecodeError, PermissionError):
                pass

        # Load per-image predictions (training) per run
        train_pred_files = list(checkpoints_dir.rglob("train_inference.json"))
        for pred_path in train_pred_files:
            try:
                with open(pred_path, "r", encoding="utf-8") as f:
                    pred_data = json.load(f)
                parent_name = pred_path.parent.name
                if len(parent_name) == 36 and parent_name.count("-") == 4:
                    dashboard_data.train_predictions_by_run[parent_name] = pred_data
                if dashboard_data.train_predictions_data is None:
                    dashboard_data.train_predictions_data = pred_data
            except (json.JSONDecodeError, PermissionError):
                pass

        # Load per-image predictions (test) per run
        test_pred_files = list(checkpoints_dir.rglob("test_inference.json"))
        for pred_path in test_pred_files:
            try:
                with open(pred_path, "r", encoding="utf-8") as f:
                    pred_data = json.load(f)
                parent_name = pred_path.parent.name
                if len(parent_name) == 36 and parent_name.count("-") == 4:
                    dashboard_data.test_predictions_by_run[parent_name] = pred_data
                if dashboard_data.test_predictions_data is None:
                    dashboard_data.test_predictions_data = pred_data
            except (json.JSONDecodeError, PermissionError):
                pass
    else:
        msg = f"Checkpoints directory not found: {checkpoints_dir}"
        logger.warning(msg)
        dashboard_data.errors.append(msg)

    # Sort runs by start_time descending (most recent first)
    dashboard_data.runs.sort(key=lambda run: run.start_time, reverse=True)

    return dashboard_data
