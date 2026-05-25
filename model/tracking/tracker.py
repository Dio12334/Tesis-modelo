"""Experiment tracking for logging runs, metrics, and results."""

import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ExperimentTracker:
    """Tracks experiment runs with configs, metrics, and results.

    Each run is stored as a separate JSON file in the output directory.
    """

    def __init__(self, output_dir: Path) -> None:
        """Initialize the experiment tracker.

        Args:
            output_dir: Directory where experiment JSON files are stored.
                        Created if it doesn't exist.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def start_run(self, config: dict, model_name: str, dataset_name: str) -> str:
        """Start a new experiment run.

        Args:
            config: Complete experiment configuration dict.
            model_name: Name/identifier of the model being trained.
            dataset_name: Name/identifier of the dataset used.

        Returns:
            Unique run ID (UUID4 string).
        """
        run_id = str(uuid.uuid4())
        run_data = {
            "run_id": run_id,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "config": config,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "metrics_history": [],
            "final_results": None,
        }
        self._save_run(run_id, run_data)
        return run_id

    def log_metrics(self, run_id: str, step: int, metrics: dict) -> None:
        """Log metrics at a given training step.

        Args:
            run_id: The run ID returned by start_run.
            step: The training step or epoch number.
            metrics: Dictionary of metric name to value.
        """
        run_data = self._load_run(run_id)
        entry = {"step": step, "timestamp": datetime.now(timezone.utc).isoformat()}
        entry.update(metrics)
        run_data["metrics_history"].append(entry)
        self._save_run(run_id, run_data)

    def end_run(self, run_id: str, results: dict) -> None:
        """Finalize an experiment run with final results.

        Args:
            run_id: The run ID returned by start_run.
            results: Dictionary of final evaluation results.
        """
        run_data = self._load_run(run_id)
        run_data["end_time"] = datetime.now(timezone.utc).isoformat()
        run_data["final_results"] = results
        self._save_run(run_id, run_data)

    def export_csv(
        self,
        run_ids: Optional[List[str]] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Export experiment results to CSV.

        Args:
            run_ids: List of run IDs to export. If None, exports all runs.
            output_path: Path for the output CSV file. If None, uses
                         output_dir/experiments.csv.

        Returns:
            Path to the generated CSV file.
        """
        if output_path is None:
            output_path = self.output_dir / "experiments.csv"
        else:
            output_path = Path(output_path)

        runs = self._get_runs(run_ids)

        # Collect all metric keys from final_results across runs
        metric_keys: List[str] = []
        for run in runs:
            if run.get("final_results"):
                for key in run["final_results"]:
                    if key not in metric_keys:
                        metric_keys.append(key)

        fieldnames = [
            "run_id",
            "model_name",
            "dataset_name",
            "start_time",
            "end_time",
        ] + metric_keys

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for run in runs:
                row: Dict[str, object] = {
                    "run_id": run["run_id"],
                    "model_name": run["model_name"],
                    "dataset_name": run["dataset_name"],
                    "start_time": run["start_time"],
                    "end_time": run.get("end_time", ""),
                }
                if run.get("final_results"):
                    for key in metric_keys:
                        row[key] = run["final_results"].get(key, "")
                writer.writerow(row)

        return output_path

    def query(
        self,
        model_type: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
    ) -> List[dict]:
        """Query experiments by filters.

        Args:
            model_type: Filter by model name. If None, no model filter applied.
            date_range: Tuple of (start_date, end_date) in ISO format strings.
                        If None, no date filter applied.

        Returns:
            List of run data dicts matching the filters.
        """
        all_runs = self._get_runs(run_ids=None)
        results = []

        for run in all_runs:
            # Filter by model type
            if model_type is not None and run.get("model_name") != model_type:
                continue

            # Filter by date range
            if date_range is not None:
                start_date, end_date = date_range
                run_start = run.get("start_time", "")
                if run_start < start_date or run_start > end_date:
                    continue

            results.append(run)

        return results

    def _run_path(self, run_id: str) -> Path:
        """Get the file path for a run's JSON file."""
        return self.output_dir / f"{run_id}.json"

    def _save_run(self, run_id: str, run_data: dict) -> None:
        """Save run data to its JSON file."""
        path = self._run_path(run_id)
        with open(path, "w") as f:
            json.dump(run_data, f, indent=2)

    def _load_run(self, run_id: str) -> dict:
        """Load run data from its JSON file."""
        path = self._run_path(run_id)
        with open(path, "r") as f:
            return json.load(f)

    def _get_runs(self, run_ids: Optional[List[str]] = None) -> List[dict]:
        """Load multiple runs. If run_ids is None, load all runs."""
        if run_ids is not None:
            return [self._load_run(rid) for rid in run_ids]

        runs = []
        for json_file in self.output_dir.glob("*.json"):
            with open(json_file, "r") as f:
                runs.append(json.load(f))
        return runs
