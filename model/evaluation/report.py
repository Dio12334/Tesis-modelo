"""Evaluation report data model and serialization."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class EvaluationReport:
    """Contains all metrics from a model evaluation run.

    Attributes:
        model_id: Identifier of the evaluated model.
        timestamp: ISO-format timestamp of when evaluation was performed.
        map_50: Mean Average Precision at IoU=0.5.
        map_50_95: Mean Average Precision at IoU=0.5:0.95.
        per_class_ap: Dict mapping class name to AP at IoU=0.5.
        precision: Overall precision at the configured confidence threshold.
        recall: Overall recall at the configured confidence threshold.
        f1_score: Overall F1-score.
        confusion_matrix: Numpy array of shape (C, C) with class confusion counts.
        class_names: Ordered list of class names (defines matrix indices).
        config: Configuration dict used for this evaluation run.
    """

    model_id: str
    timestamp: str
    map_50: float
    map_50_95: float
    per_class_ap: Dict[str, float]
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    class_names: List[str]
    config: Dict = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize the report to a JSON string.

        Handles numpy arrays (confusion_matrix) by converting to nested lists.

        Returns:
            JSON string representation of the report.
        """
        data = {
            "model_id": self.model_id,
            "timestamp": self.timestamp,
            "map_50": self.map_50,
            "map_50_95": self.map_50_95,
            "per_class_ap": self.per_class_ap,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "class_names": self.class_names,
            "config": self.config,
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "EvaluationReport":
        """Deserialize from JSON string back to EvaluationReport.

        Converts confusion_matrix back to numpy array.

        Args:
            json_str: JSON string to deserialize.

        Returns:
            EvaluationReport instance.
        """
        data = json.loads(json_str)
        data["confusion_matrix"] = np.array(data["confusion_matrix"])
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save the report to a JSON file.

        Args:
            path: File path to save the report to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EvaluationReport":
        """Load a report from a JSON file.

        Args:
            path: File path to load the report from.

        Returns:
            EvaluationReport instance.
        """
        path = Path(path)
        json_str = path.read_text(encoding="utf-8")
        return cls.from_json(json_str)
