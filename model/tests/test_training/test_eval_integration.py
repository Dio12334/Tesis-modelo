"""End-to-end integration test for the generic evaluation pipeline.

Feature: generic-evaluation-script
Task 14.1: Write end-to-end integration test on the sample dataset

Using a lightweight fake BaseDetector registered in ModelRegistry and the
sample dataset at ``model/data/rdd2022/sample``, this test runs one evaluation
per split (train, val, test), asserting:

* Checkpoint resolution succeeds with a temporary checkpoint file.
* Inference runs to completion on real images from the sample dataset.
* Metrics are computed.
* Output files exist with split-tagged names (e.g., ``val_evaluation_report.json``,
  ``val_inference.json``).
* The report contains all required fields from Requirement 16.2.
* Metrics contain ``map_50``, ``map_50_95``, ``precision``, ``recall``,
  ``f1_score``, ``per_class_ap``.

**Validates: Requirements 8.1, 8.6, 9.1, 9.2, 9.3, 16.2, 16.6**
"""

import json
from pathlib import Path
from typing import List

import pytest
import torch

from model.models.registry import BaseDetector, ModelRegistry
from model.training.evaluate_detection import evaluate


# ---------------------------------------------------------------------------
# Lightweight fake detector for integration testing
# ---------------------------------------------------------------------------


class _FakeModule:
    """Minimal nn.Module stand-in with train/eval/to methods."""

    def train(self):
        pass

    def eval(self):
        pass

    def to(self, device):
        pass


class IntegrationFakeDetector(BaseDetector):
    """A lightweight fake detector that returns normalized bounding boxes.

    Returns one detection per image with a fixed box in [0, 1] range,
    a label index of 1, and a confidence score of 0.9. This is sufficient
    to exercise the full pipeline without requiring a real model or GPU.
    """

    def __init__(self, config: dict):
        self._config = config
        self._model = _FakeModule()

    def forward(self, images: torch.Tensor) -> List[dict]:
        batch_size = images.shape[0]
        results = []
        for _ in range(batch_size):
            results.append({
                "boxes": torch.tensor(
                    [[0.1, 0.2, 0.5, 0.6]], dtype=torch.float32
                ),
                "labels": torch.tensor([1], dtype=torch.int64),
                "scores": torch.tensor([0.9], dtype=torch.float32),
            })
        return results

    def get_config_schema(self) -> dict:
        return {"num_classes": {"type": "int", "required": True}}

    def load_checkpoint(self, path: Path) -> None:
        # No-op: we don't need real weights for integration testing.
        pass

    def save_checkpoint(self, path: Path) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DATASET_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "rdd2022" / "sample"
)


@pytest.fixture(autouse=True)
def register_fake_detector():
    """Register the fake detector in ModelRegistry for the duration of the test."""
    original_models = ModelRegistry._models.copy()
    ModelRegistry._models["integration_fake"] = IntegrationFakeDetector
    yield
    ModelRegistry._models = original_models


@pytest.fixture
def checkpoint_file(tmp_path):
    """Create a temporary checkpoint file."""
    ckpt = tmp_path / "fake_checkpoint.pt"
    torch.save({"state": "fake"}, ckpt)
    return ckpt


# ---------------------------------------------------------------------------
# Required report fields (Requirement 16.2)
# ---------------------------------------------------------------------------

REQUIRED_REPORT_FIELDS = [
    "checkpoint",
    "model_type",
    "model_config",
    "dataset",
    "split",
    "num_images",
    "num_classes",
    "class_names",
    "confidence_threshold",
    "iou_threshold",
    "metrics",
    "confusion_matrix",
    "errors",
]

REQUIRED_METRIC_FIELDS = [
    "map_50",
    "map_50_95",
    "precision",
    "recall",
    "f1_score",
    "per_class_ap",
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvalIntegration:
    """End-to-end integration tests running evaluate() on the sample dataset.

    **Validates: Requirements 8.1, 8.6, 9.1, 9.2, 9.3, 16.2, 16.6**
    """

    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_evaluate_produces_correct_outputs_per_split(
        self, split, tmp_path, checkpoint_file
    ):
        """Run evaluate() for each split and assert output files exist with
        split-tagged names and all required fields.

        **Validates: Requirements 8.1, 8.6, 9.1, 9.2, 9.3, 16.2, 16.6**
        """
        output_dir = tmp_path / "output" / split
        output_dir.mkdir(parents=True, exist_ok=True)

        overrides = {
            "model": {
                "type": "integration_fake",
                "config": {"num_classes": 4, "input_size": 640},
            },
            "dataset": {"path": str(SAMPLE_DATASET_PATH)},
            "evaluation": {
                "split": split,
                "confidence_threshold": 0.25,
                "iou_threshold": 0.5,
                "output_dir": str(output_dir),
                "val_split": 0.2,
                "seed": 42,
            },
            "checkpoint": {"path": str(checkpoint_file)},
        }

        # Run the full evaluation pipeline.
        report = evaluate(config_path=None, overrides=overrides)

        # -----------------------------------------------------------------
        # Assert output files exist with split-tagged names (Req 16.6)
        # -----------------------------------------------------------------
        report_file = output_dir / f"{split}_evaluation_report.json"
        predictions_file = output_dir / f"{split}_inference.json"

        assert report_file.exists(), (
            f"Expected report file {report_file} to exist for split '{split}'"
        )
        assert predictions_file.exists(), (
            f"Expected predictions file {predictions_file} to exist for split '{split}'"
        )

        # -----------------------------------------------------------------
        # Assert report contains all required fields (Req 16.2)
        # -----------------------------------------------------------------
        with open(report_file, "r", encoding="utf-8") as f:
            written_report = json.load(f)

        for field in REQUIRED_REPORT_FIELDS:
            assert field in written_report, (
                f"Report missing required field '{field}' for split '{split}'"
            )

        # -----------------------------------------------------------------
        # Assert metrics contain required metric fields (Req 8.1, 8.6)
        # -----------------------------------------------------------------
        metrics = written_report["metrics"]
        for field in REQUIRED_METRIC_FIELDS:
            assert field in metrics, (
                f"Metrics missing required field '{field}' for split '{split}'"
            )

        # -----------------------------------------------------------------
        # Assert the report's split matches the requested split (Req 9.1-9.3)
        # -----------------------------------------------------------------
        assert written_report["split"] == split

        # -----------------------------------------------------------------
        # Assert basic structural correctness
        # -----------------------------------------------------------------
        assert written_report["model_type"] == "integration_fake"
        assert written_report["num_images"] > 0
        assert written_report["num_classes"] >= 0
        assert isinstance(written_report["class_names"], list)
        assert isinstance(written_report["confusion_matrix"], list)
        assert isinstance(written_report["errors"], dict)
        assert "count" in written_report["errors"]
        assert "items" in written_report["errors"]

        # For train/val splits the sample dataset has annotated classes;
        # the test split may have no ground-truth annotations (empty objects).
        if split in ("train", "val"):
            assert written_report["num_classes"] > 0
            assert len(written_report["class_names"]) > 0

        # -----------------------------------------------------------------
        # Assert metric values are in valid range [0, 1]
        # -----------------------------------------------------------------
        for metric_name in ["map_50", "map_50_95", "precision", "recall", "f1_score"]:
            value = metrics[metric_name]
            assert 0.0 <= value <= 1.0, (
                f"Metric '{metric_name}' = {value} is out of [0, 1] range "
                f"for split '{split}'"
            )

        # per_class_ap should be a dict (may be empty for test split with no GT)
        per_class_ap = metrics["per_class_ap"]
        assert isinstance(per_class_ap, dict)
        if split in ("train", "val"):
            assert len(per_class_ap) > 0

        # -----------------------------------------------------------------
        # Assert predictions file has correct structure
        # -----------------------------------------------------------------
        with open(predictions_file, "r", encoding="utf-8") as f:
            predictions_data = json.load(f)

        assert "images" in predictions_data
        assert "checkpoint" in predictions_data
        assert "model_type" in predictions_data
        assert len(predictions_data["images"]) == written_report["num_images"]

        # Each image entry should have the expected structure
        for img_entry in predictions_data["images"][:5]:  # Check first few
            assert "image_id" in img_entry
            assert "ground_truth" in img_entry
            assert "predictions" in img_entry
            assert "boxes" in img_entry["ground_truth"]
            assert "labels" in img_entry["ground_truth"]
            assert "boxes" in img_entry["predictions"]
            assert "labels" in img_entry["predictions"]
            assert "scores" in img_entry["predictions"]

        # -----------------------------------------------------------------
        # Assert the returned report dict matches what was written
        # -----------------------------------------------------------------
        assert report["split"] == split
        assert report["model_type"] == "integration_fake"
        for field in REQUIRED_REPORT_FIELDS:
            assert field in report
