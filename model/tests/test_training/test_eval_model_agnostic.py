"""Smoke test for model-agnostic control flow in the evaluation pipeline.

Feature: generic-evaluation-script
Task 12.3: Write smoke test for model-agnostic control flow

Run the orchestration against two distinct fake models and assert identical
control flow / no ``model_type``-keyed branches. The test:

* Creates two different fake BaseDetector implementations (FakeModelA and
  FakeModelB) with distinct internal structures.
* Registers them in ModelRegistry under different names.
* Runs ``evaluate()`` with each model type.
* Asserts that the control flow (sequence of calls to ``forward``,
  ``load_checkpoint``, ``set_eval_mode``, ``to_device``) is identical for both
  models.

This proves no ``model_type``-keyed branches exist in the inference path.

**Validates: Requirements 1.3**
"""

import json
import logging
from pathlib import Path
from typing import List
from unittest.mock import patch

import torch
from PIL import Image as PILImage

import model.training.evaluate_detection as eval_mod
from model.models.registry import BaseDetector, ModelRegistry
from model.training.evaluate_detection import evaluate


# ---------------------------------------------------------------------------
# Call-recording mixin
# ---------------------------------------------------------------------------


class _CallRecorder:
    """Mixin that records the sequence of BaseDetector method calls."""

    def __init__(self):
        self.call_log: List[str] = []


# ---------------------------------------------------------------------------
# Two distinct fake BaseDetector implementations
# ---------------------------------------------------------------------------


class FakeModelA(_CallRecorder, BaseDetector):
    """A fake detector using a ``_model`` attribute (like SSD MobileNetV3).

    Internally stores weights as a simple dict and uses ``_model`` for the
    underlying model reference. Returns fixed detections from ``forward``.
    """

    def __init__(self, config: dict):
        _CallRecorder.__init__(self)
        self._config = config
        self._weights = {"layer1": 1.0, "layer2": 2.0}
        self._model = _FakeModule("model_a_module")

    def forward(self, images: torch.Tensor) -> List[dict]:
        self.call_log.append("forward")
        batch = images.shape[0]
        return [
            {
                "boxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
                "scores": torch.tensor([0.85], dtype=torch.float32),
            }
            for _ in range(batch)
        ]

    def get_config_schema(self) -> dict:
        return {"num_classes": {"type": "int", "required": True}}

    def load_checkpoint(self, path: Path) -> None:
        self.call_log.append("load_checkpoint")

    def save_checkpoint(self, path: Path) -> None:
        pass

    def set_eval_mode(self) -> None:
        self.call_log.append("set_eval_mode")
        super().set_eval_mode()

    def to_device(self, device) -> None:
        self.call_log.append("to_device")
        super().to_device(device)


class FakeModelB(_CallRecorder, BaseDetector):
    """A fake detector using a ``model`` attribute (like YOLOv6).

    Internally stores weights as a list and uses ``model`` for the underlying
    model reference. Returns different fixed detections from ``forward`` (two
    boxes per image instead of one), proving the orchestration doesn't branch
    on model internals.
    """

    def __init__(self, config: dict):
        _CallRecorder.__init__(self)
        self._config = config
        self._params = [0.5, 0.6, 0.7]
        self.model = _FakeModule("model_b_module")

    def forward(self, images: torch.Tensor) -> List[dict]:
        self.call_log.append("forward")
        batch = images.shape[0]
        return [
            {
                "boxes": torch.tensor(
                    [[0.2, 0.3, 0.6, 0.7], [0.1, 0.1, 0.4, 0.4]],
                    dtype=torch.float32,
                ),
                "labels": torch.tensor([1, 2], dtype=torch.int64),
                "scores": torch.tensor([0.9, 0.7], dtype=torch.float32),
            }
            for _ in range(batch)
        ]

    def get_config_schema(self) -> dict:
        return {"num_classes": {"type": "int", "required": True}}

    def load_checkpoint(self, path: Path) -> None:
        self.call_log.append("load_checkpoint")

    def save_checkpoint(self, path: Path) -> None:
        pass

    def set_eval_mode(self) -> None:
        self.call_log.append("set_eval_mode")
        super().set_eval_mode()

    def to_device(self, device) -> None:
        self.call_log.append("to_device")
        super().to_device(device)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeModule:
    """A minimal nn.Module stand-in with train/eval/to methods."""

    def __init__(self, name: str):
        self.name = name
        self._training = True

    def train(self):
        self._training = True

    def eval(self):
        self._training = False

    def to(self, device):
        pass


def _fake_image_open(path, *args, **kwargs):
    """Return a small in-memory RGB image regardless of path."""
    return PILImage.new("RGB", (8, 8), color=(100, 100, 100))


def _make_fake_dataset(num_images: int):
    """Create a fake dataset class that returns a fixed number of annotations."""
    from model.datasets.base import Annotation, BoundingBox

    class _FakeDataset:
        def __init__(self, *args, **kwargs):
            self._annotations = [
                Annotation(
                    image_path=Path(f"/fake/img_{i:04d}.png"),
                    bounding_boxes=[
                        BoundingBox(0.1, 0.1, 0.5, 0.5, "crack"),
                    ],
                )
                for i in range(num_images)
            ]

        def load(self, path):
            pass

        def get_annotations(self):
            return self._annotations

        def get_class_names(self):
            return ["crack", "pothole"]

        def split(self, train_ratio, val_ratio, test_ratio, seed=42):
            return self, self, self

        def __len__(self):
            return len(self._annotations)

    return _FakeDataset


# Global references to capture the detector instances created during evaluate()
_created_detectors: List = []

# Store a reference to the real build_detector before any patching occurs.
_real_build_detector = eval_mod.build_detector


def _capturing_build_detector(model_type, model_config):
    """Wrapper around build_detector that captures the created detector."""
    detector = _real_build_detector(model_type, model_config)
    _created_detectors.append(detector)
    return detector


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestModelAgnosticControlFlow:
    """Smoke test: two distinct fake models produce identical control flow.

    **Validates: Requirements 1.3**
    """

    def test_identical_control_flow_for_different_model_types(self, tmp_path):
        """evaluate() calls the same BaseDetector methods in the same order
        regardless of which model type is configured, proving no model_type-keyed
        branches exist in the inference path.

        **Validates: Requirements 1.3**
        """
        global _created_detectors

        # Save the original registry state and restore after test.
        original_models = ModelRegistry._models.copy()
        try:
            # Register both fake models.
            ModelRegistry._models["fake_model_a"] = FakeModelA
            ModelRegistry._models["fake_model_b"] = FakeModelB

            num_images = 3
            FakeDataset = _make_fake_dataset(num_images)

            call_logs = {}

            for model_name in ("fake_model_a", "fake_model_b"):
                _created_detectors = []

                # Create a checkpoint file so resolve_checkpoint succeeds.
                ckpt_path = tmp_path / model_name / "best_model.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({}, ckpt_path)

                # Build config overrides for this model.
                overrides = {
                    "model": {
                        "type": model_name,
                        "config": {"num_classes": 2},
                    },
                    "dataset": {"path": str(tmp_path / "dataset")},
                    "evaluation": {
                        "split": "val",
                        "confidence_threshold": 0.25,
                        "iou_threshold": 0.5,
                        "confidence_thresholds_sweep": [0.05, 0.25, 0.50],
                        "output_dir": str(tmp_path / "output" / model_name),
                    },
                    "checkpoint": {"path": str(ckpt_path)},
                }

                # Create the dataset directory so path existence check passes.
                (tmp_path / "dataset").mkdir(exist_ok=True)
                # Create output directory.
                (tmp_path / "output" / model_name).mkdir(parents=True, exist_ok=True)

                # Patch collaborators to isolate the orchestration control flow.
                with patch.object(eval_mod.Image, "open", _fake_image_open), \
                     patch(
                         "model.training.evaluate_detection.RDD2022Dataset",
                         FakeDataset,
                     ), \
                     patch(
                         "model.training.evaluate_detection.build_detector",
                         _capturing_build_detector,
                     ), \
                     patch(
                         "model.training.evaluate_detection.compute_map",
                         return_value={
                             "map_50": 0.5,
                             "map_50_95": 0.4,
                             "per_class_ap": {"crack": 0.5, "pothole": 0.4},
                         },
                     ), \
                     patch(
                         "model.training.evaluate_detection.compute_precision_recall_f1",
                         return_value={
                             "precision": 0.6,
                             "recall": 0.5,
                             "f1": 0.55,
                         },
                     ), \
                     patch(
                         "model.training.evaluate_detection.compute_confusion_matrix",
                         return_value=[[1, 0], [0, 1]],
                     ):
                    evaluate(config_path=None, overrides=overrides)

                # Capture the call log from the detector that was created.
                assert len(_created_detectors) == 1, (
                    f"Expected exactly one detector created for {model_name}, "
                    f"got {len(_created_detectors)}"
                )
                detector = _created_detectors[0]
                call_logs[model_name] = detector.call_log

            # ---------------------------------------------------------------
            # Assert identical control flow
            # ---------------------------------------------------------------
            log_a = call_logs["fake_model_a"]
            log_b = call_logs["fake_model_b"]

            # Both models must have been exercised.
            assert len(log_a) > 0, "FakeModelA received no calls"
            assert len(log_b) > 0, "FakeModelB received no calls"

            # The sequence of method calls must be identical, proving no
            # model_type-keyed branches exist in the inference path.
            assert log_a == log_b, (
                f"Control flow differs between models!\n"
                f"  FakeModelA calls: {log_a}\n"
                f"  FakeModelB calls: {log_b}\n"
                f"This indicates a model_type-keyed branch in the inference path."
            )

            # Verify the expected call sequence: load_checkpoint, set_eval_mode,
            # to_device, then forward once per image.
            expected_prefix = ["load_checkpoint", "set_eval_mode", "to_device"]
            expected_forwards = ["forward"] * num_images
            expected = expected_prefix + expected_forwards

            assert log_a == expected, (
                f"Unexpected control flow sequence.\n"
                f"  Expected: {expected}\n"
                f"  Got:      {log_a}"
            )

        finally:
            # Restore the original registry state.
            ModelRegistry._models = original_models

    def test_no_model_type_inspection_in_evaluate(self):
        """The evaluate() function source does not contain model_type-keyed
        conditional branches (static analysis complement to the dynamic test).

        **Validates: Requirements 1.3**
        """
        import inspect

        source = inspect.getsource(eval_mod.evaluate)

        # The orchestration should not branch on model_type values.
        # Check for common patterns that would indicate model-specific branching.
        suspicious_patterns = [
            'if model_type ==',
            'if model_type !=',
            'elif model_type',
            'model_type in [',
            'model_type in (',
            'model_type == "ssd',
            'model_type == "yolo',
            'isinstance(detector,',
            'type(detector)',
            'detector.__class__',
        ]

        for pattern in suspicious_patterns:
            assert pattern not in source, (
                f"Found suspicious model_type-keyed branch pattern in evaluate(): "
                f"'{pattern}'. This violates Requirement 1.3 (no conditional "
                f"branches keyed on model type in inference orchestration)."
            )
