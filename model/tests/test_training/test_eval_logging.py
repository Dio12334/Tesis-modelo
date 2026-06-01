"""Smoke tests for logging side-effects in the evaluation pipeline.

Feature: generic-evaluation-script
Task 12.4: Write smoke tests for logging side-effects

These example-based tests verify that the evaluation pipeline emits the
correct log messages at the correct levels for key observable side-effects:

* DEBUG normalization-decision log per batch — whether pixel or normalized
  coordinates were detected (Req 6.3).
* INFO device log before detector load — the selected device is logged
  (Req 10.2).
* INFO progress at multiples of 50 — progress logged at 50, 100, 150, etc.
  (Req 7.5).
* INFO absolute report/predictions paths after writing (Req 16.7).

All tests use conforming fakes and monkeypatching so they run without GPUs,
real checkpoints, real datasets, or real image files.

_Requirements: 6.3, 7.5, 10.2, 16.7_
"""

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image as PILImage

import model.training.evaluate_detection as eval_mod
from model.training.evaluate_detection import (
    run_inference,
    select_device,
    write_outputs,
)


# ---------------------------------------------------------------------------
# Conforming fakes
# ---------------------------------------------------------------------------


class _PixelDetector:
    """A fake detector that returns pixel-coordinate boxes (values > 1.0)."""

    def forward(self, images):
        batch = images.shape[0]
        return [
            {
                "boxes": torch.tensor([[10.0, 20.0, 50.0, 60.0]]),
                "labels": torch.tensor([1], dtype=torch.int64),
                "scores": torch.tensor([0.9]),
            }
            for _ in range(batch)
        ]


class _NormalizedDetector:
    """A fake detector that returns normalized-coordinate boxes (all <= 1.0)."""

    def forward(self, images):
        batch = images.shape[0]
        return [
            {
                "boxes": torch.tensor([[0.1, 0.2, 0.5, 0.6]]),
                "labels": torch.tensor([1], dtype=torch.int64),
                "scores": torch.tensor([0.85]),
            }
            for _ in range(batch)
        ]


class _EmptyDetector:
    """A fake detector that returns no detections (empty boxes)."""

    def forward(self, images):
        batch = images.shape[0]
        return [
            {
                "boxes": torch.zeros((0, 4)),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,)),
            }
            for _ in range(batch)
        ]


class _FakeAnnotation:
    """A minimal Annotation stand-in with an image path and no ground-truth boxes."""

    def __init__(self, image_path: str):
        self.image_path = Path(image_path)
        self.bounding_boxes = []


class _FakeSplit:
    """A split dataset exposing ``get_annotations()`` with ``n`` annotations."""

    def __init__(self, n: int):
        self._annotations = [_FakeAnnotation(f"img_{i:04d}.jpg") for i in range(n)]

    def get_annotations(self):
        return self._annotations


def _fake_open(path):
    """Return a small in-memory RGB image regardless of the requested path."""
    return PILImage.new("RGB", (8, 8), color=(127, 127, 127))


@pytest.fixture
def patched_image_open(monkeypatch):
    """Monkeypatch ``Image.open`` used by evaluate_detection to load fake images."""
    monkeypatch.setattr(eval_mod.Image, "open", _fake_open)


# ---------------------------------------------------------------------------
# Req 6.3: DEBUG normalization-decision log per batch
# ---------------------------------------------------------------------------


class TestNormalizationDecisionDebugLog:
    """DEBUG log records whether pixel or normalized coordinates were detected.

    _Requirements: 6.3_
    """

    def test_pixel_coordinates_logged(self, patched_image_open, caplog):
        """When detector returns pixel coords, DEBUG log mentions 'pixel'. (Req 6.3)"""
        detector = _PixelDetector()
        split_ds = _FakeSplit(2)
        device = torch.device("cpu")

        with caplog.at_level(logging.DEBUG, logger="model.training.evaluate_detection"):
            run_inference(
                detector, split_ds, device, input_size=100, idx_to_class={1: "crack"}
            )

        debug_messages = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.DEBUG
            and "normalization" in record.getMessage().lower()
        ]
        # One normalization-decision log per image that has detections.
        assert len(debug_messages) == 2
        for msg in debug_messages:
            assert "pixel" in msg

    def test_normalized_coordinates_logged(self, patched_image_open, caplog):
        """When detector returns normalized coords, DEBUG log mentions 'normalized'. (Req 6.3)"""
        detector = _NormalizedDetector()
        split_ds = _FakeSplit(3)
        device = torch.device("cpu")

        with caplog.at_level(logging.DEBUG, logger="model.training.evaluate_detection"):
            run_inference(
                detector, split_ds, device, input_size=100, idx_to_class={1: "crack"}
            )

        debug_messages = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.DEBUG
            and "normalization" in record.getMessage().lower()
        ]
        assert len(debug_messages) == 3
        for msg in debug_messages:
            assert "normalized" in msg

    def test_no_normalization_log_for_empty_detections(self, patched_image_open, caplog):
        """When detector returns no boxes, no normalization DEBUG log is emitted. (Req 6.3)"""
        detector = _EmptyDetector()
        split_ds = _FakeSplit(2)
        device = torch.device("cpu")

        with caplog.at_level(logging.DEBUG, logger="model.training.evaluate_detection"):
            run_inference(
                detector, split_ds, device, input_size=100, idx_to_class={1: "crack"}
            )

        debug_norm_messages = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.DEBUG
            and "normalization" in record.getMessage().lower()
        ]
        # No boxes means no normalization decision to log.
        assert len(debug_norm_messages) == 0


# ---------------------------------------------------------------------------
# Req 10.2: INFO device log before detector load
# ---------------------------------------------------------------------------


class TestDeviceInfoLog:
    """INFO log records the selected device before the detector is loaded.

    _Requirements: 10.2_
    """

    def test_device_logged_at_info_cuda(self, caplog):
        """When CUDA is available, INFO log mentions 'cuda'. (Req 10.2)"""
        with patch("torch.cuda.is_available", return_value=True):
            with caplog.at_level(logging.INFO, logger="model.training.evaluate_detection"):
                device = select_device()

        assert device == torch.device("cuda")
        info_messages = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.INFO
        ]
        assert any("cuda" in msg.lower() for msg in info_messages)

    def test_device_logged_at_info_cpu(self, caplog):
        """When CUDA is not available, INFO log mentions 'cpu'. (Req 10.2)"""
        with patch("torch.cuda.is_available", return_value=False):
            with caplog.at_level(logging.INFO, logger="model.training.evaluate_detection"):
                device = select_device()

        assert device == torch.device("cpu")
        info_messages = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.INFO
        ]
        assert any("cpu" in msg.lower() for msg in info_messages)

    def test_device_log_contains_device_keyword(self, caplog):
        """The INFO device log contains the word 'device'. (Req 10.2)"""
        with patch("torch.cuda.is_available", return_value=False):
            with caplog.at_level(logging.INFO, logger="model.training.evaluate_detection"):
                select_device()

        info_messages = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.INFO
        ]
        assert any("device" in msg.lower() for msg in info_messages)


# ---------------------------------------------------------------------------
# Req 7.5: INFO progress at multiples of 50
# ---------------------------------------------------------------------------


class TestProgressInfoLog:
    """INFO progress logs appear at positive multiples of 50.

    _Requirements: 7.5_
    """

    def _progress_messages(self, caplog):
        """Return INFO progress-log messages from the inference loop."""
        return [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.INFO and "Processed" in record.getMessage()
        ]

    def test_progress_at_50(self, patched_image_open, caplog):
        """50 images -> progress logged at 50. (Req 7.5)"""
        detector = _EmptyDetector()
        split_ds = _FakeSplit(50)
        device = torch.device("cpu")

        with caplog.at_level(logging.INFO, logger="model.training.evaluate_detection"):
            run_inference(
                detector, split_ds, device, input_size=8, idx_to_class={1: "crack"}
            )

        messages = self._progress_messages(caplog)
        assert len(messages) == 1
        assert "50/50" in messages[0]

    def test_progress_at_100_and_150(self, patched_image_open, caplog):
        """150 images -> progress logged at 50, 100, and 150. (Req 7.5)"""
        detector = _EmptyDetector()
        split_ds = _FakeSplit(150)
        device = torch.device("cpu")

        with caplog.at_level(logging.INFO, logger="model.training.evaluate_detection"):
            run_inference(
                detector, split_ds, device, input_size=8, idx_to_class={1: "crack"}
            )

        messages = self._progress_messages(caplog)
        assert len(messages) == 3
        assert any("50/150" in m for m in messages)
        assert any("100/150" in m for m in messages)
        assert any("150/150" in m for m in messages)

    def test_no_progress_below_50(self, patched_image_open, caplog):
        """Fewer than 50 images -> no progress logs. (Req 7.5)"""
        detector = _EmptyDetector()
        split_ds = _FakeSplit(25)
        device = torch.device("cpu")

        with caplog.at_level(logging.INFO, logger="model.training.evaluate_detection"):
            run_inference(
                detector, split_ds, device, input_size=8, idx_to_class={1: "crack"}
            )

        messages = self._progress_messages(caplog)
        assert messages == []


# ---------------------------------------------------------------------------
# Req 16.7: INFO absolute report/predictions paths after writing
# ---------------------------------------------------------------------------


class TestOutputPathsInfoLog:
    """INFO logs record absolute report and predictions paths after writing.

    _Requirements: 16.7_
    """

    def test_report_path_logged_at_info(self, tmp_path, caplog):
        """After writing, the absolute report path is logged at INFO. (Req 16.7)"""
        report = {
            "checkpoint": "/fake/checkpoint.pt",
            "model_type": "test_model",
            "model_config": {},
            "dataset": "/fake/dataset",
            "split": "val",
            "num_images": 10,
            "num_classes": 2,
            "class_names": ["crack", "pothole"],
            "confidence_threshold": 0.25,
            "iou_threshold": 0.5,
            "metrics": {"map_50": 0.5, "map_50_95": 0.3},
            "confusion_matrix": [[0, 0], [0, 0]],
            "errors": {"count": 0, "items": []},
        }
        predictions = [
            {"image_id": "img_0.jpg", "boxes": [], "labels": [], "scores": []}
        ]
        ground_truths = [
            {"image_id": "img_0.jpg", "boxes": [], "labels": []}
        ]

        with caplog.at_level(logging.INFO, logger="model.training.evaluate_detection"):
            report_path, preds_path = write_outputs(
                report=report,
                predictions=predictions,
                ground_truths=ground_truths,
                output_dir=str(tmp_path),
                split="val",
                checkpoint_path=Path("/fake/checkpoint.pt"),
            )

        info_messages = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.INFO
        ]

        # The report path should be logged.
        assert any("report" in msg.lower() and str(report_path) in msg for msg in info_messages)

    def test_predictions_path_logged_at_info(self, tmp_path, caplog):
        """After writing, the absolute predictions path is logged at INFO. (Req 16.7)"""
        report = {
            "checkpoint": "/fake/checkpoint.pt",
            "model_type": "test_model",
            "model_config": {},
            "dataset": "/fake/dataset",
            "split": "test",
            "num_images": 5,
            "num_classes": 1,
            "class_names": ["crack"],
            "confidence_threshold": 0.25,
            "iou_threshold": 0.5,
            "metrics": {"map_50": 0.0, "map_50_95": 0.0},
            "confusion_matrix": [[0]],
            "errors": {"count": 0, "items": []},
        }
        predictions = [
            {"image_id": "img_0.jpg", "boxes": [], "labels": [], "scores": []}
        ]
        ground_truths = [
            {"image_id": "img_0.jpg", "boxes": [], "labels": []}
        ]

        with caplog.at_level(logging.INFO, logger="model.training.evaluate_detection"):
            report_path, preds_path = write_outputs(
                report=report,
                predictions=predictions,
                ground_truths=ground_truths,
                output_dir=str(tmp_path),
                split="test",
                checkpoint_path=Path("/fake/checkpoint.pt"),
            )

        info_messages = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.INFO
        ]

        # The predictions path should be logged.
        assert any("predictions" in msg.lower() and str(preds_path) in msg for msg in info_messages)

    def test_both_paths_are_absolute(self, tmp_path, caplog):
        """Both logged paths are absolute paths. (Req 16.7)"""
        report = {
            "checkpoint": "/fake/checkpoint.pt",
            "model_type": "test_model",
            "model_config": {},
            "dataset": "/fake/dataset",
            "split": "train",
            "num_images": 1,
            "num_classes": 1,
            "class_names": ["crack"],
            "confidence_threshold": 0.25,
            "iou_threshold": 0.5,
            "metrics": {"map_50": 0.0, "map_50_95": 0.0},
            "confusion_matrix": [[0]],
            "errors": {"count": 0, "items": []},
        }
        predictions = [
            {"image_id": "img_0.jpg", "boxes": [], "labels": [], "scores": []}
        ]
        ground_truths = [
            {"image_id": "img_0.jpg", "boxes": [], "labels": []}
        ]

        with caplog.at_level(logging.INFO, logger="model.training.evaluate_detection"):
            report_path, preds_path = write_outputs(
                report=report,
                predictions=predictions,
                ground_truths=ground_truths,
                output_dir=str(tmp_path),
                split="train",
                checkpoint_path=Path("/fake/checkpoint.pt"),
            )

        # Both returned paths should be absolute.
        assert report_path.is_absolute()
        assert preds_path.is_absolute()

        # The files should actually exist.
        assert report_path.exists()
        assert preds_path.exists()
