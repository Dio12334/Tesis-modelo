"""Unit tests for inference-loop runtime behavior.

Feature: generic-evaluation-script
Task 8.4: no_grad, device placement, and progress logging

These are example-based (NOT property-based) unit tests for the runtime
behaviors of ``run_inference`` in ``model/training/evaluate_detection.py`` that
are best verified with spies rather than universally-quantified properties:

* the forward pass executes inside a ``torch.no_grad()`` context, so
  ``torch.is_grad_enabled()`` is ``False`` while ``detector.forward`` runs
  (Req 7.6);
* every input image tensor is moved to the selected Device *before*
  ``detector.forward`` is invoked, so the tensor the detector receives reports
  the requested device (Req 10.4);
* progress is logged at INFO level after processing counts that are positive
  multiples of 50 (50, 100, 150, ...) and not at other counts (Req 7.5).

A conforming fake :class:`_RecordingDetector` records, on every ``forward``
call, the ambient ``torch.is_grad_enabled()`` flag and the ``.device`` of the
input tensor it received. A conforming fake split dataset
(:class:`_FakeSplit`) yields a configurable number of annotations so progress
logging at multiples of 50 can be exercised deterministically. ``PIL.Image``
loading is monkeypatched in the evaluate_detection module to return a small
in-memory RGB image, so the tests run without any real image files, GPUs, real
checkpoints, or real datasets. A CPU device is used throughout.

_Requirements: 7.5, 7.6, 10.4_
"""

import logging
from pathlib import Path

import pytest
import torch
from PIL import Image as PILImage

import model.training.evaluate_detection as eval_mod
from model.training.evaluate_detection import run_inference


# ---------------------------------------------------------------------------
# Conforming fakes
# ---------------------------------------------------------------------------


class _RecordingDetector:
    """A minimal BaseDetector-conforming fake that spies on ``forward``.

    On every ``forward`` call it records the ambient ``torch.is_grad_enabled()``
    flag and the ``.device`` of the input tensor, then returns one empty
    detection dict per image in the batch (shape-conforming per Req 2.5). Empty
    detections keep the test focused on the loop's runtime behavior rather than
    on coordinate post-processing.
    """

    def __init__(self):
        self.grad_enabled_seen = []
        self.input_devices_seen = []
        self.call_count = 0

    def forward(self, images):
        self.call_count += 1
        # Spy on the ambient autograd state and the input tensor's device.
        self.grad_enabled_seen.append(torch.is_grad_enabled())
        self.input_devices_seen.append(images.device)

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
    """Return a small in-memory RGB image regardless of the requested path.

    The returned object is a real ``PIL.Image`` so the torchvision ``Resize`` +
    ``ToTensor`` transform applied by ``run_inference`` works unchanged, and its
    ``.convert("RGB")`` is a no-op copy.
    """
    return PILImage.new("RGB", (8, 8), color=(127, 127, 127))


@pytest.fixture
def patched_image_open(monkeypatch):
    """Monkeypatch ``Image.open`` used by evaluate_detection to load fake images."""
    monkeypatch.setattr(eval_mod.Image, "open", _fake_open)
    return _fake_open


# ---------------------------------------------------------------------------
# Req 7.6: forward runs under torch.no_grad()
# ---------------------------------------------------------------------------


class TestForwardRunsWithGradDisabled:
    """``detector.forward`` executes inside a ``torch.no_grad()`` context.

    _Requirements: 7.6_
    """

    def test_grad_disabled_during_forward(self, patched_image_open):
        """``torch.is_grad_enabled()`` is False for every forward call. (Req 7.6)"""
        detector = _RecordingDetector()
        split_ds = _FakeSplit(3)
        device = torch.device("cpu")

        # Sanity: autograd is enabled in the ambient context before the loop, so
        # observing it disabled inside forward is attributable to no_grad().
        assert torch.is_grad_enabled() is True

        run_inference(detector, split_ds, device, input_size=8, idx_to_class={1: "crack"})

        assert detector.call_count == 3
        assert detector.grad_enabled_seen == [False, False, False]

        # The no_grad() context is properly exited: grad is enabled again after.
        assert torch.is_grad_enabled() is True


# ---------------------------------------------------------------------------
# Req 10.4: input tensor moved to selected device before forward
# ---------------------------------------------------------------------------


class TestInputTensorOnSelectedDevice:
    """Input image tensors are moved to the selected Device before forward.

    _Requirements: 10.4_
    """

    def test_input_tensor_on_requested_device(self, patched_image_open):
        """Each tensor the detector receives is on the requested device. (Req 10.4)"""
        detector = _RecordingDetector()
        split_ds = _FakeSplit(4)
        device = torch.device("cpu")

        run_inference(detector, split_ds, device, input_size=8, idx_to_class={1: "crack"})

        assert detector.call_count == 4
        assert len(detector.input_devices_seen) == 4
        for seen_device in detector.input_devices_seen:
            assert seen_device == device


# ---------------------------------------------------------------------------
# Req 7.5: progress logged at positive multiples of 50
# ---------------------------------------------------------------------------


def _progress_messages(caplog):
    """Return the INFO progress-log messages emitted by the inference loop."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO and "Processed" in record.getMessage()
    ]


class TestProgressLoggingAtMultiplesOf50:
    """INFO progress logs appear at positive multiples of 50 only.

    _Requirements: 7.5_
    """

    def test_logs_at_50_and_100_for_100_images(self, patched_image_open, caplog):
        """100 images -> progress logged exactly at 50 and 100. (Req 7.5)"""
        detector = _RecordingDetector()
        split_ds = _FakeSplit(100)
        device = torch.device("cpu")

        with caplog.at_level(logging.INFO, logger="model.training.evaluate_detection"):
            run_inference(
                detector, split_ds, device, input_size=8, idx_to_class={1: "crack"}
            )

        messages = _progress_messages(caplog)
        assert len(messages) == 2
        assert any("50/100" in m for m in messages)
        assert any("100/100" in m for m in messages)

    def test_logs_at_50_100_150_for_175_images(self, patched_image_open, caplog):
        """175 images -> progress logged at 50, 100, and 150 (not 175). (Req 7.5)"""
        detector = _RecordingDetector()
        split_ds = _FakeSplit(175)
        device = torch.device("cpu")

        with caplog.at_level(logging.INFO, logger="model.training.evaluate_detection"):
            run_inference(
                detector, split_ds, device, input_size=8, idx_to_class={1: "crack"}
            )

        messages = _progress_messages(caplog)
        assert len(messages) == 3
        assert any("50/175" in m for m in messages)
        assert any("100/175" in m for m in messages)
        assert any("150/175" in m for m in messages)
        # No progress log at the final partial count (175 is not a multiple of 50).
        assert not any("175/175" in m for m in messages)

    def test_no_progress_log_below_50(self, patched_image_open, caplog):
        """Fewer than 50 images -> no progress logs are emitted. (Req 7.5)"""
        detector = _RecordingDetector()
        split_ds = _FakeSplit(30)
        device = torch.device("cpu")

        with caplog.at_level(logging.INFO, logger="model.training.evaluate_detection"):
            run_inference(
                detector, split_ds, device, input_size=8, idx_to_class={1: "crack"}
            )

        assert _progress_messages(caplog) == []
