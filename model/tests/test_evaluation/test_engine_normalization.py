"""Tests for the evaluation engine's prediction normalization and label filtering.

These tests cover the rt-detr-num-classes Phase 4 fix:
- Predictions returned in pixel coordinates by detector wrappers must be
  normalized to [0, 1] to match the project convention (BoundingBox stores
  normalized coords; metrics.compute_iou expects normalized).
- Predictions whose label index is outside the configured class_names range
  must be dropped with a warning, instead of being silently stringified.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


from model.evaluation.engine import EvaluationEngine


@pytest.fixture
def fake_image(tmp_path):
    """Create a small RGB image on disk and return its path with known dimensions."""
    pytest.importorskip("PIL")
    from PIL import Image

    img_path = tmp_path / "fake.jpg"
    img = Image.new("RGB", (640, 480), color=(123, 45, 67))
    img.save(img_path)
    return img_path, 640, 480


def _make_dataset(image_path, class_names):
    """Build a minimal dataset stand-in compatible with EvaluationEngine."""
    annotation = MagicMock()
    annotation.image_path = image_path
    annotation.bounding_boxes = []  # GT not relevant for these tests
    annotation.metadata = {}

    dataset = MagicMock()
    dataset.__iter__ = lambda self: iter([annotation])
    dataset.__len__ = lambda self: 1
    dataset.get_class_names = lambda: class_names
    return dataset


def _make_model_returning(boxes, labels, scores):
    """Build a mock detector whose forward() returns the given pixel-space boxes."""
    import torch

    model = MagicMock()
    model.forward = MagicMock(
        return_value=[
            {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "scores": torch.tensor(scores, dtype=torch.float32),
            }
        ]
    )
    return model


class TestPredictionNormalization:
    """Verify _run_inference normalizes pixel-space predictions to [0, 1]."""

    def test_predictions_normalized_to_unit_square(self, fake_image):
        """A pixel-space [320, 240, 480, 360] in a 640x480 image becomes [0.5, 0.5, 0.75, 0.75]."""
        img_path, W, H = fake_image
        dataset = _make_dataset(img_path, class_names=["a", "b", "c"])
        model = _make_model_returning(
            boxes=[[320.0, 240.0, 480.0, 360.0]],
            labels=[1],
            scores=[0.9],
        )
        engine = EvaluationEngine()
        preds = engine._run_inference(model, dataset)

        assert len(preds) == 1
        boxes = preds[0]["boxes"]
        assert len(boxes) == 1
        x1, y1, x2, y2 = boxes[0]
        assert x1 == pytest.approx(320.0 / W)
        assert y1 == pytest.approx(240.0 / H)
        assert x2 == pytest.approx(480.0 / W)
        assert y2 == pytest.approx(360.0 / H)

    def test_normalized_predictions_within_unit_range(self, fake_image):
        """All normalized coords must lie in [0, 1] for boxes inside the image."""
        img_path, _, _ = fake_image
        dataset = _make_dataset(img_path, class_names=["a", "b"])
        model = _make_model_returning(
            boxes=[[0.0, 0.0, 100.0, 100.0], [200.0, 100.0, 600.0, 400.0]],
            labels=[0, 1],
            scores=[0.7, 0.8],
        )
        engine = EvaluationEngine()
        preds = engine._run_inference(model, dataset)

        for box in preds[0]["boxes"]:
            for v in box:
                assert 0.0 <= v <= 1.0


class TestOutOfRangeLabelFiltering:
    """Verify out-of-range label indices are dropped with a warning."""

    def test_out_of_range_label_dropped(self, fake_image, caplog):
        """Label index 99 with 3 class_names must be dropped (not stringified)."""
        import logging

        img_path, _, _ = fake_image
        dataset = _make_dataset(img_path, class_names=["a", "b", "c"])
        model = _make_model_returning(
            boxes=[[10.0, 10.0, 50.0, 50.0]],
            labels=[99],
            scores=[0.9],
        )
        engine = EvaluationEngine()

        with caplog.at_level(logging.WARNING, logger="model.evaluation.engine"):
            preds = engine._run_inference(model, dataset)

        # The prediction was dropped: empty boxes/labels/scores.
        assert preds[0]["boxes"] == []
        assert preds[0]["labels"] == []
        assert preds[0]["scores"] == []

        # A warning was logged.
        assert any(
            "out-of-range label index 99" in record.message
            for record in caplog.records
        ), f"Expected warning about label 99, got: {[r.message for r in caplog.records]}"

    def test_in_range_labels_pass_through(self, fake_image):
        """Label indices < len(class_names) are mapped to class names normally."""
        img_path, _, _ = fake_image
        dataset = _make_dataset(img_path, class_names=["a", "b", "c"])
        model = _make_model_returning(
            boxes=[[0.0, 0.0, 100.0, 100.0]],
            labels=[2],
            scores=[0.8],
        )
        engine = EvaluationEngine()
        preds = engine._run_inference(model, dataset)

        assert preds[0]["labels"] == ["c"]
        assert len(preds[0]["boxes"]) == 1
        assert len(preds[0]["scores"]) == 1

    def test_mixed_in_and_out_of_range_only_in_range_kept(self, fake_image):
        """When both valid and invalid label indices appear, only valid ones survive."""
        img_path, _, _ = fake_image
        dataset = _make_dataset(img_path, class_names=["a", "b"])
        model = _make_model_returning(
            boxes=[
                [0.0, 0.0, 100.0, 100.0],
                [50.0, 50.0, 150.0, 150.0],
                [200.0, 200.0, 300.0, 300.0],
            ],
            labels=[0, 99, 1],
            scores=[0.8, 0.7, 0.6],
        )
        engine = EvaluationEngine()
        preds = engine._run_inference(model, dataset)

        # The middle prediction (label=99) is dropped.
        assert preds[0]["labels"] == ["a", "b"]
        assert len(preds[0]["boxes"]) == 2
        assert preds[0]["scores"] == [pytest.approx(0.8), pytest.approx(0.6)]


class TestEndToEndScaleConsistency:
    """Smoke test: when predictions match GT in scale, IoU is non-zero."""

    def test_normalized_pred_matches_normalized_gt(self, fake_image):
        """A pixel-space prediction equal to the GT in pixels yields IoU=1 after normalization."""
        from model.evaluation.metrics import compute_iou

        img_path, W, H = fake_image
        # GT box stored in normalized coords (project convention)
        gt_box = [0.1, 0.1, 0.5, 0.5]
        # Same box in pixel coords (what the detector returns)
        pred_pixel = [0.1 * W, 0.1 * H, 0.5 * W, 0.5 * H]

        dataset = _make_dataset(img_path, class_names=["a"])
        model = _make_model_returning(
            boxes=[pred_pixel], labels=[0], scores=[0.99]
        )
        engine = EvaluationEngine()
        preds = engine._run_inference(model, dataset)

        normalized_pred = preds[0]["boxes"][0]
        # IoU of two identical boxes (after normalization) must be 1.0
        iou = compute_iou(gt_box, normalized_pred)
        assert iou == pytest.approx(1.0, abs=1e-5)
