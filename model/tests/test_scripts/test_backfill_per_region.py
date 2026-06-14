"""Tests for the per-region backfill CLI.

Covers the user-facing behaviours of
:mod:`model.scripts.backfill_per_region`:

* ``backfill_one`` reads a report + inference pair, computes per-region
  metric blocks and patches the report file in place.
* The function is idempotent: running again on an already-upgraded report
  is a no-op unless ``--force`` is supplied.
* ``--dry-run`` reports what would change without touching the file.
* ``--run-id`` and ``--split`` filters restrict the discovery walk.
* The CLI exits non-zero when at least one file fails.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model.scripts.backfill_per_region import (
    _discover_report_pairs,
    _predictions_from_inference,
    backfill_one,
    main,
)


CLASS_NAMES = [
    "alligator crack",
    "longitudinal crack",
    "other corruption",
    "pothole",
    "transverse crack",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _img(image_id: str, label: str = "pothole", score: float = 0.9) -> dict:
    return {
        "image_id": image_id,
        "ground_truth": {
            "boxes": [[0.1, 0.1, 0.3, 0.3]],
            "labels": [label],
        },
        "predictions": {
            "boxes": [[0.11, 0.11, 0.29, 0.29]],
            "labels": [label],
            "scores": [score],
        },
    }


def _make_pair(
    run_dir: Path,
    *,
    split: str = "val",
    images: list[dict] | None = None,
    include_per_region: bool = False,
) -> tuple[Path, Path]:
    """Write a report/inference pair into ``run_dir`` and return their paths."""
    run_dir.mkdir(parents=True, exist_ok=True)
    if images is None:
        images = [
            _img("Japan_000001.jpg", "pothole"),
            _img("Czech_000001.jpg", "pothole"),
            _img("United_States_000001.jpg", "longitudinal crack"),
        ]

    report = {
        "checkpoint": str(run_dir / "best.pt"),
        "model_type": "yolo26",
        "model_config": {"input_size": 640},
        "dataset": "model/data/rdd2022/complete",
        "split": split,
        "num_images": len(images),
        "num_classes": len(CLASS_NAMES),
        "class_names": list(CLASS_NAMES),
        "display_class_names": list(CLASS_NAMES),
        "confidence_threshold": 0.25,
        "iou_threshold": 0.5,
        "metrics": {
            "map_50": 0.5,
            "map_50_95": 0.3,
            "per_class_ap": {name: 0.5 for name in CLASS_NAMES},
            "mAP@0.5": 0.5,
            "mAP@0.5:0.95": 0.3,
            "precision": 0.7,
            "recall": 0.6,
            "f1_score": 0.65,
        },
        "confusion_matrix": [[0] * len(CLASS_NAMES) for _ in CLASS_NAMES],
        "errors": {"count": 0, "items": []},
    }
    if include_per_region:
        report["per_region"] = {"already": "here"}

    inference = {
        "checkpoint": str(run_dir / "best.pt"),
        "model_type": "yolo26",
        "dataset": "model/data/rdd2022/complete",
        "confidence_threshold": 0.25,
        "class_names": list(CLASS_NAMES),
        "display_class_names": list(CLASS_NAMES),
        "images": images,
    }

    report_path = run_dir / f"{split}_evaluation_report.json"
    inference_path = run_dir / f"{split}_inference.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    inference_path.write_text(json.dumps(inference), encoding="utf-8")
    return report_path, inference_path


# ---------------------------------------------------------------------------
# _predictions_from_inference
# ---------------------------------------------------------------------------


def test_predictions_from_inference_aligns_lists() -> None:
    inference = {
        "images": [
            _img("Japan_000001.jpg"),
            _img("Czech_000002.jpg"),
        ]
    }
    preds, gts = _predictions_from_inference(inference)
    assert len(preds) == len(gts) == 2
    assert preds[0]["image_id"] == "Japan_000001.jpg"
    assert preds[0]["scores"] == [0.9]
    assert gts[1]["labels"] == ["pothole"]


def test_predictions_from_inference_handles_empty_payload() -> None:
    preds, gts = _predictions_from_inference({})
    assert preds == [] and gts == []


# ---------------------------------------------------------------------------
# _discover_report_pairs
# ---------------------------------------------------------------------------


def test_discover_finds_all_splits(tmp_path: Path) -> None:
    run_a = tmp_path / "model_a" / "11111111-1111-1111-1111-111111111111"
    run_b = tmp_path / "model_b" / "22222222-2222-2222-2222-222222222222"
    _make_pair(run_a, split="val")
    _make_pair(run_a, split="test")
    _make_pair(run_b, split="val")
    pairs = _discover_report_pairs(tmp_path)
    assert len(pairs) == 3


def test_discover_filters_by_run_id(tmp_path: Path) -> None:
    run_a = tmp_path / "model_a" / "11111111-1111-1111-1111-111111111111"
    run_b = tmp_path / "model_b" / "22222222-2222-2222-2222-222222222222"
    _make_pair(run_a, split="val")
    _make_pair(run_b, split="val")
    pairs = _discover_report_pairs(
        tmp_path, run_id="22222222-2222-2222-2222-222222222222"
    )
    assert len(pairs) == 1
    assert pairs[0][0].parent == run_b


def test_discover_filters_by_split(tmp_path: Path) -> None:
    run = tmp_path / "model_a" / "11111111-1111-1111-1111-111111111111"
    _make_pair(run, split="val")
    _make_pair(run, split="test")
    pairs = _discover_report_pairs(tmp_path, split="test")
    assert len(pairs) == 1
    assert pairs[0][2] == "test"


def test_discover_skips_reports_without_inference(tmp_path: Path) -> None:
    run = tmp_path / "model_a" / "11111111-1111-1111-1111-111111111111"
    run.mkdir(parents=True)
    # Write only the report, no sibling inference.
    (run / "val_evaluation_report.json").write_text("{}", encoding="utf-8")
    pairs = _discover_report_pairs(tmp_path)
    assert pairs == []


# ---------------------------------------------------------------------------
# backfill_one
# ---------------------------------------------------------------------------


def test_backfill_one_adds_per_region_field(tmp_path: Path) -> None:
    run = tmp_path / "model_a" / "11111111-1111-1111-1111-111111111111"
    report_path, inference_path = _make_pair(run)
    outcome = backfill_one(report_path, inference_path)
    assert outcome == "updated"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "per_region" in payload
    assert set(payload["per_region"].keys()) == {"Japan", "Czech", "United_States"}
    block = payload["per_region"]["Japan"]
    assert block["num_images"] == 1
    assert block["num_gt_boxes"] == 1
    for key in ("map_50", "precision", "recall", "f1_score", "per_class_ap"):
        assert key in block["metrics"]


def test_backfill_one_is_idempotent(tmp_path: Path) -> None:
    run = tmp_path / "model_a" / "11111111-1111-1111-1111-111111111111"
    report_path, inference_path = _make_pair(run, include_per_region=True)
    outcome = backfill_one(report_path, inference_path)
    assert outcome == "skipped"
    # Pre-existing per_region was untouched.
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["per_region"] == {"already": "here"}


def test_backfill_one_force_overwrites(tmp_path: Path) -> None:
    run = tmp_path / "model_a" / "11111111-1111-1111-1111-111111111111"
    report_path, inference_path = _make_pair(run, include_per_region=True)
    outcome = backfill_one(report_path, inference_path, force=True)
    assert outcome == "updated"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    # The placeholder is gone; real region keys are present.
    assert "already" not in payload["per_region"]
    assert "Japan" in payload["per_region"]


def test_backfill_one_dry_run_does_not_write(tmp_path: Path) -> None:
    run = tmp_path / "model_a" / "11111111-1111-1111-1111-111111111111"
    report_path, inference_path = _make_pair(run)
    before = report_path.read_text(encoding="utf-8")
    outcome = backfill_one(report_path, inference_path, dry_run=True)
    assert outcome == "would-update"
    after = report_path.read_text(encoding="utf-8")
    assert before == after


def test_backfill_one_skips_when_no_regions_match(tmp_path: Path) -> None:
    """All-non-matching filenames -> nothing to backfill."""
    run = tmp_path / "model_a" / "11111111-1111-1111-1111-111111111111"
    bad_images = [
        {
            "image_id": "garbage.jpg",
            "ground_truth": {"boxes": [[0, 0, 1, 1]], "labels": ["pothole"]},
            "predictions": {
                "boxes": [[0, 0, 1, 1]],
                "labels": ["pothole"],
                "scores": [0.5],
            },
        }
    ]
    report_path, inference_path = _make_pair(run, images=bad_images)
    outcome = backfill_one(report_path, inference_path)
    assert outcome == "skipped"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "per_region" not in payload


# ---------------------------------------------------------------------------
# main / CLI exit code
# ---------------------------------------------------------------------------


def test_main_returns_zero_on_success(tmp_path: Path) -> None:
    run = tmp_path / "model_a" / "11111111-1111-1111-1111-111111111111"
    _make_pair(run, split="val")
    rc = main(["--checkpoints-dir", str(tmp_path)])
    assert rc == 0


def test_main_returns_zero_when_no_pairs_found(tmp_path: Path) -> None:
    rc = main(["--checkpoints-dir", str(tmp_path)])
    assert rc == 0


def test_main_dry_run_leaves_files_untouched(tmp_path: Path) -> None:
    run = tmp_path / "model_a" / "11111111-1111-1111-1111-111111111111"
    report_path, _ = _make_pair(run, split="val")
    before = report_path.read_text(encoding="utf-8")
    rc = main(["--checkpoints-dir", str(tmp_path), "--dry-run"])
    assert rc == 0
    assert report_path.read_text(encoding="utf-8") == before


def test_main_returns_nonzero_on_unreadable_report(tmp_path: Path) -> None:
    run = tmp_path / "model_a" / "11111111-1111-1111-1111-111111111111"
    report_path, inference_path = _make_pair(run, split="val")
    # Corrupt the report so json.load raises.
    report_path.write_text("not-json", encoding="utf-8")
    rc = main(["--checkpoints-dir", str(tmp_path)])
    assert rc == 1
