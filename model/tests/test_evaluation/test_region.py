"""Unit tests for :mod:`model.evaluation.region`.

Covers:

* :func:`extract_region` matches every RDD2022 region prefix, including
  multi-segment prefixes like ``China_Drone`` and ``United_States``, and
  rejects malformed identifiers.
* :func:`group_by_region` partitions aligned predictions/ground-truths and
  silently drops images whose filename does not match the pattern.
* :func:`compute_region_metrics` returns a populated metric block whose
  scalars fall in ``[0, 1]`` and whose confusion matrix has shape
  ``(C+1, C+1)``.
* The empty-slice path returns zeroed metrics with the correct keys (and
  zeroed sweep entries when a sweep list is supplied).
* :func:`compute_per_region_metrics` returns one entry per discovered region
  in deterministic (sorted) order.
"""

from __future__ import annotations

import pytest

from model.evaluation.region import (
    REGION_RE,
    compute_per_region_metrics,
    compute_region_metrics,
    extract_region,
    group_by_region,
)


CLASS_NAMES = [
    "alligator crack",
    "longitudinal crack",
    "other corruption",
    "pothole",
    "transverse crack",
]


# ---------------------------------------------------------------------------
# extract_region
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "image_id,expected",
    [
        ("Japan_001234.jpg", "Japan"),
        ("Czech_000001.jpg", "Czech"),
        ("India_009999.png", "India"),
        ("Norway_000007.jpeg", "Norway"),
        ("United_States_002354.jpg", "United_States"),
        ("China_Drone_000000.jpg", "China_Drone"),
        ("China_MotorBike_000123.jpg", "China_MotorBike"),
        (
            "model/data/rdd2022/complete/train/img/United_States_002354.jpg",
            "United_States",
        ),
    ],
)
def test_extract_region_matches_known_prefixes(image_id: str, expected: str) -> None:
    assert extract_region(image_id) == expected


@pytest.mark.parametrize(
    "image_id",
    [
        "",
        "no_digits.jpg",
        "trailing_underscore_.jpg",
        "noextension_001234",
        "weird.name.jpg",
    ],
)
def test_extract_region_returns_none_for_unrecognised(image_id: str) -> None:
    assert extract_region(image_id) is None


def test_region_regex_is_anchored_at_end() -> None:
    """Sanity: the regex must require ``_<digits>.<ext>`` at the very end."""
    # The trailing ``$`` anchor is the load-bearing part: any text *after*
    # the digits/extension must invalidate the match. ``^`` is also required
    # so the prefix capture starts at position 0 of ``Path(...).name``.
    assert REGION_RE.match("Japan_001.jpg.bak") is None
    assert REGION_RE.match("Japan_001") is None
    # Non-greedy capture still picks up multi-segment prefixes.
    match = REGION_RE.match("United_States_002354.jpg")
    assert match is not None
    assert match.group(1) == "United_States"


# ---------------------------------------------------------------------------
# group_by_region
# ---------------------------------------------------------------------------


def _pred(image_id: str, *, label: str = "pothole", score: float = 0.9) -> dict:
    return {
        "image_id": image_id,
        "boxes": [[0.1, 0.1, 0.3, 0.3]],
        "labels": [label],
        "scores": [score],
    }


def _gt(image_id: str, *, label: str = "pothole") -> dict:
    return {
        "image_id": image_id,
        "boxes": [[0.11, 0.11, 0.29, 0.29]],
        "labels": [label],
    }


def test_group_by_region_partitions_aligned_lists() -> None:
    preds = [
        _pred("Japan_000001.jpg"),
        _pred("Japan_000002.jpg"),
        _pred("Czech_000001.jpg"),
        _pred("United_States_000010.jpg"),
        _pred("China_Drone_000003.jpg"),
    ]
    gts = [
        _gt("Japan_000001.jpg"),
        _gt("Japan_000002.jpg"),
        _gt("Czech_000001.jpg"),
        _gt("United_States_000010.jpg"),
        _gt("China_Drone_000003.jpg"),
    ]
    groups = group_by_region(preds, gts)
    assert set(groups.keys()) == {"Japan", "Czech", "United_States", "China_Drone"}
    assert len(groups["Japan"][0]) == 2
    assert len(groups["Czech"][0]) == 1


def test_group_by_region_skips_unrecognised_filenames() -> None:
    preds = [
        _pred("Japan_000001.jpg"),
        _pred("garbage.jpg"),
        _pred("Czech_000001.jpg"),
    ]
    gts = [
        _gt("Japan_000001.jpg"),
        _gt("garbage.jpg"),
        _gt("Czech_000001.jpg"),
    ]
    groups = group_by_region(preds, gts)
    assert set(groups.keys()) == {"Japan", "Czech"}


def test_group_by_region_rejects_misaligned_inputs() -> None:
    preds = [_pred("Japan_000001.jpg")]
    gts: list = []
    with pytest.raises(ValueError, match="aligned 1:1"):
        group_by_region(preds, gts)


def test_group_by_region_empty_inputs() -> None:
    assert group_by_region([], []) == {}


# ---------------------------------------------------------------------------
# compute_region_metrics
# ---------------------------------------------------------------------------


def test_compute_region_metrics_populated_slice() -> None:
    preds = [_pred("Japan_000001.jpg", label="pothole", score=0.95)]
    gts = [_gt("Japan_000001.jpg", label="pothole")]
    block = compute_region_metrics(
        predictions=preds,
        ground_truths=gts,
        class_names=CLASS_NAMES,
        confidence_threshold=0.25,
        iou_threshold=0.5,
    )
    assert block["num_images"] == 1
    assert block["num_gt_boxes"] == 1
    metrics = block["metrics"]
    for key in ("map_50", "map_50_95", "precision", "recall", "f1_score"):
        assert 0.0 <= metrics[key] <= 1.0
    # The pothole match is correct -> per-class AP for pothole > 0.
    assert metrics["per_class_ap"]["pothole"] > 0.0
    size = len(CLASS_NAMES) + 1
    assert len(block["confusion_matrix"]) == size
    assert all(len(row) == size for row in block["confusion_matrix"])


def test_compute_region_metrics_empty_slice_returns_zero_block() -> None:
    block = compute_region_metrics(
        predictions=[],
        ground_truths=[],
        class_names=CLASS_NAMES,
        confidence_threshold=0.25,
        iou_threshold=0.5,
    )
    assert block["num_images"] == 0
    assert block["num_gt_boxes"] == 0
    metrics = block["metrics"]
    assert metrics["map_50"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
    assert metrics["per_class_ap"] == {name: 0.0 for name in CLASS_NAMES}
    size = len(CLASS_NAMES) + 1
    assert len(block["confusion_matrix"]) == size


def test_compute_region_metrics_includes_sweep_when_requested() -> None:
    preds = [_pred("Japan_000001.jpg", label="pothole", score=0.95)]
    gts = [_gt("Japan_000001.jpg", label="pothole")]
    sweep_thresholds = [0.05, 0.25, 0.5, 0.9]
    block = compute_region_metrics(
        predictions=preds,
        ground_truths=gts,
        class_names=CLASS_NAMES,
        confidence_threshold=0.25,
        iou_threshold=0.5,
        confidence_thresholds_sweep=sweep_thresholds,
    )
    metrics = block["metrics"]
    assert "f1_sweep" in metrics
    assert len(metrics["f1_sweep"]) == len(sweep_thresholds)
    assert metrics["f1_sweep"][0]["confidence"] == pytest.approx(0.05)
    assert "best_f1" in metrics
    assert metrics["per_class_best_f1"].keys() == set(CLASS_NAMES)
    assert metrics["default_confidence_threshold"] == pytest.approx(0.25)


def test_compute_region_metrics_empty_slice_with_sweep_zeroes_everything() -> None:
    sweep_thresholds = [0.1, 0.5]
    block = compute_region_metrics(
        predictions=[],
        ground_truths=[],
        class_names=CLASS_NAMES,
        confidence_threshold=0.3,
        iou_threshold=0.5,
        confidence_thresholds_sweep=sweep_thresholds,
    )
    metrics = block["metrics"]
    assert len(metrics["f1_sweep"]) == 2
    assert all(entry["f1"] == 0.0 for entry in metrics["f1_sweep"])
    assert metrics["best_f1"]["f1"] == 0.0
    assert all(
        entry["f1"] == 0.0 for entry in metrics["per_class_best_f1"].values()
    )


# ---------------------------------------------------------------------------
# compute_per_region_metrics
# ---------------------------------------------------------------------------


def test_compute_per_region_metrics_returns_one_entry_per_region() -> None:
    preds = [
        _pred("Japan_000001.jpg", label="pothole", score=0.95),
        _pred("Czech_000001.jpg", label="pothole", score=0.9),
        _pred("United_States_000010.jpg", label="longitudinal crack", score=0.7),
    ]
    gts = [
        _gt("Japan_000001.jpg", label="pothole"),
        _gt("Czech_000001.jpg", label="pothole"),
        _gt("United_States_000010.jpg", label="longitudinal crack"),
    ]
    out = compute_per_region_metrics(
        predictions=preds,
        ground_truths=gts,
        class_names=CLASS_NAMES,
        confidence_threshold=0.25,
        iou_threshold=0.5,
    )
    # Sorted ordering must be deterministic.
    assert list(out.keys()) == ["Czech", "Japan", "United_States"]
    assert out["Japan"]["num_images"] == 1
    assert out["Czech"]["num_images"] == 1
    assert out["United_States"]["num_images"] == 1


def test_compute_per_region_metrics_skips_unmatched_images() -> None:
    preds = [
        _pred("Japan_000001.jpg"),
        _pred("garbage.jpg"),
    ]
    gts = [
        _gt("Japan_000001.jpg"),
        _gt("garbage.jpg"),
    ]
    out = compute_per_region_metrics(
        predictions=preds,
        ground_truths=gts,
        class_names=CLASS_NAMES,
        confidence_threshold=0.25,
        iou_threshold=0.5,
    )
    assert list(out.keys()) == ["Japan"]
