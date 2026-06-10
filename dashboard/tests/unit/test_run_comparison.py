"""Unit tests for dashboard.components.run_comparison Phase-2 enhancements.

Tests cover the Tier-1 (table enrichment) and Tier-2 (sweep plots) additions
made for the freeze-schedule ablation campaign:

- mAP@0.5:0.95, Best F1, Best F1 conf, P@best, R@best, Default conf columns.
- 4-decimal precision for metric values.
- Radio-driven highlight metric (mAP@0.5 / mAP@0.5:0.95 / Best F1) backing
  ``_find_best_run_index``.
- F1-vs-confidence overlay figure built from ``f1_sweep``.
- Parametric Precision-vs-Recall overlay figure built from ``f1_sweep``.
- Per-class grouped-bar figure builder reused for ``per_class_best_f1`` and
  ``per_class_ap``; x-axis labels use ``display_class_names`` when present.
- Backwards-compat: when sweep keys are absent the corresponding cells fall
  back to ``"N/A"`` and figure builders skip the run silently.
"""

from typing import Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import plotly.graph_objects as go
import pytest

from dashboard.components.run_comparison import (
    _build_comparison_dataframe,
    _build_f1_vs_conf_figure,
    _build_per_class_grouped_bars,
    _build_pr_parametric_figure,
    _find_best_run_index,
    _get_class_display_names,
)
from dashboard.data_loader import EvaluationReport, ExperimentRun, FinalResults


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


CLASS_NAMES_EN = [
    "alligator crack",
    "longitudinal crack",
    "other corruption",
    "pothole",
    "transverse crack",
]

CLASS_NAMES_ES = [
    "piel_de_cocodrilo",
    "fisura_longitudinal",
    "otros",
    "bache",
    "fisura_transversal",
]


def _make_run(run_id: str, model_name: str = "yolo26s") -> ExperimentRun:
    """Build a minimal ExperimentRun usable by the comparison helpers."""
    return ExperimentRun(
        run_id=run_id,
        model_name=model_name,
        dataset_name="rdd2022",
        config={"name": "test"},
        start_time="2026-01-01T00:00:00.000000+00:00",
        end_time="2026-01-01T01:00:00.000000+00:00",
        metrics_history=[],
        final_results=FinalResults(
            final_train_loss=0.5,
            final_val_loss=0.4,
            best_val_loss=0.3,
            best_epoch=10,
            total_epochs=20,
        ),
    )


def _make_f1_sweep(
    confidences=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
    *,
    p_offset: float = 0.0,
    r_offset: float = 0.0,
) -> list[dict]:
    """Build a synthetic f1_sweep list (10 entries by default)."""
    sweep = []
    for c in confidences:
        # Synthetic but plausible: precision grows with conf, recall shrinks.
        precision = min(0.95, 0.30 + 1.2 * c + p_offset)
        recall = max(0.05, 0.85 - 1.0 * c + r_offset)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        sweep.append(
            {"confidence": c, "precision": precision, "recall": recall, "f1": f1}
        )
    return sweep


def _make_full_report(
    *,
    map_50: float = 0.5176,
    map_50_95: float = 0.2521,
    precision: float = 0.5967,
    recall: float = 0.4887,
    f1_score: float = 0.5373,
    default_conf: float = 0.25,
    best_f1: Optional[dict] = None,
    f1_sweep: Optional[list] = None,
    per_class_best_f1: Optional[dict] = None,
    per_class_ap: Optional[dict] = None,
    display_class_names: Optional[list[str]] = None,
) -> EvaluationReport:
    """Build an EvaluationReport with full Phase-2 sweep keys."""
    if best_f1 is None:
        best_f1 = {
            "confidence": 0.25,
            "precision": 0.5967,
            "recall": 0.4887,
            "f1": 0.5373,
        }
    if f1_sweep is None:
        f1_sweep = _make_f1_sweep()
    if per_class_best_f1 is None:
        per_class_best_f1 = {
            "alligator crack": {
                "confidence": 0.25, "precision": 0.61, "recall": 0.55, "f1": 0.58,
            },
            "longitudinal crack": {
                "confidence": 0.25, "precision": 0.61, "recall": 0.46, "f1": 0.53,
            },
            "other corruption": {
                "confidence": 0.40, "precision": 0.68, "recall": 0.59, "f1": 0.63,
            },
            "pothole": {
                "confidence": 0.20, "precision": 0.52, "recall": 0.44, "f1": 0.47,
            },
            "transverse crack": {
                "confidence": 0.25, "precision": 0.55, "recall": 0.46, "f1": 0.50,
            },
        }
    if per_class_ap is None:
        per_class_ap = {
            "alligator crack": 0.55,
            "longitudinal crack": 0.50,
            "other corruption": 0.60,
            "pothole": 0.43,
            "transverse crack": 0.48,
        }

    return EvaluationReport(
        checkpoint="checkpoints/yolo26/test.pt",
        dataset="rdd2022",
        num_val_images=7677,
        num_classes=5,
        class_names=list(CLASS_NAMES_EN),
        confidence_threshold=default_conf,
        iou_threshold=0.5,
        metrics={
            "mAP@0.5": map_50,
            "mAP@0.5:0.95": map_50_95,
            "map_50": map_50,
            "map_50_95": map_50_95,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "default_confidence_threshold": default_conf,
            "best_f1": best_f1,
            "f1_sweep": f1_sweep,
            "per_class_best_f1": per_class_best_f1,
            "per_class_ap": per_class_ap,
        },
        confusion_matrix=[[0] * 5 for _ in range(5)],
        display_class_names=display_class_names,
    )


def _make_minimal_report(map_50: float = 0.45) -> EvaluationReport:
    """Build a Phase-1-style report (no sweep keys, no display_class_names)."""
    return EvaluationReport(
        checkpoint="checkpoints/yolo26/legacy.pt",
        dataset="rdd2022",
        num_val_images=100,
        num_classes=5,
        class_names=list(CLASS_NAMES_EN),
        confidence_threshold=0.25,
        iou_threshold=0.5,
        metrics={
            "mAP@0.5": map_50,
            "precision": 0.50,
            "recall": 0.40,
            "f1_score": 0.44,
        },
        confusion_matrix=[[0] * 5 for _ in range(5)],
        display_class_names=None,
    )


# ---------------------------------------------------------------------------
# Tier 1: comparison-table enrichment
# ---------------------------------------------------------------------------


class TestComparisonDataFramePhase2Columns:
    def test_dataframe_includes_phase2_columns_when_sweep_present(self):
        runs = [_make_run("aaaaaaaa-1111-2222-3333-444444444444")]
        report = _make_full_report()
        evaluation_reports = {runs[0].run_id: report}

        df = _build_comparison_dataframe(runs, None, evaluation_reports)

        for col in (
            "mAP@0.5",
            "mAP@0.5:0.95",
            "Best F1",
            "Best F1 conf",
            "P @ best F1",
            "R @ best F1",
            "Default conf",
        ):
            assert col in df.columns, f"Missing column {col!r}; got {list(df.columns)}"

    def test_dataframe_omits_or_NA_phase2_columns_when_sweep_absent(self):
        runs = [_make_run("bbbbbbbb-1111-2222-3333-444444444444")]
        report = _make_minimal_report()
        evaluation_reports = {runs[0].run_id: report}

        df = _build_comparison_dataframe(runs, None, evaluation_reports)

        # mAP@0.5 must still be present and populated; sweep-derived columns
        # may be present (the column always renders) but must hold "N/A".
        assert df.loc[0, "mAP@0.5"] == "0.4500"
        for col in ("Best F1", "Best F1 conf", "P @ best F1", "R @ best F1"):
            assert col in df.columns, f"Missing column {col!r}"
            assert df.loc[0, col] == "N/A", (
                f"Expected {col} to be 'N/A' when f1_sweep absent; got {df.loc[0, col]!r}"
            )

    def test_dataframe_uses_4_decimal_precision_for_metrics(self):
        runs = [_make_run("cccccccc-1111-2222-3333-444444444444")]
        report = _make_full_report(
            map_50=0.51763355,
            map_50_95=0.25208672,
            best_f1={
                "confidence": 0.25,
                "precision": 0.59666503,
                "recall": 0.48871213,
                "f1": 0.53732002,
            },
        )
        evaluation_reports = {runs[0].run_id: report}

        df = _build_comparison_dataframe(runs, None, evaluation_reports)

        assert df.loc[0, "mAP@0.5"] == "0.5176"
        assert df.loc[0, "mAP@0.5:0.95"] == "0.2521"
        assert df.loc[0, "Best F1"] == "0.5373"
        assert df.loc[0, "P @ best F1"] == "0.5967"
        assert df.loc[0, "R @ best F1"] == "0.4887"
        assert df.loc[0, "Best F1 conf"] == "0.25"
        assert df.loc[0, "Default conf"] == "0.25"


# ---------------------------------------------------------------------------
# Highlight metric switching (radio-backed)
# ---------------------------------------------------------------------------


class TestFindBestRunIndex:
    def test_picks_highest_map_when_metric_is_map_05(self):
        runs = [_make_run(f"{i:08d}-1111-2222-3333-444444444444") for i in range(3)]
        reports = {
            runs[0].run_id: _make_full_report(map_50=0.40),
            runs[1].run_id: _make_full_report(map_50=0.55),  # winner
            runs[2].run_id: _make_full_report(map_50=0.50),
        }

        idx = _find_best_run_index(runs, None, reports, metric_key="mAP@0.5")

        assert idx == 1

    def test_picks_highest_map_50_95_when_selected(self):
        runs = [_make_run(f"{i:08d}-aaaa-bbbb-cccc-dddddddddddd") for i in range(3)]
        reports = {
            runs[0].run_id: _make_full_report(map_50=0.55, map_50_95=0.20),
            runs[1].run_id: _make_full_report(map_50=0.40, map_50_95=0.30),  # winner
            runs[2].run_id: _make_full_report(map_50=0.50, map_50_95=0.25),
        }

        idx = _find_best_run_index(runs, None, reports, metric_key="mAP@0.5:0.95")

        assert idx == 1

    def test_picks_highest_best_f1_when_selected(self):
        runs = [_make_run(f"{i:08d}-eeee-ffff-0000-111111111111") for i in range(3)]
        reports = {
            runs[0].run_id: _make_full_report(
                best_f1={"confidence": 0.25, "precision": 0.6, "recall": 0.4, "f1": 0.48}
            ),
            runs[1].run_id: _make_full_report(
                best_f1={"confidence": 0.30, "precision": 0.7, "recall": 0.5, "f1": 0.58}  # winner
            ),
            runs[2].run_id: _make_full_report(
                best_f1={"confidence": 0.20, "precision": 0.55, "recall": 0.45, "f1": 0.495}
            ),
        }

        idx = _find_best_run_index(runs, None, reports, metric_key="Best F1")

        assert idx == 1

    def test_returns_none_when_metric_absent_in_all_runs(self):
        runs = [_make_run(f"{i:08d}-2222-3333-4444-555555555555") for i in range(2)]
        reports = {
            runs[0].run_id: _make_minimal_report(),
            runs[1].run_id: _make_minimal_report(),
        }

        idx = _find_best_run_index(runs, None, reports, metric_key="Best F1")

        assert idx is None

    def test_default_metric_key_is_map_05_for_backwards_compat(self):
        """The legacy 3-arg call site (used in property tests) must still work."""
        runs = [_make_run(f"{i:08d}-3333-4444-5555-666666666666") for i in range(2)]
        reports = {
            runs[0].run_id: _make_full_report(map_50=0.30),
            runs[1].run_id: _make_full_report(map_50=0.60),
        }

        idx = _find_best_run_index(runs, None, reports)

        assert idx == 1


# ---------------------------------------------------------------------------
# Display-name helper
# ---------------------------------------------------------------------------


class TestGetClassDisplayNames:
    def test_prefers_display_class_names_when_present(self):
        report = _make_full_report(display_class_names=list(CLASS_NAMES_ES))

        names = _get_class_display_names(report)

        assert names == CLASS_NAMES_ES

    def test_falls_back_to_class_names_when_display_absent(self):
        report = _make_minimal_report()

        names = _get_class_display_names(report)

        assert names == CLASS_NAMES_EN

    def test_returns_empty_list_when_report_is_none(self):
        names = _get_class_display_names(None)

        assert names == []


# ---------------------------------------------------------------------------
# Tier 2 figure: F1-vs-confidence overlay
# ---------------------------------------------------------------------------


class TestF1VsConfFigure:
    def test_one_trace_per_run_with_sweep(self):
        runs = [
            _make_run("aaaaaaaa-1111-2222-3333-444444444444"),
            _make_run("bbbbbbbb-1111-2222-3333-444444444444"),
        ]
        reports = {
            runs[0].run_id: _make_full_report(),
            runs[1].run_id: _make_full_report(map_50=0.48),
        }

        fig = _build_f1_vs_conf_figure(runs, None, reports)

        # Allow extra "best F1" marker traces, but at least one line trace per run.
        line_traces = [t for t in fig.data if getattr(t, "mode", None) == "lines"]
        assert len(line_traces) == 2, (
            f"Expected one F1-vs-conf line trace per run; got {len(line_traces)}"
        )

    def test_skips_runs_without_sweep(self):
        runs = [
            _make_run("aaaaaaaa-1111-2222-3333-444444444444"),
            _make_run("bbbbbbbb-1111-2222-3333-444444444444"),
        ]
        reports = {
            runs[0].run_id: _make_full_report(),
            runs[1].run_id: _make_minimal_report(),  # no f1_sweep
        }

        fig = _build_f1_vs_conf_figure(runs, None, reports)

        line_traces = [t for t in fig.data if getattr(t, "mode", None) == "lines"]
        assert len(line_traces) == 1

    def test_x_is_confidence_y_is_f1(self):
        runs = [_make_run("aaaaaaaa-1111-2222-3333-444444444444")]
        sweep = _make_f1_sweep()
        report = _make_full_report(f1_sweep=sweep)
        reports = {runs[0].run_id: report}

        fig = _build_f1_vs_conf_figure(runs, None, reports)

        line_trace = next(t for t in fig.data if getattr(t, "mode", None) == "lines")
        expected_x = [pt["confidence"] for pt in sweep]
        expected_y = [pt["f1"] for pt in sweep]
        assert list(line_trace.x) == pytest.approx(expected_x)
        assert list(line_trace.y) == pytest.approx(expected_y)


# ---------------------------------------------------------------------------
# Tier 2 figure: parametric Precision-vs-Recall overlay
# ---------------------------------------------------------------------------


class TestPRParametricFigure:
    def test_x_is_recall_y_is_precision(self):
        runs = [_make_run("aaaaaaaa-1111-2222-3333-444444444444")]
        sweep = _make_f1_sweep()
        report = _make_full_report(f1_sweep=sweep)
        reports = {runs[0].run_id: report}

        fig = _build_pr_parametric_figure(runs, None, reports)

        # The parametric curve uses lines (with optional markers); accept both modes.
        line_trace = next(
            t
            for t in fig.data
            if isinstance(t, go.Scatter) and "lines" in (getattr(t, "mode", "") or "")
        )
        expected_x = [pt["recall"] for pt in sweep]
        expected_y = [pt["precision"] for pt in sweep]
        assert list(line_trace.x) == pytest.approx(expected_x)
        assert list(line_trace.y) == pytest.approx(expected_y)

    def test_skips_runs_without_sweep(self):
        runs = [
            _make_run("aaaaaaaa-1111-2222-3333-444444444444"),
            _make_run("bbbbbbbb-1111-2222-3333-444444444444"),
        ]
        reports = {
            runs[0].run_id: _make_full_report(),
            runs[1].run_id: _make_minimal_report(),
        }

        fig = _build_pr_parametric_figure(runs, None, reports)

        line_traces = [
            t
            for t in fig.data
            if isinstance(t, go.Scatter) and "lines" in (getattr(t, "mode", "") or "")
        ]
        assert len(line_traces) == 1


# ---------------------------------------------------------------------------
# Tier 2 figure: per-class grouped bars (best-F1 + AP)
# ---------------------------------------------------------------------------


class TestPerClassGroupedBars:
    def test_uses_spanish_display_names_when_available(self):
        runs = [_make_run("aaaaaaaa-1111-2222-3333-444444444444")]
        report = _make_full_report(display_class_names=list(CLASS_NAMES_ES))
        reports = {runs[0].run_id: report}

        fig = _build_per_class_grouped_bars(
            runs, None, reports, metric_key="per_class_best_f1"
        )

        bar = next(t for t in fig.data if isinstance(t, go.Bar))
        assert list(bar.x) == CLASS_NAMES_ES

    def test_falls_back_to_english_when_display_absent(self):
        runs = [_make_run("aaaaaaaa-1111-2222-3333-444444444444")]
        report = _make_full_report(display_class_names=None)
        reports = {runs[0].run_id: report}

        fig = _build_per_class_grouped_bars(
            runs, None, reports, metric_key="per_class_ap"
        )

        bar = next(t for t in fig.data if isinstance(t, go.Bar))
        assert list(bar.x) == CLASS_NAMES_EN

    def test_skips_runs_missing_metric(self):
        runs = [
            _make_run("aaaaaaaa-1111-2222-3333-444444444444"),
            _make_run("bbbbbbbb-1111-2222-3333-444444444444"),
        ]
        reports = {
            runs[0].run_id: _make_full_report(),
            runs[1].run_id: _make_minimal_report(),  # no per_class_best_f1
        }

        fig = _build_per_class_grouped_bars(
            runs, None, reports, metric_key="per_class_best_f1"
        )

        bars = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bars) == 1

    def test_per_class_best_f1_uses_f1_sub_key(self):
        runs = [_make_run("aaaaaaaa-1111-2222-3333-444444444444")]
        per_class = {
            "alligator crack": {"confidence": 0.25, "precision": 0.6, "recall": 0.55, "f1": 0.575},
            "longitudinal crack": {"confidence": 0.25, "precision": 0.6, "recall": 0.45, "f1": 0.515},
            "other corruption": {"confidence": 0.40, "precision": 0.68, "recall": 0.58, "f1": 0.626},
            "pothole": {"confidence": 0.20, "precision": 0.51, "recall": 0.44, "f1": 0.473},
            "transverse crack": {"confidence": 0.25, "precision": 0.55, "recall": 0.46, "f1": 0.501},
        }
        report = _make_full_report(per_class_best_f1=per_class)
        reports = {runs[0].run_id: report}

        fig = _build_per_class_grouped_bars(
            runs, None, reports, metric_key="per_class_best_f1"
        )

        bar = next(t for t in fig.data if isinstance(t, go.Bar))
        expected = [per_class[name]["f1"] for name in CLASS_NAMES_EN]
        assert list(bar.y) == pytest.approx(expected)

    def test_per_class_ap_uses_scalar_value(self):
        runs = [_make_run("aaaaaaaa-1111-2222-3333-444444444444")]
        per_class_ap = {
            "alligator crack": 0.55,
            "longitudinal crack": 0.50,
            "other corruption": 0.60,
            "pothole": 0.43,
            "transverse crack": 0.48,
        }
        report = _make_full_report(per_class_ap=per_class_ap)
        reports = {runs[0].run_id: report}

        fig = _build_per_class_grouped_bars(
            runs, None, reports, metric_key="per_class_ap"
        )

        bar = next(t for t in fig.data if isinstance(t, go.Bar))
        expected = [per_class_ap[name] for name in CLASS_NAMES_EN]
        assert list(bar.y) == pytest.approx(expected)
