"""Confusion matrix component for the Streamlit Results Dashboard.

Renders the confusion matrix as an annotated heatmap with class labels
on rows (ground truth) and columns (predicted), using a sequential color
scale where higher values have more intense color.

When raw per-image predictions (``*_inference.json``) are available, the
component exposes a confidence-threshold slider and recomputes the
matrix live on every slider change. The slider's default value is the
run's best-F1 threshold (from ``report.metrics["best_f1"]``) so the
landing view matches the report's headline operating point; users can
explore other thresholds without rerunning evaluation.

When raw predictions are not available (legacy runs, partial loads, or
when the ``model`` package can't be imported), the component falls back
to the precomputed ``report.confusion_matrix`` and shows a caption.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import plotly.figure_factory as ff
import streamlit as st

from data_loader import EvaluationReport
from region_metrics import ALL_REGIONS_LABEL, filter_predictions_by_region

try:
    from model.evaluation.metrics import compute_confusion_matrix
    _HAS_LIVE_RECOMPUTE = True
except Exception:  # pragma: no cover - import guard for stripped envs
    _HAS_LIVE_RECOMPUTE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _predictions_to_eval_inputs(
    data: Optional[dict],
) -> Tuple[List[dict], List[dict]]:
    """Flatten an inference JSON into the per-image lists expected by
    :func:`compute_confusion_matrix`.

    The inference JSON groups ground truth and predictions under each
    image entry; the metrics module expects two parallel lists keyed by
    ``image_id``. Unknown labels are not filtered here — the consumer
    already ignores labels outside ``class_names``.
    """
    images = (data or {}).get("images", [])
    predictions: List[dict] = []
    ground_truths: List[dict] = []
    for img in images:
        image_id = img.get("image_id", "")
        gt = img.get("ground_truth", {}) or {}
        pred = img.get("predictions", {}) or {}
        predictions.append({
            "image_id": image_id,
            "boxes": pred.get("boxes", []) or [],
            "labels": pred.get("labels", []) or [],
            "scores": pred.get("scores", []) or [],
        })
        ground_truths.append({
            "image_id": image_id,
            "boxes": gt.get("boxes", []) or [],
            "labels": gt.get("labels", []) or [],
        })
    return predictions, ground_truths


def _resolve_default_confidence(report: EvaluationReport) -> float:
    """Resolve the slider's default confidence threshold.

    Priority: best-F1 confidence > evaluator's default_confidence_threshold
    > hardcoded 0.25. The best-F1 path is preferred because it lands the
    user on the operating point the report headlines.
    """
    best = report.metrics.get("best_f1")
    if isinstance(best, dict) and best.get("confidence") is not None:
        try:
            return float(best["confidence"])
        except (TypeError, ValueError):
            pass
    default_conf = report.metrics.get("default_confidence_threshold")
    if default_conf is not None:
        try:
            return float(default_conf)
        except (TypeError, ValueError):
            pass
    return 0.25


def _resolve_iou(report: EvaluationReport, override: Optional[float]) -> float:
    """Resolve the IoU threshold for live recomputation."""
    if override is not None:
        try:
            return float(override)
        except (TypeError, ValueError):
            pass
    iou = getattr(report, "iou_threshold", None)
    if iou is not None:
        try:
            return float(iou)
        except (TypeError, ValueError):
            pass
    return 0.5


def _display_names_for_matrix(
    matrix, class_names: Sequence[str]
) -> List[str]:
    """Return axis labels matching the matrix shape.

    When the matrix is ``(C+1, C+1)``, the trailing row/column captures
    background (false positives / missed detections).
    """
    if len(matrix) == len(class_names) + 1:
        return list(class_names) + ["background"]
    return list(class_names)


def _render_heatmap(
    matrix, display_names: Sequence[str], display_mode: str
) -> None:
    """Render the annotated heatmap. Pure presentation, no state."""
    matrix_np = np.array(matrix, dtype=float)
    if matrix_np.size == 0:
        st.info("Confusion matrix is empty.")
        return

    if np.all(matrix_np == 0):
        st.info(
            "Insufficient detections for confusion analysis at this "
            "confidence threshold. Try lowering the slider."
        )
        return

    if display_mode == "Percentage (row-wise)":
        row_sums = matrix_np.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        normalized = (matrix_np / row_sums) * 100
        z_values = normalized.tolist()
        annotation_text = [[f"{v:.1f}%" for v in row] for row in normalized]
        title = "Confusion Matrix (Percentage)"
    else:
        z_values = matrix_np.astype(int).tolist()
        annotation_text = [[str(int(v)) for v in row] for row in matrix_np]
        title = "Confusion Matrix"

    fig = ff.create_annotated_heatmap(
        z=z_values,
        x=list(display_names),
        y=list(display_names),
        annotation_text=annotation_text,
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Class",
        yaxis_title="Ground Truth Class",
        xaxis=dict(side="bottom"),
        height=500,
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def render_confusion_matrix(
    report: Optional[EvaluationReport],
    *,
    predictions_data: Optional[dict] = None,
    region: str = ALL_REGIONS_LABEL,
    iou_threshold: Optional[float] = None,
    key_prefix: str = "",
) -> None:
    """Render the confusion matrix with an interactive confidence slider.

    When ``predictions_data`` is provided, the matrix is recomputed live
    on each slider change from the raw per-image predictions filtered by
    ``region``. The slider defaults to the run's best-F1 confidence so
    the landing view matches the report headline; the IoU threshold is
    kept locked at the report value because the evaluator's
    ``confusion_matrix`` is anchored to that IoU and the F1 sweep does
    not vary IoU.

    Args:
        report: The evaluation report. Used to resolve ``class_names``,
            the default slider value, the IoU threshold, and as the
            fallback matrix source when ``predictions_data`` is absent.
        predictions_data: Raw inference JSON (loaded
            ``*_inference.json``) for the relevant split. ``None``
            disables live recomputation.
        region: Active region label (``ALL_REGIONS_LABEL`` for all).
            Predictions are filtered to this region before recomputation.
        iou_threshold: Optional IoU override; defaults to the report's
            stored ``iou_threshold`` (or 0.5).
        key_prefix: Namespace for stateful widgets (slider, radio). Pass
            a unique value per render site to avoid Streamlit's
            ``DuplicateElementId`` error when the component appears in
            multiple region/split tabs.
    """
    st.header("Confusion Matrix")

    if report is None:
        st.info(
            "No evaluation report available. "
            "Run model evaluation to see the confusion matrix here."
        )
        return

    class_names = list(report.class_names) if report.class_names else []
    iou = _resolve_iou(report, iou_threshold)
    default_conf = _resolve_default_confidence(report)

    can_recompute = (
        _HAS_LIVE_RECOMPUTE
        and predictions_data is not None
        and bool(class_names)
    )

    # --- Confidence slider + matrix source -------------------------------
    if can_recompute:
        slider_key = f"{key_prefix}_cm_conf" if key_prefix else None
        confidence = st.slider(
            "Confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(default_conf),
            step=0.05,
            key=slider_key,
            help=(
                "Confidence cutoff used to recompute the confusion matrix "
                "from raw predictions. The default is the run's best-F1 "
                f"threshold ({default_conf:.2f}); IoU is held at {iou:.2f}."
            ),
        )
        filtered = filter_predictions_by_region(predictions_data, region)
        predictions, ground_truths = _predictions_to_eval_inputs(filtered)
        matrix = compute_confusion_matrix(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=class_names,
            iou_threshold=iou,
            confidence_threshold=confidence,
            include_background=True,
        )
        st.caption(
            f"Recomputed live - {len(predictions)} image(s) - "
            f"conf={confidence:.2f} - IoU={iou:.2f}"
        )
    else:
        matrix = report.confusion_matrix
        if not _HAS_LIVE_RECOMPUTE:
            st.caption(
                "Live recomputation unavailable (model package not "
                "importable). Showing the precomputed matrix from the "
                f"evaluation report at conf={default_conf:.2f}."
            )
        elif predictions_data is None:
            st.caption(
                "Raw predictions not available for this split or run - "
                "showing the precomputed matrix from the evaluation "
                f"report at conf={default_conf:.2f}."
            )
        elif not class_names:
            st.caption(
                "Class names missing from the report - showing the "
                "precomputed matrix without live recomputation."
            )

    # --- Display-mode toggle and heatmap ---------------------------------
    radio_key = f"{key_prefix}_cm_display_mode" if key_prefix else None
    display_mode = st.radio(
        "Display mode",
        options=["Nominal", "Percentage (row-wise)"],
        horizontal=True,
        help=(
            "Nominal shows absolute counts. Percentage shows row-wise "
            "normalization (% of each ground truth class)."
        ),
        key=radio_key,
    )

    display_names = _display_names_for_matrix(matrix, class_names)
    _render_heatmap(matrix, display_names, display_mode)
