"""Exponential Moving Average (EMA) support for the detection training loop.

Wraps Ultralytics' ``ModelEMA`` so the custom training loop in
``train_detection.py`` can maintain a moving average of model weights and
evaluate / checkpoint *those* weights instead of the raw training weights.

EMA is one of the highest-ROI training tricks for detection (typically
+1-3 mAP) and was previously absent from the custom loop. See the thesis
diagnosis notes.

Design:
    - ``create_ema(model)`` builds a ``ModelEMA`` for the underlying
      ``nn.Module`` of an Ultralytics-style wrapper (YOLO26, RT-DETR).
      Returns ``None`` for wrappers that do not expose ``_model.model``
      (e.g. torchvision SSD) so callers can no-op cleanly.
    - ``ema_weights(model, ema)`` is a context manager that temporarily loads
      the EMA weights into the live module, yields, then restores the training
      weights. Used to wrap validation, mAP evaluation, and checkpoint saving
      so all three see the averaged weights without disturbing optimisation.

``ModelEMA`` internally calls ``unwrap_model`` on whatever it is given, so it
is robust to ``torch.compile`` (``_orig_mod``) and ``channels_last``: the EMA
state-dict keys always match the eager module's keys.
"""

import contextlib
import copy
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised indirectly in the thesis env
    from ultralytics.utils.torch_utils import ModelEMA, unwrap_model
except Exception:  # ultralytics missing or API drift
    ModelEMA = None  # type: ignore[assignment]
    unwrap_model = None  # type: ignore[assignment]

try:
    import torch.nn as _nn
except Exception:  # pragma: no cover
    _nn = None  # type: ignore[assignment]


def _underlying_nn_module(model):
    """Return the trainable ``nn.Module`` inside an Ultralytics-style wrapper.

    Returns ``None`` when the wrapper does not follow the ``_model.model``
    convention or the inner object is not a real ``nn.Module`` (e.g. a test
    ``MagicMock``), signalling that EMA is not supported for this model type.
    """
    inner = getattr(model, "_model", None)
    if inner is None or not hasattr(inner, "model"):
        return None
    candidate = inner.model
    if _nn is not None and not isinstance(candidate, _nn.Module):
        return None
    return candidate


def create_ema(model, decay: float = 0.9999, tau: int = 2000):
    """Build a ``ModelEMA`` for ``model`` or return ``None`` if unsupported.

    Args:
        model: A ``BaseDetector`` wrapper.
        decay: Maximum EMA decay rate (Ultralytics default 0.9999).
        tau: EMA decay ramp time constant (Ultralytics default 2000).

    Returns:
        A ``ModelEMA`` instance, or ``None`` when EMA cannot be applied
        (ultralytics unavailable or wrapper has no underlying ``nn.Module``).
    """
    if ModelEMA is None:
        logger.warning("ultralytics ModelEMA unavailable; training without EMA.")
        return None
    target = _underlying_nn_module(model)
    if target is None:
        logger.info(
            "Model %s has no usable nn.Module at _model.model; training without EMA.",
            type(model).__name__,
        )
        return None
    try:
        ema = ModelEMA(target, decay=decay, tau=tau)
    except Exception as e:  # defensive: never let EMA setup break training
        logger.warning("Failed to initialise ModelEMA (%s); training without EMA.", e)
        return None
    logger.info("ModelEMA enabled (decay=%.4f, tau=%d).", decay, tau)
    return ema


def update_ema(ema, model) -> None:
    """Update ``ema`` from the live ``model`` weights. No-op if ``ema`` is None."""
    if ema is None:
        return
    target = _underlying_nn_module(model)
    if target is not None:
        ema.update(target)


@contextlib.contextmanager
def ema_weights(model, ema):
    """Temporarily swap the live module's weights for the EMA weights.

    On exit the original training weights are always restored, even if the
    body raises. No-op (transparent) when ``ema`` is ``None`` or disabled.
    """
    target = _underlying_nn_module(model) if ema is not None else None
    if ema is None or not getattr(ema, "enabled", False) or target is None:
        yield
        return

    live = unwrap_model(target)
    backup = {k: v.detach().clone() for k, v in live.state_dict().items()}
    live.load_state_dict(ema.ema.state_dict(), strict=True)
    try:
        yield
    finally:
        live.load_state_dict(backup, strict=True)


@contextlib.contextmanager
def ema_eval_model(model, ema):
    """Point ``model._model`` at a deep copy holding the EMA weights for eval.

    Inference through Ultralytics ``predict()`` fuses Conv+BN, which mutates the
    module structure. Running that on the live training model both corrupts
    continued training and breaks the simple state-dict swap in
    :func:`ema_weights`. So for mAP evaluation we swap in a *deep copy* of the
    underlying Ultralytics model loaded with the EMA weights — fusion happens on
    the throwaway copy and the live training model is never touched.

    No-op (transparent) when ``ema`` is ``None``/disabled or the wrapper has no
    ``_model.model``.
    """
    inner = getattr(model, "_model", None)
    if (
        ema is None
        or not getattr(ema, "enabled", False)
        or inner is None
        or not hasattr(inner, "model")
        or unwrap_model is None
    ):
        yield
        return

    clone = copy.deepcopy(inner)
    unwrap_model(clone.model).load_state_dict(ema.ema.state_dict(), strict=True)
    clone.model.eval()
    model._model = clone
    try:
        yield
    finally:
        model._model = inner


def ema_state_dict(ema) -> Optional[dict]:
    """Return the EMA weights state-dict for resume persistence, or None."""
    if ema is None:
        return None
    return {k: v.detach().cpu().clone() for k, v in ema.ema.state_dict().items()}


def load_ema_state(ema, state: Optional[dict], updates: Optional[int]) -> None:
    """Restore EMA weights + update count from a resumed checkpoint."""
    if ema is None or not state:
        return
    ema.ema.load_state_dict(state, strict=False)
    if updates is not None:
        ema.updates = int(updates)
