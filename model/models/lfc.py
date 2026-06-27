"""Layer-wise Feature Compression (LFC) for YOLO-RD.

Implements the YOLO-RD paper's LFC strategy (Wang et al., Neurocomputing 662
(2026), Sec. 3.1): halve the channel dimension of the stride-32 (P5) feature
maps — the deepest, widest stage — to cut parameters/compute while preserving
representational capacity. The paper applies it to "layers 7-9" of YOLOv8s
(512 -> 256 channels); this is the source of YOLO-RD's ~40% parameter reduction
(CSAF/LGECA themselves are near-free).

Implementation: rather than in-place surgery (channel changes ripple through
every downstream layer), we take the base architecture's YAML, halve every
``1024`` channel argument (the P5/stride-32 stage in both YOLOv8 and YOLO26 —
for the small scale 1024*0.5 = 512 -> 256, exactly the paper's reduction), and
let Ultralytics' ``parse_model`` re-derive all downstream channel bookkeeping.
Compatible pretrained weights are then transferred (the unchanged backbone/neck
layers load; the narrowed P5 stage starts fresh).
"""

import copy
import logging
import os
import tempfile
from pathlib import Path

import yaml

try:
    import ultralytics
except ImportError:  # pragma: no cover
    ultralytics = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# The P5/stride-32 channel arg (base, pre-width-scaling) that LFC halves.
_P5_CHANNELS = 1024
_LFC_CHANNELS = 512


def build_lfc_model(base_weights: str):
    """Build a YOLO model with LFC applied and pretrained weights transferred.

    Args:
        base_weights: Path/name of the base checkpoint (e.g. ``"yolo26s.pt"`` or
            ``"yolov8s.pt"``). Its filename encodes the scale, which Ultralytics
            uses to size the model.

    Returns:
        An ``ultralytics.YOLO`` whose P5 stage is half-width, with compatible
        pretrained weights loaded.
    """
    if ultralytics is None:
        raise ImportError("ultralytics is required for LFC.")

    base = ultralytics.YOLO(base_weights)
    cfg = copy.deepcopy(base.model.yaml)

    halved = 0
    for section in ("backbone", "head"):
        for layer in cfg.get(section, []):
            args = layer[3]  # [from, repeats, module, args]
            for i, a in enumerate(args):
                if a == _P5_CHANNELS:
                    args[i] = _LFC_CHANNELS
                    halved += 1

    # Write to a filename that encodes the scale (e.g. "yolo26s_lfc.yaml") so
    # Ultralytics resolves the correct compound-scaling from the stem; building
    # from a dict alone loses the scale and defaults to 'l'.
    stem = Path(str(base_weights)).stem  # e.g. "yolo26s"
    tmp_dir = tempfile.mkdtemp(prefix="lfc_")
    yaml_path = os.path.join(tmp_dir, f"{stem}_lfc.yaml")
    with open(yaml_path, "w") as fh:
        yaml.safe_dump(cfg, fh)

    model = ultralytics.YOLO(yaml_path)
    # Transfer compatible pretrained weights (unchanged layers load; the
    # narrowed P5 stage and shape-changed consumers start fresh).
    model.load(base_weights)

    # Building from a YAML leaves model-level ``nc`` as None (unlike loading a
    # .pt). Propagate the head's class count so the wrapper's head-reshape
    # (which gates on an int ``nc``) fires and resizes 80 -> num_classes.
    if getattr(model.model, "nc", None) is None:
        model.model.nc = model.model.model[-1].nc
    logger.info(
        "LFC applied: halved %d P5/stride-32 channel args (1024->512 base) "
        "from %s; pretrained weights transferred.",
        halved, base_weights,
    )
    return model
