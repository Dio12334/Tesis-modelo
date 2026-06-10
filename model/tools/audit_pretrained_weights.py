"""Audit pretrained-weight loading for the YOLO26 wrapper.

Verifies that the standard training entry point actually loads COCO-pretrained
weights into the backbone, that the detection head is correctly reshaped to
``num_classes`` (random-init for the classification convs is expected), and
that ``freeze_layers`` freezes the parameters it claims to freeze.

Run from repo root:

    python -m model.tools.audit_pretrained_weights

Outputs a structured report to stdout. Exits non-zero if any verification fails.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_PT = REPO_ROOT / "yolo26s.pt"
CONFIG_PATH = REPO_ROOT / "model" / "configs" / "train_yolo26s.yaml"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("audit")


def _flatten_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip a leading ``model.`` prefix if present so two state_dicts can be
    compared key-for-key."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        nk = k
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        out[nk] = v
    return out


def load_canonical_backbone(pt_path: Path) -> Dict[str, torch.Tensor]:
    """Load ``yolo26s.pt`` and return its raw backbone+neck tensors keyed by
    their position in the inner ``DetectionModel.model`` Sequential."""
    log.info("Loading canonical pretrained checkpoint: %s", pt_path)
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    # Ultralytics packs the actual nn.Module under "model" or "ema"
    inner = ckpt.get("model") or ckpt.get("ema")
    if inner is None:
        raise RuntimeError(f"Could not find 'model' or 'ema' in checkpoint {pt_path}")
    if hasattr(inner, "float"):
        inner = inner.float()
    state = inner.state_dict()
    log.info("  Canonical state_dict has %d tensors", len(state))
    return _flatten_state_dict(state)


def build_via_wrapper() -> Tuple[Dict[str, torch.Tensor], object]:
    """Build a ``YOLO26Detector`` via the exact path used by training and
    return its inner model state_dict plus the detector instance itself."""
    log.info("Building model via training entry path (ModelRegistry.create)")
    sys.path.insert(0, str(REPO_ROOT))
    from model.models.registry import ModelRegistry  # noqa: E402
    import model.models.yolo26_wrapper  # noqa: F401  (registers "yolo26")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    model_cfg = dict(full_cfg["model"]["config"])
    model_cfg["num_classes"] = 5  # same as train_detection.py:616

    detector = ModelRegistry.create("yolo26", model_cfg)
    inner = detector._model.model  # ultralytics DetectionModel
    state = _flatten_state_dict(inner.state_dict())
    log.info("  Wrapper-built state_dict has %d tensors", len(state))
    return state, detector


def categorize_keys(keys) -> Dict[str, List[str]]:
    """Group state_dict keys by top-level module index (Ultralytics layer idx)."""
    buckets: Dict[str, List[str]] = {}
    for k in keys:
        # Keys look like "model.0.conv.weight" or "0.conv.weight" after flatten.
        # After _flatten_state_dict (strip leading "model.") they are like
        # "0.conv.weight" or potentially deeper. Take the first segment.
        head = k.split(".", 1)[0]
        buckets.setdefault(head, []).append(k)
    return buckets


def compare_tensors(
    canonical: Dict[str, torch.Tensor],
    wrapper: Dict[str, torch.Tensor],
) -> Dict[str, Dict]:
    """For each top-level layer index, report whether tensors are identical,
    differ, or are missing on either side."""
    canon_buckets = categorize_keys(canonical.keys())
    wrap_buckets = categorize_keys(wrapper.keys())
    all_buckets = sorted(
        set(canon_buckets) | set(wrap_buckets),
        key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x),
    )

    results: Dict[str, Dict] = {}
    for idx in all_buckets:
        canon_keys = set(canon_buckets.get(idx, []))
        wrap_keys = set(wrap_buckets.get(idx, []))
        common = canon_keys & wrap_keys
        only_canon = canon_keys - wrap_keys
        only_wrap = wrap_keys - canon_keys

        identical = 0
        differ = 0
        max_abs_diff = 0.0
        sample_diff_key = None
        for k in common:
            ca, wa = canonical[k], wrapper[k]
            if ca.shape != wa.shape:
                differ += 1
                max_abs_diff = float("inf")
                sample_diff_key = sample_diff_key or k
                continue
            if torch.equal(ca, wa):
                identical += 1
            else:
                differ += 1
                d = (ca - wa).abs().max().item()
                if d > max_abs_diff:
                    max_abs_diff = d
                    sample_diff_key = k

        results[idx] = {
            "common": len(common),
            "identical": identical,
            "differ": differ,
            "only_canon": sorted(only_canon),
            "only_wrap": sorted(only_wrap),
            "max_abs_diff": max_abs_diff,
            "sample_diff_key": sample_diff_key,
        }
    return results


def audit_freeze(detector) -> Dict[str, int]:
    """Tally trainable vs frozen parameters in the inner model."""
    frozen = trainable = 0
    frozen_top: Dict[str, int] = {}
    trainable_top: Dict[str, int] = {}
    for name, p in detector._model.model.named_parameters():
        # Strip leading "model." if present to get the layer-index segment
        nk = name
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        top = nk.split(".", 1)[0]
        if p.requires_grad:
            trainable += p.numel()
            trainable_top[top] = trainable_top.get(top, 0) + p.numel()
        else:
            frozen += p.numel()
            frozen_top[top] = frozen_top.get(top, 0) + p.numel()
    return {
        "frozen_params": frozen,
        "trainable_params": trainable,
        "frozen_by_layer": frozen_top,
        "trainable_by_layer": trainable_top,
    }


def main() -> int:
    print("=" * 78)
    print("YOLO26 PRETRAINED-WEIGHT AUDIT")
    print("=" * 78)
    print(f"Repo root      : {REPO_ROOT}")
    print(f"Pretrained .pt : {PRETRAINED_PT}  ({PRETRAINED_PT.stat().st_size / 1e6:.2f} MB)")
    print(f"Config         : {CONFIG_PATH}")
    print()

    canonical = load_canonical_backbone(PRETRAINED_PT)
    wrapper_state, detector = build_via_wrapper()
    print()

    # Quick sanity: head should differ (reshaped 80 -> 5)
    # Backbone (layers 0..9) and neck (10..N-2) should be identical.
    results = compare_tensors(canonical, wrapper_state)

    print("-" * 78)
    print("Per-layer comparison (canonical .pt vs wrapper-built model)")
    print("-" * 78)
    print(f"{'Layer':>6} | {'common':>6} | {'identical':>9} | {'differ':>6} | "
          f"{'only_canon':>10} | {'only_wrap':>10} | max|diff|")
    print("-" * 78)
    total_identical = 0
    total_differ = 0
    total_common = 0
    for idx, r in results.items():
        print(
            f"{idx:>6} | {r['common']:>6} | {r['identical']:>9} | {r['differ']:>6} | "
            f"{len(r['only_canon']):>10} | {len(r['only_wrap']):>10} | "
            f"{r['max_abs_diff']:.4g}"
        )
        total_identical += r["identical"]
        total_differ += r["differ"]
        total_common += r["common"]
    print("-" * 78)
    print(f"TOTAL  | {total_common:>6} | {total_identical:>9} | {total_differ:>6}")
    print()

    # Determine which layers are head vs body
    layer_indices = [int(k) for k in results.keys() if k.isdigit()]
    if layer_indices:
        last_idx = max(layer_indices)
    else:
        last_idx = -1

    body_layers = [k for k in results.keys() if k.isdigit() and int(k) < last_idx]
    head_layer = str(last_idx) if last_idx >= 0 else None

    body_identical_pct = 0.0
    body_common = 0
    body_identical = 0
    for k in body_layers:
        body_common += results[k]["common"]
        body_identical += results[k]["identical"]
    if body_common:
        body_identical_pct = 100.0 * body_identical / body_common

    print("-" * 78)
    print("Diagnosis")
    print("-" * 78)
    print(f"Backbone+neck (layers 0..{last_idx - 1 if last_idx > 0 else '?'}): "
          f"{body_identical}/{body_common} tensors identical "
          f"({body_identical_pct:.1f}%)")
    if head_layer is not None:
        h = results[head_layer]
        print(f"Detect head   (layer {head_layer}): {h['identical']}/{h['common']} "
              f"identical, {h['differ']} differ "
              f"(differ is EXPECTED — head is reshaped 80->5)")

    # Verdict on backbone
    if body_common == 0:
        verdict_backbone = "INCONCLUSIVE (no body tensors compared)"
        ok_backbone = False
    elif body_identical_pct >= 99.5:
        verdict_backbone = "PASS — pretrained backbone weights ARE loaded"
        ok_backbone = True
    elif body_identical_pct >= 50.0:
        verdict_backbone = "PARTIAL — some body tensors match, some do not (suspicious)"
        ok_backbone = False
    else:
        verdict_backbone = "FAIL — backbone appears to be RANDOM-INITIALISED"
        ok_backbone = False
    print()
    print(f"VERDICT (backbone): {verdict_backbone}")

    # Verdict on head
    if head_layer is not None:
        h = results[head_layer]
        # cv3 / one2one_cv3 must differ; cv2 (box reg) should be identical
        head_diff_keys = [
            k for k in canonical.keys()
            if k.startswith(head_layer + ".")
            and k in wrapper_state
            and not torch.equal(canonical[k], wrapper_state[k])
            if canonical[k].shape == wrapper_state[k].shape
        ]
        head_differ_total = h["differ"]
        # Count keys that should be reshaped (cv3, one2one_cv3 last conv)
        cv3_keys = [k for k in canonical if k.startswith(head_layer + ".cv3")]
        cv3_present_in_wrapper = [k for k in cv3_keys if k in wrapper_state]
        # After head reshape, the wrapper's cv3 keys may not even match canonical
        # keys verbatim because shapes change. We expect SOME mismatch / shape diff.
        if h["differ"] > 0 or h["only_canon"] or h["only_wrap"]:
            verdict_head = (
                "PASS — head differs from canonical (expected: reshaped to 5 classes)"
            )
            ok_head = True
        else:
            verdict_head = (
                "FAIL — head matches canonical exactly (it should be reshaped to 5 classes)"
            )
            ok_head = False
        print(f"VERDICT (head):     {verdict_head}")
    else:
        ok_head = False
        print("VERDICT (head):     SKIPPED — could not identify head layer")

    # Freeze audit
    print()
    print("-" * 78)
    print("Freeze audit (config: freeze_layers=5)")
    print("-" * 78)
    fz = audit_freeze(detector)
    total = fz["frozen_params"] + fz["trainable_params"]
    pct_frozen = 100.0 * fz["frozen_params"] / total if total else 0.0
    print(f"  Frozen params    : {fz['frozen_params']:>12,} ({pct_frozen:5.2f}%)")
    print(f"  Trainable params : {fz['trainable_params']:>12,} ({100 - pct_frozen:5.2f}%)")
    print(f"  Total params     : {total:>12,}")
    print()
    print("  Frozen layers (layer_idx -> param count):")
    for k in sorted(fz["frozen_by_layer"], key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)):
        print(f"    layer {k:>3} : {fz['frozen_by_layer'][k]:>10,}")
    print("  Trainable layers (layer_idx -> param count):")
    for k in sorted(fz["trainable_by_layer"], key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)):
        print(f"    layer {k:>3} : {fz['trainable_by_layer'][k]:>10,}")

    frozen_layer_idxs = sorted(
        [int(k) for k in fz["frozen_by_layer"] if k.isdigit()]
    )
    if frozen_layer_idxs == list(range(5)):
        print()
        print("VERDICT (freeze):   PASS — exactly layers 0..4 are frozen")
        ok_freeze = True
    else:
        print()
        print(f"VERDICT (freeze):   QUESTIONABLE — frozen layer indices = {frozen_layer_idxs}")
        ok_freeze = False

    # Tiny forward sanity
    print()
    print("-" * 78)
    print("Forward-pass sanity check")
    print("-" * 78)
    detector._model.model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 640, 640)
        try:
            y = detector._model.model(x)
            shapes = []
            def collect(o):
                if isinstance(o, torch.Tensor):
                    shapes.append(tuple(o.shape))
                elif isinstance(o, (list, tuple)):
                    for s in o:
                        collect(s)
            collect(y)
            print(f"  Forward pass succeeded. Output tensor shapes (first 6): "
                  f"{shapes[:6]}")
            print(f"  Total output tensors: {len(shapes)}")
            ok_forward = True
        except Exception as e:
            print(f"  FAIL — forward pass raised: {type(e).__name__}: {e}")
            ok_forward = False

    print()
    print("=" * 78)
    print("OVERALL")
    print("=" * 78)
    overall_ok = ok_backbone and ok_head and ok_freeze and ok_forward
    print(f"  backbone : {'PASS' if ok_backbone else 'FAIL'}")
    print(f"  head     : {'PASS' if ok_head else 'FAIL'}")
    print(f"  freeze   : {'PASS' if ok_freeze else 'FAIL/CHECK'}")
    print(f"  forward  : {'PASS' if ok_forward else 'FAIL'}")
    print()
    print(f"  RESULT   : {'PASS' if overall_ok else 'FAIL — investigate'}")
    print("=" * 78)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
