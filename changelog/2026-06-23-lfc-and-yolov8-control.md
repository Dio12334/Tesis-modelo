# 2026-06-23 — LFC compression + YOLOv8 control

Completes the YOLO-RD architecture (adds the missing LFC efficiency trick) and
adds a YOLOv8s control to test cross-family transfer.

## LFC (Layer-wise Feature Compression)

The paper's ~40% parameter reduction comes entirely from LFC (CSAF/LGECA are
near-free, ~10K params each). Implemented in **`model/models/lfc.py`**
(`build_lfc_model`):
- Reads the base architecture's YAML, halves every `1024` channel arg → `512`.
  This precisely targets the P5/stride-32 stage in **both** YOLOv8 and YOLO26
  (1024*width → halved; for the 's' scale 512 → 256, the paper's reduction).
- Builds via `YOLO(yaml).load(weights)` so `parse_model` re-derives all
  downstream channel bookkeeping; compatible pretrained weights transfer (the
  narrowed P5 stage starts fresh).
- Enabled by model-config `lfc: true` (wired in `yolo26_wrapper`).

**Verified param counts (5-class):**
- yolov8s baseline 9.84 M → **+LFC 6.73 M** (paper full YOLO-RD: 6.5 M ✓)
- yolo26s baseline 9.95 M → +LFC 5.46 M
- CSAF+LGECA add only ~10 K, so the size reduction is all LFC.

### Two bugs found via smoke-testing (both fixed)
1. **Head reshape skipped (loss = 0.0):** building from a YAML leaves model-level
   `nc = None`, so the wrapper's 80→`num_classes` head reshape (gated on an int
   `nc`) bailed → 80-class head vs 5-class targets → every batch errored and was
   skipped. Fix: `build_lfc_model` propagates the head's `nc` to model level.
2. **Per-epoch mAP eval crash (`cannot pickle _thread.lock`):** YAML-built models
   carry a thread lock, so deep-copying the whole Ultralytics `YOLO` wrapper in
   `ema_eval_model` failed. Fix: deepcopy only the inner `nn.Module`
   (`DetectionModel`) and swap it in (nulling the cached predictor). Verified it
   still fusion-isolates the live model on the non-LFC path. LFC configs keep
   `use_torch_compile: false` (deepcopy of a compiled module is unreliable).

### New configs
- `train_yolo26s_4country_full_lfc.yaml` — complete YOLO-RD on YOLO26.
- `train_yolov8s_4country_full_lfc.yaml` — complete YOLO-RD on YOLOv8 (matches
  the paper's full model + size).

## YOLOv8 control

The wrapper loads whatever architecture `pretrained_weights` encodes
(`ultralytics.YOLO("yolov8s.pt")` → YOLOv8). Verified empirically that YOLOv8s
has **no C2PSA** (`Conv`/`C2f`/`SPPF` only) while YOLO26-S has `C2PSA` + `C3k2` —
the structural basis for the "LGECA is redundant on YOLO26" hypothesis. CSAF/LGECA
surgery auto-adapts (YOLOv8 detect inputs are [15,18,21] vs YOLO26's [16,19,22]).
Configs: `train_yolov8s_4country_{bce,full,full_lfc}.yaml`. Full pipeline
smoke-tested end-to-end on YOLOv8 (SR-WBCE on `v8DetectionLoss`, per-epoch mAP
eval, EMA, checkpointing).

## Caveat

LFC narrows the P5 backbone, which starts from random init in this
pretrained-transfer regime — expect a **further accuracy drop**. LFC's purpose
here is the parameter/efficiency comparison with the paper, not accuracy.
