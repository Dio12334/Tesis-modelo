"""A/B robusto: ¿`torch.compile` (con triton) acelera RT-DETR en condiciones reales?

Entrenamiento compute-bound. Aquí comparamos eager vs compile con **shapes
variables** (nº de cajas aleatorio por imagen/paso), que es lo que estresa la
recompilación de torch.compile / CUDA-graph igual que el dataset real. Un conteo
fijo de cajas nunca re-traza y sobreestima el beneficio de compile.

Controles de ruido (GPU de laptop con throttling térmico):
  - se regenera el batch en cada paso,
  - se reporta el **mínimo** (mejor caso sin throttle) y la **mediana**,
  - eager se mide DOS veces (antes y después de compile) para acotar la deriva.

Uso:
    python -m model.tools.profile_compute_variants --config model/configs/train_rt_detr.yaml
"""

import argparse
import statistics
import time
from pathlib import Path

import torch

from model.config.manager import ConfigManager
from model.models import ModelRegistry


def build_model(mtype, model_cfg, device):
    model = ModelRegistry.create(mtype, model_cfg)
    if hasattr(model, "_model") and hasattr(model._model, "model"):
        model._model.model.to(device)
    elif hasattr(model, "_model"):
        model._model.to(device)
    elif hasattr(model, "model"):
        model.model.to(device)
    model.set_train_mode()
    return model


def underlying(model):
    if hasattr(model, "_model") and hasattr(model._model, "model"):
        return model._model.model
    if hasattr(model, "_model"):
        return model._model
    return getattr(model, "model", model)


def make_batch(batch_size, input_size, num_classes, device, vary_boxes=True):
    import random as _r
    images = [torch.rand(3, input_size, input_size, device=device) for _ in range(batch_size)]
    targets = []
    for _ in range(batch_size):
        n = _r.randint(1, 12) if vary_boxes else 5
        xy = torch.rand(n, 2, device=device) * (input_size * 0.6)
        wh = torch.rand(n, 2, device=device) * (input_size * 0.3) + 10
        boxes = torch.cat([xy, xy + wh], dim=1).clamp(0, input_size)
        labels = torch.randint(0, num_classes, (n,), device=device)
        targets.append({"boxes": boxes, "labels": labels})
    return images, targets


def run_variant(name, model, batch_size, input_size, num_classes, device, use_amp,
                n=40, warmup=10, vary_boxes=True):
    optimizer = torch.optim.AdamW(model.get_parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    def step():
        images, targets = make_batch(batch_size, input_size, num_classes, device, vary_boxes)
        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast('cuda'):
                loss = model.train_step(images, targets)["loss_tensor"]
        else:
            loss = model.train_step(images, targets)["loss_tensor"]
        if loss.item() != 0.0:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    times = []
    for i in range(n + warmup):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        try:
            step()
        except Exception as e:
            print(f"  [{name}] error en step {i}: {type(e).__name__}: {e}")
            return None
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.time() - t0)
    times.sort()
    return {"min": times[0], "median": statistics.median(times), "p90": times[int(len(times) * 0.9)]}


def _recompiles():
    try:
        from torch._dynamo.utils import counters
        return dict(counters.get("stats", {}))
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("-n", type=int, default=40)
    args = ap.parse_args()

    cm = ConfigManager()
    config = cm.resolve_env_vars(cm.load(Path(args.config)))
    mcfg = config.get("model", {}).get("config", {})
    mtype = config.get("model", {}).get("type")
    tcfg = config.get("training", {})
    input_size = mcfg.get("input_size", 640)
    num_classes = mcfg.get("num_classes", 5)
    batch_size = tcfg.get("batch_size", 8)
    use_amp = tcfg.get("use_amp", True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = dict(mcfg); model_cfg["num_classes"] = num_classes
    if tcfg.get("loss"):
        model_cfg["loss"] = tcfg["loss"]

    has_triton = False
    try:
        import triton  # noqa: F401
        has_triton = True
    except ImportError:
        pass

    print(f"[env] {mtype} input={input_size} batch={batch_size} amp={use_amp} "
          f"torch={torch.__version__} triton={'si' if has_triton else 'NO'} | shapes VARIABLES")

    R = {}

    # eager (1ª medición)
    m = build_model(mtype, model_cfg, device)
    R["eager_1"] = run_variant("eager_1", m, batch_size, input_size, num_classes, device, use_amp, n=args.n)
    del m; torch.cuda.empty_cache()

    # compile (warmup largo para absorber compilación + recompilaciones por shape)
    try:
        from torch._dynamo.utils import counters
        counters.clear()
    except Exception:
        pass
    m = build_model(mtype, model_cfg, device)
    try:
        if hasattr(m, "_model") and hasattr(m._model, "model"):
            m._model.model = torch.compile(underlying(m), mode="reduce-overhead")
        R["compile"] = run_variant("compile", m, batch_size, input_size, num_classes, device, use_amp,
                                   n=args.n, warmup=25)
    except Exception as e:
        print(f"  compile error: {e}")
    recompile_stats = _recompiles()
    del m; torch.cuda.empty_cache()

    # eager (2ª medición, para acotar deriva térmica)
    m = build_model(mtype, model_cfg, device)
    R["eager_2"] = run_variant("eager_2", m, batch_size, input_size, num_classes, device, use_amp, n=args.n)
    del m; torch.cuda.empty_cache()

    print("\n================ COMPUTO POR BATCH (ms) ================")
    print(f"  {'variante':14s} {'min':>8s} {'mediana':>9s} {'p90':>8s}")
    for name in ["eager_1", "eager_2", "compile"]:
        v = R.get(name)
        if v is None:
            print(f"  {name:14s}   (no medido)")
            continue
        print(f"  {name:14s} {v['min']*1000:8.1f} {v['median']*1000:9.1f} {v['p90']*1000:8.1f}")
    print("========================================================")

    if R.get("eager_1") and R.get("eager_2") and R.get("compile"):
        eager_min = min(R["eager_1"]["min"], R["eager_2"]["min"])
        eager_med = statistics.median([R["eager_1"]["median"], R["eager_2"]["median"]])
        comp_min, comp_med = R["compile"]["min"], R["compile"]["median"]
        drift = abs(R["eager_1"]["min"] - R["eager_2"]["min"]) / eager_min * 100
        print(f"  deriva eager (ruido)      : {drift:.1f}%  (min1={R['eager_1']['min']*1000:.1f} "
              f"min2={R['eager_2']['min']*1000:.1f})")
        print(f"  speedup compile (por min) : {eager_min/comp_min:.2f}x")
        print(f"  speedup compile (mediana) : {eager_med/comp_med:.2f}x")
        if recompile_stats:
            print(f"  dynamo stats              : {recompile_stats}")
        verdict = ("compile AYUDA" if eager_min / comp_min > 1.1
                   else "compile NO ayuda claramente (dentro del ruido)")
        print(f"  veredicto                 : {verdict}")


if __name__ == "__main__":
    main()
