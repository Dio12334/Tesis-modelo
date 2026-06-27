"""Micro-benchmark: ¿el entrenamiento está limitado por DATOS o por CÓMPUTO?

Reconstruye el camino caliente de ``train_detection.train`` (mismo dataset, mismo
DataLoader, mismo ``model.train_step`` + backward) para ~N batches y mide, por batch:

  - t_data    : tiempo esperando el siguiente batch del DataLoader (data-starvation)
  - t_step    : H2D + train_step (forward+loss) + backward + optimizer.step (cómputo real)

Si ``t_data`` ≈ 0 y ``t_step`` domina  -> limitado por CÓMPUTO (GPU).
Si ``t_data`` ~ ``t_step`` o mayor      -> limitado por DATOS / IO.

NO guarda checkpoints ni corre épocas completas. No usa torch.compile (para medir el
cómputo base sin la latencia de warmup); el cómputo real con compile será algo menor.

Uso:
    python -m model.tools.profile_training_bottleneck --config model/configs/train_rt_detr.yaml
    python -m model.tools.profile_training_bottleneck --config model/configs/train_yolo26.yaml -n 40
"""

import argparse
import time
from pathlib import Path

import torch

from model.config.manager import ConfigManager
from model.training.augmentation import build_augmentation_pipeline
from model.training.train_detection import RDD2022TorchDataset, collate_fn
from model.datasets.rdd2022 import RDD2022Dataset
from model.models import ModelRegistry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("-n", "--num-batches", type=int, default=30,
                    help="Batches medidos tras el warmup.")
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    cm = ConfigManager()
    config = cm.resolve_env_vars(cm.load(Path(args.config)))
    mcfg = config.get("model", {}).get("config", {})
    mtype = config.get("model", {}).get("type", "ssd_mobilenetv3")
    dcfg = config.get("dataset", {})
    tcfg = config.get("training", {})

    input_size = mcfg.get("input_size", 320)
    num_classes = mcfg.get("num_classes", 5)
    dataset_path = dcfg.get("path", "model/data/rdd2022/sample")
    country_filter = dcfg.get("country_filter")
    batch_size = tcfg.get("batch_size", 16)
    num_workers = tcfg.get("num_workers", 4)
    prefetch = int(tcfg.get("prefetch_factor", 2))
    use_amp = tcfg.get("use_amp", True)
    val_split = tcfg.get("val_split", 0.2)
    aug_config = tcfg.get("augmentation", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} torch={torch.__version__} "
          f"model={mtype} input={input_size} batch={batch_size} "
          f"workers={num_workers} prefetch={prefetch} amp={use_amp}")

    # --- dataset load (startup cost) ---
    t0 = time.time()
    ds = RDD2022Dataset(country_filter=country_filter)
    ds.load(Path(dataset_path))
    train_ds, _val, _ = ds.split(1.0 - val_split, val_split, 0.0, seed=42)
    t_load = time.time() - t0
    print(f"[load] dataset parse+split: {t_load:.1f}s  | train images: {len(train_ds)}")

    mosaic_p = float(aug_config.get("mosaic", 0.0))
    mixup_p = float(aug_config.get("mixup", 0.0))
    aug = build_augmentation_pipeline(aug_config) if aug_config else None
    train_torch = RDD2022TorchDataset(
        train_ds, input_size=input_size, augmentation=aug,
        mosaic=mosaic_p, mixup=mixup_p,
    )

    mp_context = "spawn" if num_workers > 0 else None
    prefetch_kwargs = {"prefetch_factor": prefetch} if num_workers > 0 else {}
    loader = torch.utils.data.DataLoader(
        train_torch, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0, multiprocessing_context=mp_context,
        collate_fn=collate_fn, **prefetch_kwargs,
    )

    # --- model ---
    model_cfg = dict(mcfg)
    model_cfg["num_classes"] = num_classes
    if tcfg.get("loss"):
        model_cfg["loss"] = tcfg["loss"]
    model = ModelRegistry.create(mtype, model_cfg)
    if hasattr(model, "_model") and hasattr(model._model, "model"):
        model._model.model.to(device)
    elif hasattr(model, "_model"):
        model._model.to(device)
    elif hasattr(model, "model"):
        model.model.to(device)
    model.set_train_mode()

    params = model.get_parameters()
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()

    print(f"[run] midiendo {args.num_batches} batches (warmup {args.warmup})...")
    data_times, step_times = [], []
    measured = 0
    t_prev = time.time()
    for i, (images, targets) in enumerate(loader):
        t_data = time.time() - t_prev

        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        try:
            if use_amp:
                with torch.amp.autocast('cuda'):
                    loss = model.train_step(images, targets)["loss_tensor"]
            else:
                loss = model.train_step(images, targets)["loss_tensor"]
            if loss.item() != 0.0:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        except Exception as e:
            print(f"  batch {i}: train_step error: {e}")
        synchronize()
        t_step = time.time() - (t_prev + t_data)

        if i >= args.warmup:
            data_times.append(t_data)
            step_times.append(t_step)
            measured += 1
        if measured >= args.num_batches:
            break
        t_prev = time.time()

    if not step_times:
        print("No se midió ningún batch.")
        return

    avg_data = sum(data_times) / len(data_times)
    avg_step = sum(step_times) / len(step_times)
    total = avg_data + avg_step
    n_batches_epoch = (len(train_ds) + batch_size - 1) // batch_size
    print("\n================ RESULTADO ================")
    print(f"  t_data (espera datos)  : {avg_data*1000:7.1f} ms/batch  ({avg_data/total*100:4.1f}%)")
    print(f"  t_step (cómputo GPU)   : {avg_step*1000:7.1f} ms/batch  ({avg_step/total*100:4.1f}%)")
    print(f"  total                  : {total*1000:7.1f} ms/batch")
    print(f"  batches/época (train)  : {n_batches_epoch}")
    print(f"  época train estimada   : {total*n_batches_epoch/60:5.1f} min")
    verdict = ("LIMITADO POR DATOS / IO" if avg_data > 0.5 * avg_step
               else "LIMITADO POR CÓMPUTO (GPU)")
    print(f"  veredicto              : {verdict}")
    print("===========================================")


if __name__ == "__main__":
    main()
