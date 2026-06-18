"""Consolida las corridas de un modelo: config probada + métricas de evaluación.

Para cada run bajo checkpoints/<modelo>/<run_id>.json (tracker) cruza:
  - config usado (lr, optimizer, batch, epochs, augmentation, loss, country_filter, ...)
  - final_results (best_val_loss, best_epoch, total_epochs)
  - métricas de evaluación de checkpoints/<modelo>/<run_id>/val_evaluation_report.json
    (mAP@0.5, mAP@0.5:0.95, precision, recall, F1, per-class AP)

Uso:
    python -m model.tools.summarize_runs --model rt_detr
"""
import argparse
import json
from pathlib import Path


def load(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rt_detr")
    ap.add_argument("--root", default="checkpoints")
    args = ap.parse_args()

    base = Path(args.root) / args.model
    rows = []
    for jf in sorted(base.glob("*.json")):
        run_id = jf.stem
        run = load(jf)
        if not run:
            continue
        cfg = run.get("config", {})
        tcfg = cfg.get("training", {})
        mcfg = cfg.get("model", {}).get("config", {})
        dcfg = cfg.get("dataset", {})
        aug = tcfg.get("augmentation", {})
        loss = tcfg.get("loss", {})
        fr = run.get("final_results") or {}

        ev = load(base / run_id / "val_evaluation_report.json") or {}
        m = ev.get("metrics", {})

        rows.append({
            "run": run_id[:8],
            "start": (run.get("start_time") or "")[:16],
            "size": mcfg.get("model_size", "?"),
            "input": mcfg.get("input_size", "?"),
            "opt": tcfg.get("optimizer", "?"),
            "lr": tcfg.get("learning_rate", "?"),
            "wd": tcfg.get("weight_decay", "?"),
            "bs": tcfg.get("batch_size", "?"),
            "ep": tcfg.get("epochs", "?"),
            "warmup": tcfg.get("warmup_epochs", "?"),
            "countries": ",".join(dcfg.get("country_filter") or []) or "ALL",
            "mosaic": aug.get("mosaic", "-"),
            "mixup": aug.get("mixup", "-"),
            "hsv_s": aug.get("hsv_s", "-"),
            "scale": aug.get("scale", "-"),
            "hflip": aug.get("horizontal_flip", "-"),
            "focal": loss.get("focal_loss", "-"),
            "no_obj": loss.get("no_object_weight", "-"),
            "bbox_w": loss.get("bbox_weight", "-"),
            "giou_w": loss.get("giou_weight", "-"),
            "best_val_loss": fr.get("best_val_loss"),
            "best_ep": fr.get("best_epoch"),
            "tot_ep": fr.get("total_epochs"),
            "eval_model": ev.get("model_type", "-"),
            "map50": m.get("map_50"),
            "map5095": m.get("map_50_95"),
            "prec": m.get("precision"),
            "rec": m.get("recall"),
            "f1": m.get("f1_score"),
            "per_class": m.get("per_class_ap"),
        })

    # Orden por mAP@0.5 desc (None al final)
    rows.sort(key=lambda r: (r["map50"] is None, -(r["map50"] or 0)))

    def fmt(x, n=3):
        return f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x)

    print(f"\n=== {len(rows)} corridas de {args.model} (orden: mAP@0.5 desc) ===\n")
    hdr = ["run", "start", "sz", "in", "opt", "lr", "bs", "ep", "tot", "bestVL",
           "evalAs", "mAP50", "mAP5095", "prec", "rec", "F1"]
    print("  ".join(f"{h:>8}" for h in hdr))
    for r in rows:
        line = [
            r["run"], r["start"][5:], str(r["size"]), str(r["input"]), str(r["opt"])[:5],
            fmt(r["lr"], 5), str(r["bs"]), str(r["ep"]), str(r["tot_ep"]),
            fmt(r["best_val_loss"], 3), str(r["eval_model"])[:8],
            fmt(r["map50"]), fmt(r["map5095"]), fmt(r["prec"]), fmt(r["rec"]), fmt(r["f1"]),
        ]
        print("  ".join(f"{c:>8}" for c in line))

    print("\n=== detalle de augmentation / loss / countries por run ===")
    for r in rows:
        print(f"\n[{r['run']}] eval_as={r['eval_model']}  mAP50={fmt(r['map50'])}  F1={fmt(r['f1'])}")
        print(f"    countries={r['countries']}")
        print(f"    aug: mosaic={r['mosaic']} mixup={r['mixup']} hsv_s={r['hsv_s']} "
              f"scale={r['scale']} hflip={r['hflip']}")
        print(f"    loss: focal={r['focal']} no_obj={r['no_obj']} bbox_w={r['bbox_w']} giou_w={r['giou_w']}")
        if r["per_class"]:
            pc = "  ".join(f"{k[:10]}={fmt(v,2)}" for k, v in r["per_class"].items())
            print(f"    per_class_AP: {pc}")


if __name__ == "__main__":
    main()
