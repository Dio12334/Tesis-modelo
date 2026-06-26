"""Runner de los experimentos rigurosos de Noruega (N0 / N1 / N2) a 15 épocas.

Para cada experimento: genera el config (desde el base), entrena y evalúa, y al final imprime
una tabla comparativa (mAP global + per-país, con foco en Noruega). Secuencial (comparten GPU).
Re-ejecutable: salta los que ya tengan reporte de evaluación (--force para rehacer).

Experimentos (todos 768, mismo recipe R; difieren solo en países + letterbox):
  N0  baseline (4 países, sin Noruega)        -> referencia "sin Noruega"
  N1  + Noruega, resize CUADRADO (naïve)      -> ¿meter Noruega "como está" arrastra?
  N2  + Noruega, LETTERBOX                     -> ¿arreglar la distorsión recupera a Noruega?

Comparación clave: mAP per-país de **Noruega** en N1 vs N2 (¿ayuda el letterbox?) y mAP global
N0 vs N1/N2 (¿incluir Noruega aporta o daña?).

Uso:
    python -m model.scripts.run_norway_experiments
    python -m model.scripts.run_norway_experiments --epochs 15 --batch 6 --only N2
"""
import argparse
import copy
import glob
import json
import os
import pathlib
import re
import subprocess
import sys

import yaml

BASE_CFG = "model/configs/train_rt_detr_norway_768.yaml"  # plantilla (recipe R + letterbox)
COUNTRIES_4 = ["Japan", "India", "United_States", "Czech"]
COUNTRIES_5 = COUNTRIES_4 + ["Norway"]
CFG_DIR = pathlib.Path("model/configs")
LOGDIR = pathlib.Path("logs/norway_experiments")

# (nombre, países, letterbox)
EXPERIMENTS = [
    ("N0_baseline", COUNTRIES_4, False),
    ("N1_norway_square", COUNTRIES_5, False),
    ("N2_norway_letterbox", COUNTRIES_5, True),
]


def make_config(name, countries, letterbox, epochs, batch):
    cfg = yaml.safe_load(open(BASE_CFG, encoding="utf-8"))
    cfg["name"] = f"exp_{name}"
    cfg["dataset"]["country_filter"] = list(countries)
    cfg["training"]["letterbox"] = bool(letterbox)
    cfg["training"]["epochs"] = int(epochs)
    cfg["training"]["batch_size"] = int(batch)
    cfg["model"]["config"]["input_size"] = 768
    ckpt = f"./checkpoints/exp_{name}"
    cfg["training"]["checkpoint_dir"] = ckpt
    cfg["output"]["checkpoint_dir"] = ckpt
    cfg["output"]["results_dir"] = f"./results/exp_{name}"
    path = CFG_DIR / f"exp_norway_{name}.yaml"
    path.write_text(yaml.dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return str(path), ckpt


def newest_run_dir(ckpt_dir):
    subs = [d for d in glob.glob(os.path.join(ckpt_dir, "*")) if os.path.isdir(d)]
    return max(subs, key=os.path.getmtime) if subs else None


def find_report(run_dir):
    reps = glob.glob(os.path.join(run_dir, "**", "*evaluation_report.json"), recursive=True)
    return max(reps, key=os.path.getmtime) if reps else None


def parse_per_country(eval_log_path):
    pc = {}
    if not os.path.exists(eval_log_path):
        return pc
    pat = re.compile(r"^\s+(Japan|India|Czech|United_States|Norway|China\w*)\s+\d+\s+([\d.]+)")
    for line in open(eval_log_path, encoding="utf-8", errors="ignore"):
        m = pat.match(line)
        if m:
            pc[m.group(1)] = float(m.group(2))
    return pc


def run(cmd, log_path):
    with open(log_path, "w", encoding="utf-8") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=6, help="768/batch6 cabe en 8GB (R lo probó); baja a 4 si OOM")
    ap.add_argument("--only", default=None, help="correr solo un experimento (p.ej. N2_norway_letterbox)")
    ap.add_argument("--force", action="store_true", help="rehacer aunque ya exista reporte")
    ap.add_argument("--dry-run", action="store_true", help="solo generar configs y mostrar el plan")
    args = ap.parse_args()

    LOGDIR.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    exps = [e for e in EXPERIMENTS if (args.only is None or e[0] == args.only)]
    results = {}

    for name, countries, letterbox in exps:
        cfg_path, ckpt_dir = make_config(name, countries, letterbox, args.epochs, args.batch)
        eval_log = LOGDIR / f"{name}_eval.log"

        if args.dry_run:
            print(f"[dry-run] {name}: config -> {cfg_path} | países={len(countries)} "
                  f"letterbox={letterbox} epochs={args.epochs} batch={args.batch} ckpt={ckpt_dir}")
            continue

        # ¿ya hecho?
        rd = newest_run_dir(ckpt_dir)
        if not args.force and rd and find_report(rd):
            print(f"[skip] {name}: ya tiene reporte de evaluación (usa --force para rehacer)")
        else:
            # Resume-on-partial: si quedó un training_state.pt (época completada) sin eval,
            # continuar en vez de reentrenar desde cero (robusto a cortes a media corrida).
            resume = []
            if not args.force and rd and os.path.exists(os.path.join(rd, "training_state.pt")):
                resume = ["--resume", os.path.basename(rd)]
                print(f"  [resume] {name}: continuando desde {os.path.basename(rd)}")
            print(f"\n===== {name}: TRAIN ({args.epochs} ép, batch {args.batch}, "
                  f"letterbox={letterbox}, {len(countries)} países) =====")
            rc = run([py, "-m", "model.training.train_detection", "--config", cfg_path, "-v"] + resume,
                     LOGDIR / f"{name}_train.log")
            if rc != 0:
                print(f"  [FALLO] train {name} (rc={rc}); ver {LOGDIR}/{name}_train.log")
                continue
            rd = newest_run_dir(ckpt_dir)
            best = os.path.join(rd, "best_model.pt") if rd else None
            if not best or not os.path.exists(best):
                print(f"  [FALLO] no se encontró best_model.pt para {name}")
                continue
            print(f"===== {name}: EVAL =====")
            run([py, "-m", "model.training.evaluate_detection", "--config", cfg_path,
                 "--checkpoint", best, "--split", "val"], eval_log)

        # recolectar resultados
        rd = newest_run_dir(ckpt_dir)
        rep = find_report(rd) if rd else None
        g = json.load(open(rep, encoding="utf-8"))["metrics"] if rep else {}
        results[name] = {
            "map_50": g.get("map_50"), "recall": g.get("recall"),
            "per_class": g.get("per_class_ap", {}),
            "per_country": parse_per_country(eval_log),
        }

    # ---- tabla comparativa ----
    def f(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else "  -   "
    print("\n\n================ COMPARACIÓN (val) ================")
    print(f"{'experimento':22} {'mAP@0.5':>8} {'recall':>7} | {'Norway':>7} {'Czech':>7} {'India':>7} {'Japan':>7} {'US':>7}")
    for name, _, _ in exps:
        r = results.get(name)
        if not r:
            print(f"{name:22}  (sin resultado)"); continue
        pc = r["per_country"]
        print(f"{name:22} {f(r['map_50']):>8} {f(r['recall']):>7} | "
              f"{f(pc.get('Norway')):>7} {f(pc.get('Czech')):>7} {f(pc.get('India')):>7} "
              f"{f(pc.get('Japan')):>7} {f(pc.get('United_States')):>7}")
    print("\nReglas: Norway en N2 >> N1 -> el letterbox recupera a Noruega.  "
          "mAP global N1/N2 >= N0 -> incluir Noruega aporta.")
    (LOGDIR / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Resumen:", LOGDIR / "summary.json")


if __name__ == "__main__":
    main()
