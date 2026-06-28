"""Runner de los experimentos rigurosos de Noruega (N0/N1/N2 @768 laptop y N3 @1024 EC2).

Para cada experimento: genera el config (desde el base), entrena y evalúa, imprime una tabla
comparativa (mAP global + per-país, foco Noruega) y **acumula** los resultados en
`logs/norway_experiments/summary.json` (merge: no pisa los de otros grupos/corridas; ese JSON es el
artefacto descargable para analizar). Secuencial (comparten GPU). Re-ejecutable: salta los que ya
tengan reporte de evaluación (--force para rehacer) y resume los cortados a media corrida.

Grupos (misma receta R; difieren en países, letterbox y resolución):
  laptop @768 (8 GB):
    N0  baseline (4 países, sin Noruega)        -> referencia "sin Noruega"
    N1  + Noruega, resize CUADRADO (naïve)      -> ¿meter Noruega "como está" arrastra?
    N2  + Noruega, LETTERBOX                     -> ¿la distorsión 2:1 era el cuello?
  ec2 @1024 (~25 GB):
    N0_baseline_1024         (4 países)         -> referencia "sin Noruega" a 1024
    N3b_norway_square_1024   (+ Noruega, square) -> aísla el efecto del letterbox a 1024
    N3_norway_letterbox_1024 (+ Noruega, LETTERBOX) -> **el experimento principal**

Hallazgo @768 (ya corrido): el letterbox NO recuperó a Noruega (0.297 square -> 0.232 letterbox); a
768 fijo el letterbox encoge la imagen 2:1 a 768x384 + padding -> baja la resolución vertical. El
cuello de Noruega es **resolución**, no distorsión. N3 lo prueba a 1024 (downsample 3.9x vs 5.26x).

Uso:
    python -m model.scripts.run_norway_experiments                       # laptop (N0/N1/N2 @768)
    python -m model.scripts.run_norway_experiments --group ec2 --workers 8   # EC2 (N3 @1024)
    python -m model.scripts.run_norway_experiments --only N3_norway_letterbox_1024
"""
import argparse
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

# (nombre, países, letterbox, input_size, batch_default)
EXPERIMENTS = [
    # --- laptop @768 (8 GB) — HECHOS ---
    ("N0_baseline",             COUNTRIES_4, False, 768, 4),
    ("N1_norway_square",        COUNTRIES_5, False, 768, 4),
    ("N2_norway_letterbox",     COUNTRIES_5, True,  768, 4),
    # --- EC2 @1024 (~25 GB) — N3 ---
    ("N0_baseline_1024",        COUNTRIES_4, False, 1024, 6),
    ("N3b_norway_square_1024",  COUNTRIES_5, False, 1024, 6),
    ("N3_norway_letterbox_1024", COUNTRIES_5, True, 1024, 6),
]

GROUPS = {
    "laptop": ["N0_baseline", "N1_norway_square", "N2_norway_letterbox"],
    "ec2": ["N0_baseline_1024", "N3b_norway_square_1024", "N3_norway_letterbox_1024"],
}


def make_config(name, countries, letterbox, epochs, batch, input_size, workers=None):
    cfg = yaml.safe_load(open(BASE_CFG, encoding="utf-8"))
    cfg["name"] = f"exp_{name}"
    cfg["dataset"]["country_filter"] = list(countries)
    cfg["training"]["letterbox"] = bool(letterbox)
    cfg["training"]["epochs"] = int(epochs)
    cfg["training"]["batch_size"] = int(batch)
    if workers is not None:
        cfg["training"]["num_workers"] = int(workers)
        cfg["training"]["prefetch_factor"] = 4
    cfg["model"]["config"]["input_size"] = int(input_size)
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
    ap.add_argument("--batch", type=int, default=None,
                    help="override del batch (def por exp: 4@768 laptop, 6@1024 EC2)")
    ap.add_argument("--group", choices=["laptop", "ec2", "all"], default="laptop",
                    help="laptop=N0/N1/N2 @768 (def); ec2=N3 @1024 (~25GB); all=todos")
    ap.add_argument("--workers", type=int, default=None, help="override num_workers (EC2: 8)")
    ap.add_argument("--only", default=None, help="correr solo un experimento por nombre")
    ap.add_argument("--force", action="store_true", help="rehacer aunque ya exista reporte")
    ap.add_argument("--dry-run", action="store_true", help="solo generar configs y mostrar el plan")
    args = ap.parse_args()

    LOGDIR.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    if args.only:
        sel = {args.only}
    elif args.group == "all":
        sel = {e[0] for e in EXPERIMENTS}
    else:
        sel = set(GROUPS[args.group])
    exps = [e for e in EXPERIMENTS if e[0] in sel]
    if not exps:
        print(f"[aviso] ningún experimento coincide (group={args.group}, only={args.only})")
        return
    results = {}

    for name, countries, letterbox, input_size, exp_batch in exps:
        batch = args.batch or exp_batch
        cfg_path, ckpt_dir = make_config(name, countries, letterbox, args.epochs, batch,
                                         input_size, args.workers)
        eval_log = LOGDIR / f"{name}_eval.log"

        if args.dry_run:
            print(f"[dry-run] {name}: -> {cfg_path} | países={len(countries)} letterbox={letterbox} "
                  f"input={input_size} epochs={args.epochs} batch={batch} ckpt={ckpt_dir}")
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
            print(f"\n===== {name}: TRAIN ({args.epochs} ép, batch {batch}, input {input_size}, "
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
            "input_size": int(input_size),
            "letterbox": bool(letterbox),
            "countries": list(countries),
            "map_50": g.get("map_50"),
            "recall": g.get("recall"),
            "per_class": g.get("per_class_ap", {}),
            "per_country": parse_per_country(eval_log),
        }

    # ---- tabla comparativa ----
    def f(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else "  -   "
    print("\n\n================ COMPARACIÓN (val) ================")
    print(f"{'experimento':26} {'mAP@0.5':>8} {'recall':>7} | {'Norway':>7} {'Czech':>7} "
          f"{'India':>7} {'Japan':>7} {'US':>7}")
    for name, *_ in exps:
        r = results.get(name)
        if not r:
            print(f"{name:26}  (sin resultado)"); continue
        pc = r["per_country"]
        print(f"{name:26} {f(r['map_50']):>8} {f(r['recall']):>7} | "
              f"{f(pc.get('Norway')):>7} {f(pc.get('Czech')):>7} {f(pc.get('India')):>7} "
              f"{f(pc.get('Japan')):>7} {f(pc.get('United_States')):>7}")
    print("\nReglas: Norway sube con resolución/letterbox -> el preproc recupera a Noruega "
          "(referencia @768: square 0.297, letterbox 0.232).  "
          "mAP global con Noruega >= sin Noruega -> incluir Noruega aporta.")

    # merge en summary.json (acumula entre grupos/corridas; artefacto descargable)
    summary_path = LOGDIR / "summary.json"
    merged = {}
    if summary_path.exists():
        try:
            merged = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            merged = {}
    merged.update(results)
    summary_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Resumen (merge):", summary_path)


if __name__ == "__main__":
    main()
