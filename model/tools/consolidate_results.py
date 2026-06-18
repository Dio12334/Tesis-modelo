"""Consolida los resultados de los experimentos RT-DETR para la tesis.

Lee los val_evaluation_report.json de cada experimento, arma:
  - tabla principal (mAP@0.5, mAP@.5:.95, precision, recall, F1, best-F1)
  - tabla por clase (AP)
y escribe CSVs + (si matplotlib está) figuras PNG en results/thesis/.

Uso:
    python -m model.tools.consolidate_results
"""
import csv
import json
import os
from pathlib import Path

# (nombre, descripción, ruta del reporte)
RUNS = [
    ("Baseline (v1)", "RT-DETR-L @640, config inicial",
     "checkpoints/rt_detr/610114d9-9669-4dea-8393-427cb3f14d4b/val_evaluation_report.json"),
    ("v2 (S1+S2+S5+EMA)", "loss pro-recall + 768px + LR disc. + EMA",
     "checkpoints/rt_detr_v2/7a4ff1e9-d47a-486c-9e57-40a1eb4af92d/val_evaluation_report.json"),
    ("R (+regularización)", "v2 + weight_decay 5e-4 + random_erasing",
     "checkpoints/rt_detr_v2_R/b5da089a-ab0e-4f69-b72c-10f28487e056/val_evaluation_report.json"),
    ("RC (+balanceo clase)", "R + balanced_sampling (NEGATIVO)",
     "checkpoints/rt_detr_v2_RC/35b949db-774e-4766-9743-ffe7d1334775/val_evaluation_report.json"),
    ("R + TTA", "R + test-time augmentation (eval)",
     "checkpoints/rt_detr_v2_R/b5da089a-ab0e-4f69-b72c-10f28487e056/tta/val_evaluation_report.json"),
]

OUT = Path("results/thesis")
OUT.mkdir(parents=True, exist_ok=True)


def load(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


rows = []
per_class = {}
classes = None
for name, desc, path in RUNS:
    d = load(path)
    if not d:
        print(f"[skip] {name}: no se encontró {path}")
        continue
    m = d["metrics"]
    bf = m.get("best_f1", {}) or {}
    rows.append({
        "experimento": name, "descripcion": desc,
        "mAP@0.5": m["map_50"], "mAP@.5:.95": m["map_50_95"],
        "precision": m["precision"], "recall": m["recall"], "F1@0.5": m["f1_score"],
        "best_F1": bf.get("f1"), "best_F1_conf": bf.get("confidence"),
    })
    per_class[name] = m.get("per_class_ap", {})
    if classes is None and per_class[name]:
        classes = list(per_class[name].keys())

# --- CSV principal ---
csv_main = OUT / "resultados_principales.csv"
with open(csv_main, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# --- CSV por clase ---
csv_pc = OUT / "resultados_por_clase.csv"
with open(csv_pc, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["clase"] + [r["experimento"] for r in rows])
    for c in (classes or []):
        w.writerow([c] + [round(per_class[r["experimento"]].get(c, float("nan")), 4) for r in rows])

# --- imprimir tablas (markdown) ---
print("\n### Tabla principal\n")
hdr = ["experimento", "mAP@0.5", "mAP@.5:.95", "precision", "recall", "F1@0.5", "best_F1"]
print("| " + " | ".join(hdr) + " |")
print("|" + "|".join(["---"] * len(hdr)) + "|")
for r in rows:
    print("| " + " | ".join([r["experimento"]] + [f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]) for k in hdr[1:]]) + " |")

print("\n### AP por clase\n")
print("| clase | " + " | ".join(r["experimento"] for r in rows) + " |")
print("|" + "|".join(["---"] * (len(rows) + 1)) + "|")
for c in (classes or []):
    print(f"| {c} | " + " | ".join(f"{per_class[r['experimento']].get(c, 0):.3f}" for r in rows) + " |")

# --- figuras (best-effort) ---
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main_runs = [r for r in rows if "NEGATIVO" not in r["descripcion"]]
    names = [r["experimento"] for r in main_runs]
    maps = [r["mAP@0.5"] for r in main_runs]
    recs = [r["recall"] for r in main_runs]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(names))
    b1 = ax.bar([i - 0.2 for i in x], maps, width=0.4, label="mAP@0.5")
    b2 = ax.bar([i + 0.2 for i in x], recs, width=0.4, label="recall")
    ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0.5, 0.8); ax.set_ylabel("score"); ax.legend()
    ax.set_title("Progresión RT-DETR: mAP@0.5 y recall (val)")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003, f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "progresion_map_recall.png", dpi=150)

    # per-class baseline vs final
    if classes:
        base = [per_class[names[0]].get(c, 0) for c in classes]
        final = [per_class[names[-1]].get(c, 0) for c in classes]
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        xc = range(len(classes))
        ax2.bar([i - 0.2 for i in xc], base, width=0.4, label=names[0])
        ax2.bar([i + 0.2 for i in xc], final, width=0.4, label=names[-1])
        ax2.set_xticks(list(xc)); ax2.set_xticklabels(classes, rotation=20, ha="right", fontsize=8)
        ax2.set_ylabel("AP@0.5"); ax2.legend(); ax2.set_title("AP por clase: baseline vs final")
        fig2.tight_layout(); fig2.savefig(OUT / "ap_por_clase.png", dpi=150)
    print(f"\n[figuras] guardadas en {OUT}/")
except Exception as e:
    print(f"\n[figuras] omitidas ({type(e).__name__}: {e})")

print(f"\n[CSV] {csv_main}\n[CSV] {csv_pc}")
