# Plan de experimentación — ¿la data de Perú mejora los modelos?

> Diseño + runbook para medir si incorporar la data propia de Perú influye positivamente en los
> modelos RT-DETR ya entrenados sobre RDD2022. **NO ejecutado aún** (a la espera del OK). El ambiente
> ya está preparado (ver §"Estado del setup"). Fecha: 2026-06-22.

## Pregunta e hipótesis
- **Pregunta:** ¿agregar la data de Perú mejora el modelo? (a) ¿generaliza mejor a calles de Perú?
  (b) ¿sin perjudicar el desempeño en RDD?
- **Hipótesis:** un modelo RDD evaluado en Perú sufre una brecha de dominio; afinarlo con Perú sube
  su mAP en Perú; si además no cae mucho en RDD, la influencia es positiva.

## El problema clave: taxonomía (resuelto)
- Modelos RDD: 5 clases EN (alligator, longitudinal, other corruption, pothole, transverse).
- Perú: 8 clases ES. Se **relabelearon a las 5 de RDD** (`relabel_to_rdd5.py`) para alinear los
  índices del head de R:

  | Perú | RDD |
  |------|-----|
  | bache | pothole |
  | fisura_longitudinal, fisura_oblicua | longitudinal crack |
  | fisura_transversal | transverse crack |
  | piel_de_cocodrilo | alligator crack |
  | reparacion, tratamiento_superficial, fisura_esquina | other corruption |

  Resultado (`model/data/peru_rdd5/`, 1956 img): longitudinal 1654, other 1630, alligator 534,
  transverse 519, pothole 408. Las 5 clases quedan pobladas → orden alfabético = índices de R.

## Test held-out de Perú
El **split de val (seed 42, 20% ≈ 391 img)** de `peru_rdd5` es el test. El fine-tune entrena solo
en el 80%; A0 (zero-shot de R) y A1 (fine-tuned) se evalúan en ESE mismo 20% → comparación limpia.

---

## Track A — Fine-tune sobre R (este experimento)

### A0 — Zero-shot: ¿qué tan mal le va a R en Perú?
Evalúa R (modelo RDD, mAP 0.642) sobre el test de Perú, **sin reentrenar**. Mide la brecha de dominio.
```bash
python -m model.training.evaluate_detection \
  --config model/configs/train_rt_detr_peru_finetune.yaml \
  --checkpoint checkpoints/rt_detr_v2_R/b5da089a-ab0e-4f69-b72c-10f28487e056/best_model.pt \
  --split val \
  --output-dir checkpoints/rt_detr_peru_finetune/A0_zeroshot
```

### A1 — Fine-tune de R sobre Perú
Carga los pesos de R (`init_checkpoint`) y afina con LR bajo (~20 ép, ~40-60 min en laptop).
```bash
python -m model.training.train_detection \
  --config model/configs/train_rt_detr_peru_finetune.yaml -v
```
Luego evalúa el fine-tuned sobre el **mismo test de Perú**:
```bash
python -m model.training.evaluate_detection \
  --config model/configs/train_rt_detr_peru_finetune.yaml \
  --checkpoint checkpoints/rt_detr_peru_finetune/<RUN_ID>/best_model.pt \
  --split val
```

### A1' — ¿El fine-tune olvidó RDD? (catastrophic forgetting)
Evalúa el modelo fine-tuned sobre el **RDD original** y compara con 0.642:
```bash
python -m model.training.evaluate_detection \
  --config model/configs/train_rt_detr_v2_R.yaml \
  --checkpoint checkpoints/rt_detr_peru_finetune/<RUN_ID>/best_model.pt \
  --split val \
  --output-dir checkpoints/rt_detr_peru_finetune/A1_on_rdd
```

### Métricas y reglas de decisión (Track A)
Comparar en el **test de Perú**: A0 (R zero-shot) vs A1 (fine-tuned).
- **A1 mAP@0.5 (Perú) > A0** → la data de Perú **mejora** el modelo en Perú (influencia positiva). ✅
- Magnitud del salto = cuánto ayuda Perú.

Comparar en **RDD-val**: R (0.642) vs A1'.
- **A1' ≈ 0.642** → el fine-tune **no daña** RDD (influencia positiva neta).
- **A1' << 0.642** → hubo *forgetting*; el fine-tune mejora Perú a costa de RDD (trade-off).

Mirar también **per-clase** (las 4 clases con datos reales de Perú: pothole, longitudinal,
transverse, alligator) y la **brecha A0→A1** por clase.

---

## Track B — Modelo conjunto desde cero (futuro, EC2)
Si A da señal positiva, escalar a un estudio más completo en 4 clases comunes:
- **M1 RDD-only**, **M2 Perú-only**, **M3 RDD+Perú** (dir combinado), todos evaluados en
  {Perú-test, RDD-test}. M3 vs M1 (¿Perú ayuda? ¿daña RDD?), M3 vs M2 (¿RDD ayuda a Perú?).
- Costo: 3 runs escala-RDD → EC2. Requiere fusionar RDD+Perú en un dir (pendiente de implementar).

---

## Estado del setup (lo que YA está listo)
- [x] `model/data/peru_rdd5/` — Perú relabeleado a 5 clases RDD (1956 img + anns + meta).
- [x] `model/configs/train_rt_detr_peru_finetune.yaml` — config de fine-tune (init desde R, LR bajo).
- [x] `model/scripts/relabel_to_rdd5.py` — el relabeleo (reproducible).
- [x] `train_detection.py` — soporta `init_checkpoint` (carga pesos sin restaurar optimizer/epoch).
- [ ] **Ejecutar A0, A1, A1'** (a la espera del OK). Todo el runbook de arriba está listo.
- [ ] Track B: fusión RDD+Perú + 3 runs (futuro).

## Notas
- Mapeo `other corruption` ← {reparacion, tratamiento_superficial, fisura_esquina}: poblar las 5
  clases es necesario para alinear índices con R; "other corruption" es catch-all. Es una decisión
  de diseño del experimento (documentada). Las 4 clases "limpias" (pothole/longitudinal/transverse/
  alligator) son las que dan la señal interpretable.
- El test de Perú es el split de val (seed 42); idéntico para A0 y A1 por construcción.
