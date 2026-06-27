# Cómo mejorar las métricas de RT-DETR (análisis de corridas)

> Basado en las 9 corridas de `checkpoints/rt_detr/` (config probada + `val_evaluation_report.json`).
> Reproducible con `python -m model.tools.summarize_runs --model rt_detr`. Fecha: 2026-06-15.

## Lo que ya probaste (ordenado por mAP@0.5)

| run | eval | mAP@0.5 | mAP@.5:.95 | prec | rec | F1 | épocas | config distintiva |
|-----|------|--------:|-----------:|-----:|----:|---:|-------:|-------------------|
| **610114d9** | rt_detr | **0.592** | **0.293** | 0.585 | 0.591 | **0.588** | **14**/100 | batch 8, **sin mosaic**, JP/IN/US/CZ |
| 800888cf | rt_detr | 0.550 | 0.265 | 0.603 | 0.509 | 0.552 | 38/100 | batch 16, **mosaic 0.5**, JP/IN/US/CZ |
| 164488fd | rt_detr | 0.512 | 0.238 | 0.638 | 0.428 | 0.512 | 23/100 | mosaic 0.5, **+China** |
| 2b46b55e | two_stage | 0.473 | 0.256 | 0.643 | 0.538 | 0.586 | 74/100 | two_stage, ALL países |
| 5eee5bb9 | two_stage | 0.346 | 0.175 | 0.617 | 0.424 | 0.503 | 19/100 | two_stage, mosaic, ALL |

(4 runs más quedaron en 0–3 épocas, sin evaluar.)

## Patrones validados por tus datos

1. **Mosaic/MixUp PERJUDICAN**: 0.592 (sin) vs 0.550 (mosaic 0.5), mismos países. El daño vial son
   objetos pequeños/finos; el downscale del mosaico 2S→S los destruye. → mantener `mosaic: 0, mixup: 0`.
2. **Incluir China baja mAP**: 0.550 → 0.512. China_Drone es aéreo (otra distribución). → mantenerlo excluido.
3. **two_stage reporta MENOS mAP**: el clasificador descarta imágenes con daño (baja recall). Además su
   F1-sweep sale **plano** (el detector pre-filtra a conf=0.5) → **subreporta**. Ver recomendación #4.
4. **Sí usas pesos COCO-pretrained** (`rtdetr-l.pt` auto-descargado) — bien, no entrenas desde cero.

## El hallazgo principal: tu mejor modelo está SUBENTRENADO

Curva de val de 610114d9 (la mejor config):
```
ep0 28.26 → ep5 21.03 → ep10 19.70 → ep13 19.52   (seguía bajando; LR aún ~9.8e-5, cosine apenas arrancó)
```
Paró en la **época 14 de 100** (interrumpido, no early-stop — patience=25). **Nunca convergió.** Y las
corridas que sí entrenaron más (38, 74 ép) usaban la config **peor** (mosaic / two_stage). Es decir:

> ~~**"Mejor config (sin mosaic) + entrenar hasta converger" NO se ha probado todavía.** Es el mayor
> margen de mejora disponible.~~

> **ACTUALIZACIÓN (resume probado, ep14→18):** se reanudó 610114d9 y **NO mejoró el mAP**.
> Comparación del best re-evaluado (ep15) vs original (ep13):
> mAP@0.5 0.592→**0.591** (plano), mAP@.5:.95 0.293→0.296, prec 0.585→**0.635**, rec 0.591→**0.562**,
> F1 0.588→0.596. El val_loss tocó fondo en ep15 y subió; el gap train–val se duplicó (1.4→3.0) →
> **overfitting**, no subentrenamiento. La curva empinada inicial de val_loss era la pérdida de
> *clasificación* asentándose, NO el mAP subiendo. **Conclusión: el modelo está en meseta para esta
> config; más épocas no ayudan (y el recall, el cuello, empeoró 0.591→0.562).** El camino a mejores
> métricas son los cambios SOTA (sección siguiente), en una corrida NUEVA, sobre todo **S1** (recall)
> y **S2** (resolución). Además, desglose por país del eval: Japan 0.60 / US 0.48 / India 0.35 /
> **Czech 0.27** → desbalance de datos por país (oversamplear Czech/India).

---

## Recomendaciones (orden por impacto esperado en métricas)

### #1 — ~~Entrenar hasta converger~~ → DESCARTADO por la evidencia
Se reanudó 610114d9 (ep14→18) y el mAP quedó **plano (0.592→0.591)** con overfitting incipiente y
**recall peor** (0.591→0.562). Más épocas con esta config NO ayudan. La nueva prioridad #1 pasa a ser
**S1** (loss pro-recall) + **S2** (resolución) en una corrida nueva — ver sección de estado del arte.
Nota de schedule: el cosine apunta a 100 épocas pero solo se corrieron ~19, así que el LR nunca decayó
(~9.5e-5); si se reentrena, fijar `epochs` a un horizonte real (30-50) para que el LR baje.

### #2 — Mantener la config ganadora (ya descubierta)
`mosaic: 0`, `mixup: 0`, países `[Japan, India, United_States, Czech]`, COCO-pretrained. No reintroducir
mosaic ni China. La augmentación útil que queda es ligera: `hflip`, `hsv`, `scale [0.8, 1.2]`, `translate 0.1`.

### #3 — Elegir y reportar el punto de operación a propósito (gratis, sin reentrenar)
El F1-sweep de 610114d9 da el trade-off precisión/recall:
```
conf 0.30 → r=0.75 p=0.33 F1=0.46     conf 0.50 → r=0.59 p=0.59 F1=0.59 (óptimo F1)
conf 0.40 → r=0.68 p=0.46 F1=0.55     conf 0.55 → r=0.54 p=0.65 F1=0.59
```
- Para **F1 máximo**: conf **0.50–0.55**.
- Si la tesis prioriza **no perder daños (recall)**: conf **0.35–0.40** (recall 0.68–0.75).
- Usar `per_class_best_f1` del reporte para **umbrales por clase** (mejora el F1 agregado).

### #4 — Arreglar la evaluación two_stage (subreporta hoy)
En el eval two_stage el detector corre a `conf=0.5` interna → el F1-sweep sale **plano** y la recall queda
tapada artificialmente. Para reportar two_stage de forma justa, el detector debe emitir detecciones de
baja confianza (conf≈0.001) y dejar que el sweep filtre (como ya hace el eval directo de rt_detr). Esto
puede subir bastante el F1 reportado del two_stage sin reentrenar.

### #5 — Atacar las clases débiles (recall)
Matriz de confusión de 610114d9 (% de GT no detectado): **bache 52 %**, fisura_longitudinal 43 %,
transversal 44 %, otros 32 %, alligator 28 %. El cuello es **recall** (+ FPs notables). Palancas:
- Bajar el umbral de confianza (recall global, ver #3).
- Más entrenamiento (#1) ayuda a todas.
- Desbalance leve (longitudinal 2628 vs bache 1098 instancias GT): probar muestreo balanceado u
  oversampling de bache si tras #1 sigue rezagado.

### #6 — Tuning secundario (probar tras #1)
- **LR discriminativo**: el wrapper soporta `backbone_lr`/`head_lr` (hoy LR único 0.0001). Para fine-tuning
  de un backbone COCO, `backbone_lr: 1e-5`, `head_lr: 1e-4` suele dar mejor mAP y menos overfit. Añadir al
  `model.config` del YAML.
- **Localización floja** (mAP@0.5:0.95=0.293 vs mAP@0.5=0.592): las cajas detectan pero ajustan mal. Más
  entrenamiento + quizá subir `giou_weight` (2.0→3.0) puede apretar cajas.

---

## Orden de acción sugerido
1. **#1 + #2**: reanudar/entrenar la config sin-mosaic hasta converger (mayor ganancia).
2. **#3**: re-evaluar y fijar el punto de operación (gratis, mientras entrena).
3. **#4**: corregir el eval two_stage si vas a reportar esa variante.
4. **#5/#6**: tras converger, atacar bache y probar LR discriminativo.

---

# Mejoras del config según el estado del arte (RDD2022 / RT-DETR)

Referencias: CRDDC'2022 mejor F1 = **76.9%** (6 países; 2º YOLOv5x **P5+P6 multiescala**
F1 0.743, 3º YOLOv7+coordinate-attention 0.741); ORDDC'2024 ~0.71. Receta **oficial de
RT-DETR**: AdamW, batch 16, **EMA decay 0.9999**, **entrenamiento multiescala** (480→800px),
aug = color distortion + expansión + crop + flip + resize. (Tu mAP/F1 ~0.59 no es comparable
directo al 76.9%: ese es sobre el test oficial de 6 países; tú evalúas con split aleatorio de
4 países.)

Lo que falta o está sesgado en tu config frente al SOTA:

### S1 — Loss sesgado a precisión, pero tu cuello es RECALL ⚠️ (config, gratis, alta confianza)
Subiste `no_object_weight: 0.3` (default RT-DETR = **0.1**) "para reducir FPs" y usas
`focal_alpha: 0.20` (bajo). Ambos **penalizan predecir cajas / down-pesan positivos** → bajan
recall. Pero tu problema medido es recall (bache 52% no detectado), no precisión (prec≈rec≈0.59).
→ Probar `no_object_weight: 0.1`, `focal_alpha: 0.25`. Empuja hacia recall, alineado con el default.

### S2 — Resolución de entrada baja para objetos pequeños (config, gran impacto; ojo VRAM)
Entrenas a `input_size: 640` fijo. El SOTA RT-DETR entrena hasta **800px** y los daños finos
(grietas, baches pequeños) ganan mucho con más resolución. → Subir a `input_size: 768` o `800`.
Trade-off: en 8 GB probablemente requiera bajar `batch_size` a 4-6 (a 800px, batch 8 puede OOM).
Es la palanca de config más fuerte para tu déficit de recall en objetos pequeños.

### S3 — EMA de pesos ✅ IMPLEMENTADO (SOTA estándar)
La receta oficial de RT-DETR usa **EMA (decay 0.9999)**. Se implementó en `train_detection.py`
(clase `ModelEMA`, opt-in vía `use_ema`): mantiene un shadow de los pesos, lo actualiza tras
cada `optimizer.step()` con decay rampante, y **guarda los pesos EMA como best/final** (mejor
generalización; el modelo evaluado es el EMA). Es resume-safe (el shadow se persiste en
`training_state.pt`). Claves: `use_ema: true`, `ema_decay: 0.9999`, `ema_tau: 2000`. Ya activo en
`train_rt_detr_v2.yaml`. Default `false` para no alterar configs existentes.

### S4 — Entrenamiento multiescala (requiere código; default de RT-DETR)
RT-DETR varía la resolución por batch (480→800). Tu `RDD2022TorchDataset` redimensiona todo a
un tamaño fijo. Multiescala mejora robustez a tamaños de daño. Implementar: resolución aleatoria
por batch (múltiplos de 32) en el collate/loader.

### S5 — LR discriminativo backbone/head (config, alineado con la receta)
RT-DETR fine-tunea el backbone COCO con LR menor que la cabeza. El wrapper lo soporta
(`backbone_lr`/`head_lr`) pero tu config usa LR único 1e-4. → Añadir a `model.config`:
`backbone_lr: 1e-5`, `head_lr: 1e-4`. Reduce overfit del backbone preentrenado.

### S6 — Test-Time Augmentation en evaluación ✅ IMPLEMENTADO
TTA manual en `RT_DETR_Detector.forward` (ultralytics `augment=True` es no-op en RT-DETR): flip +
multiescala (640/768/896) + merge NMS. Flag `--tta`/`--tta-scales` en evaluate_detection.
**Resultado sobre R:** mAP@0.5 0.642→**0.649 (+0.007)**, recall 0.656→**0.725 (+0.069)**, pero
**best F1 igual (0.626)** — TTA mueve el punto de operación a recall (óptimo conf 0.55→0.60), no
sube el pico de F1. Caveat: 6× más lento en inferencia → técnica de **reporte**, no de despliegue
real-time. Reportar mAP con TTA (0.649); desplegar sin.

### S7 — Metodología de evaluación (comparabilidad)
Usas split aleatorio 20% de train (4 países). El SOTA evalúa sobre el **test oficial RDD2022**
(server del challenge). Tus números absolutos no son comparables al 76.9%; documenta el split o
usa el test oficial para reportar.

### Diff de config sugerido (YAML, lo aplicable sin tocar código)
```yaml
model:
  config:
    input_size: 768          # S2 (640 -> 768/800; baja batch si OOM)
    backbone_lr: 0.00001     # S5
    head_lr: 0.0001          # S5
training:
  batch_size: 6              # S2 (compensa la mayor resolución en 8 GB)
  loss:
    no_object_weight: 0.1    # S1 (0.3 -> 0.1, recupera recall)
    focal_alpha: 0.25        # S1 (0.20 -> 0.25)
```
**EMA (S3) ya implementado** y activo en `model/configs/train_rt_detr_v2.yaml` (que aplica
S1+S2+S5+S3+schedule). Pendiente de código: **multiescala** (S4).

> Nota: S1/S2 cambian la dinámica respecto a 610114d9, así que conviene probarlos en una corrida
> nueva (no en el resume), para no mezclar regímenes a media curva.

### ✅ Resultado validado (v2, run 7a4ff1e9 — best EMA ep16/40 @768px)

| métrica | v1 (610, 640px) | v2 (768px) | Δ |
|---|---:|---:|---:|
| mAP@0.5 | 0.592 | **0.6353** | +0.043 |
| mAP@.5:.95 | 0.293 | **0.3246** | +0.032 |
| recall | 0.591 | **0.6549** | **+0.064** |
| precision | 0.585 | 0.587 | ~= |
| F1 | 0.588 | **0.6191** | +0.031 |
| AP bache | 0.43 | **0.516** | +0.086 |

El bundle SOTA funcionó y de forma interpretable: **S1 → recall +0.064**, **S2 → bache (objeto
pequeño) +0.086**. Es solo ep16/40 (sin la fase de LR bajo ep30-40) → queda techo por encima de
0.635 al reanudar/converger. Para atribución por-cambio en la tesis, hacen falta ablaciones
(solo-S1, solo-S2, etc.); como salto agregado, está validado.

> **Actualización (resume de v2):** se reanudó hasta ep22 y el val EMPEORÓ (19.4→20.1) con LR
> decayendo y gap train-val explotando (3.6→5.75). El mejor sigue siendo ep16 (0.635). Conclusión:
> el techo de generalización está ~ep16; **más épocas NO ayudan, overfittean.** El lever es
> datos/regularización (siguiente sección).

---

# Plan de experimentos: datos / regularización (romper el techo ~0.635)

**Diagnóstico (overfitting + desbalance por país):**

| País | imgs train (% subset) | mAP@0.5 (v1) | lectura |
|---|---:|---:|---|
| Japan | 10.506 (40.6%) | 0.60 | mucha data → mejor |
| India | 7.706 (29.8%) | 0.35 | **data-rica pero DIFÍCIL** (no es cantidad) |
| US | 4.805 (18.6%) | 0.48 | media |
| Czech | 2.829 (10.9%) | 0.27 | poca data → peor |

Czech = data-starved (oversampling ayudaría). **India rompe la hipótesis de cantidad**: tiene ~3×
los datos de Czech y aun así 0.35 → dificultad intrínseca / ruido de etiquetas. Por eso balancear
países a ciegas es arriesgado (subir Czech sí, bajar Japan no).

**Estrategia:** una variable a la vez (atribución limpia). Baseline = v2 (0.635 @768). Trackear
mAP global + **por país** + **por clase** + recall. Ablar idealmente a 640/epochs cortos para
rankear; confirmar ganador a 768.

| Exp | Qué cambia vs v2 | Hipótesis | Estado |
|---|---|---|---|
| **R** regularización | `weight_decay` 1e-4→5e-4 + `random_erasing` 0.3 | retrasa overfit (probado ~ep16) → techo más alto | ✅ **HECHO: mAP 0.635→0.6420 (+0.0067)**, bache +0.025. Win real pero marginal (regularización casi agotada). best ep15, early-stop ep27. |
| **C** balanceo clase | `balanced_sampling: class` (sampler ∝ 1/sqrt(n_img clase)) | sube AP de bache/other → mAP | ❌ **NEGATIVO: mAP 0.642→0.639 (−0.0026), bache −0.008.** Solo movió prec↓/recall↑. Desbalance leve (2.4×) + oversampling con reemplazo = overfit; el cuello de bache es resolución, no frecuencia. |
| **P** balanceo país | `balanced_sampling: country` (mismo sampler) | cierra gap Czech; riesgo Japan/India | **listo** (solo cambiar `balanced_sampling: country` en un config) |
| **I** diagnóstico India | inspeccionar imágenes/etiquetas de India | entender el 0.35 antes de intervenir | en cola |

**Lanzar R:** `python -m model.training.train_detection --config model/configs/train_rt_detr_v2_R.yaml -v`
