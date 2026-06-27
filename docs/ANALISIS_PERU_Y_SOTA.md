# Análisis: transferencia con data de Perú + comparación con el estado del arte

> Análisis del experimento de transferencia de dominio (RDD2022 → Perú) y comparación de nuestro
> RT-DETR con el estado del arte en road damage detection, **con foco en modelos RT-DETR, derivados
> y otros transformers**. Complementa `RESULTADOS_TESIS.md` y `EXPERIMENTO_PERU_PLAN.md`.
> Fecha: 2026-06-22.

---

## 1. Resumen ejecutivo

1. **La data de Perú SÍ influye positivamente — en Perú.** Afinar el mejor modelo RDD (R, mAP 0.642)
   sobre Perú casi **duplicó** el mAP en calles peruanas (0.110 → 0.208).
2. **Pero el fine-tune naïve tiene un costo: olvido de RDD** (0.642 → 0.477, −0.165). La data de Perú
   ayuda a Perú a costa de RDD → para un modelo bueno en *ambos* hace falta entrenamiento conjunto
   (Track B), no fine-tune simple.
3. **La brecha de dominio es enorme**: un modelo fuerte en RDD (0.64) colapsa a **0.11** en otro país.
   Road damage detection es duramente dependiente del dominio.
4. **Frente al estado del arte de modelos transformer single-model, estamos en rango** (mAP50
   ~0.62–0.70). La distancia al 0.769 del concurso es de **ensembles + pseudo-labeling**, no de modelo.

---

## 2. Experimento de transferencia de dominio (RDD2022 → Perú)

**Setup:** el mejor modelo RDD (**R**, RT-DETR-L, mAP@0.5 0.642) se evalúa y afina sobre el dataset
propio de Perú (1956 imágenes dashcam de Mapillary, relabeleadas a las 5 clases de RDD). Test =
split de val held-out de Perú (391 img, seed 42), idéntico para A0 y A1.

| Exp | Qué | mAP@0.5 | mAP@.5:.95 | precision | recall | F1 |
|-----|-----|--------:|-----------:|----------:|-------:|---:|
| **A0** | R **zero-shot** en Perú | 0.110 | 0.033 | 0.293 | 0.129 | 0.179 |
| **A1** | R **fine-tuneado** en Perú | **0.208** | 0.066 | 0.518 | 0.156 | 0.240 |
| Δ (A1−A0) | | **+0.097 (×1.9)** | +0.033 | +0.225 | +0.027 | +0.061 |

**AP por clase (Perú), A0 → A1:** longitudinal 0.22→0.31 · alligator 0.13→0.27 · pothole 0.14→0.17 ·
transverse 0.05→0.16 · other 0.01→0.13. **Todas mejoran.**

| Exp | Qué | mAP@0.5 | recall |
|-----|-----|--------:|-------:|
| R (original) | en RDD-val | 0.642 | 0.656 |
| **A1'** | R fine-tuneado, **evaluado en RDD-val** | **0.477** | 0.407 |
| Δ (forgetting) | | **−0.165** | −0.249 |

### Lectura

- **Influencia positiva de Perú (en Perú): confirmada.** El fine-tune casi duplica el mAP (×1.9). La
  data local cierra parcialmente la brecha de dominio.
- **Pero el nivel absoluto es bajo (0.21).** Causas: (a) **brecha de dominio severa** (R cae a 0.11 en
  Perú — features de RDD casi no transfieren: cámara, escena urbana peruana, render 360°→perspectiva,
  16:9 aplastado a cuadrado); (b) **dataset chico** (1565 img de train) para un RT-DETR-L de 32M params;
  (c) recall aún muy bajo (0.16 → pierde 84% del daño): el modelo quedó **preciso pero conservador**.
- **Olvido catastrófico (A1'):** el fine-tune, aun con LR bajo (1e-5, "suave"), bajó RDD 0.642→0.477.
  La clase más golpeada fue *other corruption* (0.73→0.48), por el mapeo conflictivo Perú↔RDD de esa
  clase. **Conclusión:** un fine-tune naïve es un **trade-off** Perú↔RDD, no una mejora gratis.

### Veredicto del experimento
> **La data de Perú influye positivamente** (mejora el desempeño en Perú casi al doble). Pero
> incorporarla **vía fine-tune simple sacrifica RDD**. Para un modelo fuerte en ambos dominios hay
> que ir a **entrenamiento conjunto** (RDD+Perú, Track B) o técnicas de *continual learning* (replay).
> El gran limitante en Perú es el **tamaño del dataset** y la **brecha de dominio**.

---

## 3. Comparación con el estado del arte

### 3.1 Nuestro resultado (recap)
RT-DETR-L **single-model** sobre RDD2022 (4 países, split de val): **mAP@0.5 0.649** (con TTA),
mAP@.5:.95 0.327, **best-F1 0.626**, recall 0.725.

### 3.2 SOTA del concurso (ensembles) — la referencia "alta"
| Solución | F1 | Setup |
|----------|---:|-------|
| **CRDDC'2022 — 1º** | **0.769** | ensemble + TTA + WBF, **test oficial 6 países** |
| CRDDC'2022 — 2º (YOLOv5x P5+P6) | 0.743 | ensemble multiescala |
| ORDDC'2024 — top | ~0.70–0.73 | optimizado precisión+velocidad |

Estas usan **ensembles de varios modelos + pseudo-labeling + el test oficial de 6 países**. No es la
clase de referencia para un modelo único.

### 3.3 SOTA de **modelos transformer / RT-DETR single-model** — la referencia apropiada
| Modelo | Tipo | mAP@0.5 | F1 | Notas |
|--------|------|--------:|---:|-------|
| **Nuestro R+TTA** | **RT-DETR-L (1 modelo)** | **0.649** | **0.626** | 4 países, val-from-train |
| RDD-YOLO (YOLOv8 mejorado) | CNN (1 modelo) | 0.625 | 0.696 | val RDD2022 |
| DETR + ResNet50 | transformer | ~0.635 | — | RDD2022 |
| Variantes YOLO / RT-DETR | mixto | 0.64 – 0.704 | — | rango reportado |
| RT-DETR-Pothole, EF-RT-DETR, Pavement-DETR, RDD-DETR | RT-DETR derivados | ~0.62 – 0.70 | ~0.70 | real-time, backbones livianos |

> Los modelos **transformer single-model** (RT-DETR y derivados) reportan típicamente **mAP@0.5
> ~0.62–0.70 y F1 ~0.70** en road damage. **Nuestro 0.649 cae justo en esa banda.** En F1 (0.626)
> quedamos algo por debajo del ~0.70 de algunos, pero el protocolo de medición difiere (ver §5).

### 3.4 Dónde estamos
- **Como modelo transformer único: competitivos / en rango** con la literatura RT-DETR de road damage.
- **Vs el concurso (0.769): por debajo, pero la diferencia es de *maquinaria*** (ensembles +
  pseudo-labeling + test de 6 países), **no de calidad del modelo base**.

---

## 4. ¿Por qué no alcanzamos el 0.769 del concurso? (síntesis)

1. **La comparación no es válida (lo principal).** Su 0.769 = F1 sobre **test oficial de 6 países**
   (server, etiquetas no públicas). Nosotros: **split de val del train, 4 países**, habiendo quitado
   **China y Noruega** (los difíciles) → tarea no estándar; nuestro número está *inflado* en su
   contexto y no es comparable. Protocolo de métrica distinto.
2. **Un modelo vs maquinaria pesada.** SOTA = **ensembles** (YOLOv5x P5+P6, YOLOv7+coord-attention)
   **+ TTA + WBF + pseudo-labeling** del test no etiquetado. Nosotros: **un RT-DETR** (+TTA) en una
   laptop de 8GB, sin pseudo-labeling (su mayor palanca).
3. **La tarea es dura y dependiente del dominio — lo que el experimento de Perú demuestra.** Un modelo
   fuerte en RDD (0.64) cae a **0.11** en otro país. Objetos pequeños, alta varianza intra-clase,
   **etiquetas ruidosas/inconsistentes entre países**, fuerte dependencia de cámara/dominio. Por eso
   incluso el SOTA (0.77) es "bajo" comparado con benchmarks tipo COCO: el problema es difícil de raíz.
4. **Presupuesto.** Laptop 8GB, val-from-train (menos datos), single-model, sin la fase de
   pseudo-labeling/ensemble que da los últimos puntos.

**En una frase:** no es un problema de modelo —es que (1) no medimos en el mismo terreno que el
concurso, (2) ellos ensamblan + pseudo-etiquetan y nosotros usamos un modelo único, y (3) la tarea es
fuertemente dependiente del dominio (Perú lo deja clarísimo: 0.64→0.11 al cambiar de país).

---

## 5. Caveats de comparabilidad (importante para la tesis)
- Los números de la literatura usan **splits, subconjuntos de países y nº de clases distintos** entre
  sí y respecto a nosotros → las comparaciones de §3 son **orientativas** (clase de referencia), no
  cabeza a cabeza. Para una comparación rigurosa habría que evaluar todos sobre el **mismo split**.
- Reportar siempre nuestro **protocolo explícito** (RT-DETR-L, 4 países, val-from-train 20%, IoU 0.5,
  1 modelo, con/sin TTA) para que el lector entienda el número.
- F1 vs mAP no son intercambiables; el concurso reporta **F1**, gran parte de la literatura reporta
  **mAP@0.5**. Nosotros damos ambos.

---

## Fuentes
- [RDD2022 dataset (Geoscience Data Journal, Arya 2024)](https://rmets.onlinelibrary.wiley.com/doi/10.1002/gdj3.260)
- [CRDDC'2022 (challenge, ensembles, F1 0.769)](https://www.researchgate.net/publication/367456896_Crowdsensing-based_Road_Damage_Detection_Challenge_CRDDC'2022)
- [ORDDC'2024 — State of the art solutions](https://www.researchgate.net/publication/388092879_ORDDC'2024_State_of_the_art_Solutions_for_Optimized_Road_Damage_Detection)
- [RT-DETR-Pothole (lightweight RT-DETR para baches)](https://www.researchgate.net/publication/392434133_RT-DETR-Pothole_Lightweight_Real-Time_Detection_Transformers_for_Improved_Road_Pothole_Detection)
- [EF-RT-DETR (J. Real-Time Image Processing)](https://link.springer.com/article/10.1007/s11554-025-01641-x)
- [Pavement-DETR (RT-DETR para defectos de pavimento)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12031190/)
- [RDD-DETR (real-time end-to-end road damage)](https://link.springer.com/chapter/10.1007/978-981-95-5755-4_27)
- [RDD-YOLO (YOLOv8 mejorado, mAP50 62.5% / F1 69.6%)](https://www.mdpi.com/2076-3417/14/8/3360)
- [YOLOv8-PD (Scientific Reports)](https://www.nature.com/articles/s41598-024-62933-z)
- [Faster R-CNN vs DETR — transfer learning para road damage](https://link.springer.com/chapter/10.1007/978-3-032-10940-8_21)
