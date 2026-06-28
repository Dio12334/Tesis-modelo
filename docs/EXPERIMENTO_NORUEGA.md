# Análisis y experimento — incluir países de alta resolución (Noruega) en el entrenamiento

> Por qué Noruega (excluido) daba mAP bajo, y un experimento para incluirlo correctamente.
> Branch: `experiment/norway-highres-inclusion` (desde `experiment/peru-experiments`). NO mergear a main.
> Fecha: 2026-06-22.

## 1. El hallazgo: Noruega es un outlier de **dimensión**, no (solo) de dominio

Análisis de las dimensiones reales de las imágenes por país (`model/data/rdd2022/train/img`):

| País | n (train) | WxH típico | aspecto (w/h) | downsample → 768 |
|------|----------:|-----------:|--------------:|-----------------:|
| China_Drone | 2401 | 512×512 | 1.00 | 0.67× (se sube) |
| China_MotorBike | 1977 | 512×512 | 1.00 | 0.67× |
| United_States | 4805 | 640×640 | 1.00 | 0.83× |
| Japan | 10506 | 600×600 | 1.00 | 0.78× |
| Czech | 2829 | 600×600 | 1.00 | 0.78× |
| India | 7706 | 720×720 | 1.00 | 0.94× |
| **Noruega** | **8161** | **4040×2035** | **1.99 (2:1)** | **5.26×** |

**Noruega es radicalmente distinta:** ~8 megapíxeles y **ancha 2:1**, vs ~0.3–0.5 MP **cuadradas** del
resto. El pipeline actual hace **resize CUADRADO** (`cv2.resize(img, (768,768))`), así que Noruega sufre
**doble destrucción de información**:

1. **Downsample 5.26×**: una grieta/bache de ~40 px en el original queda en ~8 px → casi irreconocible.
2. **Distorsión de aspecto 2:1 → 1:1**: lo horizontal se comprime a la mitad; las formas se deforman.

> **Conclusión:** el bajo mAP de Noruega es en gran parte un **artefacto del preprocesamiento**, no de
> que el dominio sea intrínsecamente más difícil. La información del daño se destruye **antes** de que
> el modelo la vea. Esto es **arreglable**.

**Importante — Czech NO es lo mismo:** Czech es 600×600 cuadrada (downsample 0.78×, sin distorsión).
Su bajo mAP (0.27) **no** es de dimensión → es por otra causa (menos datos: 2829 img; o dominio/etiquetas).
Es decir, "Noruega y Czech bajos" tienen **causas distintas**.

## 2. Hipótesis
Con un preprocesamiento que **preserve la información** de Noruega, su mAP sube y **incluirla aporta
datos útiles** (8161 imágenes ≈ +30% de datos) en vez de meter ruido. Las técnicas:
- **Letterbox** (redimensionar preservando aspecto + padding gris) → elimina la distorsión 2:1.
- **Mayor resolución** (768 → 1024+) → reduce el downsample (4040→1024 = 3.9× vs 5.26×).
- **(Avanzado) Tiling / SAHI**: partir la imagen 4040×2035 en tiles y detectar por tile → preserva
  el detalle al 100%. Es el método estándar para imágenes de alta resolución.

## 3. Experimento propuesto (per-country eval es la métrica clave)

Todos parten de la receta R (S1 loss + EMA + regularización). Se mide **mAP por país** (sobre todo el
de **Noruega**) y el **mAP global**.

| Run | country_filter | input | preproc | qué prueba |
|-----|----------------|------:|---------|-----------|
| **N0** baseline | JP, IN, US, CZ | 768 | square | referencia (sin Noruega) |
| **N1** naïve | + Noruega | 768 | square | ¿meter Noruega "como está" arrastra el promedio? (confirma el artefacto) |
| **N2** letterbox | + Noruega | 768 | **letterbox** | ¿arreglar la distorsión 2:1 sube el mAP de Noruega? |
| **N3** letterbox+res | + Noruega | **1024** | **letterbox** | + menos downsample → ¿más mejora? (**el experimento principal**) |
| **N4** tiling (futuro) | + Noruega | tiles | SAHI | el techo real para alta resolución |

**Reglas de decisión:**
- **mAP(Noruega) en N2/N3 ≫ N1** → confirma que el problema era el preprocesamiento, no el dominio.
- **mAP global en N3 ≥ N0** → incluir Noruega (bien tratada) **aporta** (más datos, sin daño).
- Si N3 sube Noruega pero baja los demás → trade-off de capacidad/resolución (evaluar).

## 4. Estado del setup
- [x] Análisis de dimensiones por país (este doc).
- [x] **Letterbox implementado y validado** en **train** (`RDD2022TorchDataset`) **y eval**
      (`evaluate_detection`), consistente (test: ambos remapean igual una caja en 2:1). Preserva
      aspecto + ajusta las cajas; como es escala uniforme + pad, **conserva el IoU/mAP**. Flag
      `training.letterbox`. Reutilizable (también ayuda a Perú 16:9).
- [x] Config `model/configs/train_rt_detr_norway_letterbox.yaml` (= N3: + Noruega, 1024, letterbox).
- [x] **N0/N1/N2 ejecutados @768 (laptop, 15 ép)** — ver §6 (Resultados). El runner
      `model/scripts/run_norway_experiments.py` los genera, entrena, evalúa y consolida en JSON.
- [ ] **N3 @1024 (EC2)**: configs generados (`exp_norway_*_1024.yaml`); pendiente de correr en la EC2.
- [ ] N4 (tiling/SAHI): requiere preprocesamiento aparte (futuro).

## 5. Cómo correr (runner reproducible)
El runner genera el config, entrena, evalúa per-país y **acumula** todo en
`logs/norway_experiments/summary.json` (artefacto descargable). Re-ejecutable (salta lo ya evaluado;
resume lo cortado a media corrida).
```bash
# Laptop @768 — N0/N1/N2 (HECHO):
python -m model.scripts.run_norway_experiments --epochs 15 --batch 4

# EC2 @1024 — N3 (referencia sin Noruega 1024 + Noruega square 1024 + Noruega letterbox 1024):
python -m model.scripts.run_norway_experiments --group ec2 --epochs 15 --batch 6 --workers 8
#   -> resultados en logs/norway_experiments/summary.json  (descargar y analizar)
#   solo el principal:  --only N3_norway_letterbox_1024
```

> Nota: a 1024 + Noruega (8k imágenes grandes) en 8 GB hará OOM — N3 es **para la EC2** (~25 GB). El
> grupo `ec2` corre tres puntos a 1024 para una comparación limpia: sin Noruega, +Noruega square y
> +Noruega letterbox (aísla, a 1024, el efecto del letterbox y el de incluir Noruega).

## 6. Resultados (N0/N1/N2 @768, laptop, 15 épocas)

Evaluación sobre el split de validación. mAP@0.5 global, recall y mAP@0.5 per-país.

| Run | preproc | mAP@0.5 global | recall | **Noruega** | Czech | India | Japan | US |
|-----|---------|---------------:|-------:|------------:|------:|------:|------:|---:|
| **N0** (sin Noruega) | square | **0.632** | 0.619 | — | 0.288 | 0.455 | 0.634 | 0.506 |
| **N1** (+Noruega) | square | 0.603 | 0.560 | **0.297** | 0.311 | 0.436 | 0.636 | 0.486 |
| **N2** (+Noruega) | **letterbox** | 0.590 | 0.541 | **0.232** | 0.330 | 0.408 | 0.635 | 0.487 |

**Dos hallazgos:**

1. **Incluir Noruega baja el mAP global con cualquier preprocesamiento** (0.632 → 0.603 → 0.590). Esto
   **valida la exclusión original** de Noruega para el modelo @768.
2. **El letterbox NO recuperó a Noruega: la empeoró** (0.297 → 0.232, −22 %). La hipótesis del §2
   (la distorsión 2:1 era el cuello) queda **refutada a 768 px**.

**Por qué (re-diagnóstico — el cuello es resolución, no distorsión).** La cuenta de píxeles lo explica:

- *Resize cuadrado (N1):* 4040×2035 → llena 768×768. Distorsiona (aplasta horizontal) pero el contenido
  vertical baja solo 2.65× (2035→768).
- *Letterbox (N2):* preserva el 2:1 → encoge a 768×**384** + barras grises. **Desperdicia media imagen**
  en padding y el contenido vertical baja 5.26× (2035→384).

Es decir, a presupuesto de píxeles fijo, el letterbox compra "no distorsión" **a costa de resolución**;
y como el daño fino de Noruega depende de la resolución (su recall cae 0.377 → 0.308), el cambio es
net-negativo. **Prueba cruzada:** en los países cuadrados (~1:1, donde letterbox ≈ square) el efecto es
neutro (Japan/US planos, Czech +0.02, India −0.03) → la implementación del letterbox es correcta y el
golpe se concentra en la única imagen 2:1 (Noruega). El cuello real es el **downsample 5.26×** que
destruye el daño antes de que el modelo lo vea.

**Conclusión y siguiente paso.** El letterbox solo puede pagar **si va con más resolución**. Por eso N3
(letterbox + **1024**, downsample 3.9× en vez de 5.26×) es la prueba decisiva, y por encima N4
(tiling/SAHI), el método estándar para imágenes de 8 MP. A 768 la laptop no puede dar a Noruega los
píxeles que necesita con ningún preprocesamiento. Mientras tanto, **Noruega se mantiene excluida** del
modelo de referencia @768.

> Reproducir: `logs/norway_experiments/summary.json` (consolidado del runner). N3 se corre en la EC2
> con `--group ec2` (§5) y vuelca su resultado al mismo JSON para descargar y comparar contra esta tabla.
