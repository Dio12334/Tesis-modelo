# Bimodelo: Pipeline de Detección en Dos Etapas

## Motivación

El dataset RDD2022 contiene aproximadamente un 33 % de imágenes sin daño alguno (background puro). Ejecutar el modelo de detección sobre estas imágenes es computacionalmente costoso e introduce falsos positivos innecesarios. El sistema **bimodelo** introduce un clasificador binario liviano como filtro previo:

```
Imagen → [Clasificador Binario] → ¿Hay daño?
                                     │
                    NO ──────────────┘   → predicciones vacías (background)
                    SÍ → [Detector RT-DETR / MMR-DETR] → bounding boxes
```

---

## Arquitectura

### Etapa 1: `BinaryDamageClassifier`

| Aspecto | Detalle |
|---------|---------|
| **Backbone** | EfficientNet-B0 (timm, ImageNet-pretrained) |
| **Cabeza** | `Linear(1280, 1)` — un solo logit por imagen |
| **Activación final** | Sigmoid → probabilidad de daño `p ∈ [0, 1]` |
| **Decisión** | `p ≥ threshold` → clase 1 (damage) |
| **Parámetros** | ~5.3M |
| **Input size** | 224×224 (configurable) |

Las etiquetas binarias se derivan automáticamente de RDD2022:
- Imagen con ≥1 anotación → **1 (damage)**
- Imagen sin anotaciones → **0 (background)**

### Etapa 2: Detector de objetos

Cualquier modelo registrado en `ModelRegistry` puede usarse como detector:
`rt_detr`, `mmr_detr`, `yolo26`, `ssd_mobilenetv3`, `mobilenetv4_ssd`.

El detector solo procesa las imágenes que la etapa 1 marcó como dañadas.

---

## Archivos creados

| Archivo | Descripción |
|---------|-------------|
| `model/models/binary_classifier.py` | Clase `BinaryDamageClassifier` |
| `model/models/two_stage_detector.py` | Clase `TwoStageDetector` (implementa `BaseDetector`) |
| `model/training/train_classifier.py` | Script de entrenamiento del clasificador |
| `model/configs/train_binary_classifier.yaml` | Config de entrenamiento del clasificador |
| `model/configs/two_stage.yaml` | Config de inferencia dos etapas |

---

## Flujo de uso

### 1. Crear la rama

```bash
git checkout -b bimodelo
```

### 2. Entrenar el clasificador binario

```bash
python -m model train-classifier --config model/configs/train_binary_classifier.yaml
```

El script:
- Carga RDD2022 y deriva etiquetas binarias automáticamente
- Usa un `WeightedRandomSampler` para manejar el desbalance de clases (~67% daño / ~33% background)
- Pérdida: `BCEWithLogitsLoss` con `pos_weight` calculado según la proporción real del split train
- Guarda `best_model.pt` (mejor AUC-ROC en validación) y `last_model.pt` por época
- Early stopping configurable por `early_stopping_patience`

Checkpoints se guardan en `./checkpoints/binary_classifier/`.

### 3. Configurar la inferencia en dos etapas

Editar `model/configs/two_stage.yaml` con los paths reales:

```yaml
model:
  config:
    classifier_checkpoint: ./checkpoints/binary_classifier/best_model.pt
    detector_checkpoint: ./checkpoints/rt_detr/<run_id>/best_model.pt
    detector_type: rt_detr
    classifier_threshold: 0.4   # ajustar según trade-off deseado
```

### 4. Inferencia

```bash
python -m model predict \
  --config model/configs/two_stage.yaml \
  --input ruta/a/imagenes/ \
  --output ruta/resultados/
```

### 5. Evaluación

```bash
python -m model.training.evaluate_detection model/configs/two_stage.yaml \
  --checkpoint placeholder   # TwoStageDetector ignora este argumento
```

---

## Trade-off del threshold del clasificador

El `classifier_threshold` controla el balance entre recall y throughput:

| Threshold | Efecto |
|-----------|--------|
| `0.3` | Muy sensible: casi ningún daño se pierde, pero más imágenes llegan al detector (más cómputo) |
| `0.4–0.5` | Balance razonable para la mayoría de los casos |
| `0.6–0.7` | Más eficiente: menos imágenes al detector, pero se arriesga a perder daños leves |

Para un sistema de monitoreo de carreteras donde no se puede perder daño, usar `0.3–0.4`. Para un sistema de triaje donde la velocidad importa, usar `0.6`.

---

## Evaluación del sistema conjunto

Para medir el impacto del clasificador sobre las métricas finales del detector:

1. Evaluar el detector solo (sin filtro) con `evaluate_detection.py` → obtener mAP y F1 de referencia.
2. Evaluar el sistema dos etapas con el mismo set de validación.
3. Comparar:
   - **Recall del clasificador** (¿cuántas imágenes con daño pasan al detector?)
   - **mAP del sistema conjunto** (¿el filtro introduce falsos negativos nuevos?)
   - **Throughput** (imágenes/segundo) antes y después del filtro.

Un clasificador bien calibrado debería aumentar el throughput sin degradar el mAP significativamente.

---

## Métricas esperadas

Con EfficientNet-B0 fine-tuneado sobre RDD2022 (~30K imágenes de entrenamiento):
- **AUC-ROC esperado**: > 0.90 (la tarea es binaria y relativamente sencilla)
- **Accuracy esperada**: > 85 % en el punto de operación `threshold=0.5`
- **Tiempo de inferencia por imagen**: ~5 ms en GPU (vs. ~30 ms del detector)

---

## Extensiones posibles

- **Threshold automático**: buscar el threshold que maximiza el recall del clasificador con una restricción de precisión mínima (e.g. precisión ≥ 0.95 → se asegura que el 95 % de las predicciones "daño" realmente lo son).
- **Ensemble**: combinar la probabilidad del clasificador con el score de confianza del detector para re-rankear predicciones finales.
- **Otros backbones**: reemplazar EfficientNet-B0 por MobileNetV3-Small para mayor velocidad, o por EfficientNet-B3 para mayor precisión.
