# Arquitectura del proyecto — Road Damage Evaluation Framework

> Documento de referencia para análisis posteriores. Describe la estructura,
> los flujos de datos y las interfaces clave del proyecto de tesis (detección
> de daños viales sobre RDD2022). Última actualización: 2026-06-15.

---

## 1. Visión general

El proyecto entrena y evalúa **modelos de detección de objetos** para identificar
daños en carreteras usando el dataset **RDD2022**. Está organizado en dos grandes
componentes:

1. **`model/`** — paquete Python con el framework de entrenamiento/evaluación/inferencia.
   Es agnóstico al modelo: todos los detectores se entrenan a través de una única
   interfaz (`BaseDetector`) y un único bucle (`train_detection.train`).
2. **`dashboard/`** — app Streamlit para visualizar métricas, pérdidas, matrices de
   confusión y predicciones de los runs de entrenamiento.

El framework soporta varios modelos (RT-DETR, YOLO26, MMR-DETR, SSD-MobileNetV3,
MobileNetV4-SSD, YOLOv6) y un **clasificador binario** (daño / fondo) que habilita
un **pipeline de dos etapas** (filtrar imágenes de fondo antes de detectar).

### Hardware/entorno de referencia
- **Entorno conda: `tesis-modelo`** ⚠️ (no usar el Python base ni el otro env `roaddmg`,
  que trae torch 2.6.0+cu124 distinto).
  - Python **3.12.13**, PyTorch **2.12.0+cu126**, `ultralytics` y `cv2` instalados.
  - Intérprete: `C:\Users\Diego\miniconda3\envs\tesis-modelo\python.exe`
- GPU: **NVIDIA RTX 4070 Laptop (8.0 GB VRAM)**, CUDA disponible.
- SO: **Windows 11** (relevante: `DataLoader` usa contexto `spawn`).
- Dataset train: **38.385 imágenes** (~281 KB c/u, 512×512 y otros tamaños por país).

---

## 2. Mapa de módulos

| Ruta | Responsabilidad |
|------|-----------------|
| `model/__main__.py`, `model/cli.py` | Entry point CLI (`python -m model ...`): subcomandos `train`, `evaluate`, `predict`, `train-classifier`, `evaluate-classifier`, `list-models`. |
| `model/config/manager.py` | Carga, merge, validación y resolución de `${ENV}` de YAML. |
| `model/config/schema.py`, `model/configs/*.yaml` | Esquemas y configuraciones de entrenamiento por modelo. |
| `model/datasets/base.py` | `BaseDataset`, `Annotation`, `BoundingBox` (coords normalizadas [0,1]). |
| `model/datasets/rdd2022.py` | Carga RDD2022 (Supervisely JSON **o** PASCAL VOC XML, autodetectado), filtro por país, `split()`. |
| `model/datasets/target_mapper.py` | Mapeo de etiquetas de clase. |
| `model/models/registry.py` | `BaseDetector` (interfaz) + `ModelRegistry` (factory por decorador). |
| `model/models/*_wrapper.py`, `ssd_mobilenet.py`, `mobilenetv4_ssd.py`, `binary_classifier.py` | Wrappers concretos por modelo. Implementan `train_step`, `forward`, `get_parameters`, `save/load_checkpoint`. |
| `model/models/two_stage_detector.py` | Detector de inferencia en dos etapas (clasificador + detector). **No entrenable** (`train_step` lanza `NotImplementedError`). |
| `model/training/train_detection.py` | **Bucle de entrenamiento REAL** (PyTorch) para detección. ⭐ |
| `model/training/train_classifier.py` | Entrenamiento del clasificador binario. |
| `model/training/augmentation.py` | Transforms de augmentación por imagen (flip, HSV, scale, translate). Mosaic/MixUp se hacen a nivel de dataset. |
| `model/training/evaluate_detection.py` | Evaluación de detección post-entrenamiento (mAP, F1-sweep, per-region). |
| `model/training/loss.py`, `callbacks.py`, `pipeline.py` | **Sistema legacy/simulado** (ver §4). |
| `model/evaluation/` | `engine.py`, `metrics.py`, `region.py`, `report.py` — motor de métricas. |
| `model/inference/pipeline.py` | Inferencia sobre imágenes/directorios + guardado anotado. |
| `model/tracking/tracker.py` | `ExperimentTracker`: un JSON por run con config + `metrics_history` + `final_results`. |
| `model/scripts/`, `model/tools/` | Conversión de datos, auditoría de pesos, diagnóstico de mosaic, etc. |
| `dashboard/src/` | App Streamlit + componentes de visualización. |

---

## 3. Interfaces clave

### `BaseDetector` (`model/models/registry.py:13`)
Contrato común de todos los detectores. Métodos relevantes para entrenamiento:
- `train_step(images, targets) -> {"loss_tensor": Tensor}` — forward + cálculo de pérdida.
- `forward(images) -> List[dict]` — inferencia, devuelve `boxes/labels/scores` por imagen.
- `get_parameters()` — parámetros para el optimizador (puede devolver *param groups* con LR discriminativos).
- `set_train_mode()`, `set_eval_mode()`, `to_device()`.
- `save_checkpoint(path)`, `load_checkpoint(path)`.

> Los wrappers estilo Ultralytics (RT-DETR, YOLO26) guardan el `nn.Module` real en
> `self._model.model`. El bucle de entrenamiento detecta este patrón por
> *duck-typing* (`hasattr(model, "_model")`).

### `ModelRegistry` (`model/models/registry.py:117`)
Registro singleton. Los wrappers se registran con `@ModelRegistry.register("nombre")`
y se importan en `model/models/__init__.py`. Instanciación: `ModelRegistry.create(type, config)`,
que valida el config contra el *schema* del modelo antes de construirlo.

### `BaseDataset` / `RDD2022Dataset`
- `load(path)` autodetecta formato (Supervisely JSON vs VOC XML) y parsea **todas** las
  anotaciones al iniciar.
- `split(train, val, test, seed)` baraja índices con `random.Random(seed)` y devuelve
  tres datasets nuevos.
- `get_annotations()` devuelve `List[Annotation]`; cada `Annotation` tiene `image_path`,
  `bounding_boxes` (normalizadas) y `metadata` (incluye `country`).

---

## 4. ⚠️ Dos sistemas de entrenamiento (importante)

El repo contiene **dos rutas de entrenamiento** y conviene no confundirlas:

| | Real (productivo) | Legacy / simulado |
|---|---|---|
| Archivo | `model/training/train_detection.py` | `model/training/pipeline.py` + `callbacks.py` |
| ¿Usa PyTorch? | **Sí** (forward/backward reales, AMP, GPU) | **No** — `_compute_batch_loss` genera una curva sintética con `2·exp(−3·progress)+0.1` (`pipeline.py:558-593`) |
| Checkpoints | `.pt` (pesos + optimizador) | `.json` (solo metadatos) |
| Invocado por | `cli.handle_train` → `train_detection.train` | Nada del flujo productivo; sobre todo tests |

**Para análisis de rendimiento, el único bucle relevante es `train_detection.py`.**
`pipeline.py`/`callbacks.py` no ejecutan cómputo real y pueden ignorarse (o retirarse).

---

## 5. Flujo de entrenamiento real (`train_detection.train`)

Entrada: `python -m model.training.train_detection --config <yaml> [--resume <id|.pt>]`

```
1. ConfigManager.load() + resolve_env_vars()            (config/manager.py)
2. Semilla, device = cuda|cpu, cudnn.benchmark=True      (train_detection.py:734-749)
3. ModelRegistry.create(type, cfg) → wrapper             (:760)
4. model → .to(device)                                   (:769-793)
   ├─ (opcional) channels_last                           (:798-808)
   └─ (cuda) torch.compile(mode="reduce-overhead")       (:810-821)
5. RDD2022Dataset().load(path)                           (:824-827)  ← parsea 38k JSON
6. dataset.split(1-val, val, 0)                          (:830-831)
7. build_augmentation_pipeline(aug_cfg)                  (:834-836)
8. RDD2022TorchDataset(train/val, mosaic, mixup, ...)    (:846-850)  ← adaptador torch
9. DataLoader(train/val, num_workers, pin_memory, ...)   (:861-882)
10. Optimizador (SGD/Adam/AdamW/MuSGD) desde get_parameters()  (:884-927)
11. CosineAnnealingLR (T_max=epochs-warmup)              (:930-932)
12. GradScaler(AMP)                                       (:935)
13. SIGINT handler (1ª=fin de época, 2ª=abort)           (:942-953)
14. ExperimentTracker.start_run()                        (:956)
15. Bucle de épocas ↓
```

### Bucle por época (`:1020-1211`)
```
Para cada época:
  - Si quedan ≤ mosaic_off_epochs → desactiva Mosaic     (:1024-1030)
  - Warmup lineal de LR si epoch < warmup_epochs         (:1033-1039)
  TRAIN (model.set_train_mode):
    Para cada batch (images, targets) del train_loader:
      images  = [img.to(device) for img in images]       (:1048)  ← sin non_blocking
      targets = [{k: v.to(device) ...} ...]              (:1049)
      optimizer.zero_grad()
      with autocast('cuda'): loss = model.train_step(...)(:1055-1059)
      scaler.scale(loss).backward()                      (:1076)
      scaler.unscale_(); clip_grad_norm_(max_norm=10)    (:1079-1081)
      scaler.step(); scaler.update()                     (:1084-1085)
  - cosine_scheduler.step() (si epoch ≥ warmup)          (:1099-1100)
  VAL (model.set_eval_mode, torch.no_grad):
    Para cada batch del val_loader: loss = model.train_step(...)  (:1110-1127)
  - tracker.log_metrics()                                (:1147)  ← reescribe JSON completo
  - Si val_loss mejora → best_model.pt                   (:1158-1171)
  - Cada 5 épocas → recovery.pt                          (:1176-1186)
  - SIEMPRE → training_state.pt (~286 MB en RT-DETR)     (:1188-1197)  ← I/O por época
  - Early stopping si epochs_without_improvement ≥ N     (:1200-1206)
Final: final_model.pt + tracker.end_run()
```

### Adaptador `RDD2022TorchDataset` (`train_detection.py:47`)
Convierte `Annotation` → `(image_tensor[C,H,W], target_dict{boxes, labels})`.
- `__getitem__` (`:411`) decide entre ruta **mosaic+mixup**, **augmentación simple** o **directa**.
- `_load_image_and_bboxes` (`:157`): `PIL.open → np.array → cv2.resize(input_size)`.
  **No hay caché**: cada acceso vuelve a leer y decodificar desde disco.
- `_build_mosaic` (`:195`): carga **4 imágenes**, las pega en un lienzo 2S×2S y aplica
  un *random affine* que lo reduce a S×S (estilo Ultralytics). Los compañeros se muestrean
  solo del *pool* de imágenes no vacías (`_nonempty_indices`).
- `collate_fn` (`:531`): mantiene `images` y `targets` como **listas** (nº de cajas variable).

### Cómputo de pérdida (`train_step`)
Para RT-DETR (`rt_detr_wrapper.py:536`) y YOLO26 (`yolo26_wrapper.py:524`):
1. `torch.stack(images)` → batch `(B,C,H,W)`.
2. Convierte cajas `xyxy` (pixeles) → `xywh` normalizado, formato batch de Ultralytics
   (`batch_idx, cls, bboxes`).
3. Llama `model_module.loss(batch_dict)`, que hace forward + criterio internamente.
4. Devuelve `{"loss_tensor": loss_escalar}`.

> Nota: `train_step` llama `model_module.train()` **incondicionalmente** al inicio
> (`yolo26_wrapper.py:553`, `rt_detr_wrapper.py:565`), lo que afecta la fase de validación
> (ver `ANALISIS_CUELLOS_DE_BOTELLA.md`, hallazgo P1-5).

---

## 6. Configuración

YAML por modelo en `model/configs/`. Estructura típica (`train_rt_detr.yaml`):
```yaml
model:    { type, config: { model_size, input_size, num_classes, backbone_layers, ... } }
dataset:  { type, path, country_filter, class_mapping }
training: { epochs, batch_size, num_workers, use_amp, learning_rate, optimizer,
            warmup_epochs, val_split, checkpoint_dir, early_stopping_patience,
            seed, prefetch_factor, use_channels_last, loss:{...}, augmentation:{...} }
evaluation: { iou_thresholds, confidence_threshold, confidence_thresholds_sweep }
output:   { checkpoint_dir, results_dir, log_dir }
```

Diferencias relevantes entre configs (impactan rendimiento):

| Config | batch | workers | prefetch | input | mosaic | mixup |
|--------|------:|--------:|---------:|------:|-------:|------:|
| `train_rt_detr.yaml` | 8 | 4 | 2 (def.) | 640 | 0.0 | 0.0 |
| `train_yolo26.yaml`  | 32 | 8 | 3 | 640 | 0.5 | 0.05 |

> ⚠️ Algunas claves de config **no se usan** en `train_detection.py` (p. ej.
> `scheduler`, `eval_interval`, `early_stopping_metric: map`,
> `evaluation.confidence_thresholds_sweep`). El *early stopping* y el *best checkpoint*
> se rigen por **`val_loss`**, no por mAP. Ver hallazgo P2-9.

---

## 7. Pipeline de dos etapas (`two_stage.yaml`)

Solo **inferencia/evaluación** (no entrenamiento conjunto):
1. **Etapa 1** — `BinaryDamageClassifier` (EfficientNet-B2 @ 260px) clasifica imagen como
   *daño* o *fondo*; descarta fondos con `classifier_threshold`.
2. **Etapa 2** — el detector (p. ej. RT-DETR) corre **solo** sobre el subconjunto marcado
   como dañado (`two_stage_detector.py:97-156`).

Cada componente se entrena por separado (clasificador con `train_classifier.py`, detector
con `train_detection.py`) y se ensamblan vía checkpoints en el config.

---

## 8. Evaluación, tracking y dashboard

- **Evaluación**: `model/training/evaluate_detection.py` (CLI con `--config`, `--run-id`,
  `--split`, `--dataset`) calcula mAP@{0.25,0.5,0.75}, barrido de F1 por umbral de
  confianza y métricas **por región/país** (`model/evaluation/region.py`).
- **Tracking**: `ExperimentTracker` escribe `<checkpoint_dir>/<run_id>.json` con config,
  `metrics_history` (una entrada por época) y `final_results`.
- **Dashboard**: `python -m streamlit run dashboard/src/app.py` lee esos JSON y los
  checkpoints para mostrar curvas de pérdida, métricas, matriz de confusión, comparación
  de runs y visor de predicciones.

---

## 9. Comandos habituales (`command.md`)

> Ejecutar siempre con el entorno activado: `conda activate tesis-modelo`.

```bash
# Entrenar
python -m model.training.train_detection --config model/configs/train_yolo26.yaml -v
python -m model.training.train_detection --config model/configs/train_rt_detr.yaml -v
python -m model.training.train_detection --config <yaml> --resume <run_id|ruta.pt>

# Evaluar
python -m model.training.evaluate_detection --config <yaml> --run-id <id> --split val \
       --dataset model/data/rdd2022/complete

# Dashboard
python -m streamlit run dashboard/src/app.py
```

---

## 10. Punteros para análisis futuros

- **Rendimiento de entrenamiento** → `docs/ANALISIS_CUELLOS_DE_BOTELLA.md`.
- El bucle caliente está en `train_detection.py:1046-1127` (train + val) y en
  `RDD2022TorchDataset.__getitem__` (`:411`).
- Para perfilar I/O de checkpoints, mirar `_save_training_state` (`:592`) y los tamaños
  reales en `checkpoints/<modelo>/<run_id>/training_state.pt`.
- Para entender el coste de datos, `_load_image_and_bboxes` (`:157`) y `_build_mosaic`
  (`:195`) son los puntos clave (sin caché, decodificación por época).
