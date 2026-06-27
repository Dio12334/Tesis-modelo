# Análisis de cuellos de botella en el entrenamiento

> Objetivo: identificar dónde se pierde tiempo/throughput en el bucle de
> entrenamiento real (`model/training/train_detection.py`) y proponer correcciones
> priorizadas. Complementa `docs/ARQUITECTURA.md`. Fecha: 2026-06-15.

---

## 0. Resumen ejecutivo (corregido con medición — 2026-06-15)

> ⚠️ La sección §0.bis (medición empírica) **reordena** las prioridades de este doc.
> El análisis inicial por lectura de código sospechaba de I/O y datos; la medición
> demuestra que el entrenamiento es **100 % limitado por cómputo de GPU**. Lee §0.bis
> antes que la tabla de abajo.

**Veredicto medido**: con RT-DETR-L @640, batch 8, en la RTX 4070 Laptop, el tiempo se
reparte **0.1 % datos / 99.9 % cómputo** (416 ms/batch). No hay *data-starvation* ni
stalls de I/O relevantes. Por tanto **los hallazgos de datos/IO (P0-2, P0-3, P1-7) NO son
los cuellos** en esta máquina; quedan como mejoras menores o preventivas (p. ej. si se
sube `mosaic` o se cambia de disco). El coste real es la **inferencia+backward del modelo**.

### Tabla original (análisis por lectura de código) — re-priorizada por la medición

| # | Prioridad **real** | Hallazgo | Ubicación | Estado tras medir |
|---|-----------|----------|-----------|---------|
| P0-1 | 🟡 Baja* | `training_state.pt` (~286 MB) escrito **síncrono cada época** | `train_detection.py` (save state) | *Era P0 por código; medido ≈1-2 s/época de ~1075 s → <0.2 %. **Corregido igual** (gratis). |
| P0-2 | ⚪ Descartado | Sin caché de imágenes (decodifica JPEG cada época) | `train_detection.py:157-188` | Datos = 0.1 % del tiempo → **no aporta**. No tocar. |
| P0-3 | 🟢 Aplicado | Transferencias H2D sin `non_blocking` | `train_detection.py` (train/val loop) | Gratis; **aplicado** aunque el impacto sea ~0 aquí. |
| P1-4 | 🟢 Aplicado | `torch.compile` + batch variable, sin `drop_last` | `train_detection.py` | **`drop_last` aplicado**; además `compile` medido **0 % de mejora** (sin triton) → ahora opt-in. |
| P1-5 | 🟠 Media | **Validación corre en modo `train()`** (bug) | `*_wrapper.py` (`train_step`) | Correctness; afecta `val_loss`/early-stop. Pendiente (cambia resultados). |
| P1-6 | 🟡 Baja | Tracker reescribe el JSON completo cada época | `tracker.py:52-64` | I/O menor; pendiente. |
| P1-7 | ⚪ Descartado | `num_workers`/`prefetch` bajos | `train_rt_detr.yaml` | Datos no son el cuello → irrelevante ahora. |
| P2-9 | 🟡 Baja | Claves de config muertas (`scheduler`, `eval_interval`, `map`) | configs | Pendiente (limpieza). |
| P2-11 | 🟠 Media | **Carga del dataset = 297 s (~5 min)** al arrancar | `rdd2022.py:120-144` | Peor de lo estimado; coste de arranque por run. Pendiente. |

\* "Baja" en *throughput* de estado estable, pero su corrección reduce ~28 GB de escritura
por run y el desgaste de SSD, así que se mantiene.

---

## 0.bis — Medición empírica (la parte que importa)

Reproducible con los scripts añadidos en `model/tools/`:

```bash
# 1) ¿Datos o cómputo? (reconstruye el hot-loop real)
python -m model.tools.profile_training_bottleneck --config model/configs/train_rt_detr.yaml
# 2) Coste por batch de cada variante de optimización (datos sintéticos)
python -m model.tools.profile_compute_variants     --config model/configs/train_rt_detr.yaml
```

**Resultado (1) — reparto datos vs cómputo:**
```
t_data (espera datos)  :   0.4 ms/batch  ( 0.1%)
t_step (cómputo GPU)   : 415.8 ms/batch  (99.9%)   →  LIMITADO POR CÓMPUTO (GPU)
batches/época (train)  : 2585   |   época train estimada: 17.9 min  (≈ los ~20 min reales)
carga del dataset      : 297 s  (parseo de ~26k JSON con country_filter)
```

**Resultado (2) — coste por batch por variante (RT-DETR-L @640, batch 8, AMP):**
```
eager                 : 411.5 ms/batch   1.00x
channels_last         : 900.9 ms/batch   0.46x   ← 2.2x MÁS LENTO (decoder transformer)
compile               : 410.8 ms/batch   1.00x   ← 0 % de mejora
compile+channels_last : 902.9 ms/batch   0.46x
```

**Conclusiones:**
1. **`channels_last` PERJUDICA** (2.2× más lento). No activar `use_channels_last` con RT-DETR.
2. **`torch.compile` no acelera RT-DETR — ni siquiera con triton.** Primero medimos sin
   triton (no instalado) → 0 % de mejora. Tras **instalar `triton-windows 3.7.0.post26`**
   (verificado funcionando: compila kernels vía MSVC, `has_triton()=True`), re-medimos con
   **shapes variables** (nº de cajas aleatorio, como el dataset real) y control de ruido:
   ```
                 min     mediana
   eager       384-393   400-404
   compile       392       403
   speedup: 0.98x (min) / 1.00x (mediana)  → dentro del ruido (deriva eager 2.4%)
   ```
   El decoder transformer + el conteo variable de cajas rompen la captura de grafo; no hay
   ganancia. Se deja **opt-in** (`use_torch_compile: false`). Triton queda instalado (inocuo;
   podría ayudar a modelos más CNN como YOLO26 — sin verificar aún).
3. El cuello es el **forward+backward del propio RT-DETR-L**, irreducible sin trade-off. Para
   acelerarlo hay que **reducir cómputo** (resolución/modelo/batch). Ver §5.

---

## 1. Contexto (para dimensionar el impacto)

- **Entorno**: conda `tesis-modelo` — Python 3.12.13, **PyTorch 2.12.0+cu126**, ultralytics + cv2.
- **GPU**: RTX 4070 Laptop, **8 GB VRAM** → margen ajustado para RT-DETR-L @640, batch 8.
- **Dataset**: 38.385 imágenes train; con `val_split=0.2` → ~30.700 train / ~7.700 val.
- **Batches/época** (RT-DETR, batch 8): ~3.840 train + ~960 val.
- **Tamaño real de checkpoints medido**:
  - `training_state.pt` (RT-DETR) = **~286 MB** (pesos + estado AdamW + scheduler + scaler).
  - `final_model.pt` = ~132 MB (solo pesos).
- **SO Windows** → `DataLoader` con `multiprocessing_context="spawn"` (`train_detection.py:855`).

---

## 2. Hallazgos detallados

### 🔴 P0-1 — Checkpoint de ~286 MB escrito de forma síncrona cada época

**Dónde**: `train_detection.py:1188-1197` (`_save_training_state`, def. en `:592`).

```python
# Training state (always, for resume): overwritten each epoch
_save_training_state(run_checkpoint_dir / "training_state.pt", model, optimizer, ...)
```

**Problema**: en **cada** época se serializa y escribe a disco el estado completo
(modelo + optimizador AdamW, que duplica el nº de parámetros en *momentos*). Medido:
**~286 MB por época** para RT-DETR. La escritura ocurre en el hilo principal, justo
después de la validación, por lo que la **GPU queda ociosa** mientras `torch.save`
serializa y vuelca a disco. En 100 épocas son **~28 GB de escritura por run** (desgaste
de SSD y, en disco lento, segundos por época).

Además se suma `best_model.pt` (132 MB cuando mejora) y `recovery.pt` (cada 5 épocas).

**Solución propuesta** (cualquiera, combinables):
- Guardar `training_state.pt` **cada N épocas** (p. ej. 5), no en todas. El *resume* fino
  por época rara vez es necesario.
- Volcar en un **hilo en segundo plano** (`torch.save` a un buffer + `threading.Thread`),
  para solapar con el inicio de la época siguiente.
- Escribir a archivo temporal y `os.replace` (atómico) para no corromper en interrupción.

> Quick win: cambiar la guarda a `if (epoch + 1) % save_state_interval == 0 or last_epoch`.

---

### 🔴 P0-2 — Sin caché: 38 k imágenes decodificadas desde disco en cada época

**Dónde**: `train_detection.py:157-188` (`_load_image_and_bboxes`) y las ramas de
`__getitem__` (`:451-515`).

```python
image = Image.open(annotation.image_path).convert("RGB")   # decodifica cada vez
image_np = np.array(image)
... cv2.resize(image_np, (input_size, input_size))          # redimensiona cada vez
```

**Problema**: no existe ninguna caché (`grep cache` → 0 ocurrencias). Cada época vuelve
a leer y **decodificar el JPEG** y a redimensionarlo. Son ~30.700 decodificaciones/época
× nº de épocas. La decodificación JPEG + `cv2.resize` es trabajo **CPU-bound** que recae
en los *workers*; si no alcanzan a alimentar la GPU, esta se queda esperando
(*data-starvation*).

**Agravante con Mosaic** (configs YOLO: `mosaic=0.5`): cada muestra con mosaic carga
**4 imágenes** (`_build_mosaic`, `:255-256`). Con p=0.5 el factor medio de I/O es
≈ `0.5·4 + 0.5·1 = 2.5×`. Con MixUp se añade alguna más.

**Solución propuesta**:
- **Caché en RAM de imágenes ya redimensionadas a `input_size` en `uint8`**: a 512² la
  imagen pesa ~786 KB; las 38 k ocupan ~30 GB (demasiado). Pero **a `input_size` real**
  (la red usa 640 o menos) y, sobre todo, cacheando **solo el split de train** o un
  subconjunto, es viable. Alternativa: `input_size=320` reduce a ~300 KB/img.
- **Pre-redimensionar una vez a disco** (carpeta `*_resized/`) y leer PNG/JPEG pequeños.
- Estilo Ultralytics: opción `cache="ram"|"disk"` con array `uint8` compartido.
- Reemplazar **PIL por `cv2.imread`** (suele ser más rápido) o `PyTurboJPEG`; ojo al orden
  BGR↔RGB.
- Mantener **`persistent_workers=True`** (ya está) para no recrear workers por época.

---

### 🔴 P0-3 — Transferencias host→device sin `non_blocking` y elemento por elemento

**Dónde**: `train_detection.py:1048-1049` (train) y `1112-1113` (val).

```python
images  = [img.to(device) for img in images]                       # 1 copia por imagen
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
```

**Problema**: `pin_memory=True` está activo en los loaders (`:866`, `:877`), pero las
copias usan `non_blocking=False` (por defecto) → **no hay solape** entre la copia
H2D y el cómputo. Además se copia **imagen por imagen** en un bucle Python en lugar de
un único tensor `(B,C,H,W)`. El beneficio de la *pinned memory* se desperdicia.

**Solución propuesta**:
```python
images = torch.stack(images).to(device, non_blocking=True)   # 1 sola copia asíncrona
targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
```
(Los `train_step` de RT-DETR/YOLO ya hacen `torch.stack` internamente; conviene mover el
*stack* al loop y entregar el tensor ya apilado, eliminando también el doble trabajo —
ver P2-8.) Requiere que el `collate_fn` o el loop apile; como las imágenes son todas
`input_size×input_size`, `torch.stack` es válido.

---

### 🟠 P1-4 — `torch.compile(mode="reduce-overhead")` con formas variables

**Dónde**: `train_detection.py:810-821`.

**Problema**: `mode="reduce-overhead"` usa **CUDA graphs**, muy sensibles a cambios de
forma. Dos fuentes de variación invalidan el grafo y fuerzan recompilaciones costosas:
1. El **último batch** de cada época tiene tamaño < `batch_size` (no hay `drop_last`).
2. El número de cajas (`targets`) varía por batch → tensores de pérdida de tamaño variable.

También RT-DETR puede no compilar limpio (de hecho hay `try/except` defensivo).

**Solución propuesta**:
- Añadir **`drop_last=True`** al `train_loader` (`:861`) para fijar el tamaño de batch.
- Evaluar `mode="default"` o `dynamic=True`; **medir** si `compile` realmente ayuda en
  este modelo/GPU (a veces el overhead de recompilación supera la ganancia).
- Considerar `torch.compile` solo del *backbone* si el resto es muy dinámico.

---

### 🟠 P1-5 — La validación se ejecuta en modo `train()`

**Dónde**: `train_detection.py:1106` hace `model.set_eval_mode()`, pero
`train_step` llama `model_module.train()` incondicionalmente
(`yolo26_wrapper.py:553`, `rt_detr_wrapper.py:565`), revirtiéndolo en el primer batch.

**Problema**: durante la validación, **BatchNorm actualiza sus *running stats*** y
**dropout queda activo**, lo que (a) contamina las estadísticas del modelo con datos de
validación y (b) hace la `val_loss` ruidosa/no comparable. No es el mayor coste de
tiempo, pero sí un **bug de correctness** que afecta el *early stopping* y el *best
checkpoint* (ambos basados en `val_loss`).

**Solución propuesta**: separar el cálculo de pérdida del cambio de modo. Opciones:
- Que `train_step` **no** fuerce `.train()` (que el modo lo gestione el loop), o
- Añadir un `eval_step`/parámetro `training: bool` a `train_step`.

---

### 🟠 P1-6 — `ExperimentTracker.log_metrics` reescribe el JSON completo cada época

**Dónde**: `tracker.py:52-64` (llamado en `train_detection.py:1147`).

```python
run_data = self._load_run(run_id)          # lee TODO el JSON
run_data["metrics_history"].append(entry)
self._save_run(run_id, run_data)           # reescribe TODO el JSON
```

**Problema**: cada época lee y reescribe el archivo completo del run. Conforme crece
`metrics_history`, el coste por época crece → **O(n²)** acumulado. Para 100 épocas es
tolerable, pero es I/O y serialización innecesarios en el camino caliente.

**Solución propuesta**: *append* a un **JSONL** (`metrics.jsonl`, una línea por época) o
mantener el historial en memoria y volcar el JSON solo al final / cada N épocas. El
dashboard puede leer el JSONL.

---

### 🟠 P1-7 — `num_workers`/`prefetch_factor` posiblemente subdimensionados (RT-DETR)

**Dónde**: `train_rt_detr.yaml` (`num_workers: 4`, `prefetch_factor: 2 por defecto`).

**Problema**: con decodificación JPEG + augmentación CPU-bound (P0-2), 4 workers pueden no
saturar la GPU. La config de YOLO ya usa `8`/`3`. En una laptop con CPU de 8–16 hilos,
subir workers ayuda **solo si** el cuello es de datos (medir primero, ver §3).

**Solución propuesta**: medir utilización GPU; si <90 % con GPU esperando datos, subir
`num_workers` (p. ej. 8) y `prefetch_factor` (3–4). Combinar con la caché de P0-2.

---

### 🟡 P2-8 — Doble transferencia de `targets` a device

`train_detection.py:1049` mueve `targets` a GPU; luego `train_step` vuelve a hacer
`target["boxes"].to(device)` / `labels.to(device)` (`yolo26_wrapper.py:564-565`,
`rt_detr_wrapper.py:579-580`). Una de las dos es redundante. Unificar (mover solo en el
loop con `non_blocking`, o solo en `train_step`).

### 🟡 P2-9 — Claves de configuración muertas

`scheduler`, `warmup` parcial, `eval_interval`, `early_stopping_metric: map` y
`evaluation.confidence_thresholds_sweep` **no se consultan** en `train_detection.py`. El
*early stopping* y el *best checkpoint* usan **`val_loss`**, no mAP. Esto confunde el
*tuning*. Recomendación: implementar la evaluación mAP intra-entrenamiento (usar
`evaluate_detection`) o eliminar las claves para evitar falsas expectativas.

> Nota adicional: el **warmup pisa los LR discriminativos** de RT-DETR. `get_parameters`
> devuelve *param groups* con LR distintos para *backbone*/*head*
> (`rt_detr_wrapper.py:522-525`), pero el warmup hace
> `param_group["lr"] = warmup_lr` para **todos** los grupos (`train_detection.py:1035-1036`),
> igualándolos durante el warmup. Es correctness de *fine-tuning*, no de velocidad.

### 🟡 P2-10 — `clip_grad_norm_` reconstruye la lista de params cada batch

`train_detection.py:1080`:
```python
_clip_params = [p for g in optimizer.param_groups for p in g["params"]]
```
Se recalcula en **cada** batch. Cachearla una vez antes del bucle de épocas.

### 🟡 P2-11 — Arranque: parseo secuencial + reserialización por *spawn*

`RDD2022Dataset._load_supervisely` parsea **38.385 JSON secuencialmente** al inicio
(`rdd2022.py:120-144`). Además, en Windows (`spawn`), cada worker **reserializa** la lista
completa de `Annotation` al arrancar (`persistent_workers` lo amortiza a una vez por run).
Es coste de **arranque**, no por época, pero alarga el *time-to-first-batch*. Opciones:
parseo en paralelo, *cachear* las anotaciones parseadas (pickle), o `index` ligero.

---

## 3. Cómo medir (antes de optimizar)

Recomendado confirmar el diagnóstico con datos reales en esta GPU:

1. **Utilización GPU**: en otra terminal, `nvidia-smi dmon -s u` durante el entrenamiento.
   Si `sm%` oscila/baja a 0 entre batches → cuello de **datos** (P0-2/P0-3/P1-7).
2. **Tiempo por época vs tamaño de checkpoint**: comparar `epoch_time_s`
   (ya se loguea, `train_detection.py:1144`) con y sin el guardado de `training_state.pt`
   para aislar P0-1.
3. **Profiler**: envolver unas iteraciones con `torch.profiler.profile(activities=[CPU,CUDA])`
   y revisar si domina `aten::copy_` (transferencias), `DataLoader` (datos) o los kernels
   del modelo (cómputo).
4. **Sanidad del DataLoader**: cronometrar un bucle que **solo** itere `train_loader`
   (sin `train_step`) para medir el techo del pipeline de datos.

---

## 4. Fixes aplicados (2026-06-15)

Cambios de bajo riesgo que **no alteran la precisión** del modelo, aplicados en
`model/training/train_detection.py`:

| Cambio | Qué hace | Efecto |
|--------|----------|--------|
| `non_blocking=True` en transferencias H2D (train y val) | Solapa copia con cómputo | Gratis (impacto ~0 aquí, correcto a futuro) |
| `drop_last=True` en `train_loader` | Fija la dimensión de batch | Evita BN en batch diminuto y re-trazas de compile; descarta <8 img/época |
| `training_state.pt` cada `training_state_interval` épocas (def. 5) + última/early-stop/interrupt | Deja de escribir ~286 MB cada época | −~80 % de escritura de estado; sin cambio de resultados |
| `torch.compile` ahora **opt-in** (`use_torch_compile`, def. `false`) | No compila si no hay triton | Quita warmup/fragilidad inútiles (medido 0 % de mejora) |

Nuevas claves de config (opcionales) en `training:`
```yaml
training:
  training_state_interval: 5     # cada cuántas épocas se persiste el estado de resume
  use_torch_compile: false       # poner true solo en entornos con triton (p. ej. Linux)
```

**No se aplicó** (rechazado por medición): `use_channels_last` (2.2× más lento con RT-DETR).

Scripts de medición añadidos: `model/tools/profile_training_bottleneck.py` y
`model/tools/profile_compute_variants.py`.

---

## 5. Cómo acelerar de verdad (compute-bound) — decisiones del usuario

Como el cuello es el cómputo del modelo, las palancas con impacto **grande** implican
**trade-offs de precisión o de entorno**. Estas NO se aplicaron sin tu visto bueno:

| Palanca | Ganancia estimada | Trade-off | Notas |
|---------|-------------------|-----------|-------|
| **`input_size` 640 → 512** | ~**35 %** menos cómputo (escala ~cuadrático) | Menos resolución → posible caída en daños pequeños | Cambio de 1 línea en config; re-evaluar mAP |
| **`model_size` "l" → "m"** | **grande** (RT-DETR-m es mucho más ligero) | Menor capacidad/precisión | Requiere re-entrenar desde cero |
| **`batch_size` 8 → 12/16** | Mejor ocupación de GPU | Riesgo **OOM** en 8 GB | Probar con AMP; vigilar `nvidia-smi` |
| **Instalar `triton`** (Windows) | Potencial **1.3–2×** vía `torch.compile` real | Instalación frágil en Windows; verificar con torch 2.12 | Luego `use_torch_compile: true`. Validar con `profile_compute_variants.py` |
| **Validación cada N épocas** | ~**8–10 %** (val ≈ 10 % de la época) | Early-stop/best-checkpoint con cadencia N | Cablear la clave muerta `eval_interval`; cambia dinámica de entrenamiento |
| **Reducir nº de épocas / early-stop más agresivo** | Lineal | Posible *underfitting* | `early_stopping_patience` ya = 25 |

**Recomendación pragmática para tesis**: medir el par precisión/tiempo de
**`input_size: 512`** primero (cambio trivial, reversible, ~35 % más rápido). Si la mAP
aguanta, es la mejor relación esfuerzo/beneficio sin reentrenar arquitectura.

---

## 6. Pendientes de correctness/limpieza (no perf)

- **P1-5**: validación en modo `train()` (cambia `val_loss`; arreglarlo altera comparaciones
  previas → decidir si se hace antes de la próxima tanda de runs).
- **P1-6**: tracker → JSONL.
- **P2-9**: claves de config muertas (`scheduler`, `eval_interval`, `early_stopping_metric: map`).
- **P2-11**: carga de dataset de **297 s**; cachear las anotaciones parseadas (pickle) reduciría
  el *time-to-first-batch* drásticamente.

> Reproducir cualquier medición con los scripts de §0.bis usando el python de
> `tesis-modelo`. Tras cada cambio, re-medir `t_step` para confirmar la mejora.
