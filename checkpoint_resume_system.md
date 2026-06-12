# Sistema de Checkpoints y Reanudación del Entrenamiento

## 1. Visión general

El entrenamiento puede ser interrumpido por cortes de luz, límites de tiempo en clusters, o errores inesperados. El sistema de checkpoints garantiza que **ningún progreso se pierda**: al relanzar el entrenamiento se retoma exactamente desde donde se quedó, con el mismo estado del optimizador, scheduler, scaler AMP, y métricas de early stopping.

Todo este código vive en `model/training/train_detection.py`.

---

## 2. Estructura de archivos en disco

Cada run genera un directorio propio dentro del directorio de checkpoints:

```
checkpoints/
└── <run_id>/                        ← UUID único por ejecución
    ├── training_state.pt            ← Estado completo, sobreescrito cada época
    ├── best_model.pt                ← Mejor modelo según métrica (val_loss o mAP)
    ├── recovery.pt                  ← Copia de seguridad cada 5 épocas
    ├── final_model.pt               ← Guardado al terminar (normal o early stop)
    └── <run_id>.json                ← Metadata del experimento (tracker)
```

- `training_state.pt` es el único archivo diseñado explícitamente para reanudar: contiene todo lo necesario para que el loop continúe sin pérdida de estado.
- `recovery.pt` es un respaldo de seguridad por si `training_state.pt` se corrompió (p.ej. el proceso se mató justo mientras escribía).
- `best_model.pt` y `final_model.pt` solo contienen pesos del modelo, no estado del optimizador; sirven para inferencia/evaluación, no para reanudar.

---

## 3. Las tres funciones auxiliares de resume

### 3.1 `_get_model_state_dict` / `_set_model_state_dict` (líneas 526–546)

El codebase maneja tres patrones de wrapper distintos:

| Patrón | Acceso al `nn.Module` real |
|--------|---------------------------|
| Ultralytics-style (RT-DETR, YOLO26) | `model._model.model` |
| Wrapper simple | `model._model` |
| Directo | `model` |

Estas dos funciones encapsulan esa lógica para que `_save_training_state` y `_load_training_state` no necesiten conocer el tipo de modelo.

### 3.2 `_save_training_state` (línea 549)

```python
def _save_training_state(path, model, optimizer, scheduler, scaler, epoch,
                         best_val_loss, best_epoch, epochs_without_improvement,
                         run_id, config_used, best_map=0.0):
```

Construye un diccionario con **todo el estado necesario para reanudar** y lo guarda con `torch.save`:

| Clave | Qué guarda | Por qué importa |
|-------|-----------|-----------------|
| `model_state_dict` | Pesos del modelo | Estado de la red neuronal |
| `optimizer_state_dict` | Buffers del optimizador (momentum, etc.) | Sin esto, las primeras épocas post-resume serían "frías" |
| `scheduler_state_dict` | Paso actual del cosine scheduler | Evita resetear el LR al valor inicial |
| `scaler_state_dict` | Factor de escala del GradScaler AMP | Evita overflow/underflow en float16 al reanudar |
| `epoch` | Índice de la última época completada (0-indexed) | Define `start_epoch = epoch + 1` al reanudar |
| `best_val_loss` | Menor val_loss visto hasta ahora | El early stopping compara contra este valor |
| `best_epoch` | Época donde se alcanzó `best_val_loss` | Para logging y métricas finales |
| `epochs_without_improvement` | Contador para early stopping | Sin esto, el early stopping se resetearía al reanudar |
| `best_map` | Mejor mAP@0.5 (si `early_stopping_metric: map`) | Igual que `best_val_loss` pero para la métrica alternativa |
| `run_id` | UUID del run | Clave para saber si el checkpoint es "nuevo formato" |
| `config_used` | Config YAML completa | Auditoría; no se usa para reanudar |

### 3.3 `_load_training_state` (línea 569)

```python
def _load_training_state(path, model, optimizer, scheduler, scaler, device):
```

Carga el archivo `.pt` y restaura cada componente en orden:

1. Siempre carga `model_state_dict` en el modelo (via `_set_model_state_dict`).
2. Si existe `optimizer_state_dict` → checkpoint nuevo formato → restaura optimizer, scheduler y scaler.
3. Si **no** existe `optimizer_state_dict` → checkpoint viejo formato (solo pesos) → emite `logger.warning` y rellena los metadatos con valores por defecto (`epoch=-1`, `best_val_loss=inf`, etc.) para que el llamador pueda detectar el caso.

La detección de "viejo formato" se hace comprobando la presencia de la clave `optimizer_state_dict`, **no** el `run_id`, para mantener compatibilidad hacia atrás con checkpoints generados antes de que existiera este sistema.

### 3.4 `_resolve_resume_path` (línea 601)

```python
def _resolve_resume_path(resume_from, checkpoint_dir, model_type):
```

Acepta dos formas de indicar un checkpoint al usuario:

1. **Ruta directa** (`--resume ./checkpoints/abc123/training_state.pt`): si termina en `.pt` y existe, lo retorna tal cual.
2. **Run ID** (`--resume abc123`): busca en orden de prioridad:
   1. `<checkpoint_dir>/abc123/training_state.pt`
   2. `<checkpoint_dir>/abc123/recovery.pt`
   3. `<checkpoint_dir>/abc123/final_model.pt`
   4. `<checkpoint_dir>/abc123/best_model.pt`

Si ninguno existe, lanza `FileNotFoundError` con la lista de paths buscados para facilitar el diagnóstico.

---

## 4. El flujo de resume dentro de `train()` (líneas 979–1027)

```
¿Se pasó --resume o resume_from en config?
        │
       SÍ
        │
        ▼
_resolve_resume_path()  ──→  ruta al .pt
        │
        ▼
_load_training_state()  ──→  state dict
        │
        ├── ¿state["run_id"] tiene valor?
        │           │
        │          SÍ  ←── Checkpoint nuevo formato
        │           │
        │           ├── start_epoch = state["epoch"] + 1
        │           ├── Restaurar best_val_loss, best_epoch,
        │           │   epochs_without_improvement, best_map
        │           ├── run_checkpoint_dir = checkpoint_dir / run_id
        │           └── (NO llamar tracker.start_run, se usa el mismo run_id)
        │
        └──────────NO  ←── Checkpoint viejo formato (solo pesos)
                    │
                    ├── start_epoch = 0  (empieza desde el principio)
                    ├── tracker.start_run()  (nuevo run_id)
                    └── Crear nuevo run_checkpoint_dir
```

Luego el loop de épocas simplemente usa `start_epoch` en lugar de `0`:

```python
for epoch in range(start_epoch, epochs):
```

---

## 5. Guardado durante el loop de épocas (líneas 1191–1264)

En cada época se ejecutan hasta cuatro operaciones de guardado:

### 5.1 `best_model.pt` — condicional

```python
if avg_val_loss < best_val_loss:          # o mAP > best_map si early_stopping_metric=map
    best_val_loss = avg_val_loss
    epochs_without_improvement = 0
    _save_checkpoint(run_checkpoint_dir / "best_model.pt", ...)
else:
    epochs_without_improvement += 1
```

Usa `model.save_checkpoint()` (método del wrapper), que guarda pesos + config del modelo, pero **no** el estado del optimizador. Sirve para inferencia, no para reanudar.

### 5.2 `recovery.pt` — cada 5 épocas

```python
if (epoch + 1) % 5 == 0:
    _save_checkpoint(run_checkpoint_dir / "recovery.pt", ...)
```

También usa `model.save_checkpoint()`. Es un respaldo de seguridad por si `training_state.pt` se corrompió.

### 5.3 `training_state.pt` — cada época (línea 1255)

```python
_save_training_state(
    run_checkpoint_dir / "training_state.pt",
    model, optimizer, cosine_scheduler, scaler,
    epoch, best_val_loss, best_epoch, epochs_without_improvement,
    run_id, config, best_map=best_map,
)
```

Este es el checkpoint de resume. Se sobreescribe en cada época (no hay historial de estados intermedios), lo que mantiene el uso de disco bajo.

### 5.4 `final_model.pt` — al terminar

Guardado en el bloque `finally` / después del loop. Usa `model.save_checkpoint()`.

---

## 6. Cómo usar el sistema

### Por línea de comandos

```bash
# Reanudar por run_id
python -m model.training.train_detection \
    --config model/configs/train_ssd_mobilenet.yaml \
    --resume a3f8-2d1a-...

# Reanudar por ruta directa
python -m model.training.train_detection \
    --config model/configs/train_ssd_mobilenet.yaml \
    --resume ./checkpoints/a3f8-2d1a-.../training_state.pt
```

### Por config YAML

```yaml
training:
  resume_from: a3f8-2d1a-...   # o ruta a .pt
```

---

## 7. Compatibilidad hacia atrás

Los modelos con `save_checkpoint()` antiguo (sin parámetros `optimizer`/`epoch`/`metrics`) siguen funcionando: la función `_save_checkpoint` local comprueba la firma del método con `inspect.signature` antes de llamarlo, y omite los parámetros extra si no los soporta.

---

---

# Plan de integración en la rama `feature/refactor`

## Contexto

La rama `feature/refactor` tiene `train_detection.py` (~906 líneas) con:
- ✅ Guardado de `best_model.pt` (cuando mejora `val_loss`)
- ✅ Guardado de `recovery.pt` (cada 5 épocas)
- ✅ Guardado de `final_model.pt`
- ✅ Early stopping básico (solo por `val_loss`)
- ❌ Sin `_save_training_state` / `_load_training_state` / `_resolve_resume_path`
- ❌ Sin `training_state.pt` (el archivo para reanudar)
- ❌ Sin parámetro `resume_from` en `train()`
- ❌ Sin `start_epoch` (el loop siempre empieza en 0)
- ❌ Sin `--resume` en el CLI
- ❌ Sin soporte de `early_stopping_metric: map`

## Pasos de implementación

### Paso 1 — Añadir las funciones auxiliares (sin tocar el loop)

Insertar estas tres funciones entre el bloque de helpers y la función `train()`, justo después de las funciones de dataset.

**Archivo**: `model/training/train_detection.py`

Funciones a insertar (copiar de `refactor2`):
- `_get_model_state_dict(model)` — línea 526 de refactor2
- `_set_model_state_dict(model, state_dict)` — línea 537 de refactor2
- `_save_training_state(path, model, optimizer, scheduler, scaler, epoch, best_val_loss, best_epoch, epochs_without_improvement, run_id, config_used, best_map=0.0)` — línea 549 de refactor2
- `_load_training_state(path, model, optimizer, scheduler, scaler, device)` — línea 569 de refactor2
- `_resolve_resume_path(resume_from, checkpoint_dir, model_type)` — línea 601 de refactor2

Estas funciones no tienen dependencias circulares y se pueden insertar directamente.

### Paso 2 — Modificar la firma de `train()` y leer `resume_from`

**Cambio en la firma**:
```python
# Antes
def train(config_path: str, verbose: bool = False) -> dict:

# Después
def train(config_path: str, verbose: bool = False, resume_from: Optional[str] = None) -> dict:
```

**Leer `resume_from` desde config** (añadir justo después de leer `checkpoint_dir`):
```python
resume_from = resume_from or training_config.get("resume_from")
```

### Paso 3 — Añadir el bloque de inicialización de variables de estado

En `feature/refactor`, el tracker y la creación del `run_id` ocurre directamente (sin rama condicional). Hay que reemplazar ese bloque:

**Antes** (feature/refactor, ~línea 632):
```python
tracker = ExperimentTracker(output_dir=checkpoint_dir)
dataset_name = dataset_config.get("name", Path(dataset_path).name)

try:
    run_id = tracker.start_run(config, model_type, dataset_name)
except Exception as e:
    ...
    return {}

run_checkpoint_dir = checkpoint_dir / run_id
run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
```

**Después** (añadir inicialización + bloque resume):
```python
tracker = ExperimentTracker(output_dir=checkpoint_dir)
dataset_name = dataset_config.get("name", Path(dataset_path).name)

# Inicializar variables de estado (se sobreescriben si se carga un checkpoint)
start_epoch = 0
best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

if resume_from:
    try:
        resume_path = _resolve_resume_path(resume_from, checkpoint_dir, model_type)
        logger.info("Loading resume state from: %s", resume_path)
        state = _load_training_state(resume_path, model, optimizer, cosine_scheduler, scaler, device)
        run_id = state.get("run_id")
        if run_id:
            start_epoch = state["epoch"] + 1
            best_val_loss = state["best_val_loss"]
            best_epoch = state["best_epoch"]
            epochs_without_improvement = state["epochs_without_improvement"]
            run_checkpoint_dir = Path(checkpoint_dir) / run_id
            logger.info(
                "Resuming run %s from epoch %d (best_val_loss=%.4f at epoch %d)",
                run_id, start_epoch, best_val_loss, best_epoch,
            )
        else:
            logger.info("Old-format checkpoint loaded. Starting new training run.")
            run_id = tracker.start_run(config, model_type, dataset_name)
            run_checkpoint_dir = checkpoint_dir / run_id
            run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error("Failed to resume from '%s': %s", resume_from, e)
        signal.signal(signal.SIGINT, original_sigint_handler)
        return {}
else:
    try:
        run_id = tracker.start_run(config, model_type, dataset_name)
    except Exception as e:
        logger.error("Failed to start experiment run: %s", e)
        signal.signal(signal.SIGINT, original_sigint_handler)
        return {}
    run_checkpoint_dir = checkpoint_dir / run_id
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
```

**Nota importante**: en `feature/refactor` el bloque de creación de `run_id` ocurre **antes** de construir el optimizer y el scheduler (a diferencia de refactor2 donde ocurre después). Hay que mover este bloque a **después** de construir `optimizer` y `cosine_scheduler`, porque `_load_training_state` los necesita para restaurar sus estados.

### Paso 4 — Cambiar `range(epochs)` por `range(start_epoch, epochs)`

En el loop principal:
```python
# Antes
for epoch in range(epochs):

# Después
for epoch in range(start_epoch, epochs):
```

### Paso 5 — Añadir el guardado de `training_state.pt` en cada época

Dentro del bloque de checkpointing, después del recovery checkpoint:

```python
# Training state (siempre, para resume): sobreescribe cada época
try:
    _save_training_state(
        run_checkpoint_dir / "training_state.pt",
        model, optimizer, cosine_scheduler, scaler,
        epoch, best_val_loss, best_epoch, epochs_without_improvement,
        run_id, config,
    )
except (IOError, OSError) as e:
    logger.warning("Failed to save training state: %s", e)
```

### Paso 6 — Actualizar el CLI para aceptar `--resume`

En el bloque `if __name__ == "__main__"`:

```python
# Antes
parser.add_argument("--config", type=str, required=True, ...)
args = parser.parse_args()
train(args.config, verbose=args.verbose)

# Después
parser.add_argument("--config", type=str, required=True, ...)
parser.add_argument("--resume", type=str, default=None,
                    help="Run ID or path to .pt checkpoint to resume from")
args = parser.parse_args()
train(args.config, verbose=args.verbose, resume_from=args.resume)
```

---

## Orden de aplicación recomendado

1. Paso 1 (funciones auxiliares) — no rompe nada, se puede hacer aislado
2. Paso 2 (firma de `train` + leer `resume_from`) — cambio mínimo, no activa nada
3. Paso 6 (CLI `--resume`) — de forma simultánea con paso 2
4. Paso 3 (bloque de resume) — requiere Paso 1 y Paso 2 completos
5. Paso 4 (cambiar `range`) — requiere Paso 3 completo
6. Paso 5 (guardar `training_state.pt`) — se puede hacer independientemente, pero conviene hacerlo junto con Paso 4

## Tests a verificar después de la integración

- Lanzar un entrenamiento, interrumpirlo con Ctrl+C después de 3 épocas, verificar que `training_state.pt` existe en el directorio del run.
- Relanzar con `--resume <run_id>`, verificar en los logs que empieza en `epoch 4`.
- Verificar que `best_val_loss` y `epochs_without_improvement` se restauran correctamente (no se resetea el early stopping).
- Test con checkpoint viejo formato (solo pesos): debe iniciar un run nuevo, no fallar.
