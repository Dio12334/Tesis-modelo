# Log de implementación — importar dataset propio (BD Supabase + Mapillary)

> Bitácora de avances de la integración del dataset propio (anotaciones en Supabase +
> imágenes renderizadas desde panorámicas 360° de Mapillary) en este proyecto. Sirve como
> referencia persistente (sobrevive al límite de contexto). Fuente: `docs/DATASET_EXPORT_GUIDE.md`
> (gitignored, contiene credenciales). **Este log NO contiene credenciales.**

## Objetivo
Descargar las anotaciones de la BD, renderizar las vistas perspectivas desde los 360° de
Mapillary, y dejar la data en el formato que consume el pipeline de entrenamiento
(`RDD2022Dataset` = **Supervisely JSON**), para entrenar con `train_detection.py` sin fricción.

## Decisiones de diseño
- **Formato de salida: Supervisely JSON** (no YOLO como la guía), porque el pipeline de este
  repo (`model/datasets/rdd2022.py`) carga ese formato. Estructura:
  `model/data/local_dataset/train/{img,ann}/`. El split train/val lo hace el pipeline
  (`val_split`), así que toda la data va bajo `train/` (un solo subset).
- **Clases: superset de 8 (Opción A de la guía)** — preserva info. `num_classes=8` en el config.
  (Se puede mapear a RDD2022 4-clases después si se quiere combinar con el dataset existente.)
- **Credenciales vía `.env`** (gitignored). El código lee de variables de entorno; sin secretos
  hardcodeados → los scripts son commiteables.
- **Región/país**: se etiqueta con un tag configurable (`--region`, default "Peru"). Caveat: el
  extractor de país del loader sólo reconoce los países de RDD; para esta data el per-region eval
  puede quedar impreciso (no afecta el entrenamiento). Mejora futura si hace falta.
- **Reanudable**: salta vistas cuyo `.jpg` + `.json` ya existen.

## Archivos
- `model/scripts/equirect.py` — proyección equirectangular 360° → perspectiva (copia exacta del
  renderer de la herramienta de anotación; NO modificar defaults fov 110×120, salida 1920×1080).
- `model/scripts/build_local_dataset.py` — pipeline: BD → Mapillary → render → Supervisely JSON.
- `.env` (gitignored) — credenciales (DB_PASSWORD, MAPILLARY_TOKEN, ...).
- `model/configs/train_rt_detr_local.yaml` — config de entrenamiento sobre la data nueva (pendiente).

## Pasos / estado
- [x] (1) `equirect.py` y `build_local_dataset.py` creados (salida Supervisely). py_compile OK.
- [x] (2) Credenciales extraídas del guide → `.env` (gitignored, verificado).
- [x] (3) `psycopg2-binary` instalado (requests/cv2 ya estaban).
- [x] (4) Conexión a la BD OK (no estaba pausada).
- [x] (5) Muestra de 3 imágenes OK + **sanity visual OK**: la caja `piel_de_cocodrilo` cae sobre
      el asfalto frente al auto → proyección y alineación correctas. Carga con `RDD2022Dataset` OK.
- [x] (6) **Descarga completa: 1956 imágenes, 4745 cajas, 0 fallos.** (Una corrida inicial bajó 1955
      con un fallo cosmético del Tee por falta de `logs/`; al reanudar completó la 1956 y escribió meta.)
- [x] (7) Carga completa verificada. 8 clases. Distribución de cajas:
      fisura_longitudinal 1536 (32%), tratamiento_superficial 1316 (28%), piel_de_cocodrilo 534,
      fisura_transversal 519, bache 408, reparacion 292, fisura_oblicua 118, **fisura_esquina 22 (rara)**.
      Dataset PEQUEÑO (1956 img) + 2 clases muy raras → overfit/clases raras serán el reto.
- [x] (8) Config `train_rt_detr_local.yaml` creado (num_classes=8, sin country_filter, basado en R).

## Caveats descubiertos
- **"País" en metadata sale como el view_id** (p.ej. `1000200018032639`) porque el extractor de
  país del loader está hecho para nombres tipo RDD (`Japan_000001`). NO afecta el entrenamiento
  (se entrena sin `country_filter`); solo el per-region eval quedaría impreciso. Fix futuro:
  reconocer el tag de región en `_extract_country_from_tags`.
- **Distorsión de aspecto**: las imágenes son 1920×1080 (16:9) y el pipeline las redimensiona a
  cuadrado (`input_size`×`input_size`). Conocido; a vigilar si la precisión sufre (fix = letterbox).
- Corregí el **bug de clipping de cajas** del guide (clampeo de esquinas, no del centro) en
  `to_supervisely`.

## Cómo correr (resumen)
```bash
# credenciales en .env (gitignored)
python -m model.scripts.build_local_dataset --out model/data/local_dataset --region Peru
# entrenar luego:
python -m model.training.train_detection --config model/configs/train_rt_detr_local.yaml -v
```

## Notas / gotchas heredados de la guía
- Supabase se auto-pausa tras ~7 días sin uso → si falla con "Tenant or user not found", hay que
  reanudar el proyecto en el dashboard de Supabase.
- Las URLs cacheadas en la BD expiran → siempre re-resolver vía Graph API de Mapillary.
- No cambiar parámetros de render o las cajas se desalinean.

## Bitácora (cronológica)
- **2026-06-18** — Implementado `equirect.py` (copia exacta del renderer) y `build_local_dataset.py`
  (BD→Mapillary→render→Supervisely, reanudable, creds por env). Extraídas creds a `.env` (gitignored).
  Instalado `psycopg2-binary`. Probada muestra de 3 imágenes: BD conecta, descarga+render OK, carga
  con `RDD2022Dataset` OK, **sanity visual confirmado** (cajas alineadas con el daño). Lanzada la
  descarga completa en background. Creado `train_rt_detr_local.yaml` (8 clases).
- **2026-06-22** — Descarga completa (1956 img / 4745 cajas). Verificada distribución de clases.
  **Smoke de entrenamiento OK**: el modelo se construye con num_classes=8 (head reshape 80→8), el
  dataset carga (1956 img, 8 clases, split 1565/391), y arranca el loop (AMP+EMA+RandomErasing).
  La integración data-propia → pipeline funciona end-to-end.
- **2026-06-22** — **Experimento Perú planteado y ambiente preparado (SIN ejecutar, por pedido).**
  Plan en `docs/EXPERIMENTO_PERU_PLAN.md` (Track A: fine-tune de R sobre Perú; runbook listo).
  Setup: `relabel_to_rdd5.py` → `model/data/peru_rdd5/` (Perú en 5 clases RDD, **orden alineado con
  R verificado**); `train_rt_detr_peru_finetune.yaml` (init desde R, LR bajo); soporte `init_checkpoint`
  en `train_detection.py` (carga pesos sin restaurar optimizer/epoch). Pendiente: lanzar A0/A1/A1'
  cuando el usuario dé el OK.
