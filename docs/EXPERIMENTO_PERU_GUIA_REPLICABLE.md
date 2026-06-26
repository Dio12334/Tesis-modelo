# Guía replicable: ¿la data de Perú mejora el modelo? (para el lado YOLO26)

> **Para quién:** el compañero que entrena **YOLO26** en otra rama. Esta guía es autocontenida:
> con ella puedes replicar el experimento de transferencia desde tu lado **sin mergear ramas**.
> Tu Claude puede adaptar los detalles. **No contiene credenciales** (esas van en el "dataset
> export guide", que te pasan aparte y por canal seguro).
>
> **Contexto:** somos dos ramas atacando la misma pregunta con dos modelos (RT-DETR y YOLO26).
> En el lado RT-DETR ya se montó este experimento; aquí está la receta para que lo repliques con
> YOLO26 y al final comparemos ambos. Resultados RT-DETR de referencia: ver `RESULTADOS_TESIS.md`.

---

## 1. La pregunta y el diseño (model-agnostic)

**Pregunta:** ¿agregar la data propia de Perú mejora tu modelo YOLO26 ya entrenado en RDD2022?
(a) ¿generaliza mejor a calles de Perú? (b) ¿sin perjudicar su desempeño en RDD?

Tres mediciones, todas sobre un **test de Perú held-out** (20% que nunca se entrena):

| Exp | Qué es | Responde |
|-----|--------|----------|
| **A0 — zero-shot** | Evalúas tu modelo RDD (sin reentrenar) en el test de Perú | la **brecha de dominio** RDD→Perú |
| **A1 — fine-tune** | Afinas tu modelo RDD sobre Perú-train (LR bajo, pocas épocas) y evalúas en el test de Perú | ¿agregar Perú **mejora** en Perú? (A1 vs A0) |
| **A1' — forgetting** | Evalúas el modelo fine-tuned sobre el **RDD original** | ¿el fine-tune **dañó** RDD? (A1' vs tu baseline RDD) |

**Veredicto:** A1 mAP@0.5 (Perú) **> A0** → Perú influye positivamente. A1' ≈ baseline RDD → sin daño colateral.

---

## 2. Conseguir la data de Perú

La data de Perú **no se genera trivialmente**: las imágenes se renderizan al vuelo proyectando
panorámicas 360° de Mapillary, según anotaciones guardadas en una BD. Dos caminos:

- **(Recomendado) Pídele a tu compañero la carpeta ya descargada** — son ~1956 imágenes
  renderizadas (1920×1080) + sus anotaciones en formato **Supervisely JSON**:
  ```
  local_dataset/train/img/<view_id>.jpg
  local_dataset/train/ann/<view_id>.jpg.json
  ```
  Te la pasa por Drive/S3 (~varios cientos de MB). Así evitas re-descargar y manejar credenciales.

- **(Alternativa) Regenerarla tú** con el "dataset export guide" (te lo pasan aparte; **contiene
  credenciales** de la BD y Mapillary). Ese guide produce la data directamente en formato YOLO,
  así que si tomas este camino puedes saltarte la conversión de §4.

> El formato Supervisely JSON por imagen es: `{"size":{"width","height"}, "objects":[{"classTitle",
> "geometryType":"rectangle", "points":{"exterior":[[x_min_px,y_min_px],[x_max_px,y_max_px]]}}]}`.
> Las cajas están en **píxeles** de la imagen renderizada (1920×1080).

---

## 3. Taxonomía: mapear Perú a las clases de TU modelo RDD (paso crítico ⚠️)

Perú tiene **8 clases en español**; RDD usa otras. Para el fine-tune, las etiquetas de Perú deben
usar **los mismos índices de clase con los que entrenaste tu YOLO26 en RDD** (si no, el fine-tune
le enseña la clase equivocada a cada canal de salida).

Mapeo semántico Perú → daño RDD (las 4 clases "limpias" + un catch-all):

| Clase Perú | Daño RDD | Código RDD |
|------------|----------|-----------|
| `bache` | pothole | D40 |
| `fisura_longitudinal`, `fisura_oblicua` | longitudinal crack | D00 |
| `fisura_transversal` | transverse crack | D10 |
| `piel_de_cocodrilo` | alligator crack | D20 |
| `reparacion`, `tratamiento_superficial`, `fisura_esquina` | (sin equivalente directo) | descartar **o** "otros" |

**Tú debes decidir** según tu modelo RDD:
- Si tu YOLO26 RDD tiene **4 clases** (D00/D10/D20/D40, lo estándar de RDD2022) → mapea las 4 y
  **descarta** reparacion/tratamiento/esquina.
- Si tiene una 5ª clase "otros/other" → mándalas ahí (como hicimos en RT-DETR para no romper el
  alineamiento de índices).
- **Lo no negociable:** el `id` de cada clase en tu YAML de Perú debe ser **idéntico** al `id` que
  usa tu modelo RDD. Revisa el `names:` de tu `dataset.yaml` de RDD y respeta ese orden.

---

## 4. Convertir a YOLO + split held-out

YOLO/Ultralytics usa `imagen.jpg` + `imagen.txt` (`class_id cx cy w h`, normalizado, centro) + un
`dataset.yaml`. Convierte el Supervisely de §2 y separa un **20% de test fijo** (seed fijo):

```python
# convert_peru_to_yolo.py
import json, pathlib, random, shutil

SRC = pathlib.Path("local_dataset/train")     # Supervisely (img/ + ann/)
DST = pathlib.Path("peru_yolo")
VAL_SPLIT, SEED = 0.20, 42

# >>> AJUSTA ESTO a los índices de TU modelo RDD de YOLO26 <<<
# clase Perú -> id de clase de tu modelo RDD (None = descartar)
CLASS_ID = {
    "bache": 3,                  # pothole / D40
    "fisura_longitudinal": 0,    # longitudinal / D00
    "fisura_oblicua": 0,
    "fisura_transversal": 1,     # transverse / D10
    "piel_de_cocodrilo": 2,      # alligator / D20
    "reparacion": None,          # descartar (o un id "otros" si tu modelo lo tiene)
    "tratamiento_superficial": None,
    "fisura_esquina": None,
}

anns = sorted((SRC / "ann").glob("*.json"))
random.Random(SEED).shuffle(anns)
n_val = int(len(anns) * VAL_SPLIT)
val = set(anns[:n_val])

for split in ("train", "val"):
    (DST / "images" / split).mkdir(parents=True, exist_ok=True)
    (DST / "labels" / split).mkdir(parents=True, exist_ok=True)

for jf in anns:
    d = json.loads(jf.read_text(encoding="utf-8"))
    W, H = d["size"]["width"], d["size"]["height"]
    lines = []
    for o in d.get("objects", []):
        cid = CLASS_ID.get(o["classTitle"])
        if cid is None:
            continue
        (x1, y1), (x2, y2) = o["points"]["exterior"]
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        bw, bh = abs(x2 - x1) / W, abs(y2 - y1) / H
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    split = "val" if jf in val else "train"
    img = jf.name[:-len(".json")]                      # "<id>.jpg"
    shutil.copy2(SRC / "img" / img, DST / "images" / split / img)
    (DST / "labels" / split / (img[:-4] + ".txt")).write_text("\n".join(lines))

# dataset.yaml — usa EXACTAMENTE los nombres/orden de tu modelo RDD de YOLO26
(DST / "peru.yaml").write_text(
    f"path: {DST.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
    "names:\n  0: longitudinal_crack\n  1: transverse_crack\n  2: alligator_crack\n  3: pothole\n")
print("listo:", DST)
```

`peru_yaml/peru.yaml` → `val` = el 20% held-out (tu **test de Perú**). El fine-tune entrena en
`train` y se evalúa en `val`. A0 y A1 usan el mismo `val` → comparable.

---

## 5. Los 3 experimentos (comandos Ultralytics / YOLO26)

Sea `rdd_yolo26.pt` tu mejor modelo entrenado en RDD2022.

```bash
# A0 — zero-shot: tu modelo RDD evaluado en el test de Perú (sin reentrenar)
yolo detect val model=rdd_yolo26.pt data=peru_yolo/peru.yaml

# A1 — fine-tune sobre Perú (LR bajo, pocas épocas; ~1956 imgs => rápido)
yolo detect train model=rdd_yolo26.pt data=peru_yolo/peru.yaml \
     epochs=40 imgsz=768 lr0=0.0005 lrf=0.01 patience=10 \
     name=peru_finetune
#   (lr0 BAJO porque es fine-tune; sube imgsz si tu GPU aguanta. Ultralytics hace train/val solo.)

# A1 — evaluar el fine-tuned en el test de Perú
yolo detect val model=runs/detect/peru_finetune/weights/best.pt data=peru_yolo/peru.yaml

# A1' — forgetting: el fine-tuned evaluado en el RDD ORIGINAL
yolo detect val model=runs/detect/peru_finetune/weights/best.pt data=rdd_original.yaml
```

> Ultralytics ya hace TTA con `augment=True` en `val`/`predict` para YOLO (a diferencia de RT-DETR);
> si quieres reportar con TTA, agrega `augment=True` al `yolo detect val`.

---

## 6. Métricas y reglas de decisión

Reporta **mAP@0.5 (y mAP@.5:.95), precision, recall, F1, y AP por clase**, en:
- **test de Perú**: A0 vs A1.
- **RDD-val**: tu baseline RDD vs A1'.

| Resultado | Interpretación |
|-----------|----------------|
| A1 (Perú) **>** A0 | la data de Perú **mejora** el modelo en Perú ✅ |
| A1' (RDD) ≈ tu baseline RDD | el fine-tune **no daña** RDD → influencia positiva neta ✅ |
| A1' (RDD) **<<** baseline | hubo *forgetting*: mejora Perú a costa de RDD (trade-off) ⚠️ |

Mira sobre todo las **4 clases con datos reales** (pothole, longitudinal, transverse, alligator);
las descartadas/"otros" no son interpretables.

---

## 7. Para comparar entre ramas (RT-DETR vs YOLO26)

Para que ambos lados sean comparables, **fijen lo mismo**:
- Mismo subconjunto de Perú y **mismo split** (seed 42, val 20%).
- Mismas 4 clases evaluables.
- Reportar la **misma tabla**: A0 / A1 (Perú) y baseline / A1' (RDD), con mAP@0.5 + recall + per-clase.

Así al final se puede decir, por modelo: *cuánta brecha de dominio había (A0)*, *cuánto la cierra el
fine-tune (A1−A0)*, y *cuánto cuesta en RDD (A1'−baseline)*.

---

## 8. TL;DR (checklist)
1. Consigue la data de Perú (carpeta Supervisely del compañero, o regenérala con el export guide).
2. Mapea Perú → los **índices de clase de tu modelo RDD** (§3) y convierte a YOLO + split 80/20 (§4).
3. Corre A0 (zero-shot), A1 (fine-tune + eval Perú), A1' (eval en RDD) (§5).
4. Llena la tabla de métricas y aplica las reglas de decisión (§6).
5. Comparen ambas ramas con el formato común (§7).
