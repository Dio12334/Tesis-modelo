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
- [ ] **Ejecutar N0–N3** (a la espera del OK). N1/N2 son tweaks del config (quitar `letterbox`, bajar a 768).
- [ ] N4 (tiling/SAHI): requiere preprocesamiento aparte (futuro).

## 5. Cómo correr (cuando se dé el OK)
```bash
# N3 (principal): + Noruega, 1024, letterbox  (EC2 recomendado por VRAM/resolución)
python -m model.training.train_detection --config model/configs/train_rt_detr_norway_letterbox.yaml -v
# luego: evaluar per-country
python -m model.training.evaluate_detection --config model/configs/train_rt_detr_norway_letterbox.yaml --run-id <ID> --split val
# N1/N2: copiar el config y poner letterbox:false (N1) o input_size:768 (N2)
```

> Nota: a 1024 + Noruega (8k imágenes grandes) en 8 GB hará OOM — este experimento es **para la EC2**
> (~25 GB). En la laptop, correr una versión reducida (768 + letterbox) para sanity.
