"""Construye el dataset propio en formato Supervisely (el que carga RDD2022Dataset).

Pipeline: lee anotaciones de la BD (Supabase/PostgreSQL) -> por cada vista descarga la
panorámica 360° de Mapillary -> la proyecta a perspectiva 1920x1080 con el yaw/pitch de la
vista -> guarda <view_id>.jpg + <view_id>.jpg.json (Supervisely) bajo <out>/train/.

Credenciales: SOLO por variables de entorno (o un archivo .env gitignored). NUNCA hardcodeadas.
Requeridas: DB_PASSWORD, MAPILLARY_TOKEN. Opcionales (con default): DB_HOST/PORT/NAME/USER.

Reanudable: salta vistas cuyo .jpg + .json ya existen.

Uso:
    # con .env en la raíz (DB_PASSWORD=..., MAPILLARY_TOKEN=...)
    python -m model.scripts.build_local_dataset --out model/data/local_dataset --region Peru
    python -m model.scripts.build_local_dataset --limit 5   # muestra de prueba
"""
import argparse
import json
import os
import pathlib
import time

# psycopg2 y requests se importan dentro de main() para dar un mensaje claro si faltan.
from model.scripts.equirect import render_view_bytes

# Parámetros de render — NO cambiar (deben coincidir con la herramienta de anotación)
FOV_H, FOV_V, OUT_W, OUT_H = 110.0, 120.0, 1920, 1080

# Superset de 8 clases (Opción A) — orden/índices estables para el config de entrenamiento.
CLASSES_SUPERSET = [
    "bache", "fisura_esquina", "fisura_longitudinal", "fisura_oblicua",
    "fisura_transversal", "piel_de_cocodrilo", "reparacion", "tratamiento_superficial",
]
# Opción B: colapso a RDD2022 (4 clases). Activar con --rdd-collapse.
RDD_MAP = {
    "fisura_longitudinal": "D00", "fisura_transversal": "D10",
    "piel_de_cocodrilo": "D20", "bache": "D40",
}


def _load_dotenv(path=".env"):
    """Carga KEY=VALUE de un .env a os.environ (sin sobrescribir lo ya seteado)."""
    p = pathlib.Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def _db_config():
    pw = os.environ.get("DB_PASSWORD")
    if not pw:
        raise SystemExit("Falta DB_PASSWORD (ponlo en .env o como variable de entorno).")
    return dict(
        host=os.environ.get("DB_HOST", "aws-1-us-west-2.pooler.supabase.com"),
        port=int(os.environ.get("DB_PORT", "5432")),
        dbname=os.environ.get("DB_NAME", "postgres"),
        user=os.environ.get("DB_USER", "postgres.mmxaxabhaabewkmvuaek"),
        password=pw,
    )


def fetch_rows(psycopg2):
    conn = psycopg2.connect(**_db_config())
    cur = conn.cursor()
    cur.execute(
        """
        SELECT iv.id, iv.parent_image_id, iv.yaw, iv.pitch,
               ia.damage_type::text,
               ia.bbox_x, ia.bbox_y, ia.bbox_width, ia.bbox_height
        FROM image_annotations ia
        JOIN image_views iv  ON iv.id = ia.image_id
        JOIN unique_images ui ON ui.id = iv.parent_image_id
        ORDER BY iv.id
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def group_by_view(rows):
    views = {}
    for vid, parent, yaw, pitch, dmg, x, y, w, h in rows:
        v = views.setdefault(vid, {
            "parent": parent,
            "yaw": float(yaw),
            "pitch": float(pitch if pitch is not None else -25.0),
            "boxes": [],
        })
        v["boxes"].append((dmg, float(x), float(y), float(w), float(h)))
    return views


def download_original(requests, mapillary_id, token):
    """Re-resuelve la URL del 360° vía Graph API (las cacheadas en la BD expiran)."""
    meta = requests.get(
        f"https://graph.mapillary.com/{mapillary_id}",
        params={"access_token": token, "fields": "thumb_original_url"},
        timeout=20,
    )
    meta.raise_for_status()
    url = meta.json().get("thumb_original_url")
    if not url:
        raise ValueError("sin thumb_original_url")
    img = requests.get(url, timeout=90)
    img.raise_for_status()
    return img.content


def to_supervisely(boxes, region, class_filter):
    """Construye el dict Supervisely de una vista. ``class_filter`` mapea/filtra clases:
    si una clase no está en el filtro -> se descarta. Devuelve (dict, n_objs)."""
    objects = []
    for dmg, x, y, w, h in boxes:
        title = class_filter(dmg)
        if title is None:
            continue
        # Clip de ESQUINAS a [0,1] (corrige el bug de clampear el centro) -> pixeles.
        x1 = min(max(x, 0.0), 1.0)
        y1 = min(max(y, 0.0), 1.0)
        x2 = min(max(x + w, 0.0), 1.0)
        y2 = min(max(y + h, 0.0), 1.0)
        if x2 <= x1 or y2 <= y1:
            continue  # caja degenerada
        objects.append({
            "classTitle": title,
            "geometryType": "rectangle",
            "points": {
                "exterior": [[round(x1 * OUT_W, 1), round(y1 * OUT_H, 1)],
                             [round(x2 * OUT_W, 1), round(y2 * OUT_H, 1)]],
                "interior": [],
            },
        })
    data = {"size": {"width": OUT_W, "height": OUT_H},
            "tags": [{"name": region}], "objects": objects}
    return data, len(objects)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="model/data/local_dataset")
    ap.add_argument("--region", default="Peru", help="tag de región para las imágenes")
    ap.add_argument("--rdd-collapse", action="store_true",
                    help="mapear a las 4 clases RDD2022 (default: superset de 8)")
    ap.add_argument("--limit", type=int, default=0, help="procesar solo N vistas (prueba)")
    ap.add_argument("--sleep", type=float, default=0.05, help="pausa entre descargas (rate limit)")
    args = ap.parse_args()

    _load_dotenv()
    try:
        import psycopg2  # noqa
        import requests  # noqa
    except ImportError as e:
        raise SystemExit(f"Falta dependencia: {e.name}. Instala: pip install psycopg2-binary requests")

    token = os.environ.get("MAPILLARY_TOKEN")
    if not token:
        raise SystemExit("Falta MAPILLARY_TOKEN (ponlo en .env o como variable de entorno).")

    if args.rdd_collapse:
        classes = ["D00", "D10", "D20", "D40"]
        def class_filter(dmg):
            return RDD_MAP.get(dmg)
    else:
        classes = CLASSES_SUPERSET
        valid = set(CLASSES_SUPERSET)
        def class_filter(dmg):
            return dmg if dmg in valid else None

    out = pathlib.Path(args.out)
    img_dir = out / "train" / "img"
    ann_dir = out / "train" / "ann"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    print("Leyendo anotaciones de la BD...")
    views = group_by_view(fetch_rows(psycopg2))
    vids = sorted(views.keys())
    if args.limit:
        vids = vids[:args.limit]
    print(f"  {len(vids)} vistas a procesar, "
          f"{sum(len(views[v]['boxes']) for v in vids)} cajas totales.")

    ok = skipped = failed = empty = 0
    for i, vid in enumerate(vids, 1):
        v = views[vid]
        img_path = img_dir / f"{vid}.jpg"
        ann_path = ann_dir / f"{vid}.jpg.json"
        if img_path.exists() and ann_path.exists():
            skipped += 1
            continue
        try:
            raw = download_original(requests, v["parent"], token)
            jpg = render_view_bytes(raw, v["yaw"], v["pitch"], FOV_H, FOV_V, OUT_W, OUT_H)
        except Exception as e:
            failed += 1
            print(f"  [FALLO] {vid}: {e}")
            continue

        ann, n = to_supervisely(v["boxes"], args.region, class_filter)
        if n == 0:
            empty += 1  # imagen sin cajas válidas (negativo); igual se guarda
        img_path.write_bytes(jpg)
        ann_path.write_text(json.dumps(ann, ensure_ascii=False), encoding="utf-8")
        ok += 1
        if i % 25 == 0:
            print(f"  {i}/{len(vids)}  ok={ok} skip={skipped} fail={failed} empty={empty}")
        time.sleep(args.sleep)

    # meta.json (opcional, informativo)
    (out / "meta.json").write_text(json.dumps({
        "classes": [{"title": c, "shape": "rectangle"} for c in classes],
        "region": args.region,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nListo. ok={ok} skip={skipped} fail={failed} (vistas sin cajas válidas: {empty})")
    print(f"Salida: {out.resolve()}  (clases: {classes})")
    print("Cargar con: RDD2022Dataset().load(Path('%s'))" % args.out)


if __name__ == "__main__":
    main()
