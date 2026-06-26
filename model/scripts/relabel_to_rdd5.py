"""Relabela el dataset de Perú (8 clases ES, Supervisely) a las 5 clases de RDD2022 que usa
el modelo R, para poder hacer fine-tuning/transfer respetando los índices del head de R.

Mapeo Perú -> RDD (nombres EXACTOS que observa R, ordenados alfabéticamente por el loader ->
índices: alligator=0, longitudinal=1, other=2, pothole=3, transverse=4):

    bache                    -> pothole
    fisura_longitudinal      -> longitudinal crack
    fisura_oblicua           -> longitudinal crack     (grieta lineal)
    fisura_transversal       -> transverse crack
    piel_de_cocodrilo        -> alligator crack
    reparacion               -> other corruption
    tratamiento_superficial  -> other corruption
    fisura_esquina           -> other corruption

Copia las imágenes (reusa las ya descargadas) y reescribe las anotaciones con los classTitle
mapeados. Las 5 clases quedan todas pobladas -> get_class_names() coincide con el orden de R.

Uso:
    python -m model.scripts.relabel_to_rdd5 --src model/data/local_dataset --dst model/data/peru_rdd5
"""
import argparse
import json
import pathlib
import shutil

MAP = {
    "bache": "pothole",
    "fisura_longitudinal": "longitudinal crack",
    "fisura_oblicua": "longitudinal crack",
    "fisura_transversal": "transverse crack",
    "piel_de_cocodrilo": "alligator crack",
    "reparacion": "other corruption",
    "tratamiento_superficial": "other corruption",
    "fisura_esquina": "other corruption",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="model/data/local_dataset")
    ap.add_argument("--dst", default="model/data/peru_rdd5")
    ap.add_argument("--region", default="Peru")
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)
    src_ann = src / "train" / "ann"
    src_img = src / "train" / "img"
    dst_ann = dst / "train" / "ann"
    dst_img = dst / "train" / "img"
    dst_ann.mkdir(parents=True, exist_ok=True)
    dst_img.mkdir(parents=True, exist_ok=True)

    from collections import Counter
    cls = Counter()
    n_ann = n_img = 0
    for jf in sorted(src_ann.glob("*.json")):
        data = json.loads(jf.read_text(encoding="utf-8"))
        for obj in data.get("objects", []):
            t = obj.get("classTitle")
            if t in MAP:
                obj["classTitle"] = MAP[t]
            cls[obj["classTitle"]] += 1
        (dst_ann / jf.name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        n_ann += 1
        # copiar la imagen correspondiente (<view_id>.jpg)
        img_name = jf.name[:-len(".json")]  # quita .json -> "<id>.jpg"
        src_i = src_img / img_name
        if src_i.exists():
            shutil.copy2(src_i, dst_img / img_name)
            n_img += 1

    (dst / "meta.json").write_text(json.dumps({
        "classes": [{"title": c, "shape": "rectangle"}
                    for c in ["alligator crack", "longitudinal crack", "other corruption",
                              "pothole", "transverse crack"]],
        "region": args.region, "source": "peru relabeled to RDD2022 5-class",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Relabel listo: {n_ann} anotaciones, {n_img} imágenes copiadas -> {dst}")
    print("Distribución (RDD 5-class):")
    for c, n in cls.most_common():
        print(f"  {c:20} {n}")


if __name__ == "__main__":
    main()
