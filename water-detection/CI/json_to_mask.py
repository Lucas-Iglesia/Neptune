from __future__ import annotations
from pathlib import Path
import argparse, json, glob
from PIL import Image, ImageDraw

WATER_KEYWORDS = ("water", "eau")

def is_water_label(lbl: str) -> bool:
    return any(k in lbl.lower() for k in WATER_KEYWORDS)

def load_json_paths(ann_dir: Path | None, json_glob: list[str] | None) -> list[Path]:
    if json_glob:
        files: list[Path] = []
        for patt in json_glob:
            files.extend(Path(p).resolve() for p in glob.glob(patt))
        files = sorted(set(files))
        if not files:
            raise SystemExit(f"Aucun JSON ne correspond au motif: {json_glob}")
        return files
    if ann_dir:
        files = sorted((ann_dir).glob("*.json"))
        if not files:
            raise SystemExit(f"Aucun JSON dans {ann_dir}")
        return files
    raise SystemExit("Spécifie --ann-dir ou --json")

def find_image_for_json(jf: Path, data: dict, img_dir: Path) -> Path | None:
    img_path = data.get("imagePath")
    if img_path:
        cand = img_dir / Path(img_path).name
        if cand.exists():
            return cand
    for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]:
        cand = img_dir / (jf.stem + ext)
        if cand.exists():
            return cand
    return None

def get_size_from_data_or_image(data: dict, jf: Path, img_dir: Path) -> tuple[int,int]:
    h = data.get("imageHeight")
    w = data.get("imageWidth")
    if h and w:
        return int(w), int(h)
    img = find_image_for_json(jf, data, img_dir)
    if not img:
        raise FileNotFoundError(f"Image introuvable pour {jf.name} (imagePath manquant et aucun fichier correspondant dans {img_dir})")
    with Image.open(img) as im:
        return im.size  # (w,h)

def draw_mask_from_labelme(data: dict, size_wh: tuple[int,int]) -> Image.Image:
    w, h = size_wh
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    n_poly = 0
    for shp in data.get("shapes", []):
        label = shp.get("label", "")
        if not is_water_label(label):
            continue
        pts = shp.get("points", [])
        if len(pts) >= 3:
            draw.polygon([tuple(p) for p in pts], fill=255)
            n_poly += 1
    if n_poly == 0:
        print(f"[WARN] Aucun polygone 'water/eau' dans {data.get('imagePath','(sans nom)')}")
    return mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", type=Path, help="Dossier contenant les JSON LabelMe")
    ap.add_argument("--json", nargs="+", help="Un ou plusieurs JSON (ou globs)")
    ap.add_argument("--img-dir", type=Path, default=Path("images"), help="Dossier des images d'origine")
    ap.add_argument("--out-dir", type=Path, default=Path("masks"), help="Dossier de sortie des PNG")
    args = ap.parse_args()

    json_files = load_json_paths(args.ann_dir, args.json)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        size_wh = get_size_from_data_or_image(data, jf, args.img_dir)
        mask = draw_mask_from_labelme(data, size_wh)

        out_path = args.out_dir / f"{jf.stem}.png"
        mask.save(out_path)
        print(f"[OK] {jf.name} -> {out_path}")

if __name__ == "__main__":
    main()
