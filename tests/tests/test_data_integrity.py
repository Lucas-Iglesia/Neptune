from pathlib import Path

# Racine = dossier CI (parent de tests)
ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "nwsd/images"
MSK_DIR = ROOT / "nwsd/masks"

def _list_pairs():
    imgs = {p.stem: p for p in IMG_DIR.glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png",".tif",".tiff",".bmp"}}
    msks = {p.stem: p for p in MSK_DIR.glob("*.png")}
    common = set(imgs) & set(msks)
    return [(imgs[k], msks[k]) for k in sorted(common)], sorted(set(imgs) - set(msks)), sorted(set(msks) - set(imgs))

pairs, missing_masks, orphan_masks = _list_pairs()

def test_images_not_empty():
    assert IMG_DIR.exists(), f"Directory not found: {IMG_DIR}"
    imgs = [p for p in IMG_DIR.glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png",".tif",".tiff",".bmp"}]
    assert imgs, f"No image in {IMG_DIR}"

def test_masks_not_empty():
    assert MSK_DIR.exists(), f"Directory not found: {MSK_DIR}"
    msks = list(MSK_DIR.glob("*.png"))
    assert msks, f"No PNG mask in {MSK_DIR}"

def test_every_image_has_a_mask():
    assert len(missing_masks) == 0, f"Missing mask for: {sorted(missing_masks)}"

def test_every_mask_has_an_image():
    assert len(orphan_masks) == 0, f"Missing image for: {sorted(orphan_masks)}"