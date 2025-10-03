import io
import numpy as np
from pathlib import Path
from PIL import Image
import pytest
import allure

from model.yolo_infer import predict_mask

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "images"
MSK_DIR = ROOT / "masks"

IOU_THRESHOLD = 0.80

def binarize(arr: np.ndarray, thr: int = 127) -> np.ndarray:
    return (arr > thr).astype(np.uint8)

def iou_score(pred: np.ndarray, gt: np.ndarray, eps=1e-7) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return (inter + eps) / (union + eps)

def list_pairs():
    imgs = {p.stem: p for p in IMG_DIR.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".tif",".tiff",".bmp"}}
    msks = {p.stem: p for p in MSK_DIR.glob("*.png")}
    common = sorted(set(imgs) & set(msks))
    return [(imgs[k], msks[k]) for k in common], set(imgs) - set(msks), set(msks) - set(imgs)

pairs, missing_masks, orphan_masks = list_pairs()

def _attach_png(pil_img: Image.Image, name: str):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    allure.attach(buf.getvalue(), name=name, attachment_type=allure.attachment_type.PNG)

def _overlay(img_rgb: Image.Image, gt01: np.ndarray, pred01: np.ndarray) -> Image.Image:
    base = np.asarray(img_rgb.convert("RGB")).astype(np.uint8)
    h, w = base.shape[:2]
    gt01  = gt01.astype(bool)
    pr01  = pred01.astype(bool)

    tp = np.logical_and(gt01, pr01)
    fp = np.logical_and(~gt01, pr01)
    fn = np.logical_and(gt01, ~pr01)

    overlay = base.copy()
    overlay[tp] = (overlay[tp] * 0.5 + np.array([0,255,0])*0.5).astype(np.uint8)   # vert
    overlay[fp] = (overlay[fp] * 0.5 + np.array([255,0,0])*0.5).astype(np.uint8)   # rouge
    overlay[fn] = (overlay[fn] * 0.5 + np.array([0,0,255])*0.5).astype(np.uint8)   # bleu
    return Image.fromarray(overlay)

@pytest.mark.parametrize(
    "img_path,msk_path",
    pairs,
    ids=[Path(img).stem for img, _ in pairs],
)
@allure.feature("Water Segmentation")
@allure.story("Per-image IoU ≥ 0.80")
def test_iou_per_image(img_path, msk_path):
    img = Image.open(img_path).convert("RGB")
    gt  = Image.open(msk_path).convert("L")
    gt01 = binarize(np.asarray(gt))

    pred01 = predict_mask(img_path, conf=0.25, iou=0.4)

    _attach_png(img, f"{Path(img_path).stem}__image")
    _attach_png(Image.fromarray((gt01*255).astype(np.uint8)), f"{Path(img_path).stem}__mask_gt")
    _attach_png(Image.fromarray((pred01*255).astype(np.uint8)), f"{Path(img_path).stem}__mask_pred")
    _attach_png(_overlay(img, gt01, pred01), f"{Path(img_path).stem}__overlay_tp_fp_fn")

    assert pred01.shape == gt01.shape, f"Shape mismatch {pred01.shape} vs {gt01.shape}"
    iou = iou_score(pred01, gt01)
    allure.dynamic.title(f"{Path(img_path).stem} — IoU={iou:.3f}")
    allure.dynamic.severity(allure.severity_level.CRITICAL if iou < IOU_THRESHOLD else allure.severity_level.NORMAL)

    assert iou >= IOU_THRESHOLD, f"{Path(img_path).name}: IoU={iou:.3f} < {IOU_THRESHOLD:.2f}"

