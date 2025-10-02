# CI/model/yolo_infer.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"           # force CPU
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

# --- poids (adapte le nom si besoin) ---
WEIGHTS = Path(__file__).with_name("nwd.pt")  # ex: "nwsd-v2.pt" si c'est ton fichier
DEVICE = "cpu"
CONF_THRES = 0.25
IOU_THRES  = 0.45

# Charge 1x (CPU)
_model = YOLO(str(WEIGHTS))
_model.to(DEVICE)

def predict_mask(
    image_path: str | Path,
    conf: float = CONF_THRES,
    iou: float = IOU_THRES,
) -> np.ndarray:
    """
    Segmentation eau binaire {0,1} alignée sur ta CLI:
      - call: model(path, conf, iou)
      - resize des masks à la taille d'origine
      - union + seuil > 0.5
    """
    image_path = str(image_path)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
    h, w = img.shape[:2]

    results = _model(
        image_path,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    if not results or results[0].masks is None:
        return np.zeros((h, w), dtype=np.uint8)

    masks = results[0].masks.data.cpu().numpy()  # (N, Hm, Wm) en [0..1]
    if masks.size == 0:
        return np.zeros((h, w), dtype=np.uint8)

    resized = [cv2.resize(m.astype(np.float32), (w, h)) for m in masks]
    combined = np.max(resized, axis=0)
    bin01 = (combined > 0.5).astype(np.uint8)  # {0,1}

    return bin01
