from __future__ import annotations

import threading
import queue
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, DFineForObjectDetection
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# Capture vidéo non bloquante
# -----------------------------------------------------------------------------
class FrameGrabber(threading.Thread):
    """Thread producteur qui lit le flux vidéo aussi vite que possible.

    Le dernier frame disponible est conservé dans self.q (Queue de taille 1).
    Si l'inférence est plus lente que la capture, les frames intermédiaires sont
    écrasées ; on évite ainsi toute dérive temporelle (« lag » accumulé).
    """

    def __init__(self, src: str | int, queue_size: int = 1, target_fps: int | None = None):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(str(src))
        if not self.cap.isOpened():
            raise IOError(f"Can't open the video source: {src}")
        # Réduit le tampon interne OpenCV quand le backend le supporte
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.q: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_size)
        self.delay = 1.0 / target_fps if target_fps else 0.0
        self.stopped = False

    def run(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            if not grabbed:
                self.stop()
                break
            if self.delay:
                # Maintient le tempo d'une webcam fixe ; inutile pour un fichier
                time.sleep(self.delay)
            if self.q.full():
                try:
                    _ = self.q.get_nowait()  # éjecte l'ancien frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self, timeout: float | None = None) -> np.ndarray | None:
        """Récupère le frame le plus récent ou None si fin de flux."""
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.stopped = True
        self.cap.release()

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_PATH = "./homography/data/IMG_6863.MOV"  # fichier ou URL webcam/RTSP
SEG_MODEL_PATH = "water-detection/model-v2/nwd-v2.pt"
CONF_THRES = 0.25
MAP_W_PX, MAP_H_PX = 400, 200
UPDATE_EVERY = 300  # on recalcule l'homographie moins souvent
MIN_WATER_AREA_PX = 5_000
DISPLAY_W, DISPLAY_H = 1280, 720
MINIMAP_W, MINIMAP_H = 320, 160
PADDING_PX = 12
TARGET_FPS = 30  # ou 60
TARGET_DT = 1.0 / TARGET_FPS

# -----------------------------------------------------------------------------
# Chargement des modèles
# -----------------------------------------------------------------------------
print("• Loading NWD (YOLO-11)…")
water_seg = YOLO(SEG_MODEL_PATH)

print("• Loading DFINE-X (transformer)…")
processor = AutoImageProcessor.from_pretrained("ustc-community/dfine_x_coco")
dfine = (
    DFineForObjectDetection.from_pretrained(
        "ustc-community/dfine_x_coco",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    .to(DEVICE)
    .eval()
)

# -----------------------------------------------------------------------------
# Détection personnes (identique)
# -----------------------------------------------------------------------------
class BoxStub:
    def __init__(self, cx: float, cy: float, w: float, h: float, conf: float):
        self.xywh = torch.tensor([[cx, cy, w, h]])
        self.cls = torch.tensor([0])  # classe 0 = personne COCO
        self.conf = torch.tensor([conf])

@torch.inference_mode()
def detect_persons_dfine(frame_bgr: np.ndarray, conf_thres: float = CONF_THRES) -> list[BoxStub]:
    inputs = processor(images=frame_bgr[:, :, ::-1], return_tensors="pt").to(DEVICE)
    outputs = dfine(**inputs)
    results = processor.post_process_object_detection(
        outputs, target_sizes=[(frame_bgr.shape[0], frame_bgr.shape[1])], threshold=conf_thres
    )[0]

    persons: list[BoxStub] = []
    for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
        if label.item() == 0:  # id COCO 0 = person
            x0, y0, x1, y1 = box.tolist()
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            persons.append(BoxStub(cx, cy, x1 - x0, y1 - y0, score.item()))
    return persons

# -----------------------------------------------------------------------------
# Homographie minimap
# -----------------------------------------------------------------------------
DST_RECT = np.array([[0, 0], [MAP_W_PX, 0], [MAP_W_PX, MAP_H_PX], [0, MAP_H_PX]], dtype=np.float32)
map_canvas_base = np.full((MAP_H_PX, MAP_W_PX, 3), 80, np.uint8)  # gris
H_latest: np.ndarray | None = None

# -----------------------------------------------------------------------------
# Lancement capture multithread
# -----------------------------------------------------------------------------
print("• Starting capture thread…")
grabber = FrameGrabber(VIDEO_PATH, queue_size=1)  # queue=1 → pas d'accumulation de retard
grabber.start()

frame_idx = 0
last_display = time.perf_counter()
print("• Processing video (ESC to quit)…")

while True:
    frame = grabber.read(timeout=1)
    if frame is None:
        break  # fin du flux vidéo

    frame_idx += 1
    tic = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Segmentation eau → homographie (toutes les UPDATE_EVERY frames)
    # ------------------------------------------------------------------
    if frame_idx % UPDATE_EVERY == 1:
        seg_res = water_seg.predict(frame, imgsz=512, task="segment", conf=0.25, verbose=False)[0]

        if seg_res.masks is not None:
            mask_small = (seg_res.masks.data.cpu().numpy() > 0.5).any(axis=0).astype(np.uint8)
            mask = cv2.resize(mask_small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                main_cnt = max(contours, key=cv2.contourArea)
                if cv2.contourArea(main_cnt) > MIN_WATER_AREA_PX:
                    pts = main_cnt.reshape(-1, 2).astype(np.float32)
                    sums = pts.sum(axis=1)
                    diffs = np.diff(pts, axis=1).reshape(-1)
                    src_quad = np.array(
                        [
                            pts[np.argmin(sums)],
                            pts[np.argmin(diffs)],
                            pts[np.argmax(sums)],
                            pts[np.argmax(diffs)],
                        ],
                        dtype=np.float32,
                    )
                    H_new, _ = cv2.findHomography(src_quad, DST_RECT, cv2.RANSAC, 3.0)
                    if H_new is not None:
                        H_latest = H_new.copy()

    # ------------------------------------------------------------------
    # 2. Si homographie dispo → détection personnes + projection
    # ------------------------------------------------------------------
    if H_latest is None:
        vis_small = cv2.resize(frame, (DISPLAY_W, DISPLAY_H), cv2.INTER_AREA)
    else:
        persons = detect_persons_dfine(frame)

        # Minimap des personnes projetées
        map_canvas = map_canvas_base.copy()
        if persons:
            centers = np.array([[[float(b.xywh[0][0]), float(b.xywh[0][1])]] for b in persons], dtype=np.float32)
            proj = cv2.perspectiveTransform(centers, H_latest)
            for x, y in proj.reshape(-1, 2):
                cv2.circle(map_canvas, (int(x), int(y)), 4, (255, 255, 255), -1)

        # Dessine les bounding boxes
        vis = frame.copy()
        for b in persons:
            cx, cy, w, h = b.xywh[0]
            x0, y0 = int(cx - w / 2), int(cy - h / 2)
            x1, y1 = int(cx + w / 2), int(cy + h / 2)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 0), 2)

        # Dessine le polygone vert (zone couverte)
        src_poly = cv2.perspectiveTransform(DST_RECT[None, :, :], np.linalg.inv(H_latest))[0].astype(int)
        cv2.polylines(vis, [src_poly], True, (0, 255, 0), 3)

        # Compose l'affichage final
        vis_small = cv2.resize(vis, (DISPLAY_W, DISPLAY_H), cv2.INTER_AREA)
        map_small = cv2.resize(map_canvas, (MINIMAP_W, MINIMAP_H), cv2.INTER_AREA)
        x0, y0 = DISPLAY_W - MINIMAP_W - PADDING_PX, PADDING_PX
        vis_small[y0 : y0 + MINIMAP_H, x0 : x0 + MINIMAP_W] = map_small

    # ------------------------------------------------------------------
    # 3. Affichage à TARGET_FPS (cadence constante)
    # ------------------------------------------------------------------
    now = time.perf_counter()
    dt = now - last_display
    if dt < TARGET_DT:
        time.sleep(TARGET_DT - dt)
    last_display = time.perf_counter()

    fps = 1.0 / (time.perf_counter() - tic)
    cv2.putText(
        vis_small,
        f"FPS: {fps:4.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Beach Homography", vis_small)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# -----------------------------------------------------------------------------
# Clean-up
# -----------------------------------------------------------------------------
grabber.stop()
cv2.destroyAllWindows()
