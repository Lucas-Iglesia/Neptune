from pathlib import Path
import time
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, DFineForObjectDetection
from ultralytics import YOLO


# Config
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_PATH       = "./homography/data/IMG_6863.MOV"
SEG_MODEL_PATH   = "water-detection/model-v2/nwd-v2.pt"
CONF_THRES       = 0.25
MAP_W_PX, MAP_H_PX = 400, 200
UPDATE_EVERY     = 300
MIN_WATER_AREA_PX = 5_000
DISPLAY_W, DISPLAY_H = 1280, 720
MINIMAP_W, MINIMAP_H = 320, 160
PADDING_PX         = 12

MODEL_ID = "ustc-community/dfine-xlarge-obj2coco" # ustc-community/dfine_x_coco

# Load models
print("• Loading NWD(YOLO11)…")
water_seg = YOLO(SEG_MODEL_PATH)

print("• Loading NHD(D-FINE)…")
processor = AutoImageProcessor.from_pretrained(MODEL_ID) # n - s - m - l - x (model sizes)
dfine = DFineForObjectDetection.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE).eval()

# Person detection
class BoxStub:
    def __init__(self, cx, cy, w, h, conf):
        self.xywh = torch.tensor([[cx, cy, w, h]])
        self.cls  = torch.tensor([0])      # classe 0 = person
        self.conf = torch.tensor([conf])

@torch.inference_mode()
def detect_persons_dfine(frame_bgr, conf_thres=CONF_THRES):
    inputs = processor(images=frame_bgr[:, :, ::-1], return_tensors="pt").to(DEVICE)
    outputs = dfine(**inputs)
    results = processor.post_process_object_detection(outputs, target_sizes=[(frame_bgr.shape[0], frame_bgr.shape[1])], threshold=conf_thres,)[0]

    persons = []
    for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
        if label.item() == 0:              # id COCO 0 = person
            x0, y0, x1, y1 = box.tolist()
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            persons.append(BoxStub(cx, cy, x1 - x0, y1 - y0, score.item()))
    return persons

# Setup minimap homography
DST_RECT = np.array(
    [[0, 0], [MAP_W_PX, 0], [MAP_W_PX, MAP_H_PX], [0, MAP_H_PX]],
    dtype=np.float32,
)
map_canvas_base = np.full((MAP_H_PX, MAP_W_PX, 3), 80, np.uint8)  # gray
H_latest: np.ndarray | None = None

# Loop
cap = cv2.VideoCapture(str(Path(VIDEO_PATH)))
if not cap.isOpened():
    raise IOError(f"Can't open the video: {VIDEO_PATH}")

frame_idx = 0
print("• Processing video (ESC to quit)")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_idx += 1
    tic = time.perf_counter()

    # Update homography
    if frame_idx % UPDATE_EVERY == 1:
        seg_res = water_seg.predict(
            frame, imgsz=512, task="segment", conf=0.25, verbose=False
        )[0]

        if seg_res.masks is not None:
            mask_small = (seg_res.masks.data.cpu().numpy() > 0.5
                          ).any(axis=0).astype(np.uint8)
            mask = cv2.resize(mask_small,
                              (frame.shape[1], frame.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                main_cnt = max(contours, key=cv2.contourArea)
                if cv2.contourArea(main_cnt) > MIN_WATER_AREA_PX:
                    pts   = main_cnt.reshape(-1, 2).astype(np.float32)
                    sums  = pts.sum(axis=1)
                    diffs = np.diff(pts, axis=1).reshape(-1)
                    src_quad = np.array(
                        [pts[np.argmin(sums)],
                         pts[np.argmin(diffs)],
                         pts[np.argmax(sums)],
                         pts[np.argmax(diffs)]],
                        dtype=np.float32,
                    )
                    H_new, _ = cv2.findHomography(src_quad, DST_RECT,
                                                  cv2.RANSAC, 3.0)
                    if H_new is not None:
                        H_latest = H_new.copy()

    if H_latest is None:
        cv2.imshow("Beach Homography", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    persons = detect_persons_dfine(frame)

    # Show human on the homography mini-map
    map_canvas = map_canvas_base.copy()
    if persons:
        centers = np.array(
            [[[float(b.xywh[0][0]), float(b.xywh[0][1])]] for b in persons],
            dtype=np.float32,
        )
        proj = cv2.perspectiveTransform(centers, H_latest)
        for x, y in proj.reshape(-1, 2):
            cv2.circle(map_canvas, (int(x), int(y)), 4, (255, 255, 255), -1)

    # Show box
    vis = frame.copy()
    for b in persons:
        cx, cy, w, h = b.xywh[0]
        x0, y0 = int(cx - w/2), int(cy - h/2)
        x1, y1 = int(cx + w/2), int(cy + h/2)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 0), 2)

    src_poly = cv2.perspectiveTransform(
        DST_RECT[None, :, :], np.linalg.inv(H_latest)
    )[0].astype(int)
    cv2.polylines(vis, [src_poly], True, (0, 255, 0), 3)

    # UI
    vis_small = cv2.resize(vis, (DISPLAY_W, DISPLAY_H), cv2.INTER_AREA)
    map_small = cv2.resize(map_canvas, (MINIMAP_W, MINIMAP_H), cv2.INTER_AREA)
    x0, y0 = DISPLAY_W - MINIMAP_W - PADDING_PX, PADDING_PX
    vis_small[y0:y0 + MINIMAP_H, x0:x0 + MINIMAP_W] = map_small

    toc = time.perf_counter()
    fps = 1.0 / (toc - tic)
    print(f"Frame {frame_idx:04d} | {1000*(toc - tic):6.1f} ms | {fps:5.1f} fps")

    cv2.imshow("Beach Homography", vis_small)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
