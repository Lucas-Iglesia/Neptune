import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, DFineForObjectDetection
from ultralytics import YOLO
from pathlib import Path
import time

# === Config ===
VIDEO_PATH = "video/IMG_6863_1080p15.mov"
SEG_MODEL_PATH = "model/nwd-v2.pt"
MODEL_ID = "ustc-community/dfine-xlarge-obj2coco"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRES = 0.4
MAP_W_PX, MAP_H_PX = 400, 200
MIN_WATER_AREA_PX = 5_000
UPDATE_EVERY = 30  # frames

# Display settings
DISPLAY_W, DISPLAY_H = 1920, 1080  # Full HD resolution
MINIMAP_W, MINIMAP_H = 480, 240  # Minimap plus grosse
PADDING_PX = 12

# === Homography target ===
DST_RECT = np.array(
    [[0, 0], [MAP_W_PX, 0], [MAP_W_PX, MAP_H_PX], [0, MAP_H_PX]],
    dtype=np.float32,
)

# === Load models ===
print("Loading NWSD...")
nwsd = YOLO(SEG_MODEL_PATH)

print("Loading D-FINE...")
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
dfine = DFineForObjectDetection.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE).eval()

# === Utils ===
class BoxStub:
    def __init__(self, cx, cy, w, h, conf):
        self.xywh = torch.tensor([[cx, cy, w, h]])
        self.conf = torch.tensor([conf])

@torch.inference_mode()
def detect_persons(frame_bgr):
    inputs = processor(images=frame_bgr[:, :, ::-1], return_tensors="pt").to(DEVICE)
    if DEVICE == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)
    outputs = dfine(**inputs)
    results = processor.post_process_object_detection(
        outputs, target_sizes=[(frame_bgr.shape[0], frame_bgr.shape[1])], threshold=CONF_THRES
    )[0]

    persons = []
    for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
        if label.item() == 0:
            x0, y0, x1, y1 = box.tolist()
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            persons.append(BoxStub(cx, cy, x1 - x0, y1 - y0, score.item()))
    return persons

# === Video setup ===
cap = cv2.VideoCapture(str(Path(VIDEO_PATH)))
if not cap.isOpened():
    raise IOError("Cannot open video")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_idx = 0
H_latest = None
water_mask_global = None  # Store water mask for visualization
src_quad_global = None  # Store source quadrilateral for visualization
is_paused = False  # Pause state
show_water_detection = True  # Toggle water detection display

# === Run water detection once at the beginning ===
print("Running initial water detection...")
ok, first_frame = cap.read()
if ok:
    seg_res = nwsd.predict(first_frame, imgsz=512, task="segment", conf=0.25, verbose=False)[0]
    if seg_res.masks is not None:
        mask = (seg_res.masks.data.cpu().numpy() > 0.5).any(axis=0).astype(np.uint8)
        mask = cv2.resize(mask, (first_frame.shape[1], first_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        water_mask_global = mask.copy()  # Store for visualization
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) > MIN_WATER_AREA_PX:
                pts = cnt.reshape(-1, 2).astype(np.float32)
                sums = pts.sum(axis=1)
                diffs = np.diff(pts, axis=1).reshape(-1)
                src_quad = np.array(
                    [pts[np.argmin(sums)],
                     pts[np.argmin(diffs)],
                     pts[np.argmax(sums)],
                     pts[np.argmax(diffs)]],
                    dtype=np.float32,
                )
                src_quad_global = src_quad.copy()  # Store for visualization
                H, _ = cv2.findHomography(src_quad, DST_RECT, cv2.RANSAC, 3.0)
                if H is not None:
                    H_latest = H.copy()
                    print("Homography matrix computed successfully!")
                else:
                    print("Failed to compute homography matrix")
            else:
                print("Water area too small")
        else:
            print("No water contours found")
    else:
        print("No water masks detected")
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
else:
    raise IOError("Cannot read first frame from video")

print("Running demo (SPACE to pause/unpause, W to toggle water detection, ESC to quit)")
# Set window to fullscreen
cv2.namedWindow("Homography Demo", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Homography Demo", 1920, 1080)
cv2.setWindowProperty("Homography Demo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    if not is_paused:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if H_latest is None:
            cv2.imshow("Homography Demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                is_paused = True
            elif key == ord('w') or key == ord('W'):  # W to toggle water detection
                show_water_detection = not show_water_detection
                print(f"Water detection display: {'ON' if show_water_detection else 'OFF'}")
            continue

        # === Detect people ===
        persons = detect_persons(frame)

        # === Minimap setup ===
        map_canvas = np.full((MAP_H_PX, MAP_W_PX, 3), 50, np.uint8)

        # === Draw detections ===
        for person in persons:
            cx, cy, w, h = person.xywh[0]
            conf = person.conf[0].item()
            x0, y0 = int(cx - w/2), int(cy - h/2)
            x1, y1 = int(cx + w/2), int(cy + h/2)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            
            # Add confidence text in black
            conf_text = f"{conf:.2f}"
            cv2.putText(frame, conf_text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Project on minimap
            pt = np.array([[[cx.item(), cy.item()]]], dtype=np.float32)
            projected = cv2.perspectiveTransform(pt, H_latest)
            x_map, y_map = projected.reshape(-1, 2)[0]
            if 0 <= x_map < MAP_W_PX and 0 <= y_map < MAP_H_PX:
                cv2.circle(map_canvas, (int(x_map), int(y_map)), 5, (0, 0, 255), -1)

        # === Draw homography quadrilateral points ===
        if src_quad_global is not None and show_water_detection:
            # Draw lines connecting the quadrilateral
            quad_points = src_quad_global.astype(np.int32)
            cv2.polylines(frame, [quad_points], True, (0, 255, 0), 3)  # Green quadrilateral

        # === Display with minimap overlay ===
        # Resize main frame to display size
        frame_resized = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
        
        # Resize minimap
        map_resized = cv2.resize(map_canvas, (MINIMAP_W, MINIMAP_H))
        
        # Overlay minimap on top-right corner (no border, no text)
        x0, y0 = DISPLAY_W - MINIMAP_W - PADDING_PX, PADDING_PX
        frame_resized[y0:y0 + MINIMAP_H, x0:x0 + MINIMAP_W] = map_resized
        
        # Add PAUSED text if paused
        if is_paused:
            cv2.putText(frame_resized, "PAUSED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 4)
        
        cv2.imshow("Homography Demo", frame_resized)
    else:
        # Paused - just redraw the same frame with boxes
        if 'frame_resized' in locals():
            # Add PAUSED text when displaying paused frame
            frame_with_pause = frame_resized.copy()
            cv2.putText(frame_with_pause, "PAUSED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 4)
            cv2.imshow("Homography Demo", frame_with_pause)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        is_paused = not is_paused
        if is_paused:
            print("PAUSED - Press SPACE to continue")
        else:
            print("RESUMED")
    elif key == ord('w') or key == ord('W'):  # W to toggle water detection
        show_water_detection = not show_water_detection
        print(f"Water detection display: {'ON' if show_water_detection else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
