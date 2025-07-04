from pathlib import Path
import time
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, DFineForObjectDetection
from ultralytics import YOLO
from collections import defaultdict
import math


# Config
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_PATH       = "./homography/data/IMG_9702[1].MOV"
SEG_MODEL_PATH   = "water-detection/model-v2/nwd-v2.pt"
CONF_THRES       = 0.25
MAP_W_PX, MAP_H_PX = 400, 200
UPDATE_EVERY     = 300
MIN_WATER_AREA_PX = 5_000
DISPLAY_W, DISPLAY_H = 1280, 720
MINIMAP_W, MINIMAP_H = 320, 160
PADDING_PX         = 12

# Tracking config
MAX_DISTANCE_THRESHOLD = 100  # pixels
MAX_FRAMES_DISAPPEARED = 30   # frames before removing a track

MODEL_ID = "ustc-community/dfine-xlarge-obj2coco"

# Load models
print("• Loading NWD(YOLO11)…")
water_seg = YOLO(SEG_MODEL_PATH)

print("• Loading NHD(D-FINE)…")
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
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

# Person tracker
class PersonTracker:
    def __init__(self, max_distance=MAX_DISTANCE_THRESHOLD, max_disappeared=MAX_FRAMES_DISAPPEARED):
        self.next_id = 1
        self.tracks = {}  # {id: {'center': (x, y), 'disappeared': int, 'history': []}}
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared

    def update(self, detections):
        """Update tracker with new detections"""
        if not detections:
            # No detections, increment disappeared counter for all tracks
            to_remove = []
            for track_id in self.tracks:
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    to_remove.append(track_id)

            for track_id in to_remove:
                del self.tracks[track_id]

            return {}

        # Get detection centers
        detection_centers = []
        for det in detections:
            cx, cy = float(det.xywh[0][0]), float(det.xywh[0][1])
            detection_centers.append((cx, cy))

        # If no existing tracks, create new ones
        if not self.tracks:
            assignments = {}
            for i, center in enumerate(detection_centers):
                track_id = self.next_id
                self.tracks[track_id] = {
                    'center': center,
                    'disappeared': 0,
                    'history': [center]
                }
                assignments[i] = track_id
                self.next_id += 1
            return assignments

        # Calculate distances between existing tracks and new detections
        track_ids = list(self.tracks.keys())
        track_centers = [self.tracks[tid]['center'] for tid in track_ids]

        distances = np.zeros((len(track_centers), len(detection_centers)))
        for i, track_center in enumerate(track_centers):
            for j, det_center in enumerate(detection_centers):
                distances[i, j] = math.sqrt(
                    (track_center[0] - det_center[0])**2 +
                    (track_center[1] - det_center[1])**2
                )

        # Hungarian-like assignment (simplified greedy approach)
        assignments = {}
        used_tracks = set()
        used_detections = set()

        # Sort by distance and assign greedily
        distance_pairs = []
        for i in range(len(track_centers)):
            for j in range(len(detection_centers)):
                if distances[i, j] < self.max_distance:
                    distance_pairs.append((distances[i, j], i, j))

        distance_pairs.sort(key=lambda x: x[0])

        for dist, track_idx, det_idx in distance_pairs:
            if track_idx not in used_tracks and det_idx not in used_detections:
                track_id = track_ids[track_idx]
                # Update existing track
                self.tracks[track_id]['center'] = detection_centers[det_idx]
                self.tracks[track_id]['disappeared'] = 0
                self.tracks[track_id]['history'].append(detection_centers[det_idx])
                # Keep history manageable
                if len(self.tracks[track_id]['history']) > 50:
                    self.tracks[track_id]['history'] = self.tracks[track_id]['history'][-50:]

                assignments[det_idx] = track_id
                used_tracks.add(track_idx)
                used_detections.add(det_idx)

        # Create new tracks for unassigned detections
        for j in range(len(detection_centers)):
            if j not in used_detections:
                track_id = self.next_id
                self.tracks[track_id] = {
                    'center': detection_centers[j],
                    'disappeared': 0,
                    'history': [detection_centers[j]]
                }
                assignments[j] = track_id
                self.next_id += 1

        # Mark unassigned tracks as disappeared
        for i in range(len(track_centers)):
            if i not in used_tracks:
                track_id = track_ids[i]
                self.tracks[track_id]['disappeared'] += 1

        # Remove tracks that have been missing too long
        to_remove = []
        for track_id in self.tracks:
            if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.tracks[track_id]

        return assignments

    def get_active_tracks(self):
        """Get all currently active tracks"""
        return {tid: track for tid, track in self.tracks.items()
                if track['disappeared'] == 0}

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

# Initialize tracker
tracker = PersonTracker()

# Colors for different person IDs
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 255), (255, 128, 0), (128, 255, 0), (255, 0, 128)
]

def get_color_for_id(person_id):
    return COLORS[person_id % len(COLORS)]

# Loop
cap = cv2.VideoCapture(str(Path(VIDEO_PATH)))
if not cap.isOpened():
    raise IOError(f"Can't open the video: {VIDEO_PATH}")

frame_idx = 0
print("• Processing video with tracking (ESC to quit)")
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
        cv2.imshow("Beach Homography with Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    # Detect persons and update tracker
    persons = detect_persons_dfine(frame)
    assignments = tracker.update(persons)
    active_tracks = tracker.get_active_tracks()

    # Show tracked persons on the homography mini-map
    map_canvas = map_canvas_base.copy()

    if persons and assignments:
        # Draw on minimap
        for det_idx, track_id in assignments.items():
            if det_idx < len(persons):
                cx, cy = float(persons[det_idx].xywh[0][0]), float(persons[det_idx].xywh[0][1])
                center = np.array([[[cx, cy]]], dtype=np.float32)
                proj = cv2.perspectiveTransform(center, H_latest)
                x, y = proj.reshape(-1, 2)[0]

                if 0 <= x < MAP_W_PX and 0 <= y < MAP_H_PX:
                    color = get_color_for_id(track_id)
                    cv2.circle(map_canvas, (int(x), int(y)), 4, color, -1)
                    cv2.putText(map_canvas, str(track_id), (int(x) + 6, int(y) - 6),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw tracks on minimap
        for track_id, track in active_tracks.items():
            if len(track['history']) > 1:
                color = get_color_for_id(track_id)
                # Convert history to map coordinates
                history_points = []
                for hist_point in track['history'][-10:]:  # Last 10 points
                    center = np.array([[[hist_point[0], hist_point[1]]]], dtype=np.float32)
                    proj = cv2.perspectiveTransform(center, H_latest)
                    x, y = proj.reshape(-1, 2)[0]
                    if 0 <= x < MAP_W_PX and 0 <= y < MAP_H_PX:
                        history_points.append((int(x), int(y)))

                # Draw track
                if len(history_points) > 1:
                    pts = np.array(history_points, dtype=np.int32)
                    cv2.polylines(map_canvas, [pts], False, color, 1)

    # Show bounding boxes with IDs on main frame
    vis = frame.copy()
    for det_idx, person in enumerate(persons):
        track_id = assignments.get(det_idx, -1)
        cx, cy, w, h = person.xywh[0]
        x0, y0 = int(cx - w/2), int(cy - h/2)
        x1, y1 = int(cx + w/2), int(cy + h/2)

        if track_id != -1:
            color = get_color_for_id(track_id)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
            # Draw ID
            cv2.putText(vis, f"ID:{track_id}", (x0, y0 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # Draw confidence
            cv2.putText(vis, f"{person.conf[0]:.2f}", (x0, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw water area outline
    src_poly = cv2.perspectiveTransform(
        DST_RECT[None, :, :], np.linalg.inv(H_latest)
    )[0].astype(int)
    cv2.polylines(vis, [src_poly], True, (0, 255, 0), 3)

    # UI
    vis_small = cv2.resize(vis, (DISPLAY_W, DISPLAY_H), cv2.INTER_AREA)
    map_small = cv2.resize(map_canvas, (MINIMAP_W, MINIMAP_H), cv2.INTER_AREA)
    x0, y0 = DISPLAY_W - MINIMAP_W - PADDING_PX, PADDING_PX
    vis_small[y0:y0 + MINIMAP_H, x0:x0 + MINIMAP_W] = map_small

    # Add tracking info
    active_count = len(active_tracks)
    cv2.putText(vis_small, f"Active tracks: {active_count}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    toc = time.perf_counter()
    fps = 1.0 / (toc - tic)
    print(f"Frame {frame_idx:04d} | {1000*(toc - tic):6.1f} ms | {fps:5.1f} fps | Tracks: {active_count}")

    cv2.imshow("Beach Homography with Tracking", vis_small)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()