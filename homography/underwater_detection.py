from pathlib import Path
import time
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, DFineForObjectDetection
from ultralytics import YOLO
from collections import defaultdict
import math
from datetime import datetime


# Config
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_PATH       = "./homography/data/IMG_6863.MOV"
SEG_MODEL_PATH   = "water-detection/model-v2/nwd-v2.pt"
CONF_THRES       = 0.4
MAP_W_PX, MAP_H_PX = 400, 200
UPDATE_EVERY     = 300
MIN_WATER_AREA_PX = 5_000
DISPLAY_W, DISPLAY_H = 1280, 720
MINIMAP_W, MINIMAP_H = 320, 160
PADDING_PX         = 12

# Tracking config
MAX_DISTANCE_THRESHOLD = 100  # pixels
MAX_FRAMES_DISAPPEARED = 30   # frames before removing a track
UNDERWATER_THRESHOLD = 15     # frames without detection to consider underwater
SURFACE_THRESHOLD = 5         # consecutive detections to consider surfaced
DANGER_TIME_THRESHOLD = 90    # seconds underwater before danger alert

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

# Enhanced Person tracker for underwater detection
class UnderwaterPersonTracker:
    def __init__(self, max_distance=MAX_DISTANCE_THRESHOLD, max_disappeared=MAX_FRAMES_DISAPPEARED):
        self.next_id = 1
        self.tracks = {}  # Enhanced track structure
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.frame_rate = 30  # Assume 30 fps, can be updated

    def _init_track(self, center, timestamp):
        """Initialize a new track"""
        return {
            'center': center,
            'disappeared': 0,
            'history': [center],
            'status': 'surface',  # 'surface', 'underwater', 'unknown'
            'frames_underwater': 0,
            'frames_on_surface': 0,
            'last_seen_surface': timestamp,
            'underwater_start_time': None,
            'underwater_duration': 0,
            'submersion_events': [],  # List of (start_time, duration) tuples
            'danger_alert_sent': False,
            'dangerosity_score': 0,
            'distance_from_shore': 0.0
        }

    def update(self, detections, frame_timestamp=None):
        """Update tracker with new detections"""
        if frame_timestamp is None:
            frame_timestamp = time.time()

        if not detections:
            # No detections, update underwater status
            to_remove = []
            for track_id in self.tracks:
                track = self.tracks[track_id]
                track['disappeared'] += 1
                track['frames_underwater'] += 1
                track['frames_on_surface'] = 0

                # Update status based on frames underwater
                if track['frames_underwater'] >= UNDERWATER_THRESHOLD:
                    if track['status'] != 'underwater':
                        # Just went underwater
                        track['status'] = 'underwater'
                        track['underwater_start_time'] = frame_timestamp
                        track['danger_alert_sent'] = False
                        print(f"🌊 Person {track_id} went UNDERWATER")

                    # Update underwater duration
                    if track['underwater_start_time']:
                        track['underwater_duration'] = frame_timestamp - track['underwater_start_time']

                        # Check for danger threshold
                        if (track['underwater_duration'] > DANGER_TIME_THRESHOLD and 
                            not track['danger_alert_sent']):
                            print(f"🚨 DANGER ALERT: Person {track_id} underwater for {track['underwater_duration']:.1f}s!")
                            track['danger_alert_sent'] = True

                if track['disappeared'] > self.max_disappeared:
                    # Person completely lost - record final submersion event if underwater
                    if track['status'] == 'underwater' and track['underwater_start_time']:
                        duration = frame_timestamp - track['underwater_start_time']
                        track['submersion_events'].append((track['underwater_start_time'], duration))
                        print(f"📊 Person {track_id} final submersion: {duration:.1f}s")
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
                self.tracks[track_id] = self._init_track(center, frame_timestamp)
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
                track = self.tracks[track_id]

                # Update existing track
                track['center'] = detection_centers[det_idx]
                track['disappeared'] = 0
                track['history'].append(detection_centers[det_idx])
                track['last_seen_surface'] = frame_timestamp
                track['frames_on_surface'] += 1
                track['frames_underwater'] = 0

                # Check if person surfaced
                if (track['status'] == 'underwater' and 
                    track['frames_on_surface'] >= SURFACE_THRESHOLD):
                    # Person surfaced
                    if track['underwater_start_time']:
                        duration = frame_timestamp - track['underwater_start_time']
                        track['submersion_events'].append((track['underwater_start_time'], duration))
                        track['underwater_duration'] = 0
                        print(f"🏄 Person {track_id} SURFACED after {duration:.1f}s underwater")

                    track['status'] = 'surface'
                    track['underwater_start_time'] = None
                    track['danger_alert_sent'] = False

                # Keep history manageable
                if len(track['history']) > 50:
                    track['history'] = track['history'][-50:]

                assignments[det_idx] = track_id
                used_tracks.add(track_idx)
                used_detections.add(det_idx)

        # Create new tracks for unassigned detections
        for j in range(len(detection_centers)):
            if j not in used_detections:
                track_id = self.next_id
                self.tracks[track_id] = self._init_track(detection_centers[j], frame_timestamp)
                assignments[j] = track_id
                self.next_id += 1

        # Mark unassigned tracks as disappeared (potential underwater)
        for i in range(len(track_centers)):
            if i not in used_tracks:
                track_id = track_ids[i]
                track = self.tracks[track_id]
                track['disappeared'] += 1
                track['frames_underwater'] += 1
                track['frames_on_surface'] = 0

        # Remove tracks that have been missing too long
        to_remove = []
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            if track['disappeared'] > self.max_disappeared:
                # Record final submersion event if underwater
                if track['status'] == 'underwater' and track['underwater_start_time']:
                    duration = frame_timestamp - track['underwater_start_time']
                    track['submersion_events'].append((track['underwater_start_time'], duration))
                    print(f"📊 Person {track_id} final submersion: {duration:.1f}s")
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.tracks[track_id]

        return assignments

    def get_active_tracks(self):
        """Get all currently active tracks"""
        return {tid: track for tid, track in self.tracks.items()
                if track['disappeared'] <= UNDERWATER_THRESHOLD}

    def get_underwater_tracks(self):
        """Get tracks of people currently underwater"""
        return {tid: track for tid, track in self.tracks.items()
                if track['status'] == 'underwater'}

    def get_danger_tracks(self):
        """Get tracks of people in danger (underwater too long)"""
        current_time = time.time()
        danger_tracks = {}
        for tid, track in self.tracks.items():
            if (track['status'] == 'underwater' and 
                track['underwater_start_time'] and
                (current_time - track['underwater_start_time']) > DANGER_TIME_THRESHOLD):
                danger_tracks[tid] = track
        return danger_tracks

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

# Initialize underwater tracker
tracker = UnderwaterPersonTracker()

# Enhanced colors for different states
SURFACE_COLORS = [
    (0, 255, 0), (0, 255, 128), (128, 255, 0), (0, 255, 255), (128, 255, 128)
]
UNDERWATER_COLORS = [
    (255, 0, 0), (255, 128, 0), (255, 0, 128), (128, 0, 255), (255, 64, 64)
]
DANGER_COLOR = (0, 0, 255)  # Bright red for danger

def calculate_dangerosity_score(track, frame_timestamp, distance_from_shore=0):
    """
    Calculate dangerosity score from 0 to 100
    
    Args:
        track: Person track data
        frame_timestamp: Current frame timestamp
        distance_from_shore: Distance from shore (0-1, where 1 is farthest)
    
    Returns:
        int: Dangerosity score (0-100)
    """
    score = 0
    
    # Base score - not in water = 0
    if track['status'] != 'underwater':
        # Surface score based on proximity to water edge
        score = min(20, int(distance_from_shore * 20))
        return score
    
    # Underwater scoring
    score = 30  # Base underwater score
    
    # Time underwater factor (0-40 points)
    if track['underwater_start_time']:
        underwater_time = frame_timestamp - track['underwater_start_time']
        if underwater_time > DANGER_TIME_THRESHOLD:
            score += 40  # Maximum time danger
        else:
            score += int((underwater_time / DANGER_TIME_THRESHOLD) * 40)
    
    # Distance from shore factor (0-20 points)
    score += int(distance_from_shore * 20)
    
    # Consecutive underwater frames factor (0-10 points)
    if track['frames_underwater'] > UNDERWATER_THRESHOLD:
        excess_frames = track['frames_underwater'] - UNDERWATER_THRESHOLD
        score += min(10, int(excess_frames / 10))
    
    return min(100, score)

def get_color_by_dangerosity(score):
    """
    Get color based on dangerosity score with gradient
    
    Args:
        score: Dangerosity score (0-100)
    
    Returns:
        tuple: BGR color tuple
    """
    if score <= 20:
        # Green gradient (dark to light green)
        # Dark green (0, 100, 0) to light green (144, 238, 144)
        ratio = score / 20.0
        b = int(144 * ratio)
        g = int(100 + (138 * ratio))
        r = int(144 * ratio)
        return (b, g, r)
    
    elif score <= 40:
        # Light green to yellow
        # Light green (144, 238, 144) to yellow (0, 255, 255)
        ratio = (score - 20) / 20.0
        b = int(144 * (1 - ratio))
        g = int(238 + (17 * ratio))
        r = int(144 + (111 * ratio))
        return (b, g, r)
    
    elif score <= 60:
        # Yellow to orange
        # Yellow (0, 255, 255) to orange (0, 165, 255)
        ratio = (score - 40) / 20.0
        b = 0
        g = int(255 - (90 * ratio))
        r = 255
        return (b, g, r)
    
    elif score <= 80:
        # Orange to red
        # Orange (0, 165, 255) to red (0, 0, 255)
        ratio = (score - 60) / 20.0
        b = 0
        g = int(165 * (1 - ratio))
        r = 255
        return (b, g, r)
    
    else:
        # Red to dark red
        # Red (0, 0, 255) to dark red (0, 0, 139)
        ratio = (score - 80) / 20.0
        b = 0
        g = 0
        r = int(255 - (116 * ratio))
        return (b, g, r)

def calculate_distance_from_shore(x, y, map_width, map_height):
    """
    Calculate normalized distance from shore (0-1)
    Assumes shore is at the bottom of the map
    
    Args:
        x, y: Position coordinates
        map_width, map_height: Map dimensions
    
    Returns:
        float: Distance from shore (0-1, where 1 is farthest)
    """
    # Simple distance from bottom edge (shore)
    distance_from_bottom = (map_height - y) / map_height
    return max(0, min(1, distance_from_bottom))

def get_color_for_track(track_id, status, is_danger=False):
    if is_danger:
        return DANGER_COLOR
    elif status == 'underwater':
        return UNDERWATER_COLORS[track_id % len(UNDERWATER_COLORS)]
    else:
        return SURFACE_COLORS[track_id % len(SURFACE_COLORS)]

# Loop
cap = cv2.VideoCapture(str(Path(VIDEO_PATH)))
if not cap.isOpened():
    raise IOError(f"Can't open the video: {VIDEO_PATH}")

# Get video properties for accurate timing
fps = cap.get(cv2.CAP_PROP_FPS) or 30
tracker.frame_rate = fps

frame_idx = 0
start_time = time.time()
print("• Processing video with underwater tracking (ESC to quit)")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_idx += 1
    frame_timestamp = start_time + (frame_idx / fps)
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
        cv2.imshow("Underwater Person Tracker", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    # Detect persons and update tracker
    persons = detect_persons_dfine(frame)
    assignments = tracker.update(persons, frame_timestamp)
    active_tracks = tracker.get_active_tracks()
    underwater_tracks = tracker.get_underwater_tracks()
    danger_tracks = tracker.get_danger_tracks()

    # Show tracked persons on the homography mini-map
    map_canvas = map_canvas_base.copy()

    # Draw all active tracks on minimap
    for track_id, track in active_tracks.items():
        if track['history']:
            cx, cy = track['history'][-1]  # Latest position
            center = np.array([[[cx, cy]]], dtype=np.float32)
            proj = cv2.perspectiveTransform(center, H_latest)
            x, y = proj.reshape(-1, 2)[0]

            if 0 <= x < MAP_W_PX and 0 <= y < MAP_H_PX:
                # Calculate distance from shore and dangerosity score
                distance_from_shore = calculate_distance_from_shore(x, y, MAP_W_PX, MAP_H_PX)
                track['distance_from_shore'] = distance_from_shore
                track['dangerosity_score'] = calculate_dangerosity_score(track, frame_timestamp, distance_from_shore)
                
                # Use dangerosity-based color
                color = get_color_by_dangerosity(track['dangerosity_score'])
                is_danger = track_id in danger_tracks

                # Different symbols for different states
                if track['status'] == 'underwater':
                    # Draw larger circle for underwater
                    cv2.circle(map_canvas, (int(x), int(y)), 6, color, -1)
                    if is_danger:
                        # Pulsing effect for danger
                        pulse_size = 8 + int(2 * math.sin(frame_idx * 0.3))
                        cv2.circle(map_canvas, (int(x), int(y)), pulse_size, DANGER_COLOR, 2)
                else:
                    # Regular circle for surface
                    cv2.circle(map_canvas, (int(x), int(y)), 4, color, -1)

                # Add ID and dangerosity score
                cv2.putText(map_canvas, f"{track_id}({track['dangerosity_score']})", 
                           (int(x) + 8, int(y) - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # Draw track history with color gradient
        if len(track['history']) > 1:
            color = get_color_by_dangerosity(track['dangerosity_score'])
            history_points = []
            for hist_point in track['history'][-15:]:  # Last 15 points
                center = np.array([[[hist_point[0], hist_point[1]]]], dtype=np.float32)
                proj = cv2.perspectiveTransform(center, H_latest)
                x, y = proj.reshape(-1, 2)[0]
                if 0 <= x < MAP_W_PX and 0 <= y < MAP_H_PX:
                    history_points.append((int(x), int(y)))

            if len(history_points) > 1:
                pts = np.array(history_points, dtype=np.int32)
                cv2.polylines(map_canvas, [pts], False, color, 1)

    # Show detection boxes and status on main frame
    vis = frame.copy()
    for det_idx, person in enumerate(persons):
        track_id = assignments.get(det_idx, -1)
        cx, cy, w, h = person.xywh[0]
        x0, y0 = int(cx - w/2), int(cy - h/2)
        x1, y1 = int(cx + w/2), int(cy + h/2)

        if track_id != -1:
            track = active_tracks.get(track_id)
            if track:
                # Use dangerosity-based color
                color = get_color_by_dangerosity(track['dangerosity_score'])
                is_danger = track_id in danger_tracks

                # Different box styles for different states
                thickness = 3 if is_danger else 2
                cv2.rectangle(vis, (x0, y0), (x1, y1), color, thickness)

                # Status, ID, and dangerosity score
                status_text = f"ID:{track_id} ({track['status'].upper()}) - Score: {track['dangerosity_score']}"
                if track['status'] == 'underwater':
                    duration = frame_timestamp - (track['underwater_start_time'] or frame_timestamp)
                    status_text += f" | {duration:.1f}s"

                cv2.putText(vis, status_text, (x0, y0 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Confidence and distance from shore
                info_text = f"Conf: {person.conf[0]:.2f} | Shore: {track['distance_from_shore']:.2f}"
                cv2.putText(vis, info_text, (x0, y1 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Draw dangerosity bar
                bar_width = int(w * 0.8)
                bar_height = 6
                bar_x = x0 + int((w - bar_width) / 2)
                bar_y = y1 + 35
                
                # Background bar
                cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                             (50, 50, 50), -1)
                
                # Dangerosity bar fill
                fill_width = int((track['dangerosity_score'] / 100.0) * bar_width)
                if fill_width > 0:
                    cv2.rectangle(vis, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                                 color, -1)

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

    # Enhanced status display
    active_count = len(active_tracks)
    underwater_count = len(underwater_tracks)
    danger_count = len(danger_tracks)
    
    # Calculate average dangerosity
    avg_dangerosity = 0
    max_dangerosity = 0
    if active_tracks:
        total_dangerosity = sum(track['dangerosity_score'] for track in active_tracks.values())
        avg_dangerosity = total_dangerosity / len(active_tracks)
        max_dangerosity = max(track['dangerosity_score'] for track in active_tracks.values())

    cv2.putText(vis_small, f"Active: {active_count} | Underwater: {underwater_count} | Avg Score: {avg_dangerosity:.1f}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if danger_count > 0:
        cv2.putText(vis_small, f"DANGER: {danger_count} person(s)! Max Score: {max_dangerosity}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw danger scale legend
    legend_x, legend_y = 10, DISPLAY_H - 80
    legend_width = 200
    legend_height = 20
    
    # Draw scale background
    cv2.rectangle(vis_small, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), 
                 (50, 50, 50), -1)
    
    # Draw color gradient
    for i in range(legend_width):
        score = (i / legend_width) * 100
        color = get_color_by_dangerosity(score)
        cv2.line(vis_small, (legend_x + i, legend_y), (legend_x + i, legend_y + legend_height), color, 1)
    
    # Scale labels
    cv2.putText(vis_small, "Dangerosity: 0", (legend_x, legend_y - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(vis_small, "100", (legend_x + legend_width - 25, legend_y - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(vis_small, "Safe", (legend_x, legend_y + legend_height + 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(vis_small, "Danger", (legend_x + legend_width - 40, legend_y + legend_height + 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    toc = time.perf_counter()
    fps_current = 1.0 / (toc - tic)
    print(f"Frame {frame_idx:04d} | {1000*(toc - tic):6.1f} ms | {fps_current:5.1f} fps | "
          f"Active: {active_count} | Underwater: {underwater_count} | Danger: {danger_count} | "
          f"Avg Score: {avg_dangerosity:.1f} | Max Score: {max_dangerosity}")

    cv2.imshow("Underwater Person Tracker", vis_small)
    if cv2.waitKey(1) & 0xFF == 27:
        break

print("\nFinal Statistics:")
for track_id, track in tracker.tracks.items():
    print(f"Person {track_id}:")
    print(f"  - Final dangerosity score: {track.get('dangerosity_score', 0)}")
    print(f"  - Distance from shore: {track.get('distance_from_shore', 0):.2f}")
    print(f"  - Total submersion events: {len(track['submersion_events'])}")
    if track['submersion_events']:
        total_time = sum(duration for _, duration in track['submersion_events'])
        avg_time = total_time / len(track['submersion_events'])
        max_time = max(duration for _, duration in track['submersion_events'])
        print(f"  - Total underwater time: {total_time:.1f}s")
        print(f"  - Average submersion: {avg_time:.1f}s")
        print(f"  - Longest submersion: {max_time:.1f}s")

cap.release()
cv2.destroyAllWindows()