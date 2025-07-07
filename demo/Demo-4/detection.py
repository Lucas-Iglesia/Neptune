import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, DFineForObjectDetection
from ultralytics import YOLO
from pathlib import Path
import time
from collections import defaultdict
import math
from datetime import datetime
import pyttsx3
import threading
from gtts import gTTS
import pygame
import io
import tempfile
import os

# === Config ===
VIDEO_PATH = "video/rozel-15-full-hd-cut.mov"
SEG_MODEL_PATH = "model/nwd-v2.pt"
MODEL_ID = "ustc-community/dfine-xlarge-obj2coco"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRES = 0.55
MAP_W_PX, MAP_H_PX = 400, 200
MIN_WATER_AREA_PX = 5_000
UPDATE_EVERY = 30  # frames

# === Voice Alert Setup ===
def speak_alert(message):
    """Function to speak alert message using gTTS for better French pronunciation"""
    def _speak():
        try:
            # Use gTTS for better French pronunciation
            tts = gTTS(text=message, lang='fr', slow=False)
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                tts.save(tmp_file.name)
                
                # Initialize pygame mixer if not already done
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                
                # Play the audio
                pygame.mixer.music.load(tmp_file.name)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                # Clean up
                pygame.mixer.music.unload()
                os.unlink(tmp_file.name)
                
        except Exception as e:
            print(f"Erreur gTTS: {e}")
            # Fallback to pyttsx3
            try:
                engine = pyttsx3.init()
                
                # Try to get French voice if available
                voices = engine.getProperty('voices')
                french_voice = None
                for voice in voices:
                    if 'fr' in voice.id.lower() or 'french' in voice.name.lower():
                        french_voice = voice.id
                        break
                
                if french_voice:
                    engine.setProperty('voice', french_voice)
                
                # Optimize speech settings for clarity
                engine.setProperty('rate', 140)
                engine.setProperty('volume', 0.9)
                
                engine.say(message)
                engine.runAndWait()
                
            except Exception as e2:
                print(f"Erreur pyttsx3 fallback: {e2}")
    
    # Run speech in separate thread to not block main program
    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()

# Display settings
DISPLAY_W, DISPLAY_H = 1920, 1080  # Full HD resolution
MINIMAP_W, MINIMAP_H = 480, 240  # Minimap plus grosse
PADDING_PX = 12

# Tracking config
MAX_DISTANCE_THRESHOLD = 100  # pixels
MAX_FRAMES_DISAPPEARED = 300  # frames before removing a track
UNDERWATER_THRESHOLD = 15     # frames without detection to consider underwater
SURFACE_THRESHOLD = 5         # consecutive detections to consider surfaced
DANGER_TIME_THRESHOLD = 5    # seconds underwater before danger alert

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
        self.cls = torch.tensor([0])  # classe 0 = person

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
            'voice_alert_sent': False,  # New: track if voice alert was sent for this danger event
            'dangerosity_score': 0,
            'distance_from_shore': 0.0,
            'dive_point': None
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
                        track['dive_point'] = track['center']
                        track['danger_alert_sent'] = False
                        track['voice_alert_sent'] = False  # Reset voice alert for new dive
                        print(f"🌊 Person {track_id} went UNDERWATER")

                    # Update underwater duration
                    if track['underwater_start_time']:
                        track['underwater_duration'] = frame_timestamp - track['underwater_start_time']

                        # Check for danger threshold (console alert only)
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
                    track['voice_alert_sent'] = False  # Reset voice alert when surfacing

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
                
                # Update status based on frames underwater
                if track['frames_underwater'] >= UNDERWATER_THRESHOLD:
                    if track['status'] != 'underwater':
                        # Just went underwater
                        track['status'] = 'underwater'
                        track['underwater_start_time'] = frame_timestamp
                        track['dive_point'] = track['center']
                        track['danger_alert_sent'] = False
                        track['voice_alert_sent'] = False  # Reset voice alert for new dive
                        print(f"🌊 Person {track_id} went UNDERWATER")

                    # Update underwater duration
                    if track['underwater_start_time']:
                        track['underwater_duration'] = frame_timestamp - track['underwater_start_time']

                        # Check for danger threshold (console alert only)
                        if (track['underwater_duration'] > DANGER_TIME_THRESHOLD and 
                            not track['danger_alert_sent']):
                            print(f"🚨 DANGER ALERT: Person {track_id} underwater for {track['underwater_duration']:.1f}s!")
                            track['danger_alert_sent'] = True

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

        # Update dangerosity scores for all tracks
        for track_id, track in self.tracks.items():
            # Calculate distance from shore (will be updated later in the main loop)
            distance_from_shore = track.get('distance_from_shore', 0)
            track['dangerosity_score'] = calculate_dangerosity_score(track, frame_timestamp, distance_from_shore)

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

    # Base score for distance from shore (always applies)
    score += int(distance_from_shore * 20)

    # Check if person is diving or underwater based on frames underwater
    if track['frames_underwater'] > 0:
        # Person is diving or underwater - calculate progressive score
        
        # Base diving score (10-30 pts based on frames underwater)
        diving_progress = min(track['frames_underwater'] / UNDERWATER_THRESHOLD, 1.0)
        score += int(10 + (diving_progress * 20))  # 10-30 pts
        
        # If officially underwater, add more points
        if track['status'] == 'underwater':
            score += 20  # Additional 20 pts for being officially underwater
            
            # Time underwater factor (0-40 pts)
            if track['underwater_start_time']:
                t = frame_timestamp - track['underwater_start_time']
                if t > DANGER_TIME_THRESHOLD:
                    score += 40
                    # Unique alert
                    if not track['danger_alert_sent']:
                        print(f"🚨 DANGER ALERT: Person underwater for {t:.1f}s!")
                        track['danger_alert_sent'] = True
                else:
                    score += int((t / DANGER_TIME_THRESHOLD) * 40)
        
        # Frames underwater excess factor (0-10 pts)
        if track['frames_underwater'] > UNDERWATER_THRESHOLD:
            excess = track['frames_underwater'] - UNDERWATER_THRESHOLD
            score += min(10, excess // 10)

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

# Enhanced colors for different states
SURFACE_COLORS = [
    (0, 255, 0), (0, 255, 128), (128, 255, 0), (0, 255, 255), (128, 255, 128)
]
UNDERWATER_COLORS = [
    (255, 0, 0), (255, 128, 0), (255, 0, 128), (128, 0, 255), (255, 64, 64)
]
DANGER_COLOR = (0, 0, 255)  # Bright red for danger

# Initialize underwater tracker
tracker = UnderwaterPersonTracker()

# === Video setup ===
cap = cv2.VideoCapture(str(Path(VIDEO_PATH)))
if not cap.isOpened():
    raise IOError("Cannot open video")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
tracker.frame_rate = fps  # Update tracker frame rate
frame_idx = 0
start_time = time.time()
H_latest = None
water_mask_global = None  # Store water mask for visualization
src_quad_global = None  # Store source quadrilateral for visualization
is_paused = False  # Pause state
show_water_detection = False  # Toggle water detection display - starts OFF

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

print("Running underwater person tracker (SPACE to pause/unpause, W to toggle water detection, T to test voice alert, ESC to quit)")
# Set window to fullscreen
cv2.namedWindow("Underwater Person Tracker", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Underwater Person Tracker", 1920, 1080)
cv2.setWindowProperty("Underwater Person Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    if not is_paused:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        frame_timestamp = start_time + (frame_idx / fps)
        tic = time.perf_counter()

        if H_latest is None:
            cv2.imshow("Underwater Person Tracker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                is_paused = True
            elif key == ord('w') or key == ord('W'):  # W to toggle water detection
                show_water_detection = not show_water_detection
                print(f"Water detection display: {'ON' if show_water_detection else 'OFF'}")
            elif key == ord('t') or key == ord('T'):  # T to test voice alert
                test_message = "Test de l'alerte vocale. Système de surveillance aquatique opérationnel."
                speak_alert(test_message)
                print("🔊 Test d'alerte vocale déclenché")
            continue

        # === Detect people and update tracker ===
        persons = detect_persons(frame)
        assignments = tracker.update(persons, frame_timestamp)
        active_tracks = tracker.get_active_tracks()
        underwater_tracks = tracker.get_underwater_tracks()
        danger_tracks = tracker.get_danger_tracks()

        # === Voice Alert System - Trigger when person enters danger status ===
        current_time = time.time()
        for track_id, track in danger_tracks.items():
            if not track['voice_alert_sent']:
                # Person just entered danger status - send voice alert immediately
                duration = track['underwater_duration'] if 'underwater_duration' in track else 0
                
                # Create short, direct French alert message
                if duration > 0:
                    message = f"Alerte ! Baigneur en danger."
                else:
                    message = f"Alerte ! Baigneur en danger."
                
                speak_alert(message)
                track['voice_alert_sent'] = True
                print(f"🔊 VOICE ALERT: Person {track_id} danger status - alert sent")

        # === Minimap setup ===
        map_canvas = np.full((MAP_H_PX, MAP_W_PX, 3), 50, np.uint8)

        # Draw dive points for all tracks
        for t_id, t in tracker.tracks.items():
            if t['dive_point'] is None:
                continue
            dp = np.array([[[t['dive_point'][0], t['dive_point'][1]]]], np.float32)
            pd = cv2.perspectiveTransform(dp, H_latest).reshape(-1, 2)[0]
            x_d, y_d = int(pd[0]), int(pd[1])
            if 0 <= x_d < MAP_W_PX and 0 <= y_d < MAP_H_PX:
                is_dang = t_id in danger_tracks
                # Use danger color (red) if person is in danger, otherwise use dangerosity color
                if is_dang:
                    col = DANGER_COLOR  # Bright red for danger
                else:
                    col = get_color_by_dangerosity(t['dangerosity_score'])
                
                cv2.drawMarker(
                    map_canvas,
                    (x_d, y_d),
                    color=col,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=10,
                    thickness=2
                )

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

            # Draw track history
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

        # Update dangerosity scores for ALL tracks (including underwater ones)
        for track_id, track in tracker.tracks.items():
            if track_id not in active_tracks and track['center']:
                # For underwater tracks, use their last known position
                cx, cy = track['center']
                center = np.array([[[cx, cy]]], dtype=np.float32)
                proj = cv2.perspectiveTransform(center, H_latest)
                x, y = proj.reshape(-1, 2)[0]
                
                if 0 <= x < MAP_W_PX and 0 <= y < MAP_H_PX:
                    distance_from_shore = calculate_distance_from_shore(x, y, MAP_W_PX, MAP_H_PX)
                    track['distance_from_shore'] = distance_from_shore
                    track['dangerosity_score'] = calculate_dangerosity_score(track, frame_timestamp, distance_from_shore)

        # === Draw detections on main frame ===
        vis = frame.copy()
        for det_idx, person in enumerate(persons):
            track_id = assignments.get(det_idx, -1)
            cx, cy, w, h = person.xywh[0]
            conf = person.conf[0].item()
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
                    if track['status'] == 'underwater':
                        status_text = f"ID:{track_id} (UNDERWATER) - Score:{track['dangerosity_score']}"
                        duration = frame_timestamp - (track['underwater_start_time'] or frame_timestamp)
                        status_text += f" | {duration:.1f}s"
                    else:
                        status_text = f"ID:{track_id} - Score:{track['dangerosity_score']}"

                    cv2.putText(vis, status_text, (x0, y0 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                # Untracked detection
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
                conf_text = f"{conf:.2f}"
                cv2.putText(vis, conf_text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # === Draw danger crosses on main frame for people in danger ===
        for track_id in danger_tracks:
            track = tracker.tracks[track_id]
            if track['dive_point'] is not None:
                # Draw red cross at dive point on main frame
                dive_x, dive_y = int(track['dive_point'][0]), int(track['dive_point'][1])
                cv2.drawMarker(
                    vis,
                    (dive_x, dive_y),
                    color=DANGER_COLOR,  # Bright red
                    markerType=cv2.MARKER_CROSS,
                    markerSize=20,
                    thickness=3
                )


        # === Draw homography quadrilateral points ===
        if src_quad_global is not None and show_water_detection:
            # Draw lines connecting the quadrilateral
            quad_points = src_quad_global.astype(np.int32)
            cv2.polylines(vis, [quad_points], True, (0, 255, 0), 3)  # Green quadrilateral

        # === Display with minimap overlay ===
        # Resize main frame to display size
        frame_resized = cv2.resize(vis, (DISPLAY_W, DISPLAY_H))
        
        # Resize minimap
        map_resized = cv2.resize(map_canvas, (MINIMAP_W, MINIMAP_H))
        
        # Overlay minimap on top-right corner
        x0, y0 = DISPLAY_W - MINIMAP_W - PADDING_PX, PADDING_PX
        frame_resized[y0:y0 + MINIMAP_H, x0:x0 + MINIMAP_W] = map_resized

        # Enhanced status display
        active_count = len(active_tracks)
        underwater_count = len(underwater_tracks)
        danger_count = len(danger_tracks)
        
        # Calculate max dangerosity and ID from ALL tracks
        max_dangerosity = 0
        max_dangerosity_id = None
        if tracker.tracks:
            for track_id, track in tracker.tracks.items():
                if track['dangerosity_score'] > max_dangerosity:
                    max_dangerosity = track['dangerosity_score']
                    max_dangerosity_id = track_id

        status_text = f"Active: {active_count} | Underwater: {underwater_count}"
        if max_dangerosity_id is not None:
            status_text += f" | Max Score: {max_dangerosity} (ID:{max_dangerosity_id})"
        
        cv2.putText(frame_resized, status_text, 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        if danger_count > 0:
            cv2.putText(frame_resized, f"DANGER: {danger_count} person(s)! Max Score: {max_dangerosity}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw danger scale legend
        legend_x, legend_y = 10, DISPLAY_H - 80
        legend_width = 200
        legend_height = 20
        
        # Draw scale background
        cv2.rectangle(frame_resized, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), 
                     (50, 50, 50), -1)
        
        # Draw color gradient
        for i in range(legend_width):
            score = (i / legend_width) * 100
            color = get_color_by_dangerosity(score)
            cv2.line(frame_resized, (legend_x + i, legend_y), (legend_x + i, legend_y + legend_height), color, 1)
        
        # Scale labels removed (white text)
        cv2.putText(frame_resized, "Safe", (legend_x, legend_y + legend_height + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(frame_resized, "Danger", (legend_x + legend_width - 40, legend_y + legend_height + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Add PAUSED text if paused
        if is_paused:
            cv2.putText(frame_resized, "PAUSED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 4)
        
        toc = time.perf_counter()
        fps_current = 1.0 / (toc - tic)
        print(f"Frame {frame_idx:04d} | {1000*(toc - tic):6.1f} ms | {fps_current:5.1f} fps | "
              f"Active: {active_count} | Underwater: {underwater_count} | Danger: {danger_count} | "
              f"Max Score: {max_dangerosity} (ID:{max_dangerosity_id})")
        
        cv2.imshow("Underwater Person Tracker", frame_resized)
    else:
        # Paused - redraw the same frame but allow toggling water detection
        if 'frame_resized' in locals():
            # Recreate the visual frame with current water detection setting
            if 'vis' in locals():
                # Redraw the frame with current water detection setting
                vis_paused = frame.copy() if 'frame' in locals() else vis.copy()
                
                # Redraw detections if available
                if 'persons' in locals() and 'assignments' in locals() and 'active_tracks' in locals():
                    for det_idx, person in enumerate(persons):
                        track_id = assignments.get(det_idx, -1)
                        cx, cy, w, h = person.xywh[0]
                        conf = person.conf[0].item()
                        x0, y0 = int(cx - w/2), int(cy - h/2)
                        x1, y1 = int(cx + w/2), int(cy + h/2)

                        if track_id != -1:
                            track = active_tracks.get(track_id)
                            if track:
                                color = get_color_by_dangerosity(track['dangerosity_score'])
                                is_danger = track_id in danger_tracks if 'danger_tracks' in locals() else False
                                thickness = 3 if is_danger else 2
                                cv2.rectangle(vis_paused, (x0, y0), (x1, y1), color, thickness)

                                # Status text
                                if track['status'] == 'underwater':
                                    status_text = f"ID:{track_id} (UNDERWATER) - Score:{track['dangerosity_score']}"
                                    if 'frame_timestamp' in locals() and track['underwater_start_time']:
                                        duration = frame_timestamp - track['underwater_start_time']
                                        status_text += f" | {duration:.1f}s"
                                else:
                                    status_text = f"ID:{track_id} - Score:{track['dangerosity_score']}"

                                cv2.putText(vis_paused, status_text, (x0, y0 - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        else:
                            # Untracked detection
                            cv2.rectangle(vis_paused, (x0, y0), (x1, y1), (0, 255, 0), 2)
                            conf_text = f"{conf:.2f}"
                            cv2.putText(vis_paused, conf_text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # Draw water detection quadrilateral if enabled
                if src_quad_global is not None and show_water_detection:
                    quad_points = src_quad_global.astype(np.int32)
                    cv2.polylines(vis_paused, [quad_points], True, (0, 255, 0), 3)

                # Resize and add minimap
                frame_resized_paused = cv2.resize(vis_paused, (DISPLAY_W, DISPLAY_H))
                
                # Add minimap if available
                if 'map_resized' in locals():
                    x0, y0 = DISPLAY_W - MINIMAP_W - PADDING_PX, PADDING_PX
                    frame_resized_paused[y0:y0 + MINIMAP_H, x0:x0 + MINIMAP_W] = map_resized

                # Add status text
                if 'active_count' in locals():
                    status_text = f"Active: {active_count} | Underwater: {underwater_count if 'underwater_count' in locals() else 0}"
                    if 'max_dangerosity_id' in locals() and max_dangerosity_id is not None:
                        status_text += f" | Max Score: {max_dangerosity if 'max_dangerosity' in locals() else 0} (ID:{max_dangerosity_id})"
                    
                    cv2.putText(frame_resized_paused, status_text, 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                    if 'danger_count' in locals() and danger_count > 0:
                        cv2.putText(frame_resized_paused, f"DANGER: {danger_count} person(s)! Max Score: {max_dangerosity if 'max_dangerosity' in locals() else 0}",
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Add legend
                legend_x, legend_y = 10, DISPLAY_H - 80
                legend_width = 200
                legend_height = 20
                cv2.rectangle(frame_resized_paused, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), 
                             (50, 50, 50), -1)
                for i in range(legend_width):
                    score = (i / legend_width) * 100
                    color = get_color_by_dangerosity(score)
                    cv2.line(frame_resized_paused, (legend_x + i, legend_y), (legend_x + i, legend_y + legend_height), color, 1)
                

                cv2.putText(frame_resized_paused, "Safe", (legend_x, legend_y + legend_height + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(frame_resized_paused, "Danger", (legend_x + legend_width - 40, legend_y + legend_height + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                # Add PAUSED text
                cv2.putText(frame_resized_paused, "PAUSED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 4)
                
                cv2.imshow("Underwater Person Tracker", frame_resized_paused)
            else:
                # Fallback: just add PAUSED text to the last frame
                frame_with_pause = frame_resized.copy()
                cv2.putText(frame_with_pause, "PAUSED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 4)
                cv2.imshow("Underwater Person Tracker", frame_with_pause)

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
    elif key == ord('t') or key == ord('T'):  # T to test voice alert
        test_message = "Test de l'alerte vocale. Système de surveillance aquatique opérationnel."
        speak_alert(test_message)
        print("🔊 Test d'alerte vocale déclenché")

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
