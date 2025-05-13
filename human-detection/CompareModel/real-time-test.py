import cv2
import time
import argparse
from collections import deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import math

# -------- CONFIGURATION --------
VIDEO_PATH        = "./video-test-converted/IMG_6865.MOV"
MODEL_PATH        = "./model/nhd-v3.pt"
CONF_THRESH       = 0.6    # Confidence threshold for YOLO detections
IOU_THRESH        = 0.3    # IoU threshold for tracking
# Tracking parameters (appearance disabled)
MAX_AGE           = 60     # Max frames to keep track without detections
N_INIT            = 3      # Frames needed to confirm a track
MAX_COSINE_DIST   = 0.8    # Disable appearance matching
NN_BUDGET         = 50     # Limited appearance descriptors
# Smoothing
SMOOTH_WINDOW     = 5      # Frames for bbox smoothing
# Velocity gating
VELOCITY_THRESH   = 50     # Max pixels movement per frame
# --------------------------------


def process_video(video_path):
    # Load YOLO model
    model = YOLO(MODEL_PATH)

    # Initialize DeepSORT tracker with IoU gating
    deepsort = DeepSort(
        max_age=MAX_AGE,
        n_init=N_INIT,
        max_cosine_distance=MAX_COSINE_DIST,
        nn_budget=NN_BUDGET,
        embedder="mobilenet",
        half=False,
        bgr=True,
        max_iou_distance=IOU_THRESH
    )

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = 1.0 / fps

    # Buffers for smoothing, velocity gating, and trajectories
    track_buffers = {}    # track_id -> deque of bboxes
    last_centers = {}     # track_id -> last (cx, cy)
    trajectories = {}     # track_id -> deque of past centers

    # Create and maximize window
    cv2.namedWindow("YOLO + DeepSORT", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("YOLO + DeepSORT", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detections
        detections = []
        for res in model(frame, stream=True):
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            cls_ids = res.boxes.cls.cpu().numpy()
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                if cls_ids[i] == 0 and confs[i] >= CONF_THRESH:
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    w, h = x2 - x1, y2 - y1
                    detections.append(([x1, y1, w, h], float(confs[i]), "person"))

        # Update tracker
        tracks = deepsort.update_tracks(detections, frame=frame)

        # Draw each confirmed track with smoothing, velocity gating, and trajectory
        for tr in tracks:
            if not tr.is_confirmed() or tr.time_since_update != 0:
                continue
            tid = tr.track_id
            l, t, r, b = map(int, tr.to_ltrb())
            # Smoothing bbox
            if tid not in track_buffers:
                track_buffers[tid] = deque(maxlen=SMOOTH_WINDOW)
            buf = track_buffers[tid]
            buf.append((l, t, r, b))
            avg_l = int(sum(v[0] for v in buf) / len(buf))
            avg_t = int(sum(v[1] for v in buf) / len(buf))
            avg_r = int(sum(v[2] for v in buf) / len(buf))
            avg_b = int(sum(v[3] for v in buf) / len(buf))
            # Center point
            cx = (avg_l + avg_r) // 2
            cy = (avg_t + avg_b) // 2
            # Velocity gating
            if tid in last_centers:
                prev_cx, prev_cy = last_centers[tid]
                if math.hypot(cx - prev_cx, cy - prev_cy) > VELOCITY_THRESH:
                    continue
            last_centers[tid] = (cx, cy)
            # Trajectory
            if tid not in trajectories:
                trajectories[tid] = deque(maxlen=SMOOTH_WINDOW * 5)
            traj = trajectories[tid]
            traj.append((cx, cy))
            # Draw trajectory
            for j in range(1, len(traj)):
                cv2.line(frame, traj[j-1], traj[j], (255, 0, 0), 2)
            # Draw bbox and ID
            cv2.rectangle(frame, (avg_l, avg_t), (avg_r, avg_b), (0, 255, 0), 2)
            cv2.putText(frame, str(tid), (avg_l, avg_t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display
        cv2.imshow("YOLO + DeepSORT", frame)
        elapsed = time.time() - start
        wait = max(int((interval - elapsed) * 1000), 1)
        if cv2.waitKey(wait) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time YOLO + DeepSORT with trajectories")
    parser.add_argument("video", help="Path to the input video file")
    args = parser.parse_args()
    process_video(args.video)
