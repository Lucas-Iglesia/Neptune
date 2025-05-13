import cv2
import time
import argparse
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -------- CONFIGURATION --------
VIDEO_PATH      = "./video-test-converted/IMG_6865.MOV"
MODEL_PATH      = "model/nhd-v3.pt"
CONF_THRESH     = 0.6   # Confidence threshold for YOLO detections
# Tracking parameters (appearance disabled)
MAX_AGE         = 60   # Max frames to keep track without detections
N_INIT          = 3     # Frames needed to confirm a track
MAX_COSINE_DIST = 0.8   # Disable appearance matching (always allow)
NN_BUDGET       = 50     # No appearance descriptors stored
# --------------------------------

def process_video(video_path):
    # Load YOLO model
    model = YOLO(MODEL_PATH)

    # Initialize DeepSORT tracker with appearance disabled
    deepsort = DeepSort(
        max_age=MAX_AGE,
        n_init=N_INIT,
        max_cosine_distance=MAX_COSINE_DIST,
        nn_budget=NN_BUDGET,
        half=False,
        bgr=True
    )

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = 1.0 / fps

    # Create and maximize window
    cv2.namedWindow("YOLO + DeepSORT", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("YOLO + DeepSORT", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference on full frame
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

        # Update tracker with detections
        tracks = deepsort.update_tracks(detections, frame=frame)

        # Draw only confirmed tracks from actual detections
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update != 0:
                continue
            l, t, r, b = track.to_ltrb()
            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
            cv2.putText(
                frame,
                str(track.track_id),
                (int(l), int(t) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Display the result
        cv2.imshow("YOLO + DeepSORT", frame)

        # Maintain original frame rate
        elapsed = time.time() - start_time
        wait_ms = max(int((interval - elapsed) * 1000), 1)
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time YOLO + DeepSORT tracking with appearance disabled"
    )
    parser.add_argument(
        "video",
        help="Path to the input video file"
    )
    args = parser.parse_args()
    process_video(args.video)
