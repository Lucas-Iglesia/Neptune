import cv2
import os
import time
from ultralytics import YOLO

# List of models to compare
models = ["./model/nhd-v5.pt"]

# Folder containing input videos
input_folder       = "video-test-converted"
output_root_folder = "result-video"
os.makedirs(output_root_folder, exist_ok=True)

# Liste de tous les .mp4
video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mp4")]

# Cible d'affichage
target_fps     = 30
sec_per_frame  = 1.0 / target_fps

for model_name in models:
    model = YOLO(model_name)
    model_output_folder = os.path.join(
        output_root_folder,
        os.path.splitext(os.path.basename(model_name))[0]
    )
    os.makedirs(model_output_folder, exist_ok=True)

    for video_file in video_files:
        video_path  = os.path.join(input_folder, video_file)
        output_path = os.path.join(model_output_folder, f"processed_{video_file}")

        cap   = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in= cap.get(cv2.CAP_PROP_FPS) or target_fps

        # On force la sortie à 30fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(output_path, fourcc,
                                 target_fps,
                                 (width, height))

        while cap.isOpened():
            t0 = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                break

            # Inference et annotation
            results     = model(frame)
            annotated   = results[0].plot()
            out.write(annotated)

            # Affiche et ajuste le délai pour atteindre 30 fps
            cv2.imshow("YOLO Detection", annotated)

            elapsed = time.perf_counter() - t0
            wait_ms = max(1, int((sec_per_frame - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        print(f"✅ Video processed with {model_name}: {video_file} → {output_path}")

cv2.destroyAllWindows()
print("🎉 All videos have been successfully processed for all models!")
