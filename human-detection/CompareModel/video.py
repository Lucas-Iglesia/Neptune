import cv2
import os
from ultralytics import YOLO

# List of models to compare
models = ["./model/nhd-v3.pt"]

# Folder containing input videos
input_folder = "video-test"
output_root_folder = "result-video"

# Get all videos in the folder
video_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]

for model_name in models:
    model = YOLO(model_name)  # Load the YOLO model
    model_output_folder = os.path.join(output_root_folder, os.path.splitext(model_name)[0])
    os.makedirs(model_output_folder, exist_ok=True)

    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(model_output_folder, f"processed_{video_file}")

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        new_width, new_height = (1920, 1080) if width > 1920 or height > 1080 else (width, height)

        # Define the writer to save the output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Make a YOLO prediction on the frame
            results = model(frame)

            # Draw the results on the image
            annotated_frame = results[0].plot()

            # Resize to HD if necessary
            if (width, height) != (new_width, new_height):
                annotated_frame = cv2.resize(annotated_frame, (new_width, new_height))

            # Save the annotated frame in the video
            out.write(annotated_frame)

            # Display the video live (optional)
            cv2.imshow("YOLO Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        out.release()
        print(f"✅ Video processed with {model_name}: {video_file} → {output_path}")

cv2.destroyAllWindows()
print("🎉 All videos have been successfully processed for all models!")