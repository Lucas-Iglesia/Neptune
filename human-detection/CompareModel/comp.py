import cv2
import os
import numpy as np
from tqdm import tqdm

# Folders containing processed videos
nhd_folder = "result-video/model/nhd-v3"
yolo_folder = "result-video/model/yolo11n"
output_folder = "result-video/comparisons"
os.makedirs(output_folder, exist_ok=True)

# Get common files
nhd_videos = sorted([f for f in os.listdir(nhd_folder) if f.endswith(".MOV")])
yolo_videos = sorted([f for f in os.listdir(yolo_folder) if f.endswith(".MOV")])

common_videos = set(nhd_videos) & set(yolo_videos)

# Define the writer for the final video
output_path = os.path.join(output_folder, "comparison_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None
fps = 30

total_frames = 0

# Calculate the total number of frames for the progress bar
for video_file in common_videos:
    cap_nhd = cv2.VideoCapture(os.path.join(nhd_folder, video_file))
    total_frames += int(cap_nhd.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_nhd.release()

# Progress bar
tqdm_bar = tqdm(total=total_frames, desc="Processing videos", unit="frame")

for video_file in common_videos:
    nhd_path = os.path.join(nhd_folder, video_file)
    yolo_path = os.path.join(yolo_folder, video_file)

    cap_nhd = cv2.VideoCapture(nhd_path)
    cap_yolo = cv2.VideoCapture(yolo_path)

    if not cap_nhd.isOpened() or not cap_yolo.isOpened():
        print(f"⚠️ Unable to open {video_file}, skipping to the next one.")
        continue

    width = int(cap_nhd.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_nhd.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_height = height * 2
    
    if out is None:
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, new_height))

    while True:
        ret_nhd, frame_nhd = cap_nhd.read()
        ret_yolo, frame_yolo = cap_yolo.read()
        
        if not ret_nhd or not ret_yolo:
            break
        
        # Add labels
        cv2.putText(frame_nhd, "NHD-V3", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame_yolo, "Yolo11", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Concatenate vertically
        stacked_frame = np.vstack((frame_nhd, frame_yolo))
        
        out.write(stacked_frame)
        tqdm_bar.update(1)

    cap_nhd.release()
    cap_yolo.release()

out.release()
tqdm_bar.close()
print(f"✅ Comparison video generated: {output_path}")