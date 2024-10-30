import cv2
import os

def extract_frames(video_file, output_folder, frame_interval=1):
    cap = cv2.VideoCapture(video_file)
    count = 0
    frame_count = 0

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if frame_count % frame_interval == 0:
                cv2.imwrite(os.path.join(output_folder, f'frame_{count:04d}.jpg'), frame)
                count += 1
            frame_count += 1
        else:
            break

    cap.release()

def process_videos_in_folder(input_folder, output_folder, frame_interval=1):
    for filename in os.listdir(input_folder):
        if filename.endswith('.avi'):
            video_file = os.path.join(input_folder, filename)
            gesture_type = os.path.basename(input_folder)
            output_subfolder = os.path.join(output_folder, gesture_type)
            extract_frames(video_file, output_subfolder, frame_interval)

# Dossiers d'entrée et de sortie
input_folders = ['normal_gestures', 'danger_gestures']
output_folder = 'data'

# Traiter toutes les vidéos dans les dossiers d'entrée
frame_interval = 0.1  # Ajustez cette valeur pour extraire plus de frames
for input_folder in input_folders:
    process_videos_in_folder(input_folder, output_folder, frame_interval)