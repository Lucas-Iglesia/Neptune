import cv2
import os
from datetime import datetime

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def capture_video(output_folder, gesture_type):
    create_directory(output_folder)
    
    # Obtenir la date et l'heure actuelles
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_folder, f"{gesture_type}_{current_time}.avi")
    
    cap = cv2.VideoCapture(0)  # 0 pour la caméra par défaut

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Dossiers pour les vidéos
normal_gestures_folder = 'normal_gestures'
danger_gestures_folder = 'danger_gestures'

# Capturez des vidéos de vous faisant des gestes normaux et des gestes indiquant un danger
capture_video(normal_gestures_folder, 'normal_gestures')
capture_video(danger_gestures_folder, 'danger_gestures')