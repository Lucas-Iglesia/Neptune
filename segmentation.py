import cv2
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

cv2.namedWindow('Pose estimation', cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture('./data/crowded_beach.mov')

if not cap.isOpened():
    print("Erreur : Impossible de lire la vidéo.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    scale_percent = 150
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    
    results = model(frame, conf=0.3, iou=0.3)
    
    annotated_frame = results[0].plot()

    cv2.imshow('Pose estimation', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
