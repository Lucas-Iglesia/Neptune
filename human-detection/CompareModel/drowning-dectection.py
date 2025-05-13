from ultralytics import YOLO
import cv2

model = YOLO("model/yolo11x-pose.pt")  # ou nano/FP16

cap = cv2.VideoCapture("video-test-converted/IMG_6864.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
delay = max(1, int(1000 / fps))

SKIP = 3            # infère 1 frame sur 3
last_annot = None
frame_idx  = 0

cv2.namedWindow("Realtime Pose", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Realtime Pose", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % SKIP == 0:
        # inférence sur la frame
        res = model(frame, half=True, imgsz=384, conf=0.1)[0]
        last_annot = res.plot()

    # affiche la dernière annotation (ou la frame brute si aucune)
    display = last_annot if last_annot is not None else frame
    cv2.imshow("Realtime Pose", display)

    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
