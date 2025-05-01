import cv2, time
import numpy as np
from ultralytics import YOLO

H = np.load("homography/H.npy")
W_p, H_p = np.load("homography/rect_shape.npy")
base_canvas = np.full((int(H_p), int(W_p), 3), 80, dtype=np.uint8)

dst_rect = np.array([[[0, 0], [W_p, 0], [W_p, H_p], [0, H_p]]], dtype=np.float32)
src_rect = cv2.perspectiveTransform(dst_rect, np.linalg.inv(H))[0].astype(int)

model = YOLO("human-detection/model/nhd-v3.pt")
model.fuse()

cap = cv2.VideoCapture("./homography/data/beach_30fps.mp4")
if not cap.isOpened():
    raise IOError("Impossible d'ouvrir la vidéo.")

DISP_W, DISP_H = 1280, 720
MAP_W,  MAP_H  = 320, 160
PAD            = 12

frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_idx += 1
    t0 = time.perf_counter()

    results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
    persons = [b for b in results.boxes if int(b.cls) == 0]

    canvas = base_canvas.copy()
    if persons:
        centers = np.array([[[float(b.xywh[0][0]), float(b.xywh[0][1])]]
                            for b in persons], dtype=np.float32)
        proj = cv2.perspectiveTransform(centers, H)
        for x, y in proj.reshape(-1, 2):
            cv2.circle(canvas, (int(x), int(y)), 4, (255, 255, 255), -1)

    vis = results.plot(conf=False, labels=False)

    cv2.polylines(vis, [src_rect], isClosed=True, color=(0, 255, 0), thickness=3)

    vis_small = cv2.resize(vis, (DISP_W, DISP_H), cv2.INTER_AREA)
    mini_map  = cv2.resize(canvas, (MAP_W, MAP_H), cv2.INTER_AREA)
    x0, y0 = DISP_W - MAP_W - PAD, PAD
    vis_small[y0:y0+MAP_H, x0:x0+MAP_W] = mini_map

    elapsed = time.perf_counter() - t0
    print(f"Frame {frame_idx:04d} | {elapsed*1000:6.1f} ms | {1/elapsed:5.1f} fps")
    cv2.imshow("Beach Homography", vis_small)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
