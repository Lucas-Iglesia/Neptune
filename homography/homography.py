"""
homography.py

Real-time people localisation on a beach:

1.  Water segmentation model (`nwd-v2.pt`) automatically finds the
    shoreline and recalculates the homography every ~3 s.

2.  Person detection model (`nhd-v3.pt`) tracks swimmers on each frame.

3.  Detected people are projected onto a top-down “mini-map” using the
    most recent homography and displayed as white dots.

4.  A green polygon on the main video shows the area currently covered
    by the homography.

"""

import time
import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH      = "./homography/data/IMG_6863.MOV"
SEG_MODEL_PATH  = "water-detection/model-v2/nwd-v2.pt"
DET_MODEL_PATH  = "human-detection/model/nhd-v3.pt"

# Homography destination rectangle (top-down map size in pixels)
MAP_W_PX, MAP_H_PX = 400, 200
DST_RECT = np.array(
    [[0, 0], [MAP_W_PX, 0], [MAP_W_PX, MAP_H_PX], [0, MAP_H_PX]],
    dtype=np.float32,
)

UPDATE_EVERY = 90
MIN_WATER_AREA_PX = 5_000   # Ignore tiny blobs

# UI sizes
DISPLAY_W, DISPLAY_H = 1280, 720
MINIMAP_W, MINIMAP_H = 320, 160
PADDING_PX           = 12

water_seg = YOLO(SEG_MODEL_PATH)          # water segmentation model
person_det = YOLO(DET_MODEL_PATH).fuse()  # person detector (weights fused)

H_latest: np.ndarray | None = None

# Grey background for the top-down map
map_canvas_base = np.full((MAP_H_PX, MAP_W_PX, 3), 80, np.uint8)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {VIDEO_PATH}")

frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_idx += 1
    tic = time.perf_counter()

    # Refresh homography every UPDATE_EVERY frames
    if frame_idx % UPDATE_EVERY == 1:
        seg_res = water_seg.predict(
            frame, imgsz=512, task="segment", conf=0.25, verbose=False
        )[0]

        if seg_res.masks is not None:
            # Merge all water masks into a single binary mask
            mask_small = (seg_res.masks.data.cpu().numpy() > 0.5).any(axis=0).astype(np.uint8)

            # Resize mask back to the original video resolution
            mask = cv2.resize(
                mask_small,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            # Extract the largest connected component (assumed to be the sea)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                main_cnt = max(contours, key=cv2.contourArea)
                if cv2.contourArea(main_cnt) > MIN_WATER_AREA_PX:
                    # Find the four extreme points of the contour
                    pts = main_cnt.reshape(-1, 2).astype(np.float32)
                    sums  = pts.sum(axis=1)
                    diffs = np.diff(pts, axis=1).reshape(-1)

                    src_quad = np.array(
                        [
                            pts[np.argmin(sums)],    # top-left
                            pts[np.argmin(diffs)],   # top-right
                            pts[np.argmax(sums)],    # bottom-right
                            pts[np.argmax(diffs)],   # bottom-left
                        ],
                        dtype=np.float32,
                    )

                    # Compute a new homography with RANSAC
                    H_new, _ = cv2.findHomography(src_quad, DST_RECT, cv2.RANSAC, 3.0)
                    if H_new is not None:
                        H_latest = H_new.copy()

    # If no homography yet, just show the raw frame
    if H_latest is None:
        cv2.imshow("Beach Homography", frame)
        if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
            break
        continue

    # Detect people on the current frame
    det_res = person_det.predict(
        frame, imgsz=640, conf=0.25, verbose=False
    )[0]
    persons = [b for b in det_res.boxes if int(b.cls) == 0]

    # Project detected people onto the mini-map
    map_canvas = map_canvas_base.copy()
    if persons:
        centers = np.array(
            [[[float(b.xywh[0][0]), float(b.xywh[0][1])]] for b in persons],
            dtype=np.float32,
        )
        proj = cv2.perspectiveTransform(centers, H_latest)
        for x, y in proj.reshape(-1, 2):
            cv2.circle(map_canvas, (int(x), int(y)), 4, (255, 255, 255), -1)

    # Draw the water area polygon on the original frame
    src_poly = cv2.perspectiveTransform(
        DST_RECT[None, :, :], np.linalg.inv(H_latest)
    )[0].astype(int)

    vis = det_res.plot(conf=False, labels=False)
    cv2.polylines(vis, [src_poly], isClosed=True, color=(0, 255, 0), thickness=3)

    vis_small = cv2.resize(vis, (DISPLAY_W, DISPLAY_H), cv2.INTER_AREA)
    map_small = cv2.resize(map_canvas, (MINIMAP_W, MINIMAP_H), cv2.INTER_AREA)

    # Overlay the mini-map in the top-right corner
    x0, y0 = DISPLAY_W - MINIMAP_W - PADDING_PX, PADDING_PX
    vis_small[y0:y0 + MINIMAP_H, x0:x0 + MINIMAP_W] = map_small

    toc = time.perf_counter()
    fps = 1.0 / (toc - tic)
    print(f"Frame {frame_idx:04d} | {1000*(toc - tic):6.1f} ms | {fps:5.1f} fps")

    cv2.imshow("Beach Homography", vis_small)
    if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
