
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# === CONFIGURATION ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_PATH = "video/rozel-1080p.mp4"
SEG_MODEL_PATH = "model/nwd-v2.pt"
UPDATE_EVERY = 300
MIN_WATER_AREA_PX = 5_000
MAP_W_PX, MAP_H_PX = 400, 200
DST_RECT = np.array([[0, 0], [MAP_W_PX, 0], [MAP_W_PX, MAP_H_PX], [0, MAP_H_PX]], dtype=np.float32)

# === INITIALIZATION ===
print("Loading water detection model (NWD)...")
water_seg = YOLO(SEG_MODEL_PATH)

cap = cv2.VideoCapture(str(Path(VIDEO_PATH)))
if not cap.isOpened():
    raise IOError(f"Unable to open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_delay = int(1000 / fps)  # Simple frame delay in milliseconds
frame_idx = 0
H_latest = None
paused = False
last_frame = None
map_canvas_base = np.full((MAP_H_PX, MAP_W_PX, 3), 80, np.uint8)
initial_detection_done = False
current_water_mask = None  # Store the current water mask for display

# FPS monitoring and control
fps_start_time = time.time()
fps_frame_count = 0
current_fps = 0
target_frame_time = 1.0 / fps  # Target time per frame (1/30 = 0.0333s)
last_frame_time = time.time()

print(f"Video FPS: {fps}, Frame delay: {frame_delay}ms")
print("Controls: [SPACE] = Pause/Resume  |  [ENTER] = Recalculate NWD  |  [W] = Toggle Water Mask  |  [ESC] = Quit")

window_created = False
show_water_mask = False  # Toggle for water mask display

while True:
    # Calculate timing first
    current_time = time.time()
    elapsed_since_last_frame = current_time - last_frame_time
    
    # Handle key input with minimal delay
    key = cv2.waitKey(1) & 0xFF
    if key == 27:         # ESC
        break
    elif key == 32:       # SPACE - Toggle pause/resume
        paused = not paused
        if paused:
            print("Video paused")
        else:
            print("Video resumed")
            last_frame_time = time.time()  # Reset timing when resuming
    elif key == ord('w') or key == ord('W'):  # W - Toggle water mask
        show_water_mask = not show_water_mask
        print(f"Water mask display: {'ON' if show_water_mask else 'OFF'}")
    elif key == 13:       # ENTER = rerun water detection
        if last_frame is not None:
            seg_res = water_seg.predict(last_frame, imgsz=512, task="segment", conf=0.25, verbose=False)[0]
            if seg_res.masks is not None:
                mask_small = (seg_res.masks.data.cpu().numpy() > 0.5).any(axis=0).astype(np.uint8)
                mask = cv2.resize(mask_small, (last_frame.shape[1], last_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                current_water_mask = mask.copy()  # Store the mask for display
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    main_cnt = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(main_cnt) > MIN_WATER_AREA_PX:
                        pts = main_cnt.reshape(-1, 2).astype(np.float32)
                        sums = pts.sum(axis=1)
                        diffs = np.diff(pts, axis=1).reshape(-1)
                        src_quad = np.array([
                            pts[np.argmin(sums)],
                            pts[np.argmin(diffs)],
                            pts[np.argmax(sums)],
                            pts[np.argmax(diffs)]
                        ], dtype=np.float32)
                        H_new, _ = cv2.findHomography(src_quad, DST_RECT, cv2.RANSAC, 3.0)
                        if H_new is not None:
                            H_latest = H_new.copy()
                            print("Water recalculated (NWD rerun)")

    if not paused:
        # Only proceed if enough time has passed for target FPS
        if elapsed_since_last_frame >= target_frame_time:
            ok, frame = cap.read()
            if not ok:
                break
            last_frame = frame.copy()
            last_frame_time = current_time  # Use the current_time we calculated earlier
            frame_idx += 1
            
            # Calculate real FPS
            fps_frame_count += 1
            if fps_frame_count % 60 == 0:  # Update FPS every 60 frames for less overhead
                fps_elapsed = current_time - fps_start_time
                current_fps = fps_frame_count / fps_elapsed
                print(f"Current FPS: {current_fps:.1f} (Target: {fps})")
                fps_start_time = current_time
                fps_frame_count = 0
            
            # Run water detection only on first frame and every UPDATE_EVERY frames
            if frame_idx == 1 or frame_idx % UPDATE_EVERY == 0:
                print("Running water detection..." if frame_idx == 1 else f"Updating water detection (frame {frame_idx})")
                seg_res = water_seg.predict(last_frame, imgsz=512, task="segment", conf=0.25, verbose=False)[0]
                if seg_res.masks is not None:
                    mask_small = (seg_res.masks.data.cpu().numpy() > 0.5).any(axis=0).astype(np.uint8)
                    mask = cv2.resize(mask_small, (last_frame.shape[1], last_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    current_water_mask = mask.copy()  # Store the mask for display
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours:
                        main_cnt = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(main_cnt) > MIN_WATER_AREA_PX:
                            pts = main_cnt.reshape(-1, 2).astype(np.float32)
                            sums = pts.sum(axis=1)
                            diffs = np.diff(pts, axis=1).reshape(-1)
                            src_quad = np.array([
                                pts[np.argmin(sums)],
                                pts[np.argmin(diffs)],
                                pts[np.argmax(sums)],
                                pts[np.argmax(diffs)]
                            ], dtype=np.float32)
                            H_new, _ = cv2.findHomography(src_quad, DST_RECT, cv2.RANSAC, 3.0)
                            if H_new is not None:
                                H_latest = H_new.copy()
                                print("Water detection completed" if frame_idx == 1 else "Water detection updated")
                initial_detection_done = True

    # Display only if we have a valid frame
    if last_frame is not None:
        # Create fullscreen window only when we have the first frame
        if not window_created:
            cv2.namedWindow("Demo NWD - Neptune", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Demo NWD - Neptune", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            window_created = True
            
        if H_latest is not None:
            src_poly = cv2.perspectiveTransform(DST_RECT[None, :, :], np.linalg.inv(H_latest))[0].astype(int)
            vis = last_frame.copy()
            cv2.polylines(vis, [src_poly], True, (0, 255, 0), 3)
        else:
            vis = last_frame.copy()
        
        # Apply water mask overlay if enabled
        if show_water_mask and current_water_mask is not None:
            # Create yellow overlay for water areas
            yellow_overlay = np.zeros_like(vis)
            yellow_overlay[current_water_mask > 0] = [0, 255, 255]  # Yellow in BGR
            # Blend with original image
            vis = cv2.addWeighted(vis, 0.7, yellow_overlay, 0.3, 0)

        # Get screen size for fullscreen display
        screen_height, screen_width = 1080, 1920  # Adjust if needed
        vis_fullscreen = cv2.resize(vis, (screen_width, screen_height), interpolation=cv2.INTER_AREA)
        
        # Display FPS on screen
        # cv2.putText(vis_fullscreen, f"FPS: {current_fps:.1f}/{fps}", (50, 50), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        if paused:
            cv2.putText(vis_fullscreen, "PAUSED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 4)

        cv2.imshow("Demo NWD - Neptune", vis_fullscreen)

cap.release()
cv2.destroyAllWindows()
