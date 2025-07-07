from pathlib import Path
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, DFineForObjectDetection
from ultralytics import YOLO

class HomographyProcessor:
    MAP_W_PX, MAP_H_PX = 400, 200
    UPDATE_EVERY = 300
    CONF_THRES = 0.4
    MIN_WATER_AREA_PX = 5_000
    INFER_EVERY = 1

    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(str(Path(video_path)))
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.water_seg = YOLO("water-detection/model-v2/nwd-v2.pt")
        self.processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-xlarge-obj2coco")
        self.dfine = DFineForObjectDetection.from_pretrained("ustc-community/dfine-xlarge-obj2coco").to(self.device).eval()
        self.DST_RECT = np.array([[0, 0], [self.MAP_W_PX, 0], [self.MAP_W_PX, self.MAP_H_PX], [0, self.MAP_H_PX]], dtype=np.float32)
        self.H_latest: np.ndarray | None = None
        self.water_bbox = None
        self.frame_idx = 0

    @torch.inference_mode()
    def _detect_persons(self, frame_bgr):
        inputs = self.processor(images=frame_bgr[:, :, ::-1], return_tensors="pt").to(self.device)
        outputs = self.dfine(**inputs)
        results = self.processor.post_process_object_detection(outputs, target_sizes=[(frame_bgr.shape[0], frame_bgr.shape[1])], threshold=self.CONF_THRES)[0]
        people = [(box.cpu().numpy(), score.item()) for box, label, score in zip(results["boxes"], results["labels"], results["scores"]) if label.item() == 0]
        return people

    def _process_frame(self, frame_bgr):
        self.frame_idx += 1
        if self.frame_idx % self.UPDATE_EVERY == 1:
            seg = self.water_seg.predict(frame_bgr, imgsz=512, task="segment", conf=0.25, verbose=False)[0]
            if seg.masks is not None:
                mask_small = (seg.masks.data.cpu().numpy() > 0.5).any(axis=0).astype(np.uint8)
                mask = cv2.resize(mask_small, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    main_cnt = max(cnts, key=cv2.contourArea)
                    if cv2.contourArea(main_cnt) > self.MIN_WATER_AREA_PX:
                        pts = main_cnt.reshape(-1, 2).astype(np.float32)
                        sums = pts.sum(axis=1)
                        diffs = np.diff(pts, axis=1).reshape(-1)
                        src_quad = np.array([pts[np.argmin(sums)], pts[np.argmin(diffs)], pts[np.argmax(sums)], pts[np.argmax(diffs)]], dtype=np.float32)
                        H, _ = cv2.findHomography(src_quad, self.DST_RECT, cv2.RANSAC, 3.0)
                        if H is not None:
                            self.H_latest = H
                        x, y, w, h = cv2.boundingRect(main_cnt)
                        self.water_bbox = (x, y, w, h)

        if self.H_latest is None:
            return frame_bgr

        if self.frame_idx % self.INFER_EVERY == 0:
            self.people_cache = self._detect_persons(frame_bgr)
        people = getattr(self, "people_cache", [])
        # people = self._detect_persons(frame_bgr)
        map_canvas = np.full((self.MAP_H_PX, self.MAP_W_PX, 3), 80, np.uint8)

        if people:
            centers = np.array([[[ (b[0][0] + b[0][2]) / 2, (b[0][1] + b[0][3]) / 2 ]] for b in people], dtype=np.float32)
            proj = cv2.perspectiveTransform(centers, self.H_latest)
            for x, y in proj.reshape(-1, 2):
                cv2.circle(map_canvas, (int(x), int(y)), 4, (255, 255, 255), -1)
            for box, _ in people:
                x0, y0, x1, y1 = box.astype(int)
                cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)

        if self.water_bbox:
            x, y, w, h = self.water_bbox
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 255), 2)

        vis = frame_bgr.copy()
        MAX_LONG_SIDE = 1280
        h, w = vis.shape[:2]
        if max(w, h) > MAX_LONG_SIDE:
            scale = MAX_LONG_SIDE / max(w, h)
            vis = cv2.resize(vis, (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_AREA)

        MINIMAP_W, MINIMAP_H, PAD = 320, 160, 12
        map_small = cv2.resize(map_canvas, (MINIMAP_W, MINIMAP_H))
        y0, x0 = PAD, vis.shape[1] - MINIMAP_W - PAD
        vis[y0:y0 + MINIMAP_H, x0:x0 + MINIMAP_W] = map_small

        return vis

    def frames(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            yield self._process_frame(frame)
