import cv2
import torch
import time
from transformers import AutoImageProcessor, DFineForObjectDetection

# Config
MODEL_ID = "ustc-community/dfine-xlarge-obj2coco"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_PATH = "video/IMG_6863_1080p15.mov"
CONF_THRES = 0.5

# Load model
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = DFineForObjectDetection.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE).eval()

@torch.inference_mode()
def detect_persons_dfine(frame_bgr, conf_thres=CONF_THRES):
    inputs = processor(images=frame_bgr[:, :, ::-1], return_tensors="pt").to(DEVICE)
    if DEVICE == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)
    outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs,
        target_sizes=[(frame_bgr.shape[0], frame_bgr.shape[1])],
        threshold=conf_thres,
    )[0]

    persons = []
    for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
        if label.item() == 0:  # class 0 = person
            x0, y0, x1, y1 = map(int, box.tolist())
            persons.append((x0, y0, x1, y1, score.item()))
    return persons

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {VIDEO_PATH}")

print("🎯 Demo NHD: ESC pour quitter.")
print("Controls: [SPACE] = Pause/Resume  |  [ESC] = Quit")

# FPS calculation variables
fps_counter = 0
fps_start_time = time.time()
fps_display_interval = 1.0  # Display FPS every second

# Pause system variables
paused = False
last_frame = None
last_detections = []
window_created = False

while True:
    # Handle key input
    key = cv2.waitKey(1) & 0xFF
    if key == 27:         # ESC
        break
    elif key == 32:       # SPACE - Toggle pause/resume
        paused = not paused
        if paused:
            print("Video paused")
        else:
            print("Video resumed")
    
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        
        last_frame = frame.copy()
        detections = detect_persons_dfine(frame)
        last_detections = detections.copy()  # Store detections for pause

        for x0, y0, x1, y1, score in detections:
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, f"{score:.2f}", (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # FPS calculation and display
        fps_counter += 1
        current_time = time.time()
        elapsed_time = current_time - fps_start_time
        
        if elapsed_time >= fps_display_interval:
            fps = fps_counter / elapsed_time
            # print(f"📊 FPS: {fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
    
    # Display the frame (paused or not)
    if last_frame is not None:
        # Create fullscreen window only when we have the first frame
        if not window_created:
            cv2.namedWindow("NHD Demo - Human Detection", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("NHD Demo - Human Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            window_created = True
            
        display_frame = last_frame.copy()
        
        # Show the last detections (frozen when paused)
        for x0, y0, x1, y1, score in last_detections:
            cv2.rectangle(display_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(display_frame, f"{score:.2f}", (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Add pause indicator
        if paused:
            cv2.putText(display_frame, "PAUSED", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 4)
        
        cv2.imshow("NHD Demo - Human Detection", display_frame)

cap.release()
cv2.destroyAllWindows()
