import cv2, numpy as np
from ultralytics import YOLO

model = YOLO("water-detection/model-v2/nwd-v2.pt")
cap   = cv2.VideoCapture("./homography/data/beach_low.mp4")
if not cap.isOpened():
    raise IOError("Impossible d'ouvrir la vidéo.")

alpha  = 0.35                 # transparence
color  = (255, 0, 0)          # bleu BGR

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # --- segmentation ---
    res  = model.predict(frame, imgsz=512, task="segment",
                         conf=0.25, verbose=False)[0]

    if res.masks is not None:
        # res.masks.data : (N, Hm, Wm)   N = nombre de régions
        mask_stack = res.masks.data.cpu().numpy()         # float32 0-1
        # fusion : au moins un masque actif → 1
        mask = (mask_stack > 0.5).any(axis=0).astype(np.uint8)  # (Hm,Wm)

        # redimensionne à la taille d’origine
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

        # overlay bleu semi-transparent
        overlay = frame.copy()
        overlay[mask == 1] = color
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.imshow("Segmentation eau (nwd-v2)", frame)
    if cv2.waitKey(1) & 0xFF == 27:   # Échap
        break

cap.release()
cv2.destroyAllWindows()
