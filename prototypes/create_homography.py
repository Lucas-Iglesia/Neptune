import cv2
import numpy as np

# --- 1. Extraction de la première frame ---
cap = cv2.VideoCapture("./homography/data/beach_30fps.mp4")
ok, frame = cap.read()
cap.release()
if not ok:
    raise IOError("Impossible de lire la vidéo")

# --- 2. Sélection interactive de 4 points ---
win = "Homography (clic gauche = point, ECHAP quand fini)"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
points = []

def on_click(event, x, y, flags, _):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
        cv2.imshow(win, frame)

cv2.setMouseCallback(win, on_click)
cv2.imshow(win, frame)
while True:
    if cv2.waitKey(20) & 0xFF == 27 or len(points) == 4:  # ESC ou 4 points
        break
cv2.destroyAllWindows()

if len(points) != 4:
    raise ValueError("Quatre points requis pour l'homographie")

src = np.array(points, dtype=np.float32)            # (4,2)

# --- 3. Rectangle cible arbitraire ---
W_p, H_p = 400, 200
dst = np.array([[0, 0], [W_p, 0], [W_p, H_p], [0, H_p]], dtype=np.float32)

# --- 4. Homographie puis rectification ---
H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)  # seuil 3 px
warp = cv2.warpPerspective(frame, H, (W_p, H_p))

cv2.imwrite("homography/map.jpg", warp)

np.save("homography/H.npy", H)           # ← sauvegarde
np.save("homography/rect_shape.npy", np.array([W_p, H_p]))  # pour le script 2
