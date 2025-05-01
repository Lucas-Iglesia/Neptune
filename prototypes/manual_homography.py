import cv2
import numpy as np

# Load source image
source_image = cv2.imread("map/beach.jpg")
if source_image is None:
    raise FileNotFoundError("image.png not found")

# Create a blank white target image (same size as source)
target_image = 255 * np.ones_like(source_image)

# Store points
source_points = []
target_points = []

def click_source(event, x, y, flags, param):
    """Callback function to collect points from the source image."""
    if event == cv2.EVENT_LBUTTONDOWN:
        source_points.append((x, y))
        print(f"Source Point {len(source_points)}: ({x}, {y})")
        cv2.circle(source_image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Source Image", source_image)

def click_target(event, x, y, flags, param):
    """Callback function to collect points from the target image."""
    if event == cv2.EVENT_LBUTTONDOWN:
        target_points.append((x, y))
        print(f"Target Point {len(target_points)}: ({x}, {y})")
        cv2.circle(target_image, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow("Target Image", target_image)

# Create windows and set mouse callbacks
cv2.imshow("Source Image", source_image)
cv2.imshow("Target Image", target_image)
cv2.setMouseCallback("Source Image", click_source)
cv2.setMouseCallback("Target Image", click_target)

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord('h') and len(source_points) == len(target_points) and len(target_points) >= 4:
        # Convert to NumPy arrays
        src_pts = np.array(source_points, dtype=np.float32)
        dst_pts = np.array(target_points, dtype=np.float32)

        # Compute homography
        H, _ = cv2.findHomography(src_pts, dst_pts)
        print("\nComputed Homography Matrix:")
        print(H)

        # Apply homography to warp source image into target
        warped_image = cv2.warpPerspective(source_image, H, (target_image.shape[1], target_image.shape[0]))

        # Display the warped image
        cv2.imshow("Warped Image", warped_image)

    elif key == ord('q'):
        break

cv2.destroyAllWindows()
