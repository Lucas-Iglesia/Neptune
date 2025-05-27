# prototype homography map2D real time for Neptune app
# Adrien Picot
# Neptune

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# 4 points in pixels in the image captured by the "Camera"
pts_image = np.array([[100, 200], [400, 200], [450, 600], [50, 600]], dtype='float32')

# the same 4 point but with real coordinates on the beach
pts_beach = np.array([[0, 0], [10, 0], [10, 20], [0, 20]], dtype='float32')

H, _ = cv2.findHomography(pts_image, pts_beach)

# function used to transform an image point to a real point (A beach point)
def transform_point(pt, H):
    pts = np.array([[pt]], dtype='float32')
    beach_pt = cv2.perspectiveTransform(pts, H)
    return beach_pt[0][0]


# 3 fake human detection
humans_image = [
    (150, 250),
    (300, 350),
    (350, 500)
]

# We transform our 3 humans pixel coordinates to 3 beach coordinates
humans_beach = [transform_point(pt, H) for pt in humans_image]

def display_map(humans_beach):
    plt.clf()
    beach_x = [pt[0] for pt in humans_beach]
    beach_y = [pt[1] for pt in humans_beach]

    plt.scatter(beach_x, beach_y, c='red', label='Humans')
    plt.title('2D Map Homography')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.xlim(-1, 12)
    plt.ylim(-1, 22)
    plt.legend()
    plt.grid()
    plt.pause(0.1)

plt.ion()
fig = plt.figure()

# Simulation of real time movements of our fake humans
for i in range(50):
    humans_image = [
        (150 + np.random.randint(-10, 10), 250 + np.random.randint(-10, 10)),
        (300 + np.random.randint(-10, 10), 350 + np.random.randint(-10, 10)),
        (350 + np.random.randint(-10, 10), 500 + np.random.randint(-10, 10))
    ]
    
    humans_beach = [transform_point(pt, H) for pt in humans_image]
    display_map(humans_beach)
    time.sleep(0.5)

plt.ioff()
plt.show()