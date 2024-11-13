import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

model = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

def is_in_water(box, water_region):
    # box = [x_min, y_min, x_max, y_max] of the person
    person_y_center = (box[1] + box[3]) / 2
    return person_y_center >= water_region

# Lire la vidéo avec OpenCV
cap = cv2.VideoCapture('videos/v.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    input_tensor = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)[0]

    for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        if score > 0.5 and label == 1:  # Label 1 = person
            x_min, y_min, x_max, y_max = box

            # define a water region
            water_region_y = frame.shape[0] * 0.75

            if is_in_water([x_min, y_min, x_max, y_max], water_region_y):
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)

            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)

    cv2.imshow('Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
