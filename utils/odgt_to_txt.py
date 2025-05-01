import json
import os
from PIL import Image

odgt_path = "annotation_train.odgt"
images_dir = "dataset/images"
labels_dir = "dataset/labels"

os.makedirs(labels_dir, exist_ok=True)

with open(odgt_path, "r") as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    if not line:
        continue

    data = json.loads(line)
    image_id = data["ID"]
    
    image_name = image_id + ".jpg"  
    image_path = os.path.join(images_dir, image_name)

    if not os.path.isfile(image_path):
        print(f"Image {image_path} not found, skipping...")
        continue
    
    with Image.open(image_path) as img:
        img_width, img_height = img.size

    label_path = os.path.join(labels_dir, image_id + ".txt")
    
    yolo_bboxes = []

    for obj in data["gtboxes"]:
        if "fbox" not in obj:
            continue
        
        box = obj["fbox"]
        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h

        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        box_w = x2 - x1
        box_h = y2 - y1

        x_center /= img_width
        y_center /= img_height
        box_w   /= img_width
        box_h   /= img_height

        class_id = 0

        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
        yolo_bboxes.append(yolo_line)
    
    with open(label_path, "w") as label_file:
        for bbox_line in yolo_bboxes:
            label_file.write(bbox_line + "\n")
