import os
import cv2
import numpy as np
import random

DATA_DIR = 'Data/train'  # Main folder containing subfolders per label
ROTATE_PROB = 0.5  # Chance of rotating each image
ANGLES = [-90, -45, 45, 90]  # Rotation angles in degrees

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_path):
        continue

    for img_name in os.listdir(label_path):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(label_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        if random.random() < ROTATE_PROB:
            angle = random.choice(ANGLES)
            rotated = rotate_image(image, angle)

            # Construct a new filename without overwriting original
            name, ext = os.path.splitext(img_name)
            new_name = f"{name}_rot{angle}{ext}"
            new_path = os.path.join(label_path, new_name)

            # Save rotated image
            cv2.imwrite(new_path, rotated)
