import os
import cv2
import numpy as np
import mediapipe as mp

# Path to your dataset
DATASET_PATH = 'Data/train'
OUTPUT_DIR = 'keypoints_by_label'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

# Process each label folder
for label_name in sorted(os.listdir(DATASET_PATH)):
    class_folder = os.path.join(DATASET_PATH, label_name)
    if not os.path.isdir(class_folder):
        continue

    keypoints_list = []

    for img_name in os.listdir(class_folder):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(class_folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y
            base_z = hand_landmarks.landmark[0].z
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            keypoints_list.append(keypoints)

    if keypoints_list:
        keypoints_array = np.array(keypoints_list)
        output_path = os.path.join(OUTPUT_DIR, f"{label_name}.npy")
        np.save(output_path, keypoints_array)
        print(f"Saved {len(keypoints_array)} samples to {output_path}")
    else:
        print(f"No hand landmarks found for label: {label_name}")
