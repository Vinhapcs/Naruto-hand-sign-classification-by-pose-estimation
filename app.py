import warnings
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter

# Load model
model = joblib.load('hand_sign_model.pkl')

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, static_image_mode=False)
mp_draw = mp.solutions.drawing_utils

# Buffer for label smoothing
prediction_buffer = deque(maxlen=15)  # Holds last 15 predictions
display_label = "..."  # Displayed on screen
CONFIDENCE_THRESHOLD = 10  # How many consistent votes needed to show label

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    predicted_label = None

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            # Draw landmarks
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Extract keypoints
            base_x = handLms.landmark[0].x
            base_y = handLms.landmark[0].y
            base_z = handLms.landmark[0].z
            keypoints = []
            for lm in handLms.landmark:
                keypoints.extend([
                    lm.x - base_x,
                    lm.y - base_y,
                    lm.z - base_z
                ])

            # Predict if all 21 landmarks are present
            if len(keypoints) == 63:
                keypoints_np = np.array(keypoints).reshape(1, -1)
                predicted_label = model.predict(keypoints_np)[0]
                prediction_buffer.append(predicted_label)

    # Update display label if consistent prediction
    if prediction_buffer:
        most_common, count = Counter(prediction_buffer).most_common(1)[0]
        if count >= CONFIDENCE_THRESHOLD:
            display_label = most_common

    # Show result
    cv2.putText(frame, f"Sign: {display_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Hand Sign Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
