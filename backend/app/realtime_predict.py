import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os

# ================= LOAD MODEL =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.npy")

model = load_model(MODEL_PATH)
labels = np.load(LABELS_PATH)

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

sequence = []
prediction_text = "Press 's' to Capture"
confidence_text = ""
sentence = []

last_prediction = None
confirm_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(sequence) < 30:
                sequence.append(landmarks)

    # ================= DISPLAY TEXT =================
    cv2.putText(frame, prediction_text,
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.putText(frame, confidence_text,
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2)

    cv2.putText(frame, "Sentence: " + " ".join(sentence),
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2)

    cv2.imshow("ISL Realtime Prediction", frame)

    key = cv2.waitKey(1) & 0xFF

    # ================= START CAPTURE =================
    if key == ord('s'):
        sequence = []
        prediction_text = "Capturing..."
        confidence_text = ""
        last_prediction = None
        confirm_count = 0

    # ================= PREDICTION =================
    if len(sequence) == 30:
        input_data = np.array(sequence).reshape(1, 30, 63)
        prediction = model.predict(input_data, verbose=0)

        predicted_class = labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        # ----- Confirmation Logic -----
        if predicted_class == last_prediction:
            confirm_count += 1
        else:
            confirm_count = 1
            last_prediction = predicted_class

        if confirm_count >= 2:
            prediction_text = f"Confirmed: {predicted_class}"
            confidence_text = f"Confidence: {confidence:.2f}"

            # Avoid duplicate consecutive words
            if len(sentence) == 0 or sentence[-1] != predicted_class:
                sentence.append(predicted_class)
        else:
            prediction_text = "Detecting..."
            confidence_text = ""

        sequence = []

    # ================= CLEAR SENTENCE =================
    if key == ord('c'):
        sentence = []

    # ================= EXIT =================
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()