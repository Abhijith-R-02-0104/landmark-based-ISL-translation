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

# ================= PARAMETERS =================
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.80
PREDICTION_WINDOW = 15

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

sequence = []
sentence = []
predictions = []

prediction_text = "Press 's' to Start"
confidence_text = ""

capturing = False

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    # ================= HAND DETECTION =================
    if capturing and results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = []

            # 🔥 SAME NORMALIZATION AS TRAINING
            wrist = hand_landmarks.landmark[0]
            wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

            for lm in hand_landmarks.landmark:
                landmarks.extend([
                    lm.x - wrist_x,
                    lm.y - wrist_y,
                    lm.z - wrist_z
                ])

            sequence.append(landmarks)

            # sliding window
            if len(sequence) > SEQUENCE_LENGTH:
                sequence = sequence[-SEQUENCE_LENGTH:]

    elif capturing:
        prediction_text = "No hand detected"
        predictions = []

    # ================= PREDICTION =================
    if len(sequence) == SEQUENCE_LENGTH:

        input_data = np.array(sequence).reshape(1, SEQUENCE_LENGTH, 63)

        prediction = model.predict(input_data, verbose=0)

        predicted_class = labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        if confidence > CONFIDENCE_THRESHOLD:

            predictions.append(predicted_class)

            if len(predictions) > PREDICTION_WINDOW:
                predictions.pop(0)

            # 🔥 Strong N-frame confirmation
            if len(predictions) == PREDICTION_WINDOW and len(set(predictions)) == 1:

                final_prediction = predictions[0]

                prediction_text = f"Confirmed: {final_prediction}"
                confidence_text = f"Confidence: {confidence:.2f}"

                # avoid duplicate words
                if len(sentence) == 0 or sentence[-1] != final_prediction:
                    sentence.append(final_prediction)

                # 🔥 reset after confirmation
                predictions = []
                sequence = []

        else:
            prediction_text = "Low confidence"
            confidence_text = ""
            predictions = []

    # ================= DISPLAY =================
    cv2.putText(
        frame,
        prediction_text,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        confidence_text,
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        "Sentence: " + " ".join(sentence),
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.imshow("ISL Realtime Prediction", frame)

    key = cv2.waitKey(1) & 0xFF

    # ================= START =================
    if key == ord('s'):
        capturing = True
        sequence = []
        predictions = []
        prediction_text = "Capturing..."
        confidence_text = ""

    # ================= PAUSE =================
    if key == ord('x'):
        capturing = False
        sequence = []
        predictions = []
        prediction_text = "Paused"

    # ================= CLEAR =================
    if key == ord('c'):
        sentence = []

    # ================= EXIT =================
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()