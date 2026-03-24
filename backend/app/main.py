from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import mediapipe as mp
import numpy as np
import base64
from tensorflow.keras.models import load_model
import os
from collections import deque

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/status")
def status():
    return {"status": "ok"}


# ================= LOAD MODEL =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.npy")

model  = load_model(MODEL_PATH)
labels = np.load(LABELS_PATH)


# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# ================= PARAMS =================
SEQUENCE_LENGTH = 30


# ================= STATE =================
sequence = deque(maxlen=SEQUENCE_LENGTH)
predictions = deque(maxlen=5)
sentence = []


class FrameData(BaseModel):
    image: str


@app.post("/predict")
async def predict(data: FrameData):

    # ===== Decode frame =====
    try:
        image_data = data.image.split(",")[1]
        decoded    = base64.b64decode(image_data)
        np_arr     = np.frombuffer(decoded, np.uint8)
        frame      = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError
    except Exception:
        return {"letter": "-", "confidence": 0.0, "gesture": None}

    frame = cv2.resize(frame, (640, 480))
    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ===== MediaPipe =====
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        sequence.clear()
        predictions.clear()
        return {"letter": "-", "confidence": 0.0, "gesture": None}

    # ===== Extract landmarks =====
    for hand_landmarks in results.multi_hand_landmarks:

        wrist = hand_landmarks.landmark[0]
        wx, wy, wz = wrist.x, wrist.y, wrist.z

        landmarks = []

        for lm in hand_landmarks.landmark:
            landmarks.extend([
                lm.x - wx,
                lm.y - wy,
                lm.z - wz
            ])

        sequence.append(landmarks)

    # ===== Prediction =====
    if len(sequence) == SEQUENCE_LENGTH:

        input_data = np.array(sequence).reshape(1, SEQUENCE_LENGTH, 63)

        prediction = model.predict(input_data, verbose=0)

        pred_index = np.argmax(prediction)
        predicted_class = str(labels[pred_index])
        confidence = float(np.max(prediction))

        print("INDEX:", pred_index)
        print("LABEL FROM FILE:", labels[pred_index])
        print("ALL LABELS:", labels)

        print(f"PRED: {predicted_class} | CONF: {confidence:.2f}")

        gesture_output = None

        # ===== STRICT CONFIRMATION =====
        if confidence > 0.7:

            predictions.append(predicted_class)

            if len(predictions) == 5 and len(set(predictions)) == 1:

                final_prediction = predictions[0]
                gesture_output = final_prediction

                # avoid duplicates
                if len(sentence) == 0 or sentence[-1] != final_prediction:
                    sentence.append(final_prediction)

                predictions.clear()
                sequence.clear()

                print(f"✅ CONFIRMED: {final_prediction}")

        return {
            "letter": predicted_class,
            "confidence": confidence,
            "gesture": gesture_output
        }

    # ===== Not enough frames yet =====
    return {"letter": "-", "confidence": 0.0, "gesture": None}