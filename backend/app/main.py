from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import mediapipe as mp
import numpy as np
import base64
from tensorflow.keras.models import load_model
import os
import time

# ================= INIT APP =================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= LOAD MODEL =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.npy")

model = load_model(MODEL_PATH, compile=False)
labels = np.load(LABELS_PATH)

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ================= PARAMETERS =================
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.85   # 🔥 stronger threshold
COOLDOWN_TIME = 0.8           # 🔥 faster switching

# ================= STATE =================
sequence = []
sentence = []
last_prediction_time = 0

# ================= REQUEST MODEL =================
class FrameData(BaseModel):
    image: str


@app.post("/predict")
async def predict(data: FrameData):

    global sequence, sentence, last_prediction_time

    # ===== Decode Image =====
    image_data = data.image.split(",")[1]
    decoded = base64.b64decode(image_data)

    np_arr = np.frombuffer(decoded, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {
            "letter": "-",
            "confidence": 0.0,
            "current_word": " ".join(sentence)
        }

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ===== MediaPipe =====
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        sequence = []
        return {
            "letter": "-",
            "confidence": 0.0,
            "current_word": " ".join(sentence)
        }

    # ===== LANDMARKS =====
    for hand_landmarks in results.multi_hand_landmarks:

        wrist = hand_landmarks.landmark[0]
        wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

        landmarks = []

        for lm in hand_landmarks.landmark:
            landmarks.extend([
                lm.x - wrist_x,
                lm.y - wrist_y,
                lm.z - wrist_z
            ])

        sequence.append(landmarks)

        if len(sequence) > SEQUENCE_LENGTH:
            sequence = sequence[-SEQUENCE_LENGTH:]

    # ===== PREDICTION =====
    if len(sequence) == SEQUENCE_LENGTH:

        input_data = np.array(sequence).reshape(1, SEQUENCE_LENGTH, 63)
        prediction = model.predict(input_data, verbose=0)

        predicted_class = labels[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        if confidence > CONFIDENCE_THRESHOLD:

            current_time = time.time()

            # 🔥 cooldown check
            if current_time - last_prediction_time > COOLDOWN_TIME:

                if len(sentence) == 0 or sentence[-1] != predicted_class:
                    sentence.append(predicted_class)

                last_prediction_time = current_time
                sequence = []   # reset for next gesture

                return {
                    "letter": predicted_class,
                    "confidence": confidence,
                    "current_word": " ".join(sentence)
                }

    return {
        "letter": "-",
        "confidence": 0.0,
        "current_word": " ".join(sentence)
    }