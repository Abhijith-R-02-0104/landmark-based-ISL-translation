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

# ================= INIT =================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status")
def status():
    return {"status": "ok"}

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
    min_detection_confidence=0.5,  # 🔥 LOWERED
    min_tracking_confidence=0.5
)

# ================= PARAMS =================
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.60  # 🔥 LOWERED

# ================= STATE =================
sequence = []
sentence = []

class FrameData(BaseModel):
    image: str


@app.post("/predict")
async def predict(data: FrameData):

    global sequence, sentence

    # ===== Decode =====
    try:
        image_data = data.image.split(",")[1]
    except:
        return {"letter": "-", "confidence": 0.0, "current_word": " ".join(sentence)}

    decoded = base64.b64decode(image_data)
    np_arr = np.frombuffer(decoded, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"letter": "-", "confidence": 0.0, "current_word": " ".join(sentence)}

    # 🔥 IMPORTANT FIX (RESIZE)
    frame = cv2.resize(frame, (640, 480))

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ===== MEDIAPIPE =====
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

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

            if len(sequence) > SEQUENCE_LENGTH:
                sequence = sequence[-SEQUENCE_LENGTH:]

    # 🔥 DEBUG
    print("SEQ LEN:", len(sequence))

    # ===== FORCE PREDICTION (KEY CHANGE) =====
    if len(sequence) > 0:

        input_data = np.array(sequence[-SEQUENCE_LENGTH:])

        # pad if needed
        if input_data.shape[0] < SEQUENCE_LENGTH:
            pad = np.zeros((SEQUENCE_LENGTH - input_data.shape[0], 63))
            input_data = np.vstack((pad, input_data))

        input_data = input_data.reshape(1, SEQUENCE_LENGTH, 63)

        prediction = model.predict(input_data, verbose=0)

        predicted_class = labels[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        print("PRED:", predicted_class, confidence)

        # only add to sentence if confident
        if confidence > CONFIDENCE_THRESHOLD:
            if len(sentence) == 0 or sentence[-1] != predicted_class:
                sentence.append(predicted_class)

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