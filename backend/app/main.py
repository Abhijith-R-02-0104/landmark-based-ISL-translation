from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import mediapipe as mp
import numpy as np
import base64
from tensorflow.keras.models import load_model
import os

# ================= INIT APP =================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for now
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

# ================= STATE =================
sequence = []
last_prediction = None
confirm_count = 0
sentence = []

# ================= REQUEST MODEL =================
class FrameData(BaseModel):
    image: str

@app.post("/predict")
async def predict(data: FrameData):
    global sequence, last_prediction, confirm_count, sentence

    # Decode base64 image
    image_data = data.image.split(",")[1]
    decoded = base64.b64decode(image_data)
    np_arr = np.frombuffer(decoded, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ===== MEDIAPIPE PROCESS =====
    results = hands.process(rgb)

    print("Hands detected:", results.multi_hand_landmarks is not None)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            sequence.append(landmarks)

        print("Sequence length:", len(sequence))

    # ===== MODEL PREDICTION =====
    print("Sequence length:", len(sequence))
    
    if len(sequence) == 30:
        input_data = np.array(sequence).reshape(1, 30, 63)
        prediction = model.predict(input_data, verbose=0)

        predicted_class = labels[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        print("Predicted:", predicted_class, "Confidence:", confidence)

        if predicted_class == last_prediction:
            confirm_count += 1
        else:
            confirm_count = 1
            last_prediction = predicted_class

        if confirm_count >= 2:
            if len(sentence) == 0 or sentence[-1] != predicted_class:
                sentence.append(predicted_class)

        sequence = []

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