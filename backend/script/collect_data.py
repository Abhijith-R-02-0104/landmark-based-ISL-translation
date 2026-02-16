import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ================= CONFIG =================
FRAMES_PER_SAMPLE = 30

# Get backend directory dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw")

# ================= MEDIAPIPE INIT =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ================= USER INPUT =================
person_name = input("Enter Person Name (Abhi/Namith): ").strip()
gesture_name = input("Enter Gesture Name: ").strip().lower()
num_samples = int(input("Number of Samples to Record: "))

gesture_path = os.path.join(DATA_PATH, person_name, gesture_name)
os.makedirs(gesture_path, exist_ok=True)

# Auto-detect next sample number
existing_samples = len(os.listdir(gesture_path))
starting_index = existing_samples + 1

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

for sample_num in range(starting_index, starting_index + num_samples):

    print(f"\nGet Ready for Sample {sample_num}")
    time.sleep(3)

    sequence = []
    frame_count = 0

    while frame_count < FRAMES_PER_SAMPLE:
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
                    landmarks.append([lm.x, lm.y, lm.z])

                sequence.append(landmarks)
                frame_count += 1

        # Display Info
        cv2.putText(frame, f"Person: {person_name}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)

        cv2.putText(frame, f"Gesture: {gesture_name}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)

        cv2.putText(frame, f"Recording: {frame_count}/{FRAMES_PER_SAMPLE}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    sequence = np.array(sequence)

    # Only save if full sequence captured
    if sequence.shape[0] == FRAMES_PER_SAMPLE:
        save_path = os.path.join(gesture_path, f"sample_{sample_num}.npy")
        np.save(save_path, sequence)
        print(f"Saved: {save_path}")
    else:
        print("Incomplete sequence detected. Sample discarded.")

cap.release()
cv2.destroyAllWindows()

print("\nData Collection Completed.")
